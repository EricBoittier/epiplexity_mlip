from __future__ import annotations

import difflib
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Literal

TrainingPhase = Literal["all", "teacher", "student"]
TEACHER_PHASE_DONE_FILENAME = "teacher_phase_done.json"

import ase
import jax
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
import pandas as pd
from tqdm import tqdm

from mmml.data.loaders import get_data_statistics
from mmml.data.rmd17 import (
    load_rmd17_npz,
    load_rmd17_official_splits,
    resolve_rmd17_splits_dir,
)
from mmml.generate.sample.compare_ensemble_entropy import (
    _make_soap,
    ase_to_chemcoord,
    construction_table_from_zmat,
    process_ensemble_frames,
)
from mmml.interfaces.chemcoordInterface.interface import patch_chemcoord_for_pandas3
from mmml.physnetjax.physnetjax.analysis.analysis import plot_stats
from mmml.physnetjax.physnetjax.data.batches import prepare_batches_jit
from mmml.physnetjax.physnetjax.models.model import EF
from mmml.physnetjax.physnetjax.training.training import train_model

from src.config import (
    EV_PER_KCAL,
    ExperimentConfig,
    ModelConfig,
    SelectionConfig,
    SoapConfig,
    TeacherNoiseConfig,
    TrainingConfig,
    resolve_save_every_epoch,
)
from src.histogram_metrics import (
    PLOT_STATS_ENERGY_KEYS,
    PLOT_STATS_FORCE_KEYS,
    compute_normalized_histogram,
    extract_array_with_fallback,
    kl_divergence,
    plot_stats_arrays_to_dataset_units,
)
from src.training_resume import epoch_from_checkpoint_path, latest_epoch_checkpoint, latest_checkpoint_epoch_from_paths

patch_chemcoord_for_pandas3()


def get_pool_array(pool: Any, key: str) -> np.ndarray:
    """Return an array from a dict-like or attribute-style pool."""
    candidates = (key, key.lower(), key.upper())

    if hasattr(pool, "keys"):
        keys = set(pool.keys())
        for candidate in candidates:
            if candidate in keys:
                return np.asarray(pool[candidate])

    for candidate in candidates:
        if hasattr(pool, candidate):
            return np.asarray(getattr(pool, candidate))

    raise KeyError(f"Could not find {key!r} in pool")


def subset_arrays(data: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    """Subset arrays whose first dimension matches the dataset length."""
    n_total = len(data["E"])
    out: dict[str, Any] = {}

    for key, value in data.items():
        value_arr = np.asarray(value) if isinstance(value, np.ndarray) else value
        if isinstance(value_arr, np.ndarray) and value_arr.shape[:1] == (n_total,):
            out[key] = value_arr[idx]
        else:
            out[key] = value

    return out


def scan_information_content(
    pool: dict[str, Any],
    *,
    window_size: int,
    stride: int,
    stop: int | None,
    soap: SoapConfig,
) -> pd.DataFrame:
    """Compute information-content metrics for sliding windows over a pool."""
    R = get_pool_array(pool, "R")
    Z = get_pool_array(pool, "Z")
    E = get_pool_array(pool, "E").reshape(len(R), -1)
    F = get_pool_array(pool, "F")

    n_frames = len(R)
    stop = n_frames if stop is None else min(stop, n_frames)

    z0 = Z[0] if Z.ndim == 2 else Z
    ref_atoms = ase.Atoms(numbers=z0, positions=R[0])
    ref_zmat = ase_to_chemcoord(ref_atoms).get_zmat()
    c_table = construction_table_from_zmat(ref_zmat)

    soap_engine = _make_soap(
        list(soap.species),
        soap.rcut,
        soap.nmax,
        soap.lmax,
        soap.sigma,
    )

    rows: list[dict[str, Any]] = []
    starts = range(0, stop - window_size + 1, stride)

    for start in tqdm(starts, desc="Scanning information windows"):
        end = start + window_size
        idx = np.arange(start, end)
        frames = [ase.Atoms(numbers=Z[i] if Z.ndim == 2 else Z, positions=R[i]) for i in idx]

        info = process_ensemble_frames(
            frames,
            label=f"window_{start}_{end}",
            soap_engine=soap_engine,
            c_table=c_table,
        )

        Ei = E[idx]
        Fi = F[idx]
        rows.append(
            {
                "start": start,
                "end": end,
                "indices": idx,
                "E_min": float(Ei.min()),
                "E_max": float(Ei.max()),
                "E_mean": float(Ei.mean()),
                "F_min": float(Fi.min()),
                "F_max": float(Fi.max()),
                "F_abs_mean": float(np.abs(Fi).mean()),
                **info,
            }
        )

    return pd.DataFrame(rows)


def make_information_split(
    info_df: pd.DataFrame,
    *,
    metric: str,
    train_fraction: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Select highest-information windows for training and remaining windows for validation."""
    if metric == "random":
        if seed is None:
            raise ValueError("random window ranking requires seed")
        ranked = info_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    elif metric not in info_df.columns:
        available = ", ".join(map(str, info_df.columns))
        raise ValueError(f"Metric {metric!r} not found in info_df. Available columns: {available}")
    else:
        ranked = info_df.sort_values(metric, ascending=False).reset_index(drop=True)
    n_train_windows = max(1, round(train_fraction * len(ranked)))
    train_windows = ranked.iloc[:n_train_windows]
    valid_windows = ranked.iloc[n_train_windows:]
    train_idx = np.unique(np.concatenate(train_windows["indices"].to_numpy()))
    valid_idx = np.unique(np.concatenate(valid_windows["indices"].to_numpy()))
    valid_idx = np.setdiff1d(valid_idx, train_idx)
    return train_idx, valid_idx, train_windows, valid_windows


def make_random_split(
    n_pool: int,
    *,
    n_train: int,
    n_valid: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_train + n_valid > n_pool:
        raise ValueError(f"Need {n_train + n_valid} samples, but pool only has {n_pool}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pool)
    train_idx = np.sort(perm[:n_train])
    valid_idx = np.sort(perm[n_train : n_train + n_valid])
    return train_idx, valid_idx


def make_selected_data(
    official_train_pool_data: dict[str, Any],
    selection: SelectionConfig,
    training: TrainingConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata: dict[str, Any] = {"selection": asdict(selection)}
    n_pool = len(official_train_pool_data["E"])

    if selection.kind == "random":
        train_idx, valid_idx = make_random_split(
            n_pool,
            n_train=training.n_train,
            n_valid=training.n_valid,
            seed=selection.seed,
        )
        metadata["info_df_path"] = None
        metadata["train_windows_path"] = None
        metadata["valid_windows_path"] = None
    elif selection.kind == "information":
        if selection.metric is None:
            raise ValueError("Information selection requires selection.metric")
        selection_train_fraction = (
            training.n_train / (training.n_train + training.n_valid)
            if selection.train_fraction is None
            else selection.train_fraction
        )
        info_df = scan_information_content(
            official_train_pool_data,
            window_size=selection.window_size,
            stride=selection.stride,
            stop=selection.stop,
            soap=selection.soap,
        )
        train_idx, valid_idx, train_windows, valid_windows = make_information_split(
            info_df,
            metric=selection.metric,
            train_fraction=selection_train_fraction,
            seed=selection.seed,
        )
        metadata["info_df"] = info_df
        metadata["train_windows"] = train_windows
        metadata["valid_windows"] = valid_windows
    else:
        raise ValueError(f"Unknown selection kind: {selection.kind}")

    train_data = subset_arrays(official_train_pool_data, train_idx)
    valid_data = subset_arrays(official_train_pool_data, valid_idx)
    train_data["idx"] = np.asarray(train_idx)
    valid_data["idx"] = np.asarray(valid_idx)
    metadata["n_train"] = int(len(train_data["E"]))
    metadata["n_valid"] = int(len(valid_data["E"]))
    metadata["train_idx"] = train_idx
    metadata["valid_idx"] = valid_idx
    return train_data, valid_data, metadata


def add_teacher_label_noise(
    data: dict[str, Any],
    *,
    scale: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Add Gaussian noise to E and F proportional to each array's standard deviation."""
    if scale <= 0.0:
        raise ValueError(f"teacher noise scale must be positive, got {scale}")

    out = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in data.items()}
    rng = np.random.default_rng(seed)

    energy = np.asarray(out["E"], dtype=float)
    forces = np.asarray(out["F"], dtype=float)
    energy_std = float(np.std(energy))
    force_std = float(np.std(forces))

    if energy_std > 0.0:
        out["E"] = energy + scale * energy_std * rng.standard_normal(energy.shape)
    if force_std > 0.0:
        out["F"] = forces + scale * force_std * rng.standard_normal(forces.shape)

    metadata = {
        "scale": float(scale),
        "energy_std": energy_std,
        "force_std": force_std,
        "energy_noise_std": float(scale * energy_std),
        "force_noise_std": float(scale * force_std),
    }
    return out, metadata


def center_energies_on_train(
    train_data: dict[str, Any],
    valid_data: dict[str, Any],
    test_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float]:
    train_e_mean = float(np.mean(train_data["E"]))
    train_data["E"] = train_data["E"] - train_e_mean
    valid_data["E"] = valid_data["E"] - train_e_mean
    test_data["E"] = test_data["E"] - train_e_mean
    return train_data, valid_data, test_data, train_e_mean


def teacher_predict_dataset(
    *,
    model: EF,
    ema_params: Any,
    dataset: dict[str, Any],
    batch_size: int,
    num_atoms: int,
    seed: int,
    set_name: str,
    convert_to_ev: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    key_eval, _ = jax.random.split(jax.random.PRNGKey(seed))
    batches = prepare_batches_jit(
        key_eval,
        dataset,
        batch_size,
        data_keys=("R", "Z", "F", "E", "N"),
        num_atoms=num_atoms,
    )
    stats = plot_stats(
        batches,
        model,
        ema_params,
        _set=set_name,
        do_kde=False,
        batch_size=batch_size,
        do_plot=False,
    )
    e_pred, e_source = extract_array_with_fallback(
        stats,
        preferred_keys=PLOT_STATS_ENERGY_KEYS,
        fallback=np.asarray(dataset["E"]),
        expected_shape=np.asarray(dataset["E"]).shape,
    )
    f_pred, f_source = extract_array_with_fallback(
        stats,
        preferred_keys=PLOT_STATS_FORCE_KEYS,
        fallback=np.asarray(dataset["F"]),
        expected_shape=np.asarray(dataset["F"]).shape,
    )
    if e_source == "fallback_ground_truth" or f_source == "fallback_ground_truth":
        print(
            f"WARNING [{set_name}]: plot_stats missing model predictions "
            f"(energy={e_source}, forces={f_source}); distillation/KL metrics use ground truth."
        )
    reference_e = np.asarray(dataset["E"])
    reference_f = np.asarray(dataset["F"])
    e_pred, f_pred = plot_stats_arrays_to_dataset_units(
        e_pred,
        f_pred,
        e_source=e_source,
        f_source=f_source,
        convert_to_ev=convert_to_ev,
        reference_e=reference_e,
        reference_f=reference_f,
    )
    if f_source != "fallback_ground_truth":
        ref_scale = float(np.median(np.abs(reference_f)))
        pred_scale = float(np.median(np.abs(f_pred)))
        if ref_scale > 0.0 and pred_scale > 0.0:
            ratio = pred_scale / ref_scale
            if ratio < 0.25 or ratio > 4.0:
                print(
                    f"WARNING [{set_name}]: teacher force scale mismatch after unit alignment "
                    f"(median |F_pred|={pred_scale:.4g}, median |F_ref|={ref_scale:.4g}, ratio={ratio:.4g}). "
                    "Check plot_stats / convert_to_ev handling."
                )
    distill_data = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in dataset.items()}
    distill_data["E"] = e_pred
    distill_data["F"] = f_pred
    return distill_data, {
        "energy_source": e_source,
        "forces_source": f_source,
        "units": "eV_eV_per_A" if convert_to_ev else "kcal_mol_kcal_per_mol_A",
    }


def build_model(model_cfg: ModelConfig, data: dict[str, Any], num_atoms: int) -> EF:
    return EF(
        features=model_cfg.features,
        max_degree=model_cfg.max_degree,
        num_iterations=model_cfg.num_iterations,
        num_basis_functions=model_cfg.num_basis_functions,
        cutoff=model_cfg.cutoff,
        natoms=num_atoms,
        max_atomic_number=int(np.asarray(data["Z"]).max()),
        charges=model_cfg.charges,
        zbl=model_cfg.zbl,
    )


def latest_run_dir(ckpt_root: Path, run_name: str) -> Path | None:
    run_dirs = sorted(ckpt_root.glob(f"{run_name}-*"))
    return run_dirs[-1] if run_dirs else None


def latest_checkpoint_epoch(run_ckpt_dir: Path | None) -> int | None:
    """Return the highest saved epoch index for a run, or None if no checkpoints."""
    path_epoch = latest_checkpoint_epoch_from_paths(run_ckpt_dir)
    if path_epoch is None or run_ckpt_dir is None:
        return path_epoch
    latest_epoch_path = latest_epoch_checkpoint(run_ckpt_dir)
    if latest_epoch_path is None:
        return path_epoch
    try:
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(latest_epoch_path.resolve())
        stored_epoch = restored.get("epoch")
        if stored_epoch is not None:
            return max(path_epoch, int(stored_epoch))
    except Exception:
        pass
    return path_epoch


def is_training_complete(run_ckpt_dir: Path | None, target_epochs: int) -> bool:
    latest_epoch = latest_checkpoint_epoch(run_ckpt_dir)
    return latest_epoch is not None and latest_epoch >= target_epochs


def maybe_resume_ema_params(run_ckpt_dir: Path | None) -> tuple[Any | None, float | None]:
    """Restore EMA params and best loss from latest checkpoint if possible."""
    if run_ckpt_dir is None:
        return None, None

    latest_epoch = latest_epoch_checkpoint(run_ckpt_dir)
    if latest_epoch is None:
        return None, None

    restored = orbax.checkpoint.PyTreeCheckpointer().restore(latest_epoch.resolve())
    ema_params = (
        restored.get("ema_params")
        or restored.get("params_ema")
        or restored.get("best_params")
        or restored.get("params")
    )
    if ema_params is None:
        return None, None

    best_loss_val = restored.get("best_loss")
    best_loss = float(best_loss_val) if best_loss_val is not None else float("nan")
    return ema_params, best_loss


def train_or_resume_model(
    *,
    phase_label: str,
    resume: bool,
    run_ckpt_dir: Path | None,
    target_epochs: int,
    key: Any,
    model: EF,
    train_data: dict[str, Any],
    valid_data: dict[str, Any],
    learning_rate: float,
    batch_size: int,
    num_atoms: int,
    energy_weight: float,
    forces_weight: float,
    run_name: str,
    ckpt_dir: Path,
    save_every_epoch: bool | int,
    log_tb: bool,
    print_freq: int,
) -> tuple[Any, float]:
    """Run train_model, skipping only when the latest checkpoint reached target_epochs."""
    latest_epoch = latest_checkpoint_epoch(run_ckpt_dir)

    if resume and is_training_complete(run_ckpt_dir, target_epochs):
        ema_params, best_loss = maybe_resume_ema_params(run_ckpt_dir)
        if ema_params is not None:
            print(
                f"{phase_label}: training complete at epoch {latest_epoch} "
                f"(target {target_epochs}); skipping train_model"
            )
            return ema_params, float(best_loss)

    restart = bool(resume and latest_epoch is not None and latest_epoch > 0)
    if restart:
        print(
            f"{phase_label}: resuming train_model from epoch {latest_epoch} "
            f"toward {target_epochs} (restart=True)"
        )
    else:
        print(f"{phase_label}: starting train_model for {target_epochs} epochs (restart=False)")

    ema_params, best_loss = train_model(
        key=key,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        num_epochs=target_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_atoms=num_atoms,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        data_keys=("R", "Z", "F", "E", "N"),
        name=run_name,
        ckpt_dir=ckpt_dir,
        best=True,
        restart=restart,
        save_every_epoch=save_every_epoch,
        batch_method="default",
        log_tb=log_tb,
        print_freq=print_freq,
    )
    return ema_params, float(best_loss)


def load_distillation_targets(
    run_output_dir: Path,
    train_data: dict[str, Any],
    valid_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild distill train/valid dicts from a saved teacher_distillation_targets.npz."""
    npz_path = run_output_dir / "teacher_distillation_targets.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing distillation targets: {npz_path}")
    saved = np.load(npz_path)
    teacher_train = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in train_data.items()}
    teacher_valid = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in valid_data.items()}
    teacher_train["E"] = saved["train_E"]
    teacher_train["F"] = saved["train_F"]
    teacher_valid["E"] = saved["valid_E"]
    teacher_valid["F"] = saved["valid_F"]
    return teacher_train, teacher_valid


def _experiment_training_complete(
    *,
    config: ExperimentConfig,
    ckpt_root: Path,
    run_name: str,
    run_output_dir: Path,
) -> bool:
    student_run_name = f"{run_name}_student"
    teacher_dir = latest_run_dir(ckpt_root, run_name)
    student_dir = latest_run_dir(ckpt_root, student_run_name)
    if not is_training_complete(teacher_dir, config.training.num_epochs):
        return False
    if not is_training_complete(student_dir, config.student_epochs):
        return False
    if not (run_output_dir / "teacher_distillation_targets.npz").exists():
        return False
    return True


def _signature_path(value: Path | str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(Path(text).resolve())


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, Path):
        return _signature_path(value)
    if isinstance(value, dict):
        return {k: _normalize_signature_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_normalize_signature_value(v) for v in value]
    if isinstance(value, list):
        return [_normalize_signature_value(v) for v in value]
    return value


def _normalize_resume_signature(signature: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_signature_value(signature)
    training = normalized.get("training")
    if isinstance(training, dict):
        training.pop("ckpt_root", None)
    dataset = normalized.get("dataset")
    if isinstance(dataset, dict):
        dataset.pop("resolved_splits_dir", None)
        if "data_path" in dataset:
            dataset["data_path"] = _signature_path(dataset["data_path"])
        if "rmd17_splits_dir" in dataset:
            dataset["rmd17_splits_dir"] = _signature_path(dataset["rmd17_splits_dir"])
    return normalized


def _resume_signatures_match(existing: dict[str, Any], current: dict[str, Any]) -> bool:
    return _normalize_resume_signature(existing) == current


def _format_resume_signature_diff(existing: dict[str, Any], current: dict[str, Any]) -> str:
    stored = json.dumps(_normalize_resume_signature(existing), indent=2, sort_keys=True).splitlines()
    live = json.dumps(current, indent=2, sort_keys=True).splitlines()
    diff = difflib.unified_diff(stored, live, fromfile="stored resume_signature.json", tofile="current run", lineterm="")
    text = "\n".join(diff)
    return text if text else "(no diff lines produced; compare normalized payloads manually)"


def _raise_resume_signature_mismatch(existing: dict[str, Any], current: dict[str, Any]) -> None:
    diff = _format_resume_signature_diff(existing, current)
    raise ValueError(
        "Resume safety check failed: existing run signature does not match current settings. "
        "Disable resume for this run, remove experiment_metadata for this run_name, or align config. "
        f"Diff:\n{diff}"
    )


def _resume_signature(
    *,
    config: ExperimentConfig,
    selection: SelectionConfig,
    num_atoms: int,
) -> dict[str, Any]:
    training = asdict(config.training)
    training.pop("ckpt_root", None)
    return _normalize_resume_signature(
        {
            "molecule": config.molecule,
            "dataset": {
                "data_path": _signature_path(config.dataset.data_path),
                "rmd17_splits_dir": _signature_path(config.dataset.rmd17_splits_dir),
                "split_id": int(config.dataset.split_id),
                "max_structures": config.dataset.max_structures,
                "convert_to_ev": bool(config.dataset.convert_to_ev),
            },
            "selection": asdict(selection),
            "training": training,
            "model": asdict(config.model),
            "student_model": asdict(config.student_model),
            "student_epochs": int(config.student_epochs),
            "student_learning_rate": float(config.student_learning_rate),
            "teacher_noise": (
                None
                if config.teacher_noise is None
                else {"scale": float(config.teacher_noise.scale), "run_suffix": config.teacher_noise.run_suffix}
            ),
            "num_atoms": int(num_atoms),
        }
    )


def evaluate_test_set(
    *,
    model: EF,
    ema_params: Any,
    test_data: dict[str, Any],
    batch_size: int,
    num_atoms: int,
    seed: int,
    run_name: str,
    split_id: int,
    splits_dir: Path,
    run_ckpt_dir: Path | None,
    eval_dir: Path,
) -> dict[str, Any]:
    key_eval, _ = jax.random.split(jax.random.PRNGKey(seed))
    test_batches = prepare_batches_jit(
        key_eval,
        test_data,
        batch_size,
        data_keys=("R", "Z", "F", "E", "N"),
        num_atoms=num_atoms,
    )
    print(f"Test batches: {len(test_batches)} × batch_size={batch_size}")
    stats = plot_stats(
        test_batches,
        model,
        ema_params,
        _set=f"Test | {run_name}",
        do_kde=False,
        batch_size=batch_size,
        do_plot=True,
    )
    metrics = {
        "n_test": int(len(test_data["E"])),
        "split_id": split_id,
        "splits_dir": str(splits_dir),
        "checkpoint_run": str(run_ckpt_dir) if run_ckpt_dir else None,
        "energy": {
            "mae_kcal_mol": float(stats["E_mae"]),
            "rmse_kcal_mol": float(stats["E_rmse"]),
            "mae_ev": float(stats["E_mae"]) * EV_PER_KCAL,
            "rmse_ev": float(stats["E_rmse"]) * EV_PER_KCAL,
        },
        "forces": {
            "mae_kcal_mol_A": float(stats["F_mae"]),
            "rmse_kcal_mol_A": float(stats["F_rmse"]),
            "mae_ev_A": float(stats["F_mae"]) * EV_PER_KCAL,
            "rmse_ev_A": float(stats["F_rmse"]) * EV_PER_KCAL,
        },
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    fig_path = eval_dir / "parity_test.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    return metrics


def load_experiment_data(config: ExperimentConfig) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, int]:
    raw = np.load(config.dataset.data_path, allow_pickle=True)
    print("NPZ keys:", list(raw.keys()))
    for key in raw.keys():
        value = raw[key]
        print(f"  {key:20s} shape={getattr(value, 'shape', '?')}")
    data = load_rmd17_npz(
        config.dataset.data_path,
        convert_to_ev=config.dataset.convert_to_ev,
        max_structures=config.dataset.max_structures,
    )
    num_atoms = int(data["N"].max())
    print(f"\nMolecule size: {num_atoms} atoms")
    print(f"Loaded {len(data['E'])} structures")
    print(get_data_statistics(data))
    splits_dir = resolve_rmd17_splits_dir(
        config.dataset.data_path,
        config.dataset.rmd17_splits_dir,
    )
    official_train_pool, test_idx = load_rmd17_official_splits(splits_dir, config.dataset.split_id)
    n_total = len(data["E"])
    if official_train_pool.max() >= n_total or test_idx.max() >= n_total:
        raise ValueError("Split indices out of range for loaded NPZ")
    test_data = subset_arrays(data, test_idx)
    official_train_pool_data = subset_arrays(data, official_train_pool)
    return data, official_train_pool_data, test_data, splits_dir, num_atoms


def collect_checkpoint_valid_losses(ckpt_dir: Path, every: int = 10) -> pd.DataFrame:
    ckpts = [p for p in ckpt_dir.glob("*") if p.is_dir() or p.exists()]

    def epoch_from_path(path: Path) -> int:
        return epoch_from_checkpoint_path(path)

    ckpts = sorted(ckpts, key=epoch_from_path)
    rows: list[dict[str, Any]] = []
    for path in ckpts[::every]:
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(path.resolve())
        objectives = restored.get("objectives", {})
        rows.append(
            {
                "checkpoint": str(path),
                "epoch": int(restored.get("epoch", epoch_from_path(path))),
                "best_loss": float(restored.get("best_loss", np.nan)),
                "valid_loss": float(objectives.get("valid_loss", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def summarize_validation_curves(curves: dict[str, Iterable[float]]) -> pd.DataFrame:
    rows = []
    for name, values in curves.items():
        val_loss = np.asarray(list(values), dtype=float)
        baseline = float(np.mean(val_loss[-2:])) if len(val_loss) >= 2 else float(val_loss[-1])
        excess = np.maximum(val_loss - baseline, 0.0)
        rows.append(
            {
                "name": name,
                "n_points": int(len(val_loss)),
                "final_valid_loss": float(val_loss[-1]),
                "baseline_last2_mean": baseline,
                "auc_excess_valid_loss": float(np.trapezoid(excess)),
                "sum_above_final": float(np.sum(val_loss[val_loss > val_loss[-1]])),
            }
        )
    return pd.DataFrame(rows)


def train_one_experiment(
    *,
    config: ExperimentConfig,
    selection: SelectionConfig,
    data: dict[str, Any],
    official_train_pool_data: dict[str, Any],
    test_data_uncentered: dict[str, Any],
    splits_dir: Path,
    num_atoms: int,
    resume: bool = False,
    phase: TrainingPhase = "all",
) -> dict[str, Any]:
    run_name = selection.run_name(config.molecule, config.dataset.split_id)
    run_output_dir = config.training.ckpt_root / "experiment_metadata" / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    result_summary_path = run_output_dir / "result_summary.json"
    teacher_phase_done_path = run_output_dir / TEACHER_PHASE_DONE_FILENAME
    resume_signature_path = run_output_dir / "resume_signature.json"
    current_signature = _resume_signature(
        config=config,
        selection=selection,
        num_atoms=num_atoms,
    )
    if resume and phase == "teacher" and teacher_phase_done_path.exists():
        if resume_signature_path.exists():
            with open(resume_signature_path) as f:
                existing_signature = json.load(f)
            if not _resume_signatures_match(existing_signature, current_signature):
                _raise_resume_signature_mismatch(existing_signature, current_signature)
        with open(teacher_phase_done_path) as f:
            return json.load(f)
    if resume and phase in ("all", "student") and result_summary_path.exists():
        if _experiment_training_complete(
            config=config,
            ckpt_root=config.training.ckpt_root,
            run_name=run_name,
            run_output_dir=run_output_dir,
        ):
            if resume_signature_path.exists():
                with open(resume_signature_path) as f:
                    existing_signature = json.load(f)
                if not _resume_signatures_match(existing_signature, current_signature):
                    _raise_resume_signature_mismatch(existing_signature, current_signature)
            with open(result_summary_path) as f:
                return json.load(f)
        print(
            f"Resume: {result_summary_path.name} exists but training is incomplete; "
            "continuing teacher/student training."
        )

    selection_metadata_path = run_output_dir / "selection_metadata.json"
    if resume and selection_metadata_path.exists():
        with open(selection_metadata_path) as f:
            loaded = json.load(f)
        run_uuid = str(loaded.get("run_uuid") or uuid.uuid4())
    else:
        run_uuid = str(uuid.uuid4())

    run_ckpt_dir = latest_run_dir(config.training.ckpt_root, run_name)
    if resume and run_ckpt_dir is not None and not resume_signature_path.exists():
        raise ValueError(
            "Resume safety check failed: found teacher checkpoints but no resume_signature.json. "
            "Disable resume for this run or remove the partial checkpoint directory."
        )
    if resume and resume_signature_path.exists():
        with open(resume_signature_path) as f:
            existing_signature = json.load(f)
        if not _resume_signatures_match(existing_signature, current_signature):
            _raise_resume_signature_mismatch(existing_signature, current_signature)
    with open(resume_signature_path, "w") as f:
        json.dump(current_signature, f, indent=2)
    train_data, valid_data, selection_metadata = make_selected_data(
        official_train_pool_data, selection, config.training
    )
    test_data = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in test_data_uncentered.items()}
    train_data, valid_data, test_data, train_e_mean = center_energies_on_train(train_data, valid_data, test_data)
    teacher_noise_metadata: dict[str, Any] | None = None
    if config.teacher_noise is not None:
        train_data, train_noise = add_teacher_label_noise(
            train_data,
            scale=config.teacher_noise.scale,
            seed=selection.seed + 10_000,
        )
        valid_data, valid_noise = add_teacher_label_noise(
            valid_data,
            scale=config.teacher_noise.scale,
            seed=selection.seed + 10_001,
        )
        teacher_noise_metadata = {
            "scale": float(config.teacher_noise.scale),
            "run_suffix": config.teacher_noise.run_suffix,
            "train": train_noise,
            "valid": valid_noise,
        }
    np.save(run_output_dir / "train_idx.npy", selection_metadata["train_idx"])
    np.save(run_output_dir / "valid_idx.npy", selection_metadata["valid_idx"])
    if "info_df" in selection_metadata:
        info_df = selection_metadata.pop("info_df")
        train_windows = selection_metadata.pop("train_windows")
        valid_windows = selection_metadata.pop("valid_windows")
        info_df.to_pickle(run_output_dir / "info_df.pkl")
        train_windows.to_pickle(run_output_dir / "train_windows.pkl")
        valid_windows.to_pickle(run_output_dir / "valid_windows.pkl")
        selection_metadata["info_df_path"] = str(run_output_dir / "info_df.pkl")
        selection_metadata["train_windows_path"] = str(run_output_dir / "train_windows.pkl")
        selection_metadata["valid_windows_path"] = str(run_output_dir / "valid_windows.pkl")
    with open(run_output_dir / "selection_metadata.json", "w") as f:
        json.dump(
            {
                **selection_metadata,
                "train_e_mean": train_e_mean,
                "splits_dir": str(splits_dir),
                "run_uuid": run_uuid,
                "teacher_noise": teacher_noise_metadata,
            },
            f,
            indent=2,
            default=str,
        )
    key = jax.random.PRNGKey(selection.seed)
    model = build_model(config.model, data, num_atoms)
    run_ckpt_dir = latest_run_dir(config.training.ckpt_root, run_name)
    save_every = resolve_save_every_epoch(config.training)
    if phase in ("all", "teacher"):
        ema_params, best_loss = train_or_resume_model(
            phase_label=f"Teacher | {run_name}",
            resume=resume,
            run_ckpt_dir=run_ckpt_dir,
            target_epochs=config.training.num_epochs,
            key=key,
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            num_atoms=num_atoms,
            energy_weight=config.training.energy_weight,
            forces_weight=config.training.forces_weight,
            run_name=run_name,
            ckpt_dir=config.training.ckpt_root,
            save_every_epoch=save_every,
            log_tb=config.training.log_tb,
            print_freq=config.training.print_freq,
        )
        run_ckpt_dir = latest_run_dir(config.training.ckpt_root, run_name)
        teacher_metrics = evaluate_test_set(
            model=model,
            ema_params=ema_params,
            test_data=test_data,
            batch_size=config.training.batch_size,
            num_atoms=num_atoms,
            seed=selection.seed + 1,
            run_name=run_name,
            split_id=config.dataset.split_id,
            splits_dir=splits_dir,
            run_ckpt_dir=run_ckpt_dir,
            eval_dir=config.training.ckpt_root / f"test_eval_{run_name}",
        )
        teacher_train_targets, teacher_train_meta = teacher_predict_dataset(
            model=model,
            ema_params=ema_params,
            dataset=train_data,
            batch_size=config.training.batch_size,
            num_atoms=num_atoms,
            seed=selection.seed + 2,
            set_name=f"Teacher train reevaluation | {run_name}",
            convert_to_ev=config.dataset.convert_to_ev,
        )
        teacher_valid_targets, teacher_valid_meta = teacher_predict_dataset(
            model=model,
            ema_params=ema_params,
            dataset=valid_data,
            batch_size=config.training.batch_size,
            num_atoms=num_atoms,
            seed=selection.seed + 3,
            set_name=f"Teacher valid reevaluation | {run_name}",
            convert_to_ev=config.dataset.convert_to_ev,
        )
        np.savez(
            run_output_dir / "teacher_distillation_targets.npz",
            train_E=teacher_train_targets["E"],
            train_F=teacher_train_targets["F"],
            train_idx=teacher_train_targets.get("idx", np.arange(len(teacher_train_targets["E"]))),
            valid_E=teacher_valid_targets["E"],
            valid_F=teacher_valid_targets["F"],
            valid_idx=teacher_valid_targets.get("idx", np.arange(len(teacher_valid_targets["E"]))),
        )
        teacher_phase_result = {
            "run_name": run_name,
            "run_uuid": run_uuid,
            "phase": "teacher",
            "best_loss": float(best_loss),
            "run_ckpt_dir": str(run_ckpt_dir) if run_ckpt_dir else None,
            "metadata_dir": str(run_output_dir),
            "teacher_epoch": latest_checkpoint_epoch(run_ckpt_dir),
            "teacher": {
                "best_loss": float(best_loss),
                "run_ckpt_dir": str(run_ckpt_dir) if run_ckpt_dir else None,
                "label_noise": teacher_noise_metadata,
                "test_metrics": teacher_metrics,
                "train_target_source": teacher_train_meta,
                "valid_target_source": teacher_valid_meta,
            },
            **teacher_metrics,
        }
        with open(teacher_phase_done_path, "w") as f:
            json.dump(teacher_phase_result, f, indent=2, default=str)
        if phase == "teacher":
            return teacher_phase_result
    else:
        run_ckpt_dir = latest_run_dir(config.training.ckpt_root, run_name)
        if not is_training_complete(run_ckpt_dir, config.training.num_epochs):
            raise ValueError(
                f"Student phase requires completed teacher training for {run_name}; "
                f"latest epoch {latest_checkpoint_epoch(run_ckpt_dir)} "
                f"< target {config.training.num_epochs}"
            )
        ema_params, best_loss = maybe_resume_ema_params(run_ckpt_dir)
        if ema_params is None:
            raise ValueError(f"Teacher checkpoints exist but could not restore EMA params for {run_name}")
        if teacher_phase_done_path.exists():
            with open(teacher_phase_done_path) as f:
                teacher_phase = json.load(f)
            teacher_metrics = teacher_phase.get("teacher", {}).get("test_metrics", {})
            best_loss = float(teacher_phase.get("best_loss", best_loss or float("nan")))
        else:
            teacher_metrics = {}
        teacher_train_targets, teacher_valid_targets = load_distillation_targets(
            run_output_dir, train_data, valid_data
        )
        teacher_train_meta = {"energy_source": "npz", "forces_source": "npz", "units": "eV_eV_per_A"}
        teacher_valid_meta = dict(teacher_train_meta)

    student_run_name = f"{run_name}_student"
    student_key = jax.random.PRNGKey(selection.seed + 1000)
    student_model = build_model(config.student_model, data, num_atoms)
    student_ckpt_dir = latest_run_dir(config.training.ckpt_root, student_run_name)
    student_ema_params, student_best_loss = train_or_resume_model(
        phase_label=f"Student | {student_run_name}",
        resume=resume,
        run_ckpt_dir=student_ckpt_dir,
        target_epochs=config.student_epochs,
        key=student_key,
        model=student_model,
        train_data=teacher_train_targets,
        valid_data=teacher_valid_targets,
        learning_rate=config.student_learning_rate,
        batch_size=config.training.batch_size,
        num_atoms=num_atoms,
        energy_weight=config.training.energy_weight,
        forces_weight=config.training.forces_weight,
        run_name=student_run_name,
        ckpt_dir=config.training.ckpt_root,
        save_every_epoch=save_every,
        log_tb=config.training.log_tb,
        print_freq=config.training.print_freq,
    )
    student_ckpt_dir = latest_run_dir(config.training.ckpt_root, student_run_name)
    student_metrics = evaluate_test_set(
        model=student_model,
        ema_params=student_ema_params,
        test_data=test_data,
        batch_size=config.training.batch_size,
        num_atoms=num_atoms,
        seed=selection.seed + 1001,
        run_name=student_run_name,
        split_id=config.dataset.split_id,
        splits_dir=splits_dir,
        run_ckpt_dir=student_ckpt_dir,
        eval_dir=config.training.ckpt_root / f"test_eval_{student_run_name}",
    )
    teacher_test_pred, teacher_test_meta = teacher_predict_dataset(
        model=model,
        ema_params=ema_params,
        dataset=test_data,
        batch_size=config.training.batch_size,
        num_atoms=num_atoms,
        seed=selection.seed + 4,
        set_name=f"Teacher test reevaluation | {run_name}",
        convert_to_ev=config.dataset.convert_to_ev,
    )
    student_test_pred, student_test_meta = teacher_predict_dataset(
        model=student_model,
        ema_params=student_ema_params,
        dataset=test_data,
        batch_size=config.training.batch_size,
        num_atoms=num_atoms,
        seed=selection.seed + 1002,
        set_name=f"Student test reevaluation | {student_run_name}",
        convert_to_ev=config.dataset.convert_to_ev,
    )
    teacher_force = np.asarray(teacher_test_pred["F"]).reshape(-1)
    student_force = np.asarray(student_test_pred["F"]).reshape(-1)
    force_min = float(min(np.min(teacher_force), np.min(student_force)))
    force_max = float(max(np.max(teacher_force), np.max(student_force)))
    if np.isclose(force_min, force_max):
        force_max = force_min + 1e-6
    n_bins = 128
    teacher_hist, edges = compute_normalized_histogram(teacher_force, bins=n_bins, vmin=force_min, vmax=force_max)
    student_hist, _ = compute_normalized_histogram(student_force, bins=n_bins, vmin=force_min, vmax=force_max)
    kl_t_to_s = kl_divergence(teacher_hist, student_hist)
    kl_s_to_t = kl_divergence(student_hist, teacher_hist)
    student_valid_curve_df = (
        collect_checkpoint_valid_losses(student_ckpt_dir, every=1)
        if student_ckpt_dir is not None
        else pd.DataFrame()
    )
    student_auc_excess_valid_loss = float("nan")
    if not student_valid_curve_df.empty and "valid_loss" in student_valid_curve_df.columns:
        valid_curve = student_valid_curve_df["valid_loss"].dropna().to_numpy(dtype=float)
        if len(valid_curve) > 0:
            summary_df = summarize_validation_curves({"student_valid_loss": valid_curve})
            student_auc_excess_valid_loss = float(summary_df.iloc[0]["auc_excess_valid_loss"])
    result = {
        "run_name": run_name,
        "run_uuid": run_uuid,
        "best_loss": float(best_loss),
        "run_ckpt_dir": str(run_ckpt_dir) if run_ckpt_dir else None,
        "metadata_dir": str(run_output_dir),
        "teacher": {
            "best_loss": float(best_loss),
            "run_ckpt_dir": str(run_ckpt_dir) if run_ckpt_dir else None,
            "label_noise": teacher_noise_metadata,
            "test_metrics": teacher_metrics,
            "train_target_source": teacher_train_meta,
            "valid_target_source": teacher_valid_meta,
            "test_target_source": teacher_test_meta,
        },
        "student": {
            "run_name": student_run_name,
            "best_loss": float(student_best_loss),
            "run_ckpt_dir": str(student_ckpt_dir) if student_ckpt_dir else None,
            "test_metrics": student_metrics,
            "test_target_source": student_test_meta,
            "auc_excess_valid_loss": student_auc_excess_valid_loss,
        },
        "distillation_metrics": {
            "kl_teacher_to_student_test_force_hist": kl_t_to_s,
            "kl_student_to_teacher_test_force_hist": kl_s_to_t,
            "force_hist_bins": n_bins,
            "force_hist_range": [force_min, force_max],
            "teacher_hist_sum": float(teacher_hist.sum()),
            "student_hist_sum": float(student_hist.sum()),
            "teacher_hist": teacher_hist.tolist(),
            "student_hist": student_hist.tolist(),
            "hist_bin_edges": edges.tolist(),
        },
        **teacher_metrics,
    }
    with open(result_summary_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def run_experiments(config: ExperimentConfig) -> None:
    config.training.ckpt_root.mkdir(parents=True, exist_ok=True)
    print("JAX devices:", jax.devices())
    data, official_train_pool_data, test_data, splits_dir, num_atoms = load_experiment_data(config)
    all_results = []
    for selection in config.selections:
        result = train_one_experiment(
            config=config,
            selection=selection,
            data=data,
            official_train_pool_data=official_train_pool_data,
            test_data_uncentered=test_data,
            splits_dir=splits_dir,
            num_atoms=num_atoms,
            resume=False,
        )
        all_results.append(result)
        results_path = config.training.ckpt_root / "experiment_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Updated {results_path}")
