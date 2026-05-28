from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

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

from src.config import EV_PER_KCAL, ExperimentConfig, ModelConfig, SelectionConfig, SoapConfig, TrainingConfig

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
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Select highest-information windows for training and remaining windows for validation."""
    if metric not in info_df.columns:
        available = ", ".join(map(str, info_df.columns))
        raise ValueError(f"Metric {metric!r} not found in info_df. Available columns: {available}")

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


def _extract_array_with_fallback(
    stats: dict[str, Any],
    preferred_keys: tuple[str, ...],
    fallback: np.ndarray,
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, str]:
    for key in preferred_keys:
        if key in stats:
            arr = np.asarray(stats[key])
            if arr.size == int(np.prod(expected_shape)):
                return arr.reshape(expected_shape), key
            if arr.shape == expected_shape:
                return arr, key
    return np.asarray(fallback).reshape(expected_shape), "fallback_ground_truth"


def teacher_predict_dataset(
    *,
    model: EF,
    ema_params: Any,
    dataset: dict[str, Any],
    batch_size: int,
    num_atoms: int,
    seed: int,
    set_name: str,
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
    e_pred, e_source = _extract_array_with_fallback(
        stats,
        preferred_keys=("E_pred", "E_preds", "E_hat", "E_model", "pred_E"),
        fallback=np.asarray(dataset["E"]),
        expected_shape=np.asarray(dataset["E"]).shape,
    )
    f_pred, f_source = _extract_array_with_fallback(
        stats,
        preferred_keys=("F_pred", "F_preds", "F_hat", "F_model", "pred_F"),
        fallback=np.asarray(dataset["F"]),
        expected_shape=np.asarray(dataset["F"]).shape,
    )
    distill_data = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in dataset.items()}
    distill_data["E"] = e_pred
    distill_data["F"] = f_pred
    return distill_data, {"energy_source": e_source, "forces_source": f_source}


def compute_normalized_histogram(
    values: np.ndarray,
    *,
    bins: int,
    vmin: float,
    vmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, range=(vmin, vmax))
    hist = hist.astype(float)
    denom = float(hist.sum())
    if denom <= 0.0:
        return np.full_like(hist, 1.0 / len(hist), dtype=float), edges
    return hist / denom, edges


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


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


def latest_epoch_checkpoint(run_ckpt_dir: Path) -> Path | None:
    """Return the latest epoch checkpoint directory inside a run directory."""
    epoch_dirs = [p for p in run_ckpt_dir.glob("epoch-*") if p.is_dir()]
    if not epoch_dirs:
        return None

    def epoch_from_path(path: Path) -> int:
        try:
            return int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    epoch_dirs = sorted(epoch_dirs, key=epoch_from_path)
    return epoch_dirs[-1]


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


def _resume_signature(
    *,
    config: ExperimentConfig,
    selection: SelectionConfig,
    splits_dir: Path,
    num_atoms: int,
) -> dict[str, Any]:
    return {
        "molecule": config.molecule,
        "dataset": {
            "data_path": str(config.dataset.data_path),
            "rmd17_splits_dir": str(config.dataset.rmd17_splits_dir) if config.dataset.rmd17_splits_dir else None,
            "split_id": int(config.dataset.split_id),
            "max_structures": config.dataset.max_structures,
            "convert_to_ev": bool(config.dataset.convert_to_ev),
            "resolved_splits_dir": str(splits_dir),
        },
        "selection": asdict(selection),
        "training": asdict(config.training),
        "model": asdict(config.model),
        "student_model": asdict(config.student_model),
        "student_epochs": int(config.student_epochs),
        "student_learning_rate": float(config.student_learning_rate),
        "num_atoms": int(num_atoms),
    }


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
        return int(path.name.split("-")[-1])

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
) -> dict[str, Any]:
    run_uuid = str(uuid.uuid4())
    run_name = selection.run_name(config.molecule, config.dataset.split_id)
    run_output_dir = config.training.ckpt_root / "experiment_metadata" / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    result_summary_path = run_output_dir / "result_summary.json"
    resume_signature_path = run_output_dir / "resume_signature.json"
    current_signature = _resume_signature(
        config=config,
        selection=selection,
        splits_dir=splits_dir,
        num_atoms=num_atoms,
    )
    if resume and resume_signature_path.exists():
        with open(resume_signature_path) as f:
            existing_signature = json.load(f)
        if existing_signature != current_signature:
            raise ValueError(
                "Resume safety check failed: existing run signature does not match current settings. "
                "Disable resume for this run or use a different checkpoint root."
            )
    if resume and not resume_signature_path.exists():
        raise ValueError(
            "Resume safety check failed: missing resume_signature.json for this run. "
            "To avoid mixing incompatible settings, rerun with resume disabled once, "
            "or start from a clean checkpoint root."
        )
    if resume and result_summary_path.exists():
        with open(result_summary_path) as f:
            return json.load(f)
    with open(resume_signature_path, "w") as f:
        json.dump(current_signature, f, indent=2, default=str)
    train_data, valid_data, selection_metadata = make_selected_data(
        official_train_pool_data, selection, config.training
    )
    test_data = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in test_data_uncentered.items()}
    train_data, valid_data, test_data, train_e_mean = center_energies_on_train(train_data, valid_data, test_data)
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
            {**selection_metadata, "train_e_mean": train_e_mean, "splits_dir": str(splits_dir), "run_uuid": run_uuid},
            f,
            indent=2,
            default=str,
        )
    key = jax.random.PRNGKey(selection.seed)
    model = build_model(config.model, data, num_atoms)
    run_ckpt_dir = latest_run_dir(config.training.ckpt_root, run_name)
    ema_params, best_loss = (None, None)
    if resume:
        ema_params, best_loss = maybe_resume_ema_params(run_ckpt_dir)
    if ema_params is None:
        ema_params, best_loss = train_model(
            key=key,
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            num_epochs=config.training.num_epochs,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            num_atoms=num_atoms,
            energy_weight=config.training.energy_weight,
            forces_weight=config.training.forces_weight,
            data_keys=("R", "Z", "F", "E", "N"),
            name=run_name,
            ckpt_dir=config.training.ckpt_root,
            best=True,
            save_every_epoch=config.training.save_every_epoch,
            batch_method="default",
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
    )
    teacher_valid_targets, teacher_valid_meta = teacher_predict_dataset(
        model=model,
        ema_params=ema_params,
        dataset=valid_data,
        batch_size=config.training.batch_size,
        num_atoms=num_atoms,
        seed=selection.seed + 3,
        set_name=f"Teacher valid reevaluation | {run_name}",
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
    student_run_name = f"{run_name}_student"
    student_key = jax.random.PRNGKey(selection.seed + 1000)
    student_model = build_model(config.student_model, data, num_atoms)
    student_ckpt_dir = latest_run_dir(config.training.ckpt_root, student_run_name)
    student_ema_params, student_best_loss = (None, None)
    if resume:
        student_ema_params, student_best_loss = maybe_resume_ema_params(student_ckpt_dir)
    if student_ema_params is None:
        student_ema_params, student_best_loss = train_model(
            key=student_key,
            model=student_model,
            train_data=teacher_train_targets,
            valid_data=teacher_valid_targets,
            num_epochs=config.student_epochs,
            learning_rate=config.student_learning_rate,
            batch_size=config.training.batch_size,
            num_atoms=num_atoms,
            energy_weight=config.training.energy_weight,
            forces_weight=config.training.forces_weight,
            data_keys=("R", "Z", "F", "E", "N"),
            name=student_run_name,
            ckpt_dir=config.training.ckpt_root,
            best=True,
            save_every_epoch=config.training.save_every_epoch,
            batch_method="default",
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
    )
    student_test_pred, student_test_meta = teacher_predict_dataset(
        model=student_model,
        ema_params=student_ema_params,
        dataset=test_data,
        batch_size=config.training.batch_size,
        num_atoms=num_atoms,
        seed=selection.seed + 1002,
        set_name=f"Student test reevaluation | {student_run_name}",
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
