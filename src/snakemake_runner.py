from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    SelectionConfig,
    TeacherNoiseConfig,
    TrainingConfig,
    default_config,
)
from src.experiment import load_experiment_data, train_one_experiment
from src.shared_storage import (
    pull_latest_checkpoints_for_resume,
    pull_metadata_for_resume,
    push_latest_checkpoints_to_shared,
    push_metadata_to_shared,
)


def build_selection_matrix(
    *,
    seeds: list[int],
    metrics: list[str],
    window_size: int,
    stride: int,
    train_fraction: float,
) -> list[SelectionConfig]:
    selections: list[SelectionConfig] = []
    for seed in seeds:
        selections.append(
            SelectionConfig(
                name=f"random_seed{seed}",
                kind="random",
                seed=seed,
            )
        )
    for seed in seeds:
        for metric in metrics:
            selections.append(
                SelectionConfig(
                    name=f"{metric}_ws{window_size}_seed{seed}",
                    kind="information",
                    seed=seed,
                    metric=metric,
                    window_size=window_size,
                    stride=stride,
                    train_fraction=train_fraction,
                )
            )
    return selections


def build_config_from_args(args: argparse.Namespace, selection: SelectionConfig) -> ExperimentConfig:
    base = default_config()
    dataset_cfg = DatasetConfig(
        data_path=Path(args.data_path),
        rmd17_splits_dir=Path(args.splits_dir) if args.splits_dir else None,
        split_id=args.split_id,
        max_structures=args.max_structures,
        convert_to_ev=bool(args.convert_to_ev),
    )
    training_cfg = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_train=args.n_train,
        n_valid=args.n_valid,
        energy_weight=args.energy_weight,
        forces_weight=args.forces_weight,
        save_every_epoch=_parse_save_every_epoch_arg(args.save_every_epoch),
        save_every_n_epochs=args.save_every_n_epochs,
        log_tb=bool(args.log_tb),
        print_freq=args.print_freq,
        ckpt_root=Path(args.ckpt_root).resolve(),
    )
    student_model_cfg = ModelConfig(
        features=args.student_features,
        max_degree=args.student_max_degree,
        num_iterations=args.student_num_iterations,
        num_basis_functions=args.student_num_basis_functions,
        cutoff=args.student_cutoff,
        charges=bool(args.student_charges),
        zbl=bool(args.student_zbl),
    )
    teacher_noise = None
    if args.teacher_noise_scale is not None and args.teacher_noise_scale > 0.0:
        teacher_noise = TeacherNoiseConfig(
            scale=float(args.teacher_noise_scale),
            run_suffix=args.teacher_noise_suffix,
        )

    teacher_model_cfg = replace(base.model, cutoff=float(args.teacher_cutoff))

    return ExperimentConfig(
        molecule=args.molecule,
        dataset=dataset_cfg,
        training=training_cfg,
        model=teacher_model_cfg,
        student_model=student_model_cfg,
        student_epochs=args.student_epochs,
        student_learning_rate=args.student_learning_rate,
        teacher_noise=teacher_noise,
        selections=(selection,),
    )


def _bool_arg(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _parse_save_every_epoch_arg(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def run_selection(args: argparse.Namespace) -> None:
    import traceback

    try:
        _run_selection_impl(args)
    except Exception:
        traceback.print_exc()
        raise


def _run_selection_impl(args: argparse.Namespace) -> None:
    selections = build_selection_matrix(
        seeds=args.seeds,
        metrics=args.metrics,
        window_size=args.window_size,
        stride=args.stride,
        train_fraction=args.train_fraction,
    )
    by_name = {s.name: s for s in selections}
    if args.selection_name not in by_name:
        available = ", ".join(sorted(by_name))
        raise ValueError(f"Unknown selection {args.selection_name!r}. Available: {available}")
    selection = by_name[args.selection_name]
    config = build_config_from_args(args, selection)
    config.training.ckpt_root.mkdir(parents=True, exist_ok=True)
    run_name = selection.run_name(config.molecule, config.dataset.split_id)
    student_run_name = f"{run_name}_student"
    shared_ckpt_root = Path(args.shared_ckpt_root).resolve() if args.shared_ckpt_root else None
    if shared_ckpt_root is not None:
        shared_ckpt_root.mkdir(parents=True, exist_ok=True)
        pull_metadata_for_resume(
            local_ckpt_root=config.training.ckpt_root,
            shared_ckpt_root=shared_ckpt_root,
            run_name=run_name,
        )
        if args.phase == "teacher":
            pull_latest_checkpoints_for_resume(
                local_ckpt_root=config.training.ckpt_root,
                shared_ckpt_root=shared_ckpt_root,
                run_names=(run_name,),
            )
        elif args.phase == "student":
            pull_latest_checkpoints_for_resume(
                local_ckpt_root=config.training.ckpt_root,
                shared_ckpt_root=shared_ckpt_root,
                run_names=(run_name, student_run_name),
            )
        elif args.phase == "all":
            pull_latest_checkpoints_for_resume(
                local_ckpt_root=config.training.ckpt_root,
                shared_ckpt_root=shared_ckpt_root,
                run_names=(run_name, student_run_name),
            )
    data, official_train_pool_data, test_data, splits_dir, num_atoms = load_experiment_data(config)
    result = train_one_experiment(
        config=config,
        selection=selection,
        data=data,
        official_train_pool_data=official_train_pool_data,
        test_data_uncentered=test_data,
        splits_dir=splits_dir,
        num_atoms=num_atoms,
        resume=bool(args.resume),
        phase=args.phase,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    done_path = Path(args.done_file)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text("ok\n")

    if shared_ckpt_root is not None:
        push_metadata_to_shared(
            local_ckpt_root=config.training.ckpt_root,
            shared_ckpt_root=shared_ckpt_root,
            run_name=run_name,
        )
        if args.phase == "teacher":
            push_latest_checkpoints_to_shared(
                local_ckpt_root=config.training.ckpt_root,
                shared_ckpt_root=shared_ckpt_root,
                run_names=(run_name,),
            )
        elif args.phase == "student":
            push_latest_checkpoints_to_shared(
                local_ckpt_root=config.training.ckpt_root,
                shared_ckpt_root=shared_ckpt_root,
                run_names=(run_name, student_run_name),
            )
        elif args.phase == "all":
            push_latest_checkpoints_to_shared(
                local_ckpt_root=config.training.ckpt_root,
                shared_ckpt_root=shared_ckpt_root,
                run_names=(run_name, student_run_name),
            )
        print(f"Synced experiment_metadata/{run_name} to {shared_ckpt_root}")


def aggregate_results(args: argparse.Namespace) -> None:
    results: list[dict[str, Any]] = []
    for path_str in args.input_json:
        with open(path_str) as f:
            results.append(json.load(f))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snakemake helpers for experiment execution.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-selection")
    run_parser.add_argument("--selection-name", required=True)
    run_parser.add_argument("--output-json", required=True)
    run_parser.add_argument("--done-file", required=True)

    run_parser.add_argument("--molecule", default="rmd17_aspirin")
    run_parser.add_argument("--data-path", required=True)
    run_parser.add_argument("--splits-dir", default="")
    run_parser.add_argument("--split-id", type=int, default=1)
    run_parser.add_argument("--max-structures", type=int, default=None)
    run_parser.add_argument("--convert-to-ev", type=_bool_arg, default=True)
    run_parser.add_argument("--ckpt-root", required=True)
    run_parser.add_argument(
        "--shared-ckpt-root",
        default="",
        help="NFS/shared checkpoint root; experiment_metadata is synced here after each job.",
    )

    run_parser.add_argument("--num-epochs", type=int, default=1000)
    run_parser.add_argument("--batch-size", type=int, default=50)
    run_parser.add_argument("--learning-rate", type=float, default=1e-3)
    run_parser.add_argument("--n-train", type=int, default=950)
    run_parser.add_argument("--n-valid", type=int, default=50)
    run_parser.add_argument("--energy-weight", type=float, default=1.0)
    run_parser.add_argument("--forces-weight", type=float, default=52.91)
    run_parser.add_argument("--save-every-epoch", type=_parse_save_every_epoch_arg, default=True)
    run_parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=None,
        help="If set, checkpoint every N epochs (passed to train_model.save_every_epoch).",
    )
    run_parser.add_argument("--teacher-cutoff", type=float, default=10.0)
    run_parser.add_argument("--log-tb", type=_bool_arg, default=False)
    run_parser.add_argument("--print-freq", type=int, default=1)

    run_parser.add_argument("--student-epochs", type=int, default=500)
    run_parser.add_argument("--student-learning-rate", type=float, default=5e-4)
    run_parser.add_argument("--student-features", type=int, default=32)
    run_parser.add_argument("--student-max-degree", type=int, default=0)
    run_parser.add_argument("--student-num-iterations", type=int, default=2)
    run_parser.add_argument("--student-num-basis-functions", type=int, default=16)
    run_parser.add_argument("--student-cutoff", type=float, default=10.0)
    run_parser.add_argument("--student-charges", type=_bool_arg, default=False)
    run_parser.add_argument("--student-zbl", type=_bool_arg, default=False)

    run_parser.add_argument("--window-size", type=int, default=10)
    run_parser.add_argument("--stride", type=int, default=10)
    run_parser.add_argument("--train-fraction", type=float, default=0.95)
    run_parser.add_argument("--seeds", nargs="+", type=int, required=True)
    run_parser.add_argument("--metrics", nargs="+", required=True)
    run_parser.add_argument("--resume", type=_bool_arg, default=False)
    run_parser.add_argument(
        "--phase",
        choices=("all", "teacher", "student"),
        default="all",
        help="Training phase: teacher-only, student-only (requires teacher artifacts), or both.",
    )
    run_parser.add_argument(
        "--teacher-noise-scale",
        type=float,
        default=None,
        help="If set and > 0, add Gaussian noise to teacher train/valid E/F scaled by this factor times each split's std.",
    )
    run_parser.add_argument(
        "--teacher-noise-suffix",
        default="teacher_noise",
        help="Suffix appended to run names for teacher-noise variants.",
    )

    agg_parser = subparsers.add_parser("aggregate")
    agg_parser.add_argument("--input-json", nargs="+", required=True)
    agg_parser.add_argument("--output-json", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run-selection":
        run_selection(args)
    elif args.command == "aggregate":
        aggregate_results(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
