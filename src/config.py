from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

EV_PER_KCAL = 0.043364106370
SelectionKind = Literal["random", "information"]


@dataclass(frozen=True)
class DatasetConfig:
    data_path: Path = Path("/scicore/home/meuwly/boitti0000/data/rmd17/npz_data/rmd17_aspirin.npz")
    rmd17_splits_dir: Path | None = None
    split_id: int = 1
    max_structures: int | None = None
    convert_to_ev: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    num_epochs: int = 1000
    batch_size: int = 50
    learning_rate: float = 1e-3
    n_train: int = 950
    n_valid: int = 50
    energy_weight: float = 1.0
    forces_weight: float = 52.91
    save_every_epoch: bool = True
    log_tb: bool = False
    print_freq: int = 1
    ckpt_root: Path = Path("checkpoints/rmd17_aspirin")


@dataclass(frozen=True)
class ModelConfig:
    features: int = 64
    max_degree: int = 0
    num_iterations: int = 3
    num_basis_functions: int = 32
    cutoff: float = 10.0
    charges: bool = False
    zbl: bool = False


@dataclass(frozen=True)
class SoapConfig:
    species: tuple[str, ...] = ("C", "H", "O")
    nmax: int = 10
    lmax: int = 10
    rcut: float = 3.0
    sigma: float = 0.5


@dataclass(frozen=True)
class SelectionConfig:
    name: str
    kind: SelectionKind
    seed: int
    metric: str | None = None
    window_size: int = 10
    stride: int = 10
    train_fraction: float | None = None
    stop: int | None = None
    soap: SoapConfig = field(default_factory=SoapConfig)

    def run_name(self, molecule: str, split_id: int) -> str:
        if self.kind == "random":
            return f"{molecule}_split{split_id:02d}_random_seed{self.seed}"
        metric = self.metric or "metric"
        return (
            f"{molecule}_split{split_id:02d}_{metric}"
            f"_ws{self.window_size}_st{self.stride}_seed{self.seed}"
        )


@dataclass(frozen=True)
class ExperimentConfig:
    molecule: str = "rmd17_aspirin"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    student_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            features=32,
            max_degree=0,
            num_iterations=2,
            num_basis_functions=16,
            cutoff=5.0,
            charges=False,
            zbl=False,
        )
    )
    student_epochs: int = 500
    student_learning_rate: float = 5e-4
    selections: tuple[SelectionConfig, ...] = field(default_factory=tuple)


def default_config() -> ExperimentConfig:
    seeds = (42, 43, 44)
    metrics = ("gzip_bytes_cart", "gzip_bytes_zmat", "gzip_bytes_soap")

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
                    name=f"{metric}_ws10_seed{seed}",
                    kind="information",
                    seed=seed,
                    metric=metric,
                    window_size=10,
                    stride=10,
                    train_fraction=0.95,
                )
            )

    return ExperimentConfig(selections=tuple(selections))
