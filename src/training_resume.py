"""Epoch-aware checkpoint resume helpers (no heavy ML imports)."""

from __future__ import annotations

from pathlib import Path


def epoch_from_checkpoint_path(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def latest_epoch_checkpoint(run_ckpt_dir: Path) -> Path | None:
    """Return the latest epoch checkpoint directory inside a run directory."""
    epoch_dirs = [p for p in run_ckpt_dir.glob("epoch-*") if p.is_dir()]
    if not epoch_dirs:
        return None
    epoch_dirs = sorted(epoch_dirs, key=epoch_from_checkpoint_path)
    return epoch_dirs[-1]


def latest_checkpoint_epoch_from_paths(run_ckpt_dir: Path | None) -> int | None:
    """Return the highest epoch-* directory index for a run, or None."""
    if run_ckpt_dir is None:
        return None
    latest_epoch_path = latest_epoch_checkpoint(run_ckpt_dir)
    if latest_epoch_path is None:
        return None
    path_epoch = epoch_from_checkpoint_path(latest_epoch_path)
    return path_epoch if path_epoch >= 0 else None


def is_training_complete_from_paths(run_ckpt_dir: Path | None, target_epochs: int) -> bool:
    latest_epoch = latest_checkpoint_epoch_from_paths(run_ckpt_dir)
    return latest_epoch is not None and latest_epoch >= target_epochs
