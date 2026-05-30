from __future__ import annotations

import shutil
from pathlib import Path

from src.training_resume import (
    epoch_from_checkpoint_path,
    latest_checkpoint_epoch_from_paths,
    latest_epoch_checkpoint,
)


def run_metadata_dir(ckpt_root: Path, run_name: str) -> Path:
    return ckpt_root / "experiment_metadata" / run_name


def shared_orbax_root(shared_ckpt_root: Path) -> Path:
    return shared_ckpt_root / "orbax_runs"


def _latest_matching_run_dir(ckpt_root: Path, run_name: str) -> Path | None:
    run_dirs = sorted(ckpt_root.glob(f"{run_name}-*"))
    return run_dirs[-1] if run_dirs else None


def _latest_shared_run_dir(shared_ckpt_root: Path, run_name: str) -> Path | None:
    orbax_root = shared_orbax_root(shared_ckpt_root)
    if not orbax_root.is_dir():
        return None
    shared_dirs = sorted(orbax_root.glob(f"{run_name}-*"))
    return shared_dirs[-1] if shared_dirs else None


def sync_latest_epoch_checkpoint(*, source_run_dir: Path, target_run_dir: Path) -> int | None:
    """Copy only the latest epoch-* checkpoint from source into target run dir."""
    latest = latest_epoch_checkpoint(source_run_dir)
    if latest is None:
        return None
    epoch = epoch_from_checkpoint_path(latest)
    target_run_dir.mkdir(parents=True, exist_ok=True)
    target_epoch = target_run_dir / latest.name
    if target_epoch.exists():
        shutil.rmtree(target_epoch)
    shutil.copytree(latest, target_epoch)
    (target_run_dir / "latest_epoch.txt").write_text(f"{epoch}\n")
    return epoch


def pull_latest_checkpoints_for_resume(
    *,
    local_ckpt_root: Path,
    shared_ckpt_root: Path,
    run_names: tuple[str, ...],
) -> None:
    """Pull the latest shared epoch checkpoint when local is missing or older."""
    orbax_root = shared_orbax_root(shared_ckpt_root)
    if not orbax_root.is_dir():
        return
    local_ckpt_root.mkdir(parents=True, exist_ok=True)
    for run_name in run_names:
        shared_dir = _latest_shared_run_dir(shared_ckpt_root, run_name)
        if shared_dir is None:
            continue
        shared_epoch = latest_checkpoint_epoch_from_paths(shared_dir)
        if shared_epoch is None:
            continue
        local_dir = _latest_matching_run_dir(local_ckpt_root, run_name)
        local_epoch = latest_checkpoint_epoch_from_paths(local_dir)
        if local_epoch is not None and local_epoch >= shared_epoch:
            continue
        target_local = local_dir if local_dir is not None else local_ckpt_root / shared_dir.name
        sync_latest_epoch_checkpoint(source_run_dir=shared_dir, target_run_dir=target_local)


def push_latest_checkpoints_to_shared(
    *,
    local_ckpt_root: Path,
    shared_ckpt_root: Path,
    run_names: tuple[str, ...],
) -> None:
    """Copy the latest local epoch checkpoint for each run to shared storage."""
    orbax_root = shared_orbax_root(shared_ckpt_root)
    orbax_root.mkdir(parents=True, exist_ok=True)
    for run_name in run_names:
        local_dir = _latest_matching_run_dir(local_ckpt_root, run_name)
        if local_dir is None:
            continue
        target = orbax_root / local_dir.name
        epoch = sync_latest_epoch_checkpoint(source_run_dir=local_dir, target_run_dir=target)
        if epoch is not None:
            print(f"Synced latest checkpoint epoch-{epoch} for {run_name} -> {target}")


def pull_metadata_for_resume(*, local_ckpt_root: Path, shared_ckpt_root: Path, run_name: str) -> None:
    """Copy shared experiment_metadata/run to local /tmp so resume checks see prior runs."""
    shared_dir = run_metadata_dir(shared_ckpt_root, run_name)
    local_dir = run_metadata_dir(local_ckpt_root, run_name)
    if not shared_dir.is_dir():
        return
    if local_dir.is_dir():
        return
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(shared_dir, local_dir)


def push_metadata_to_shared(*, local_ckpt_root: Path, shared_ckpt_root: Path, run_name: str) -> None:
    """Copy local experiment_metadata/run to shared storage after a successful job."""
    local_dir = run_metadata_dir(local_ckpt_root, run_name)
    if not local_dir.is_dir():
        return
    shared_dir = run_metadata_dir(shared_ckpt_root, run_name)
    shared_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_dir, shared_dir, dirs_exist_ok=True)
