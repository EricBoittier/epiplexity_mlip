from __future__ import annotations

import shutil
from pathlib import Path


def run_metadata_dir(ckpt_root: Path, run_name: str) -> Path:
    return ckpt_root / "experiment_metadata" / run_name


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
