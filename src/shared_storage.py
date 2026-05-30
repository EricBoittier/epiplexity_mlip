from __future__ import annotations

import shutil
from pathlib import Path


def run_metadata_dir(ckpt_root: Path, run_name: str) -> Path:
    return ckpt_root / "experiment_metadata" / run_name


from __future__ import annotations

import shutil
from pathlib import Path


def run_metadata_dir(ckpt_root: Path, run_name: str) -> Path:
    return ckpt_root / "experiment_metadata" / run_name


def shared_orbax_root(shared_ckpt_root: Path) -> Path:
    return shared_ckpt_root / "orbax_runs"


def _latest_matching_run_dir(ckpt_root: Path, run_name: str) -> Path | None:
    run_dirs = sorted(ckpt_root.glob(f"{run_name}-*"))
    return run_dirs[-1] if run_dirs else None


def pull_orbax_runs_for_resume(
    *,
    local_ckpt_root: Path,
    shared_ckpt_root: Path,
    run_names: tuple[str, ...],
) -> None:
    """Copy shared Orbax run dirs to local /tmp when missing (split teacher/student jobs)."""
    orbax_root = shared_orbax_root(shared_ckpt_root)
    if not orbax_root.is_dir():
        return
    local_ckpt_root.mkdir(parents=True, exist_ok=True)
    for run_name in run_names:
        if _latest_matching_run_dir(local_ckpt_root, run_name) is not None:
            continue
        shared_dirs = sorted(orbax_root.glob(f"{run_name}-*"))
        if not shared_dirs:
            continue
        shutil.copytree(shared_dirs[-1], local_ckpt_root / shared_dirs[-1].name, dirs_exist_ok=True)


def push_orbax_runs_to_shared(
    *,
    local_ckpt_root: Path,
    shared_ckpt_root: Path,
    run_names: tuple[str, ...],
) -> None:
    """Copy local Orbax run dirs to shared storage after a successful job."""
    orbax_root = shared_orbax_root(shared_ckpt_root)
    orbax_root.mkdir(parents=True, exist_ok=True)
    for run_name in run_names:
        local_dir = _latest_matching_run_dir(local_ckpt_root, run_name)
        if local_dir is None:
            continue
        target = orbax_root / local_dir.name
        shutil.copytree(local_dir, target, dirs_exist_ok=True)


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
