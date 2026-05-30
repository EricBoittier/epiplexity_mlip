"""Tests for shared latest-checkpoint sync."""

from __future__ import annotations

from pathlib import Path

from src.shared_storage import (
    pull_latest_checkpoints_for_resume,
    push_latest_checkpoints_to_shared,
    sync_latest_epoch_checkpoint,
)


def test_sync_latest_epoch_checkpoint_copies_only_latest(tmp_path: Path) -> None:
    source = tmp_path / "run-abc"
    source.mkdir()
    (source / "epoch-50").mkdir()
    (source / "epoch-50" / "params.txt").write_text("old")
    (source / "epoch-200").mkdir()
    (source / "epoch-200" / "params.txt").write_text("new")

    target = tmp_path / "shared" / "run-abc"
    epoch = sync_latest_epoch_checkpoint(source_run_dir=source, target_run_dir=target)
    assert epoch == 200
    assert (target / "epoch-200" / "params.txt").read_text() == "new"
    assert not (target / "epoch-50").exists()
    assert (target / "latest_epoch.txt").read_text().strip() == "200"


def test_push_and_pull_latest_checkpoint_roundtrip(tmp_path: Path) -> None:
    local = tmp_path / "local_ckpt"
    shared = tmp_path / "shared_ckpt"
    run_dir = local / "mol_split01_random_seed42-uuid"
    run_dir.mkdir(parents=True)
    (run_dir / "epoch-100").mkdir()
    (run_dir / "epoch-100" / "marker.txt").write_text("ckpt")

    push_latest_checkpoints_to_shared(
        local_ckpt_root=local,
        shared_ckpt_root=shared,
        run_names=("mol_split01_random_seed42",),
    )

    fresh_local = tmp_path / "fresh_local"
    pull_latest_checkpoints_for_resume(
        local_ckpt_root=fresh_local,
        shared_ckpt_root=shared,
        run_names=("mol_split01_random_seed42",),
    )
    pulled = fresh_local / "mol_split01_random_seed42-uuid" / "epoch-100" / "marker.txt"
    assert pulled.read_text() == "ckpt"


if __name__ == "__main__":
    test_sync_latest_epoch_checkpoint_copies_only_latest(Path("/tmp/epiplexity_sync_test"))
    print("ok")
