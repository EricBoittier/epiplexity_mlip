"""Tests for epoch-aware training resume helpers."""

from __future__ import annotations

from pathlib import Path

from src.training_resume import (
    epoch_from_checkpoint_path,
    is_training_complete_from_paths,
    latest_checkpoint_epoch_from_paths,
)


def test_epoch_from_checkpoint_path() -> None:
    assert epoch_from_checkpoint_path(Path("epoch-999")) == 999
    assert epoch_from_checkpoint_path(Path("epoch-50")) == 50
    assert epoch_from_checkpoint_path(Path("bad-name")) == -1


def test_latest_checkpoint_epoch_uses_highest_epoch_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-uuid"
    run_dir.mkdir()
    (run_dir / "epoch-50").mkdir()
    (run_dir / "epoch-200").mkdir()
    (run_dir / "epoch-100").mkdir()
    assert latest_checkpoint_epoch_from_paths(run_dir) == 200


def test_is_training_complete(tmp_path: Path) -> None:
    run_dir = tmp_path / "complete-uuid"
    run_dir.mkdir()
    (run_dir / "epoch-999").mkdir()
    assert is_training_complete_from_paths(run_dir, 1000) is True
    assert is_training_complete_from_paths(run_dir, 1001) is False

    partial = tmp_path / "partial-uuid"
    partial.mkdir()
    (partial / "epoch-50").mkdir()
    assert is_training_complete_from_paths(partial, 1000) is False
    assert is_training_complete_from_paths(None, 1000) is False


if __name__ == "__main__":
    test_epoch_from_checkpoint_path()
    print("ok")
