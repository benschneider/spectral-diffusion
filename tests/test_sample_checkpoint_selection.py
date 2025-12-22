from __future__ import annotations

from pathlib import Path

from src.cli.sample import _find_latest_checkpoint


def test_find_latest_checkpoint_sorts_by_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    (ckpt_dir / "checkpoint_step_200.pt").write_bytes(b"")
    (ckpt_dir / "checkpoint_step_1000.pt").write_bytes(b"")
    (ckpt_dir / "checkpoint_step_800.pt").write_bytes(b"")

    latest = _find_latest_checkpoint(run_dir)
    assert latest.name == "checkpoint_step_1000.pt"

