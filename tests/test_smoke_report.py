import os
import subprocess
from pathlib import Path

import pytest


def test_smoke_report_generates_summary(tmp_path):
    cifar_root = Path("data/cifar-10-batches-py")
    if not cifar_root.exists():
        pytest.skip("CIFAR-10 dataset not available")

    output_dir = tmp_path / "smoke_report"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    result = subprocess.run(
        ["bash", "scripts/run_smoke_report.sh", str(output_dir)],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    figures_dir = output_dir / "figures"
    if not figures_dir.exists():
        candidates = sorted(
            [p for p in output_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        assert candidates, result.stdout + result.stderr
        figures_dir = candidates[0] / "figures"
    summary_path = figures_dir / "summary.md"
    assert summary_path.exists(), result.stdout + result.stderr
    content = summary_path.read_text()
    assert "Synthetic Benchmark" in content
    assert "CIFAR-10 Reconstruction Benchmark" in content
