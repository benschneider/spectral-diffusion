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
    summary_path = output_dir / "figures" / "summary.md"
    assert summary_path.exists(), result.stdout + result.stderr
    content = summary_path.read_text()
    assert "Synthetic Benchmark" in content
    assert "CIFAR-10 Reconstruction Benchmark" in content
