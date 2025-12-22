from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_make_paper_figures_dry_run_writes_manifest(tmp_path: Path) -> None:
    out_root = tmp_path / "paper_out"
    script = Path("scripts/make_paper_figures.py")
    assert script.exists()

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-root",
            str(out_root),
            "--dry-run",
            "--force",
            "--profile",
            "fast",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0

    manifest_path = out_root / "paper" / "manifest.json"
    commands_path = out_root / "paper" / "commands.sh"
    status_path = out_root / "paper" / "status.json"
    assert manifest_path.exists()
    assert commands_path.exists()
    assert status_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["out_root"] == str(out_root.resolve())
    assert payload["profile"] == "fast"
    assert payload["figure_map"] == {}
    assert "executed" in payload
