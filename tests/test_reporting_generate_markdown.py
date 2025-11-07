from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reporting.generate_markdown import (
    FigureMetadata,
    _figure_gallery_section,
    _summary_section,
)


def test_figure_metadata_from_payload_normalises_fields() -> None:
    payload = {
        "dataset": "CIFAR",
        "run_mapping": {"cfg": 1},
        "notes": "targets_missing",
        "extra": "value",
    }

    metadata = FigureMetadata.from_payload(payload)

    assert metadata.dataset == "CIFAR"
    assert metadata.run_mapping == {"cfg": "1"}
    assert metadata.note == "targets_missing"
    # ``raw`` should retain original keys for caption heuristics.
    assert metadata.raw["extra"] == "value"


def test_figure_gallery_section_deduplicates_and_formats(tmp_path: Path) -> None:
    figs_dir = tmp_path
    (figs_dir / "fig1.png").write_bytes(b"image-a")
    # Duplicate content should be skipped even when appearing later in traversal.
    (figs_dir / "zzz_duplicate.png").write_bytes(b"image-a")
    nested = figs_dir / "nested"
    nested.mkdir()
    (nested / "fig2.png").write_bytes(b"image-b")

    metadata = {
        "fig1.png": {
            "dataset": "cifar",
            "run_mapping": {"baseline": "run-42"},
            "notes": "targets_missing",
        },
        "fig2.png": {"dataset": "synthetic"},
    }

    lines = _figure_gallery_section(figs_dir, metadata)

    image_lines = [line for line in lines if line.startswith("![](")]
    assert len(image_lines) == 2
    assert any("*Run mapping:* baseline → run-42" in line for line in lines)
    assert any("Target samples were unavailable" in line for line in lines)


def test_summary_section_includes_tables_and_notes() -> None:
    synthetic_df = pd.DataFrame(
        [
            {
                "run_id": "synth-1",
                "loss_final": 0.12345,
                "images_per_second": 12.345,
                "runtime_seconds": 10.0,
            }
        ]
    )
    cifar_df = pd.DataFrame(
        [
            {
                "run_id": "cifar-1",
                "loss_final": 0.22345,
                "images_per_second": 6.789,
                "runtime_seconds": 20.0,
            }
        ]
    )
    metric_notes = {"cifar": {"high_freq_psnr_missing": True}}

    lines = _summary_section(synthetic_df, cifar_df, metric_notes)

    assert lines[0] == "## Summary Table"
    # Ensure the markdown table rendered both datasets.
    table_body = "\n".join(lines)
    assert "Synthetic" in table_body
    assert "CIFAR-10" in table_body
    assert any("Some PSNR values" in line for line in lines)
