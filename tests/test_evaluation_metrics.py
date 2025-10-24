import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.evaluation.metrics import FID_AVAILABLE, compute_dataset_metrics


def _write_image(path: Path, array: np.ndarray) -> None:
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(path)


def test_compute_dataset_metrics_identity(tmp_path):
    gen_dir = tmp_path / "gen"
    ref_dir = tmp_path / "ref"
    gen_dir.mkdir()
    ref_dir.mkdir()

    rng = np.random.default_rng(seed=0)
    for idx in range(3):
        array = rng.random((32, 32, 3))
        name = f"img_{idx}.png"
        _write_image(gen_dir / name, array)
        _write_image(ref_dir / name, array)

    metrics = compute_dataset_metrics(gen_dir, ref_dir)
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-8)
    assert np.isinf(metrics["psnr"])


def test_compute_dataset_metrics_mismatch(tmp_path):
    gen_dir = tmp_path / "gen"
    ref_dir = tmp_path / "ref"
    gen_dir.mkdir()
    ref_dir.mkdir()
    _write_image(gen_dir / "a.png", np.zeros((8, 8, 3)))
    _write_image(ref_dir / "b.png", np.zeros((8, 8, 3)))

    with pytest.raises(ValueError):
        compute_dataset_metrics(gen_dir, ref_dir)


def test_compute_dataset_metrics_fid_optional(tmp_path):
    gen_dir = tmp_path / "gen"
    ref_dir = tmp_path / "ref"
    gen_dir.mkdir()
    ref_dir.mkdir()
    for idx in range(2):
        array = np.zeros((16, 16, 3)) + idx * 0.25
        name = f"img_{idx}.png"
        _write_image(gen_dir / name, array)
        _write_image(ref_dir / name, array)

    if FID_AVAILABLE:
        metrics = compute_dataset_metrics(gen_dir, ref_dir, use_fid=True, strict_filenames=False)
        assert "fid" in metrics
    else:
        with pytest.raises(RuntimeError):
            compute_dataset_metrics(gen_dir, ref_dir, use_fid=True, strict_filenames=False)
