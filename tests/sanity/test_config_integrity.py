from pathlib import Path

import pytest
import yaml


CONFIG_BENCHMARK = Path("configs/benchmark_spectral_cifar.yaml")
CONFIG_TAGUCHI = Path("configs/taguchi_smoke_base.yaml")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def test_config_files_exist():
    for path in (CONFIG_BENCHMARK, CONFIG_TAGUCHI):
        assert path.exists(), f"Missing required config file: {path}"


def test_config_consistency():
    benchmark = _load_yaml(CONFIG_BENCHMARK)
    taguchi = _load_yaml(CONFIG_TAGUCHI)

    bench_data = benchmark.get("data", {})
    taguchi_data = taguchi.get("data", {})
    assert bench_data.get("channels") == 3, "Benchmark CIFAR config must specify 3 channels."
    assert (
        bench_data.get("channels") == taguchi_data.get("channels")
    ), "Channel count mismatch between benchmark and Taguchi configs."
    assert (
        bench_data.get("height"), bench_data.get("width")
    ) == (
        taguchi_data.get("height"),
        taguchi_data.get("width"),
    ), "Image resolution mismatch between benchmark and Taguchi configs."

    bench_diff = benchmark.get("diffusion", {})
    taguchi_diff = taguchi.get("diffusion", {})
    assert (
        bench_diff.get("beta_schedule") == taguchi_diff.get("beta_schedule")
    ), "Diffusion beta schedule differs between configs."

    for cfg_name, diff in [("benchmark", bench_diff), ("taguchi", taguchi_diff)]:
        assert "snr_ratio" in diff, f"{cfg_name} config must set diffusion.snr_ratio"
        assert "spectral_operator_mode" in diff, f"{cfg_name} config must set diffusion.spectral_operator_mode"
        assert float(diff["snr_ratio"]) > 0.0
        assert str(diff["spectral_operator_mode"]) in {"none", "radial", "radial_squared"}

    # Optional normalization block – skip gracefully if missing
    normalization = bench_data.get("normalization")
    if normalization is not None:
        for key in ("mean", "std"):
            assert normalization.get(key) is not None, f"Missing data normalization {key}."
    else:
        pytest.skip("Benchmark CIFAR config has no explicit normalization section")
