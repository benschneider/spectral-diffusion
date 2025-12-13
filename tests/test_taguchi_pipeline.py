from pathlib import Path

import pandas as pd
import pytest

from src.experiments.run_experiment import (
    TaguchiExperimentRunner,
    apply_factor_to_config,
    build_factor_column_mapping,
    load_factor_registry,
)


def _design_matrix() -> pd.DataFrame:
    return pd.read_csv(Path("configs/taguchi/L27_extended.csv"))


def test_factor_registry_loads_levels():
    registry = load_factor_registry(Path("configs/taguchi/factor_registry.yaml"))
    expected = {
        "snr_ratio",
        "spectral_operator_mode",
        "sampler_type",
        "sampling_steps",
        "train_steps",
        "image_resolution",
    }
    assert set(registry.keys()) == expected
    assert "spectral_operator_mode" in registry
    assert registry["snr_ratio"]["levels"] == [0.8, 1.0, 1.4]


def test_build_factor_mapping_matches_cardinality():
    registry = load_factor_registry(Path("configs/taguchi/factor_registry.yaml"))
    design = _design_matrix()
    mapping = build_factor_column_mapping(registry, design)
    expected_columns = {
        col
        for col in design.columns
        if col.lower() not in {"run", "row"}
    }
    assert set(mapping.keys()) == expected_columns
    assert set(mapping.values()) == set(registry.keys())


def test_runner_builds_config_from_row():
    registry = load_factor_registry(Path("configs/taguchi/factor_registry.yaml"))
    runner = TaguchiExperimentRunner(
        design_matrix_path=Path("configs/taguchi/L27_extended.csv"),
        base_config={
            "model": {"type": "unet_tiny"},
            "data": {"height": 32, "width": 32},
            "spectral": {},
            "diffusion": {},
            "sampling": {},
            "training": {"epochs": 1, "num_batches": 50},
            "optim": {},
        },
    )
    runner.set_factor_registry(registry, randomize=False, seed=0)

    row = runner.design.iloc[0]
    config = runner._build_config_from_row(row=row, row_number=int(row["run"]))

    assert config["diffusion"]["spectral_operator_mode"] in {"none", "radial", "radial_squared"}
    assert "operator_mode" not in config.get("spectral", {})
    assert config["sampling"]["sampler_type"] == "ddim"
    assert config["sampling"]["sampling_steps"] == 10
    assert config["training"]["num_batches"] == 50
    assert config["data"]["height"] == 32
    assert config["data"]["width"] == 32

    taguchi_meta = config["taguchi"]
    assert taguchi_meta["row_number"] == 1
    assert taguchi_meta["factor_levels"]["spectral_operator_mode"]["level_label"] is not None
    assert set(taguchi_meta["factor_mapping"].values()) == set(registry.keys())
    assert "curriculum" not in config.get("training", {})


def test_randomised_mapping_respects_level_counts():
    registry = load_factor_registry(Path("configs/taguchi/factor_registry.yaml"))
    runner = TaguchiExperimentRunner(
        design_matrix_path=Path("configs/taguchi/L27_extended.csv"),
        base_config={},
    )
    runner.set_factor_registry(registry, randomize=True, seed=42)
    mapping = runner.column_mapping
    assert mapping is not None
    assert set(mapping.values()) == set(registry.keys())


def test_sampler_alias_accepts_dpm_solver_pp():
    registry = load_factor_registry(Path("configs/taguchi/factor_registry.yaml"))
    cfg = {}
    sampler_levels = registry["sampler_type"]["levels"]
    assert "dpm_solver++" in sampler_levels
    level_index = sampler_levels.index("dpm_solver++") + 1
    apply_factor_to_config(cfg, "sampler_type", level_index, registry)
    assert cfg["sampling"]["sampler_type"] == "dpm_solver++"


def test_registry_requires_all_factors(tmp_path):
    bad_registry = tmp_path / "bad.yaml"
    bad_registry.write_text(
        "factors:\n"
        "  snr_ratio:\n"
        "    levels: [0.8, 1.0]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_factor_registry(bad_registry)
