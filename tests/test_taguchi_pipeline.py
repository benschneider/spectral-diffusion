from pathlib import Path

import pandas as pd

from src.experiments.run_experiment import (
    TaguchiExperimentRunner,
    build_factor_column_mapping,
    load_factor_registry,
)


def _design_matrix() -> pd.DataFrame:
    return pd.read_csv(Path("configs/taguchi/L27_extended.csv"))


def test_factor_registry_loads_levels():
    registry = load_factor_registry(Path("configs/taguchi/factor_registry.yaml"))
    assert "spectral_adapter_placement" in registry
    assert registry["lr_schedule_mode"]["levels"] == [
        "constant",
        "cosine",
        "cosine_warmup",
    ]


def test_build_factor_mapping_matches_cardinality():
    registry = load_factor_registry(Path("configs/taguchi/factor_registry.yaml"))
    design = _design_matrix()
    mapping = build_factor_column_mapping(registry, design)
    assert set(mapping.keys()) == set("ABCDEFGHIJ")
    assert set(mapping.values()) == set(registry.keys())


def test_runner_builds_config_from_row():
    registry = load_factor_registry(Path("configs/taguchi/factor_registry.yaml"))
    runner = TaguchiExperimentRunner(
        design_matrix_path=Path("configs/taguchi/L27_extended.csv"),
        base_config={
            "model": {"type": "unet_spectral"},
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

    assert config["spectral"]["apply_to"] == []
    assert config["spectral"]["weighting"] == "none"
    assert config["diffusion"]["uniform_corruption"] is False
    assert config["model"]["enable_phase_attention"] is False
    assert config["sampling"]["sampler_type"] == "ddim"
    assert config["sampling"]["num_steps"] == 30
    assert "curriculum" not in config["training"]
    assert config["optim"]["lr_schedule"] == "constant"
    assert config["training"]["num_batches"] == 50
    assert config["data"]["height"] == 32
    assert config["data"]["width"] == 32

    taguchi_meta = config["taguchi"]
    assert taguchi_meta["row_number"] == 1
    assert taguchi_meta["factor_levels"]["spectral_adapter_placement"]["level_label"] == "none"
    assert "lr_schedule_mode" in taguchi_meta["factor_mapping"].values()


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
