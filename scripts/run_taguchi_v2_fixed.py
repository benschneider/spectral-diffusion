#!/usr/bin/env python
"""Deterministic Taguchi V2 verification for spectral diffusion."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.debug.record_training_steps import record_training_steps
from src.cli.common import load_config
from src.cli.train import apply_variant_override
from src.training.data.text_encoder_decoder import encode_text_to_image_dense


DEFAULT_BASE_CONFIG = Path("configs/benchmark_spectral_cifar.yaml")
DEFAULT_MATRIX = Path("configs/taguchi_matrix.json")
DEFAULT_OUTPUT = Path("results/taguchi_v2")


@dataclass
class TaguchiRun:
    row_index: int
    parameters: Dict[str, object]
    output_dir: Path


def _set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_taguchi_rows(matrix_path: Path) -> List[Dict[str, object]]:
    if matrix_path.exists():
        with matrix_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, list):
            return [dict(row) for row in raw]
    return [
        {"fft_norm": "ortho", "snr_ratio": 1.0, "dc_scale_factor": 0.1, "uniform_corruption": True},
        {"fft_norm": "backward", "snr_ratio": 0.7, "dc_scale_factor": 0.2, "uniform_corruption": True},
        {"fft_norm": "forward", "snr_ratio": 1.4, "dc_scale_factor": 0.1, "uniform_corruption": False},
        {"fft_norm": "ortho", "snr_ratio": 0.5, "dc_scale_factor": 0.0, "uniform_corruption": True},
    ]


def _generate_fixed_inputs(
    cache_path: Path,
    *,
    use_text: bool,
    num_samples: int,
    image_size: int,
) -> torch.Tensor:
    if cache_path.exists():
        return torch.load(cache_path)

    samples: List[torch.Tensor] = []
    for i in range(num_samples):
        if use_text:
            prompt = f"Prompt {i}"
            answer = f"Response {i}"
            tensor = encode_text_to_image_dense(prompt, answer, image_size=(image_size, image_size))
        else:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(i)
            tensor = torch.rand((3, image_size, image_size), generator=gen)
        samples.append(tensor)
    stacked = torch.stack(samples)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stacked, cache_path)
    return stacked


def _build_loader(tensor: torch.Tensor, batch_size: int) -> DataLoader:
    targets = tensor.clone()
    dataset = TensorDataset(tensor, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)


def _prepare_config(base_config: Path, overrides: Dict[str, object], destination: Path) -> Path:
    config = load_config(config_path=base_config)
    apply_variant_override(config, overrides.get("variant"))

    data_cfg = config.setdefault("data", {})
    data_cfg["source"] = "synthetic"
    data_cfg.setdefault("channels", 3)
    data_cfg.setdefault("height", overrides.get("image_size", 32))
    data_cfg.setdefault("width", overrides.get("image_size", 32))

    training_cfg = config.setdefault("training", {})
    if "batch_size" in overrides:
        training_cfg["batch_size"] = int(overrides["batch_size"])

    diffusion_cfg = config.setdefault("diffusion", {})
    spectral_cfg = config.setdefault("spectral", {})
    for key in ("fft_norm", "uniform_corruption", "snr_ratio", "dc_scale_factor"):
        if key in overrides:
            diffusion_cfg[key] = overrides[key]
            spectral_cfg[key] = overrides[key]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return destination


def _summarise_metrics(run_dir: Path) -> None:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        target = run_dir / "metrics.json"
        target.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")


def _loss_curve(step_metrics: Path, output_path: Path) -> None:
    if not step_metrics.exists():
        return
    steps: List[int] = []
    losses: List[float] = []
    with step_metrics.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in record and "loss" in record:
                steps.append(int(record["step"]))
                losses.append(float(record["loss"]))
    if not steps:
        return
    plt.figure(figsize=(6, 3))
    plt.plot(steps, losses, marker="o", linewidth=1.5)
    plt.title("Training Loss over Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def _collect_metrics(run_dirs: Iterable[TaguchiRun]) -> List[Dict[str, object]]:
    collected: List[Dict[str, object]] = []
    for entry in run_dirs:
        summary_path = entry.output_dir / "summary.json"
        payload: Dict[str, object] = {"row": entry.row_index, **entry.parameters}
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                info = json.load(handle)
            payload.update(info)
        else:
            payload["error"] = "missing_summary"
        collected.append(payload)
    return collected


def _write_summary_csv(rows: List[Dict[str, object]], destination: Path) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            values = [str(row.get(key, "")) for key in keys]
            handle.write(",".join(values) + "\n")


def _plot_main_effects(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    loss_column = next((c for c in ["mean_loss", "final_loss", "loss"] if c in df.columns), None)
    if loss_column is None:
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(df["snr_ratio"], df[loss_column], c="tab:blue", label=loss_column)
    plt.title("Taguchi V2: Loss vs SNR Ratio")
    plt.xlabel("SNR Ratio")
    plt.ylabel(loss_column.replace("_", " ").title())
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run(matrix_path: Path, base_config: Path, output_root: Path, *, seed: int) -> None:
    _set_global_seed(seed)
    output_root.mkdir(parents=True, exist_ok=True)

    cache_path = output_root / "cached_inputs.pt"
    inputs = _generate_fixed_inputs(cache_path, use_text=True, num_samples=128, image_size=32)

    taguchi_rows = _load_taguchi_rows(matrix_path)
    run_descriptors: List[TaguchiRun] = []

    for index, row_params in enumerate(taguchi_rows):
        run_dir = output_root / f"row_{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        batch_size = int(row_params.get("batch_size", 32))
        loader = _build_loader(inputs, batch_size=batch_size)

        config_path = _prepare_config(
            base_config=base_config,
            overrides=row_params,
            destination=run_dir / "config.yaml",
        )

        print(f"\n=== Running row_{index:02d} ===")
        print(json.dumps(row_params, indent=2))

        record_training_steps(
            config_path=config_path,
            variant=row_params.get("variant"),
            steps=150,
            output_dir=run_dir,
            snr_ratio=float(row_params.get("snr_ratio", 1.0)),
            dc_scale_factor=float(row_params.get("dc_scale_factor", 0.1)),
            loader=loader,
            log_interval=10,
        )

        _summarise_metrics(run_dir)
        _loss_curve(run_dir / "step_metrics.jsonl", run_dir / "loss_curve.png")
        run_descriptors.append(TaguchiRun(index, row_params, run_dir))

    summary_rows = _collect_metrics(run_descriptors)
    csv_path = output_root / "taguchi_v2_results.csv"
    _write_summary_csv(summary_rows, csv_path)
    print(f"[Summary] Wrote results to {csv_path}")
    _plot_main_effects(csv_path, output_root / "main_effects_v2.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic Taguchi V2 verification")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX, help="Taguchi matrix JSON")
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE_CONFIG, help="Base config path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination for artefacts")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(matrix_path=args.matrix, base_config=args.config, output_root=args.output, seed=int(args.seed))


if __name__ == "__main__":  # pragma: no cover
    main()
