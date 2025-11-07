#!/usr/bin/env python
"""Generate a compact multi-panel report directly from an HDF5 archive."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.hdf5_packager import dataset_to_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path, help="Consolidated HDF5 archive")
    parser.add_argument(
        "--output",
        type=Path,
        help="Target image path (default: <archive_dir>/compact_report.png)",
    )
    return parser.parse_args()


def _load_table(h5: h5py.File, relative_path: str) -> pd.DataFrame:
    index_ds = h5["index"]["files"]
    index_df = dataset_to_dataframe(index_ds)
    matches = index_df[index_df["relative_path"] == relative_path]
    if matches.empty:
        raise FileNotFoundError(f"No artefact named {relative_path} in archive")
    dataset_path = matches.iloc[0]["dataset_path"]
    return dataset_to_dataframe(h5[dataset_path])


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def render_compact_report(hdf5_path: Path, output_path: Path) -> None:
    with h5py.File(hdf5_path, "r") as h5:
        synthetic = _load_table(h5, "synthetic/summary.csv")
        cifar = _load_table(h5, "cifar/summary.csv")
        taguchi = _load_table(h5, "figures/taguchi_contrib.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    syn_df = synthetic.copy()
    syn_df["eval_psnr"] = _coerce_numeric(syn_df.get("eval_psnr"))
    syn_df = syn_df.dropna(subset=["eval_psnr"])
    axes[0].bar(syn_df["display_name"], syn_df["eval_psnr"], color="#4472c4")
    axes[0].set_title("Synthetic PSNR by model")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].tick_params(axis="x", labelrotation=30)
    for label in axes[0].get_xticklabels():
        label.set_horizontalalignment("right")

    cifar_df = cifar.copy()
    cifar_df["eval_psnr"] = _coerce_numeric(cifar_df.get("eval_psnr"))
    cifar_df = cifar_df.dropna(subset=["eval_psnr"])
    axes[1].bar(cifar_df["display_name"], cifar_df["eval_psnr"], color="#ed7d31")
    axes[1].set_title("CIFAR-10 PSNR by model")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].tick_params(axis="x", labelrotation=30)
    for label in axes[1].get_xticklabels():
        label.set_horizontalalignment("right")

    syn_df["images_per_second"] = _coerce_numeric(syn_df.get("images_per_second"))
    syn_df["loss_drop_per_second"] = _coerce_numeric(syn_df.get("loss_drop_per_second"))
    axes[2].scatter(
        syn_df["images_per_second"],
        syn_df["loss_drop_per_second"],
        c="#70ad47",
        s=80,
    )
    for _, row in syn_df.iterrows():
        axes[2].annotate(
            row["display_name"],
            (row["images_per_second"], row["loss_drop_per_second"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    axes[2].set_xlabel("Images / second")
    axes[2].set_ylabel("Loss drop / second")
    axes[2].set_title("Throughput vs convergence (Synthetic)")

    taguchi_df = taguchi.copy()
    taguchi_df["contrib_pct"] = _coerce_numeric(taguchi_df.get("contrib_pct"))
    taguchi_df = taguchi_df.sort_values("contrib_pct", ascending=False).head(8)
    axes[3].barh(taguchi_df["factor"], taguchi_df["contrib_pct"], color="#9e480e")
    axes[3].invert_yaxis()
    axes[3].set_xlabel("Contribution (%)")
    axes[3].set_title("Taguchi contribution summary")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    hdf5_path: Path = args.hdf5
    output_path: Optional[Path] = args.output
    if output_path is None:
        output_path = hdf5_path.with_name("compact_report.png")

    render_compact_report(hdf5_path, output_path)
    print(f"Saved compact report to {output_path}")


if __name__ == "__main__":
    main()
