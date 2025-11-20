import json
import logging
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

from src.cli.common import (
    append_run_summary,
    configure_run_logger,
    ensure_directories,
    save_config_snapshot,
    save_metrics,
)
from src.training import TrainingPipeline

try:  # pragma: no cover - optional dependency already handled downstream
    from src.analysis.taguchi_stats import generate_taguchi_report
except Exception:  # pragma: no cover
    generate_taguchi_report = None  # type: ignore


def load_factor_registry(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the factor registry YAML describing Taguchi factor levels."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    factors = data.get("factors")
    if not isinstance(factors, dict) or not factors:
        raise ValueError(f"No factors found in registry {path}")
    return factors


def randomize_factor_mapping(factors: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, str]:
    """
    Produce a randomized mapping from Taguchi column letters to factor names.

    Primarily useful for shuffling factor assignments before distributing a batch.
    """
    keys = list(factors.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    return {chr(65 + idx): name for idx, name in enumerate(keys)}


def _design_columns(design: pd.DataFrame) -> List[str]:
    return [col for col in design.columns if col.lower() not in {"run", "row"}]


def _column_cardinality(design: pd.DataFrame, columns: Sequence[str]) -> Dict[str, int]:
    return {col: int(design[col].nunique()) for col in columns}


def _factor_level_count(factor_info: Dict[str, Any]) -> int:
    levels = factor_info.get("levels", [])
    if not isinstance(levels, list):
        raise ValueError("Each factor in the registry must define a list of 'levels'.")
    return len(levels)


def build_factor_column_mapping(
    factors: Dict[str, Dict[str, Any]],
    design: pd.DataFrame,
    randomize: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, str]:
    """
    Assign Taguchi design columns to factors, respecting level cardinality.

    Columns with <=2 distinct levels are treated as binary, while those with >=3
    are treated as ternary factors. Randomisation shuffles factor order prior to
    assignment while keeping column ordering stable.
    """
    columns = _design_columns(design)
    if not columns:
        raise ValueError("Design matrix must contain Taguchi columns (A, B, ...).")

    cardinality = _column_cardinality(design, columns)
    binary_columns = [col for col in columns if cardinality[col] <= 2]
    ternary_columns = [col for col in columns if cardinality[col] > 2]

    if not ternary_columns and not binary_columns:
        raise ValueError("Unable to infer column cardinalities for Taguchi mapping.")

    factor_names = list(factors.keys())
    ordered_factors = (
        list(randomize_factor_mapping(factors, seed).values())
        if randomize
        else factor_names
    )

    ternary_factors = [
        name for name in ordered_factors if _factor_level_count(factors[name]) >= 3
    ]
    binary_factors = [
        name for name in ordered_factors if _factor_level_count(factors[name]) <= 2
    ]

    if len(ternary_factors) != len(ternary_columns):
        raise ValueError(
            f"Design expects {len(ternary_columns)} ternary factors but registry "
            f"contains {len(ternary_factors)} (levels >=3)."
        )
    if len(binary_factors) != len(binary_columns):
        raise ValueError(
            f"Design expects {len(binary_columns)} binary factors but registry "
            f"contains {len(binary_factors)} (levels <=2)."
        )

    mapping: Dict[str, str] = {}
    for col, factor in zip(sorted(ternary_columns), ternary_factors):
        mapping[col] = factor
    for col, factor in zip(sorted(binary_columns), binary_factors):
        mapping[col] = factor
    return mapping


def _apply_adapter_placement(cfg: Dict[str, Any], label: str) -> None:
    spectral_cfg = cfg.setdefault("spectral", {})
    placement_map = {
        "none": [],
        "input_only": ["input"],
        "input_and_output": ["input", "output"],
    }
    apply_to = placement_map.get(label)
    if apply_to is None:
        raise ValueError(f"Unknown spectral_adapter_placement level '{label}'")
    spectral_cfg["apply_to"] = apply_to
    spectral_cfg["enabled"] = bool(apply_to)


def _apply_loss_weighting(cfg: Dict[str, Any], label: str) -> None:
    spectral_cfg = cfg.setdefault("spectral", {})
    spectral_cfg.pop("bandpass_inner", None)
    spectral_cfg.pop("bandpass_outer", None)
    if label == "none":
        spectral_cfg["weighting"] = "none"
    elif label == "radial_highfreq":
        spectral_cfg["weighting"] = "radial"
    elif label == "aggressive_highfreq":
        spectral_cfg["weighting"] = "bandpass"
        spectral_cfg.setdefault("bandpass_inner", 0.25)
        spectral_cfg.setdefault("bandpass_outer", 0.75)
    else:
        raise ValueError(f"Unknown spectral_loss_weighting level '{label}'")


def _apply_noise_shaping(cfg: Dict[str, Any], label: str) -> None:
    diffusion_cfg = cfg.setdefault("diffusion", {})
    spectral_cfg = cfg.setdefault("spectral", {})
    if label == "off":
        diffusion_cfg["uniform_corruption"] = False
        spectral_cfg["freq_equalized_noise"] = False
    elif label == "mild_equalize":
        diffusion_cfg["uniform_corruption"] = True
        spectral_cfg["freq_equalized_noise"] = False
    elif label == "strong_equalize":
        diffusion_cfg["uniform_corruption"] = True
        spectral_cfg["freq_equalized_noise"] = True
    else:
        raise ValueError(f"Unknown spectral_noise_shaping_strength level '{label}'")


def _apply_snr_ratio(cfg: Dict[str, Any], label: str) -> None:
    diffusion_cfg = cfg.setdefault("diffusion", {})
    spectral_cfg = cfg.setdefault("spectral", {})
    try:
        value = float(label)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"snr_ratio level '{label}' is not numeric") from exc
    diffusion_cfg["snr_ratio"] = value
    spectral_cfg["snr_ratio"] = value


def _apply_phase_capacity(cfg: Dict[str, Any], label: str) -> None:
    model_cfg = cfg.setdefault("model", {})
    if label == "off":
        model_cfg["enable_phase_attention"] = False
        model_cfg["phase_heads"] = 0
    elif label == "tiny":
        model_cfg["enable_phase_attention"] = True
        model_cfg["phase_heads"] = 1
    elif label == "full":
        model_cfg["enable_phase_attention"] = True
        model_cfg["phase_heads"] = 4
    else:
        raise ValueError(f"Unknown phase_attention_capacity level '{label}'")


def _apply_sampler(cfg: Dict[str, Any], label: str) -> None:
    sampling_cfg = cfg.setdefault("sampling", {})
    sampler_map = {
        "ddim": "ddim",
        "dpm_solver_pp": "dpm_solver++",
        "spectral_guided": "masf",
    }
    sampler = sampler_map.get(label)
    if sampler is None:
        raise ValueError(f"Unknown sampler_type level '{label}'")
    sampling_cfg["sampler_type"] = sampler


def _apply_sampling_steps(cfg: Dict[str, Any], label: str) -> None:
    sampling_cfg = cfg.setdefault("sampling", {})
    step_map = {
        "30": 30,
        "50": 50,
        "100": 100,
    }
    if str(label) not in step_map:
        raise ValueError(f"Unknown sampling_steps level '{label}'")
    sampling_cfg["num_steps"] = step_map[str(label)]


def _apply_curriculum(cfg: Dict[str, Any], label: str) -> None:
    training_cfg = cfg.setdefault("training", {})
    if label == "none":
        training_cfg.pop("curriculum", None)
    elif label == "lowres_warmup":
        training_cfg["curriculum"] = {
            "mode": "lowres_warmup",
            "warmup_epochs": 1,
            "resolution": 8,
        }
    elif label == "spectral_first":
        training_cfg["curriculum"] = {
            "mode": "spectral_first",
            "spectral_epochs": 1,
        }
    else:
        raise ValueError(f"Unknown curriculum_mode level '{label}'")


def _apply_train_steps(cfg: Dict[str, Any], label: str) -> None:
    training_cfg = cfg.setdefault("training", {})
    steps = int(label)
    if steps <= 0:
        raise ValueError("Training steps must be positive.")
    training_cfg["num_batches"] = steps
    training_cfg.setdefault("epochs", 1)


def _apply_image_resolution(cfg: Dict[str, Any], label: str) -> None:
    data_cfg = cfg.setdefault("data", {})
    res = int(label)
    if res <= 0:
        raise ValueError("Image resolution must be positive.")
    data_cfg["height"] = res
    data_cfg["width"] = res


_FACTOR_APPLIERS = {
    "spectral_adapter_placement": _apply_adapter_placement,
    "spectral_loss_weighting": _apply_loss_weighting,
    "spectral_noise_shaping_strength": _apply_noise_shaping,
    "snr_ratio": _apply_snr_ratio,
    "phase_attention_capacity": _apply_phase_capacity,
    "sampler_type": _apply_sampler,
    "sampling_steps": _apply_sampling_steps,
    "curriculum_mode": _apply_curriculum,
    "train_steps": _apply_train_steps,
    "image_resolution": _apply_image_resolution,
}


def apply_factor_to_config(
    cfg: Dict[str, Any],
    factor_name: str,
    level_index: int,
    factor_registry: Dict[str, Dict[str, Any]],
) -> str:
    """
    Mutate a configuration dictionary in-place based on a factor level.

    Returns the human-readable level label for bookkeeping.
    """
    factor_info = factor_registry.get(factor_name)
    if factor_info is None:
        raise KeyError(f"Factor '{factor_name}' not found in registry.")
    levels = factor_info.get("levels", [])
    if not levels:
        raise ValueError(f"Factor '{factor_name}' does not define any levels.")
    if level_index < 1 or level_index > len(levels):
        raise ValueError(
            f"Level index {level_index} out of range for factor '{factor_name}' "
            f"(1..{len(levels)})."
        )
    label = str(levels[level_index - 1])
    applier = _FACTOR_APPLIERS.get(factor_name)
    if applier is None:
        raise KeyError(f"No applier registered for factor '{factor_name}'.")
    applier(cfg, label)
    return label

class TaguchiExperimentRunner:
    """Automate Taguchi design experiments across spectral diffusion variants."""

    def __init__(
        self,
        design_matrix_path: Path,
        base_config: Dict[str, Any],
        factor_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        mapping_seed: Optional[int] = None,
    ) -> None:
        self.design_matrix_path = Path(design_matrix_path)
        self.base_config = deepcopy(base_config)
        self.design = self._load_design_matrix()
        self.summary_filename = f"{self.design_matrix_path.stem}_summary.csv"

        self.factor_registry: Dict[str, Dict[str, Any]] = factor_registry or {}
        self.column_mapping: Optional[Dict[str, str]] = None
        self.mapping_seed = mapping_seed
        self.mapping_generated_at: Optional[str] = None

        if factor_registry:
            if column_mapping:
                self.column_mapping = column_mapping
                self.mapping_generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            else:
                self.set_factor_registry(factor_registry, randomize=False, seed=mapping_seed)
        elif column_mapping:
            self.column_mapping = column_mapping
            self.mapping_generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _load_design_matrix(self) -> pd.DataFrame:
        """Load the Taguchi orthogonal array describing the experiment batch."""
        return pd.read_csv(self.design_matrix_path)

    def set_factor_registry(
        self,
        factors: Dict[str, Dict[str, Any]],
        randomize: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """Register Taguchi factors and compute a column mapping."""
        self.factor_registry = factors
        self.mapping_seed = seed
        self.column_mapping = build_factor_column_mapping(
            factors=factors,
            design=self.design,
            randomize=randomize,
            seed=seed,
        )
        self.mapping_generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _ensure_mapping(self) -> None:
        if not self.factor_registry:
            return
        if self.column_mapping is None:
            self.set_factor_registry(self.factor_registry, randomize=False, seed=self.mapping_seed)

    def _mapping_payload(self) -> Optional[Dict[str, Any]]:
        if not self.factor_registry or not self.column_mapping:
            return None
        if self.mapping_generated_at is None:
            self.mapping_generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {
            "design_matrix": str(self.design_matrix_path),
            "generated_at": self.mapping_generated_at,
            "mapping": self.column_mapping,
            "seed": self.mapping_seed,
            "factors": {
                name: {"levels": info.get("levels", [])}
                for name, info in self.factor_registry.items()
            },
        }

    def _write_mapping_file(self, path: Path, payload: Optional[Dict[str, Any]]) -> None:
        if payload is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _collect_rows(self, row_indices: Optional[Sequence[int]]) -> List[Tuple[int, pd.Series]]:
        if row_indices is None:
            return list(self.design.iterrows())

        selected: List[Tuple[int, pd.Series]] = []
        for identifier in row_indices:
            idx_val = int(identifier)
            if "run" in self.design.columns:
                matches = self.design[self.design["run"] == idx_val]
                if matches.empty:
                    raise ValueError(f"Row {idx_val} not found in design matrix.")
                design_idx = int(matches.index[0])
            else:
                design_idx = idx_val - 1
                if design_idx < 0 or design_idx >= len(self.design):
                    raise ValueError(f"Row index {idx_val} out of bounds for design matrix.")
            selected.append((design_idx, self.design.loc[design_idx]))
        return selected

    def run_batch(
        self,
        output_dir: Path,
        logger=None,
        report_metric: Optional[str] = None,
        report_mode: str = "larger",
        report_path: Optional[Path] = None,
        row_indices: Optional[Sequence[int]] = None,
        finalize: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run experiments for the specified rows of the Taguchi design."""
        results: List[Dict[str, Any]] = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log = logger or logging.getLogger("spectral_diffusion.taguchi")

        self._ensure_mapping()
        mapping_payload = self._mapping_payload()
        self._write_mapping_file(output_dir / "factor_mapping.json", mapping_payload)

        rows = self._collect_rows(row_indices)
        summary_path = output_dir / self.summary_filename
        finalize_batch = finalize and len(rows) == len(self.design)

        for design_idx, row in rows:
            row_number = int(row.get("run", design_idx + 1))
            run_config = self._build_config_from_row(row=row, row_number=row_number)
            factor_levels = (
                run_config.get("taguchi", {}).get("factor_levels", {})
                if isinstance(run_config.get("taguchi"), dict)
                else {}
            )
            run_id = self._make_run_id(row_number=row_number, factor_levels=factor_levels)
            dirs = ensure_directories(output_dir=output_dir, run_id=run_id)
            self._write_mapping_file(dirs["run_dir"] / "factor_mapping.json", mapping_payload)

            config_copy_path = dirs["run_dir"] / "config.yaml"
            save_config_snapshot(config=run_config, destination=config_copy_path)

            run_logger = logger or logging.getLogger(f"spectral_diffusion.taguchi.{run_id}")
            configure_run_logger(run_logger, dirs["logs_dir"] / "train.log")
            summary_note = ", ".join(
                f"{name}={meta.get('level_label', meta.get('level_index'))}"
                for name, meta in sorted(factor_levels.items())
            )
            log.info(
                "Running Taguchi row %d/%d (run_id=%s%s)",
                row_number,
                len(self.design),
                run_id,
                f", factors: {summary_note}" if summary_note else "",
            )

            pipeline = TrainingPipeline(config=run_config, work_dir=dirs["run_dir"], logger=run_logger)
            metrics = pipeline.run()

            metrics_path = dirs["metrics_dir"] / f"{run_id}.json"
            save_metrics(metrics=metrics, destination=metrics_path)
            flat_metrics_path = output_dir / f"run_{row_number}_metrics.json"
            self._write_flat_metrics(flat_metrics_path, metrics)

            append_run_summary(
                run_id=run_id,
                config_path=config_copy_path,
                metrics_path=metrics_path,
                summary_path=summary_path,
                metrics=metrics,
            )

            results.append(
                {
                    "run_id": run_id,
                    "row_number": row_number,
                    "config_path": config_copy_path,
                    "metrics_path": metrics_path,
                    "metrics": metrics,
                }
            )

        if summary_path.exists():
            summary_copy = output_dir / "summary.csv"
            if summary_copy != summary_path:
                summary_copy.write_text(summary_path.read_text())
            stem_prefix = self.design_matrix_path.stem.split("_")[0]
            alt_summary = output_dir / f"{stem_prefix}_summary.csv"
            if alt_summary != summary_path and alt_summary != summary_copy:
                alt_summary.write_text(summary_path.read_text())

        if finalize_batch and report_metric and generate_taguchi_report is not None and summary_path.exists():
            report_target = report_path or (output_dir / "taguchi_report.csv")
            try:
                generate_taguchi_report(
                    summary_path=summary_path,
                    metric=report_metric,
                    mode=report_mode,
                    output_path=report_target,
                )
                log.info("Taguchi report written to %s", report_target)
            except ValueError as exc:
                log.warning("Unable to generate Taguchi report: %s", exc)
        elif finalize_batch and report_metric:
            log.warning("Taguchi report requested but dependency not available. Install pandas/yaml.")
        return results

    def _legacy_build_config(self, row: pd.Series, row_number: int) -> Dict[str, Any]:
        cfg = deepcopy(self.base_config)
        cfg.setdefault("model", {})
        cfg.setdefault("spectral", {})
        cfg.setdefault("sampling", {})
        cfg.setdefault("initialization", {})

        if "A" in row:
            cfg["spectral"]["freq_equalized_noise"] = int(row["A"]) == 2
        if "B" in row:
            cfg["spectral"]["freq_attention"] = int(row["B"]) == 2
        if "C" in row:
            cfg["sampling"]["sampler_type"] = "dpm_solver++" if int(row["C"]) == 2 else "ddim"
        if "D" in row:
            cfg["spectral"]["enabled"] = int(row["D"]) == 2
        if "E" in row:
            level = int(row["E"])
            init_cfg = cfg.setdefault("initialization", {})
            if level == 2:
                init_cfg.update(
                    {
                        "strategy": "cross_domain_flat",
                        "scale": init_cfg.get("scale", 0.02),
                        "recycle": True,
                    }
                )
                if "source" not in init_cfg:
                    init_cfg["source"] = {
                        "type": "constant",
                        "values": [0.0, 1.0, -1.0, 0.5],
                    }
            else:
                init_cfg.setdefault("strategy", "default")

        taguchi_meta = cfg.setdefault("taguchi", {})
        taguchi_meta["row"] = row.to_dict()
        taguchi_meta["row_number"] = row_number
        return cfg

    def _build_config_from_row(self, row: pd.Series, row_number: int) -> Dict[str, Any]:
        """
        Merge base configuration with row-specific overrides from the design matrix.
        """
        if not self.factor_registry or not self.column_mapping:
            return self._legacy_build_config(row=row, row_number=row_number)

        cfg = deepcopy(self.base_config)
        cfg.setdefault("model", {})
        cfg.setdefault("spectral", {})
        cfg.setdefault("sampling", {})
        cfg.setdefault("initialization", {})
        cfg.setdefault("diffusion", {})
        cfg.setdefault("training", {})
        cfg.setdefault("optim", {})

        design_levels: Dict[str, int] = {}
        factor_levels: Dict[str, Dict[str, Any]] = {}
        for column, factor_name in self.column_mapping.items():
            if column not in row:
                continue
            level_value = int(row[column])
            label = apply_factor_to_config(
                cfg=cfg,
                factor_name=factor_name,
                level_index=level_value,
                factor_registry=self.factor_registry,
            )
            design_levels[column] = level_value
            factor_levels[factor_name] = {
                "column": column,
                "level_index": level_value,
                "level_label": label,
            }

        taguchi_meta = cfg.setdefault("taguchi", {})
        taguchi_meta["row"] = design_levels
        taguchi_meta["row_number"] = row_number
        if factor_levels:
            taguchi_meta["factor_levels"] = factor_levels
        if self.column_mapping:
            taguchi_meta["factor_mapping"] = self.column_mapping
        if self.mapping_generated_at:
            taguchi_meta["mapping_generated_at"] = self.mapping_generated_at
        if self.mapping_seed is not None:
            taguchi_meta["mapping_seed"] = self.mapping_seed
        return cfg

    @staticmethod
    def _write_flat_metrics(destination: Path, metrics: Dict[str, Any]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    @staticmethod
    def _make_run_id(
        row_number: int,
        factor_levels: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        suffix = ""
        if factor_levels:
            snr_meta = factor_levels.get("snr_ratio")
            if isinstance(snr_meta, dict):
                label = str(
                    snr_meta.get("level_label", snr_meta.get("level_index", ""))
                )
                sanitized = label.replace(" ", "")
                sanitized = sanitized.replace(".", "p").replace("/", "_")
                if not sanitized:
                    sanitized = "lvl"
                suffix = f"_snr{sanitized}"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S%f")
        return f"taguchi_row{row_number:02d}{suffix}_{timestamp}"


def run_experiments(
    design_matrix: Path,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    report_metric: Optional[str] = None,
    report_mode: str = "larger",
    report_path: Optional[Path] = None,
    factor_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    randomize_mapping: bool = False,
    mapping_seed: Optional[int] = None,
    row_indices: Optional[Sequence[int]] = None,
    finalize: bool = True,
) -> List[Dict[str, Any]]:
    """Convenience wrapper function for running a Taguchi batch."""
    runner = TaguchiExperimentRunner(
        design_matrix_path=design_matrix,
        base_config=config,
    )
    if factor_registry:
        runner.set_factor_registry(
            factors=factor_registry,
            randomize=randomize_mapping,
            seed=mapping_seed,
        )
    return runner.run_batch(
        output_dir=output_dir or Path("results"),
        report_metric=report_metric,
        report_mode=report_mode,
        report_path=report_path,
        row_indices=row_indices,
        finalize=finalize,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Taguchi experiments")
    parser.add_argument("--config", type=Path, required=True, help="Base YAML configuration")
    parser.add_argument("--array", type=Path, required=True, help="Taguchi design CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/taguchi_run"),
        help="Directory to store run artifacts",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument(
        "--report-metric",
        type=str,
        default="loss_drop_per_second",
        help="Metric column to analyze for Taguchi S/N reporting",
    )
    parser.add_argument(
        "--report-mode",
        choices=["larger", "smaller"],
        default="larger",
        help="S/N mode for Taguchi report",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional override for Taguchi report output path",
    )
    parser.add_argument(
        "--factor-registry",
        type=Path,
        default=None,
        help="Path to factor registry YAML describing factor levels.",
    )
    parser.add_argument(
        "--randomize-mapping",
        action="store_true",
        help="Randomly assign factors to Taguchi columns before running.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for factor-to-column assignment.",
    )
    parser.add_argument(
        "--row",
        type=int,
        action="append",
        help="Specific design row(s) to execute (1-indexed). Can be set multiple times.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Generate Taguchi summaries even when running a subset of rows.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    with args.config.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}

    factor_registry = None
    if args.factor_registry is not None:
        factor_registry = load_factor_registry(args.factor_registry)

    row_indices = args.row
    should_finalize = args.finalize or (row_indices is None)

    results = run_experiments(
        design_matrix=args.array,
        config=base_config,
        output_dir=args.output_dir,
        report_metric=args.report_metric,
        report_mode=args.report_mode,
        report_path=args.report_path,
        factor_registry=factor_registry,
        randomize_mapping=args.randomize_mapping,
        mapping_seed=args.seed,
        row_indices=row_indices,
        finalize=should_finalize,
    )

    summary_name = f"{args.array.stem}_summary.csv"
    summary_path = args.output_dir / summary_name
    if row_indices:
        print(
            f"Completed {len(results)} Taguchi run(s) for rows {row_indices}. "
            f"Artifacts written to {args.output_dir}."
        )
    else:
        print(f"Completed {len(results)} Taguchi runs. Summary stored in {summary_path}")


if __name__ == "__main__":
    main()
