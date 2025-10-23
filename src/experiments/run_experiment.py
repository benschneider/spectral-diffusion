import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.training import TrainingPipeline
from train_model import (
    append_run_summary,
    configure_run_logger,
    ensure_directories,
    save_config_snapshot,
    save_metrics,
)


class TaguchiExperimentRunner:
    """Automate Taguchi design experiments across spectral diffusion variants."""

    def __init__(self, design_matrix_path: Path, base_config: Dict[str, Any]) -> None:
        self.design_matrix_path = design_matrix_path
        self.base_config = base_config
        self.design = self._load_design_matrix()

    def _load_design_matrix(self) -> pd.DataFrame:
        """Load the Taguchi orthogonal array describing the experiment batch."""
        return pd.read_csv(self.design_matrix_path)

    def run_batch(self, output_dir: Path, logger=None) -> List[Dict[str, Any]]:
        """Run a batch of experiments defined by the Taguchi design."""
        results: List[Dict[str, Any]] = []
        output_dir = Path(output_dir)
        summary_path = output_dir / "summary.csv"

        for idx, row in self.design.iterrows():
            run_id = self._make_run_id(index=idx)
            run_config = self._build_config_from_row(row=row)
            dirs = ensure_directories(output_dir=output_dir, run_id=run_id)

            config_copy_path = dirs["run_dir"] / "config.yaml"
            save_config_snapshot(config=run_config, destination=config_copy_path)

            run_logger = logger or logging.getLogger(f"spectral_diffusion.taguchi.{run_id}")
            configure_run_logger(run_logger, dirs["run_dir"] / "run.log")

            pipeline = TrainingPipeline(config=run_config, work_dir=dirs["run_dir"], logger=run_logger)
            metrics = pipeline.run()

            metrics_path = dirs["metrics_dir"] / f"{run_id}.json"
            save_metrics(metrics=metrics, destination=metrics_path)
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
                    "config_path": config_copy_path,
                    "metrics_path": metrics_path,
                    "metrics": metrics,
                }
            )
        return results

    def _build_config_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Merge base configuration with row-specific overrides from the design matrix.

        Expected columns (example L8):
          A (freq_equalized_noise): 1=off, 2=on
          B (freq_attention):       1=off, 2=on
          C (sampler):              1=ddim, 2=dpm-solver
        """
        cfg = deepcopy(self.base_config)

        cfg.setdefault("model", {})
        cfg.setdefault("spectral", {})
        cfg.setdefault("sampling", {})

        if "A" in row:
            cfg["spectral"]["freq_equalized_noise"] = int(row["A"]) == 2
        if "B" in row:
            cfg["spectral"]["freq_attention"] = int(row["B"]) == 2
        if "C" in row:
            cfg["sampling"]["sampler_type"] = "dpm-solver" if int(row["C"]) == 2 else "ddim"

        taguchi_meta = cfg.setdefault("taguchi", {})
        taguchi_meta["row"] = row.to_dict()
        return cfg

    @staticmethod
    def _make_run_id(index: int) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S%f")
        return f"taguchi_{index:03d}_{timestamp}"


def run_experiments(design_matrix: Path, config: Dict[str, Any], output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Convenience wrapper function for running a Taguchi batch."""
    runner = TaguchiExperimentRunner(design_matrix_path=design_matrix, base_config=config)
    return runner.run_batch(output_dir=output_dir or Path("results"))
