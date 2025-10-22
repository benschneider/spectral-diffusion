from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.training import TrainingPipeline


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
        results = []
        for _, row in self.design.iterrows():
            run_config = self._build_config_from_row(row=row)
            pipeline = TrainingPipeline(config=run_config, work_dir=output_dir, logger=logger)
            metrics = pipeline.run()
            results.append(metrics)
        return results

    def _build_config_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Merge base configuration with row-specific overrides from the design matrix.

        Expected columns (example L8):
          A (freq_equalized_noise): 1=off, 2=on
          B (freq_attention):       1=off, 2=on
          C (sampler):              1=ddim, 2=dpm-solver
        """
        cfg = {**self.base_config}

        cfg.setdefault("model", {})
        cfg.setdefault("spectral", {})
        cfg.setdefault("sampling", {})

        if "A" in row:
            cfg["spectral"]["freq_equalized_noise"] = int(row["A"]) == 2
        if "B" in row:
            cfg["spectral"]["freq_attention"] = int(row["B"]) == 2
        if "C" in row:
            cfg["sampling"]["sampler_type"] = "dpm-solver" if int(row["C"]) == 2 else "ddim"

        cfg.setdefault("taguchi", {})["row"] = row.to_dict()
        return cfg


def run_experiments(design_matrix: Path, config: Dict[str, Any], output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Convenience wrapper function for running a Taguchi batch."""
    runner = TaguchiExperimentRunner(design_matrix_path=design_matrix, base_config=config)
    return runner.run_batch(output_dir=output_dir or Path("results"))
