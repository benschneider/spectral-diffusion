"""Tests for figure generation functionality."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from src.visualization.figures import (
    _collect_loss_histories,
    plot_loss_metrics,
    plot_taguchi_metric_distribution,
    generate_figures
)


class TestCollectLossHistories:
    """Test loss history collection from metrics."""

    def test_collect_with_valid_data(self, tmp_path):
        """Test collecting loss histories from valid metrics files."""
        # Create mock metrics directory
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()

        # Create sample metrics JSON
        sample_metrics = {
            "loss_history": [1.0, 0.8, 0.6, 0.4],
            "mae_history": [0.9, 0.7, 0.5, 0.3]
        }

        import json
        with open(metrics_dir / "run1.json", "w") as f:
            json.dump(sample_metrics, f)

        # Create summary CSV
        summary_df = pd.DataFrame({
            "run_id": ["run1"],
            "metrics_path": [str(metrics_dir / "run1.json")],
            "loss_final": [0.4]
        })

        result = _collect_loss_histories(summary_df)

        assert len(result) == 1
        assert result[0]["label"] == "run1"
        assert len(result[0]["loss_history"]) == 4
        assert result[0]["loss_history"] == [1.0, 0.8, 0.6, 0.4]

    def test_collect_with_missing_metrics(self, tmp_path):
        """Test handling of missing metrics files."""
        summary_df = pd.DataFrame({
            "run_id": ["run1"],
            "metrics_path": [str(tmp_path / "nonexistent.json")],
            "loss_final": [0.4]
        })

        result = _collect_loss_histories(summary_df)

        assert len(result) == 0

    def test_collect_with_empty_dataframe(self):
        """Test handling of empty summary dataframe."""
        summary_df = pd.DataFrame()

        result = _collect_loss_histories(summary_df)

        assert len(result) == 0


class TestPlotLossMetrics:
    """Test loss metrics plotting."""

    def test_plot_loss_metrics_basic(self, tmp_path):
        """Test basic loss metrics plotting."""
        df = pd.DataFrame({
            "run_id": ["run1", "run2"],
            "loss_drop": [0.5, 0.3],
            "loss_final": [0.3, 0.4],
            "display_name": ["Run 1", "Run 2"]
        })

        out_path = tmp_path / "test.png"
        plot_loss_metrics(df, "Test Title", out_path)

        # Verify file was created
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_loss_metrics_empty_df(self, tmp_path):
        """Test plotting with empty dataframe."""
        df = pd.DataFrame()

        out_path = tmp_path / "test.png"

        # Should raise KeyError for missing columns
        with pytest.raises(KeyError):
            plot_loss_metrics(df, "Test Title", out_path)


class TestPlotTaguchiMetricDistribution:
    """Test Taguchi metric distribution plotting."""

    def test_plot_taguchi_distribution_valid(self, tmp_path):
        """Test Taguchi distribution plotting with valid data."""
        taguchi_df = pd.DataFrame({
            "factor": ["factor_A", "factor_A"],
            "level": [1, 2],
            "mean_metric": [0.5, 0.3]
        })

        out_path = tmp_path / "test.png"
        plot_taguchi_metric_distribution(
            taguchi_df, "loss_drop_per_second", out_path
        )

        # Verify file was created
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_taguchi_distribution_no_factors(self, tmp_path):
        """Test plotting when no factor columns exist."""
        taguchi_df = pd.DataFrame({
            "run_id": ["run1"],
            "loss_drop_per_second": [0.5]
        })

        out_path = tmp_path / "test.png"

        # Should not raise an error and should not create file
        plot_taguchi_metric_distribution(
            taguchi_df, "loss_drop_per_second", out_path
        )

        # File should not be created due to missing columns
        assert not out_path.exists()


class TestGenerateFigures:
    """Test overall figure generation."""

    def test_generate_figures_integration(self, tmp_path):
        """Test the main figure generation integration."""
        # Create minimal CSV files to avoid warnings
        synthetic_dir = tmp_path / "synthetic"
        cifar_dir = tmp_path / "cifar"
        taguchi_dir = tmp_path / "taguchi"
        figures_dir = tmp_path / "figures"

        synthetic_dir.mkdir(parents=True)
        cifar_dir.mkdir(parents=True)
        taguchi_dir.mkdir(parents=True)

        # Create minimal summary CSVs
        pd.DataFrame({
            "run_id": ["syn1"],
            "loss_drop": [0.5],
            "loss_final": [0.4],
            "images_per_second": [100.0],
            "runtime_seconds": [1.0],
            "steps_per_second": [10.0]
        }).to_csv(synthetic_dir / "summary.csv", index=False)

        pd.DataFrame({
            "run_id": ["cifar1"],
            "loss_drop": [0.3],
            "loss_final": [0.3],
            "images_per_second": [50.0],
            "runtime_seconds": [2.0],
            "steps_per_second": [5.0]
        }).to_csv(cifar_dir / "summary.csv", index=False)

        generate_figures(
            synthetic_dir=synthetic_dir,
            cifar_dir=cifar_dir,
            taguchi_dir=taguchi_dir,
            output_dir=figures_dir,
            descriptions_path=None,
            generated_at="2023-01-01T00:00:00"
        )

        # Verify output directory was created and has content
        assert figures_dir.exists()
        assert (figures_dir / "summary.md").exists()


class TestFigureStyling:
    """Test figure styling and formatting."""

    def test_setup_style_applies_settings(self):
        """Test that style setup applies matplotlib settings."""
        from src.visualization.figures import _setup_style

        # Reset matplotlib settings
        plt.rcdefaults()

        _setup_style()

        # Check that our custom settings were applied
        assert plt.rcParams["axes.titlesize"] == 10
        assert plt.rcParams["figure.dpi"] == 300
        assert plt.rcParams["figure.constrained_layout.use"] == True

    def test_color_palette_generation(self):
        """Test color palette generation."""
        from src.visualization.figures import _color_palette

        colors = _color_palette(5)
        assert len(colors) == 5
        assert all(isinstance(c, str) for c in colors)

    def test_color_palette_large_n(self):
        """Test color palette with large n."""
        from src.visualization.figures import _color_palette

        colors = _color_palette(20)
        assert len(colors) == 20