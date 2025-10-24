# Taguchi Analysis Quick Tips

## Auto-generated reports

- `src.experiments.run_experiment` writes `taguchi_report.csv` automatically when `--report-metric` is provided.
- Summary CSV: `results/<run>/summary.csv`; S/N report: `results/<run>/taguchi_report.csv`.

## Example summary

- Use `scripts/figures/generate_figures.py` to produce publication-ready plots.
- Example batch: `results/taguchi_spectral_docs/` contains both files.

## Notebook integration

- Load reports with `pandas.read_csv` for dashboards or notebooks.
- Factors are ranked by the `rank` column in `taguchi_report.csv`.