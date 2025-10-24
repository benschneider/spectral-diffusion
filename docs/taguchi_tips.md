# Taguchi Analysis Quick Tips

## Auto-generated reports

- `src.experiments.run_experiment` now writes `taguchi_report.csv` automatically when `--report-metric` is provided. 
  Use scripts (`run_taguchi_smoke.sh`, `run_taguchi_comparison.sh`, `run_taguchi_minimal.sh`) to trigger the auto report.
- `taguchi_report.csv` contains factor/level S/N results; the `rank` column highlights the most influential factors.

## Example summary

Artifacts from `results/taguchi_spectral_docs/` include:
- `summary.csv`: raw per-run metrics
- `taguchi_report.csv`: ranked S/N table

## Generating notebooks

Use `pandas.read_csv("results/taguchi_spectral_docs/taguchi_report.csv")` to load the table and build charts.