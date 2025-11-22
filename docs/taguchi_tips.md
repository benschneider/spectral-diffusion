# Taguchi Analysis Quick Tips

## Auto-generated reports

- `src.experiments.run_experiment` writes `taguchi_report.csv` automatically when `--report-metric` is provided.
- Summary CSV: `results/<run>/summary.csv`; S/N report: `results/<run>/taguchi_report.csv`.

## Example summary

- Use `scripts/generate_report_v2.py` to produce the cleaned report bundle (Taguchi plots + summary).
- Example batch: `results/taguchi_spectral_docs/` contains both files.

## Notebook integration

- Load reports with `pandas.read_csv` for dashboards or notebooks.
- Factors are ranked by the `rank` column in `taguchi_report.csv`.
