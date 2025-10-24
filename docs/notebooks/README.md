# Notebooks (coming soon)

This folder will host interactive demos. Suggested pipeline once data exists:

1. Run `scripts/run_full_report.sh` to generate benchmark artefacts.
2. Open `docs/notebooks/spectral_vs_spatial.ipynb` (to be authored) to explore:
   - loss vs. time curves for TinyUNet vs SpectralUNet
   - frequency-domain visualisations
   - Taguchi factor effects

If you create a notebook, please keep imports modular (use `src.visualization` helpers rather than rewriting plotting code).
