# Notebooks (coming soon)

This folder will host interactive demos. Suggested pipeline once data exists:

1. Run a smoke pipeline (`scripts/run_taguchi_smoke.sh` or `python train.py --config configs/baseline.yaml`) to generate artefacts.
2. Open `docs/notebooks/spectral_vs_spatial.ipynb` (to be authored) to explore:
   - loss vs. time curves for TinyUNet
   - frequency-domain visualisations
   - Taguchi factor effects using the six active knobs

If you create a notebook, please keep imports modular (use `src.visualization` helpers rather than rewriting plotting code).
