# Report HDF5 Layout

This document defines the canonical structure for consolidated experiment
artefacts stored inside ``.h5`` archives. The packager introduced in
``scripts/pack_report_to_hdf5.py`` ingests the legacy directory layout (for
example ``results/full_report_32x32_*``) and produces a single file with the
following guarantees:

- Every CSV, JSON, YAML, Markdown, and HTML artefact from the original report is
  stored as a structured dataset or UTF-8 text block.
- Figures (PNG, JPG, PDF, SVG, GIF) are inlined as binary datasets when the
  ``--no-figures`` flag is not supplied.
- A machine-readable index links each dataset back to its original location for
  traceability.

The goal is to replace the scattered ``summary.csv`` / ``*.json`` / ``*.md``
tree with a one-stop container that downstream notebooks, scripts, and paper
figure pipelines can query.

## High-level organisation

```
/<root>
├─ attrs: created_utc, report_root, generator, generator_version
├─ /artefacts
│  ├─ /cifar
│  │  ├─ /summary.csv
│  │  │  └─ table                ← structured dataset of run metrics
│  │  └─ /sanity
│  │     ├─ cifar_32x32_tiny_sanity_cifar10.json
│  │     │  └─ json             ← pretty-printed JSON payload
│  │     └─ ...
│  ├─ /synthetic
│  │  ├─ /summary.csv
│  │  │  └─ table
│  │  └─ /runs
│  │     └─ piecewise_32x32_tiny
│  │        ├─ config.yaml
│  │        │  └─ yaml          ← YAML text
│  │        └─ metrics.json
│  │           └─ json
│  ├─ /figures
│  │  ├─ taguchi_main_effects.csv
│  │  │  └─ table
│  │  ├─ taguchi_main_effects.png
│  │  │  └─ binary              ← uint8 buffer of the PNG file
│  │  └─ summary.md
│  │     └─ text
│  └─ ... (ablation, taguchi, synthetic configs, etc.)
└─ /index
   └─ files                      ← structured dataset describing every artefact
```

Every file from the original report directory maps to a **group** inside
``/artefacts`` whose name matches the source filename (including the extension).
The group's attributes expose ``source_path`` and ``file_type``. The actual
payload is stored in a dataset whose name indicates the representation:

| Representation | Dataset name | Notes |
| -------------- | ------------ | ----- |
| CSV table      | ``table``    | Structured array with typed columns. Column names and row count are mirrored in dataset attributes. |
| JSON payload   | ``json``     | Pretty-printed UTF-8 text. Top-level keys are stored in ``keys`` attribute. |
| YAML payload   | ``yaml``     | Preserves original ordering. Top-level keys recorded in ``keys`` attribute. |
| Markdown/HTML  | ``text``     | UTF-8 text stored as a length-1 dataset with a ``length`` attribute. |
| Figures        | ``binary``   | Raw bytes (``uint8``) compressed with the packager's codec. |

The ``/index/files`` dataset contains the canonical mapping table:

| Column         | Description |
| -------------- | ----------- |
| ``relative_path`` | Path relative to the source report directory. |
| ``dataset_path``  | Absolute path of the dataset inside the HDF5 archive. |
| ``file_type``     | ``table`` / ``json`` / ``yaml`` / ``text`` / ``figure``. |
| ``size``          | File size on disk (bytes). |
| ``sha256``        | SHA-256 digest of the original file for provenance. |

## Migrating away from the legacy layout

The previous workflow relied on the folder tree under ``results/<report>/``. The
packager keeps that structure discoverable via ``relative_path`` so that we can
remove the legacy tree once downstream consumers have switched to HDF5. For
tracking purposes, keep the old directory on disk until dependent notebooks have
been updated. After validation, future runs can write **only** the ``.h5``
archive alongside lightweight previews (such as ``compact_report.png``).

Key mappings:

| Legacy location                                              | HDF5 dataset                                       |
| ------------------------------------------------------------ | -------------------------------------------------- |
| ``results/.../summary.csv``                                  | ``/artefacts/summary.csv/table``                   |
| ``results/.../cifar/summary.csv``                            | ``/artefacts/cifar/summary.csv/table``             |
| ``results/.../synthetic/runs/*/config.yaml``                 | ``/artefacts/synthetic/runs/*/config.yaml/yaml``   |
| ``results/.../cifar/runs/*/metrics/*.json``                  | ``/artefacts/cifar/runs/*/metrics/*.json/json``    |
| ``results/.../figures/taguchi_main_effects.png``             | ``/artefacts/figures/taguchi_main_effects.png/binary`` |
| ``results/.../figures/taguchi_main_effects.csv``             | ``/artefacts/figures/taguchi_main_effects.csv/table`` |
| ``results/.../figures/summary.md``                           | ``/artefacts/figures/summary.md/text``             |

## Generating visual summaries from HDF5

``scripts/render_compact_report.py`` consumes the archive and produces a concise
4-panel figure that combines:

1. Synthetic PSNR comparison.
2. CIFAR-10 PSNR comparison.
3. Throughput versus convergence scatter plot.
4. Top Taguchi factor contributions.

This figure is meant to replace the sprawling figure directory with a quick
status snapshot. It can be regenerated at any time:

```
scripts/pack_report_to_hdf5.py results/full_report_32x32_20251102_142304
scripts/render_compact_report.py results/full_report_32x32_20251102_142304.h5
```

The render script relies solely on the HDF5 archive; no raw CSV/JSON files are
needed once the archive is produced.

## Follow-up actions

- Update downstream notebooks to read from ``/index/files`` instead of walking
  the filesystem.
- Remove bespoke ``json`` / ``csv`` loaders once the migration is complete.
- Extend ``render_compact_report.py`` with additional panels (e.g. FID, SSIM)
  as new experiments arrive.

Documenting these mappings ensures that future refactors can delete the legacy
folders with confidence while preserving experiment provenance inside a single
source of truth.
