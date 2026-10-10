# Reproduction materials for Monthly conditional flow distributions reveal seasonal mismatches in hydrological donor selection

This repository contains the code, derived inputs, plotted data, and verification records for the study **Monthly conditional flow distributions reveal seasonal mismatches in hydrological donor selection** by Zijie Deng, Yu Xia, Jiyang Yu, and Zhenyu Liao.

The analysis compares an annual flow-duration representation with monthly conditional flow distributions that retain the joint structure of flow magnitude and month of occurrence. The repository contains climate-stratified evaluations, resolution and donor-count controls, temporal-transfer tests, prediction controls, and the workflow for rebuilding derived inputs from the Caravan data release.

## Repository layout

- `reproduction/` — fast replay from the distributed derived inputs.
- `reproduction/temporal_transfer/` — time-separated donor retrieval, signature persistence, donor blending, and prediction controls.
- `reproduction/benchmark_controls/` — resolution and seasonal-target comparisons.
- `reproduction/sensitivity_checks/` — monthly-support and extraction-quality checks.
- `reproduction/robustness_checks/` — repeated splits, spatial exclusions, and epsilon diagnostics.
- `reproduction/climate_process/` — climate-stratified scores and the data for the climate-process figure.
- `figures/` — plotting scripts, source tables, reference exports, and publication-width figures.
- `raw_pipeline/` — Caravan import, feature construction, transport reconstruction, analysis, and plotted-data export.
- `provenance/` — metadata for the source data release.

## Environment

Python 3.11 or later is recommended. Install the dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The versions used for the verification run are recorded in `requirements-tested.txt`.

## Fast reproduction

Run these commands from the repository root:

```bash
python reproduction/test_seasonal_transport.py
python reproduction/reproduce_main.py
python reproduction/benchmark_controls/reproduce.py
python reproduction/sensitivity_checks/reproduce_sensitivity.py
python reproduction/climate_process/reproduce.py
```

The primary replay recomputes the annual FDC, monthly conditional, and allocation distances, fits the transfer mixture on calibration records, checks the physical-station split, and evaluates the held-out records. The archived primary scores are:

- monthly conditional representation: `0.2904733733`
- conventional FDC representation: `0.4528633063`
- relative reduction: `35.8585%`

The compact input archive supplies the original joint-transport distances to the donor pool. The corresponding raw-data route for recomputing those distances is in `raw_pipeline/`.

## Temporal transfer and prediction analyses

The `reproduction/temporal_transfer/` directory contains the time-separated analyses used to test whether early-period hydrological signatures support later-period donor selection and prediction. Its README gives the exact commands for:

- temporal donor retrieval;
- direct-target signature persistence;
- calibrated donor blends;
- early-signature mismatch rules;
- cross-domain holdouts;
- monthly-support diagnostics; and
- prediction-control figures.

All cached inputs, per-target predictions, and aggregate summaries are supplied alongside the scripts.

## Robustness and benchmark analyses

The fixed-split benchmark controls compare quantile resolution and seasonal targets. The robustness suite repeats group splits, applies spatial exclusions, and evaluates reduced-cohort entropic-transport settings. The sensitivity suite checks monthly support and record-level extraction quality. Each analysis directory contains its protocol, inputs, results, and replay script.

## Figures

```bash
python figures/redraw_figures.py
python figures/export_widths.py
python figures/plot_climate_process.py
```

The scripts write to `figures/redrawn_figures/`. Source tables used by the figures are in `figures/figure_data/`, and reference PDFs are in `figures/reference/`. The climate-process export is in `figures/climate_process/`.

## Rebuilding derived inputs from Caravan

The complete rebuild route is documented in [`raw_pipeline/README.md`](raw_pipeline/README.md). It imports the Caravan v1.6 CSV release, constructs the retained 1981–2010 cohort, generates hydrological signatures and conditional distributions, reconstructs the transport matrix, reruns the analyses, and exports the plotted tables.

Raw Caravan observations are not included in this repository. Download them from the official source listed in `DATA_PROVENANCE.md`, then follow the documented directory layout. The compact replay can be run without downloading the raw archive.

## Data, software, and citation

The original software is released under the MIT License. Caravan and the derived data products remain subject to their source terms; see `RIGHTS.md` and `DATA_PROVENANCE.md`.

Use `CITATION.cff` for the author list and citation metadata. `CHECKSUMS.sha256` records the checksums of the distributed inputs, source files, figures, and result tables.
