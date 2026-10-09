# Data provenance and reproduction scope

The source observations are from the Caravan v1.6 CSV release: [10.5281/zenodo.15530022](https://doi.org/10.5281/zenodo.15530022). The dataset publication is Kratzert et al. (2023), *Caravan — A global community dataset for large-sample hydrology*, **Scientific Data**, 10, 61. The archived provider metadata are in `provenance/caravan_zenodo_metadata.json`.

The study combines records from CAMELS-US, CAMELS-BR, LamaH-CE, HYSETS, and CAMELS-CL for 1981–2010. The retained cohort contains 2,416 records. Physical-station identifiers are retained for the station-separated evaluation.

The distributed derived inputs contain:

- `reproduction/benchmark_inputs.npz`: record and station identifiers, evaluation attributes, annual and monthly representations, donor/calibration/test indices, and archived transport distances;
- `reproduction/robustness_checks/extension_inputs.npz`: representations, coordinates, and distances for repeated splits and spatial exclusions;
- `reproduction/benchmark_controls/inputs.npz`: inputs for resolution and seasonal-target controls;
- `reproduction/sensitivity_checks/monthly_controls.npz` and `per_record_qc.csv`: monthly controls and extraction-quality counts;
- `reproduction/temporal_transfer/`: compact inputs and per-target outputs for time-separated transfer and prediction analyses.

The primary replay recomputes conditional, FDC, and allocation distances and the calibrated transfer scores. Archived joint-transport distances are supplied for the compact replay; `raw_pipeline/` provides the full route for rebuilding them from the raw observations. The reduced-cohort epsilon calculations are reproduced only for the cohorts specified in their protocol.

Raw observations are not redistributed. The import and feature-building scripts require a separately downloaded Caravan archive and preserve the provider identifiers and licensing notices. No credentials or private contact information are required by the replay scripts.
