# Sources and transformations

Source observations: Caravan v1.6 CSV release, https://doi.org/10.5281/zenodo.15530022. Dataset article: Kratzert et al. (2023), *Caravan – A global community dataset for large-sample hydrology*, Scientific Data 10, 61, https://doi.org/10.1038/s41597-023-01975-w. Credit Caravan and the constituent CAMELS-US, CAMELS-BR, CAMELS-CL, HYSETS and LamaH-CE providers. Source terms and attribution are described in the repository-level RIGHTS.md. The software MIT licence does not replace third-party data terms.

- `short_record_inputs.npz`: 2,522-record QC universe; gauge/site/region identifiers; 1981–1983, 1981–1985, 1981–1990 and 1986–1990 normalized/log-transformed annual quantiles, 12×32 monthly quantiles and monthly runoff-share profiles; 2001–2010 BFI, flashiness, runoff ratio and snow proxy; coverage flags/counts; frozen donor/calibration/test physical-site split. No raw daily series are included.
- `record_index.csv`: gauge_id, site_id and region only, in that frozen row order, for locating source files during regeneration.
- `prediction_controls/early_signature_labels.npz`: four early-window signatures, calculated from the same source observations and signature functions; row identity is asserted against the temporal input before replay. BFI and flashiness use discharge, while runoff ratio and snow proxy also use early forcing.
- `diagnostics_inputs/benchmark_inputs.npz`: complete-record derived attributes and representations, retaining the older benchmark's frozen physical-site split and joint-transport distances.
- `diagnostics_inputs/blindspot_pairs.npz`: frozen sampled catchment pairs, annual-distance/rank membership and timing-distance fields used by the snow-regime diagnostic.
- `results/` and `prediction_controls/*.csv`/`*.json`: derived cohort counts, error summaries, per-target predictions, physical-site bootstrap intervals, locked specifications and monthly sample-support summaries.

The earlier complete-record derived inputs, raw rebuilding scripts and numerical transport checks remain in their original repository locations. Later labels at donor/calibration sites are training information for the temporal transfer experiment; later test-site labels are final-scoring information. The diagnostic and prediction tests do not create new observed data. Machine-specific file locations in public metadata are replaced by logical source descriptions; frozen arrays and numerical outputs are preserved.

This revision release independently replayed temporal scores, direct-target persistence, monthly blending and the annual-FDC/allocation/monthly three-family controls from the distributed caches. Raw observation regeneration remains an explicit separate route requiring the original release download.
