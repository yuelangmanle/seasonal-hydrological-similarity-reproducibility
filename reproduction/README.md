# Reproduction materials

These files reproduce the main hydrological-similarity comparison from the distributed derived inputs.

## Requirements

Python 3.11 or later, NumPy, and SciPy are required for the main replay. pandas is used by the extended analyses. The tested environment is listed in the repository-level `requirements-tested.txt`.

## Main comparison

From this directory, or by using the paths below from the repository root:

```bash
python reproduction/test_seasonal_transport.py
python reproduction/reproduce_main.py
```

The main script reconstructs the annual FDC, monthly conditional, and monthly-allocation distances, calibrates the donor mixture on calibration records, checks station separation, and evaluates the held-out records. The supplied transport archive contains the original joint-transport distances to the donor pool.

`benchmark_inputs.npz` contains catchment and station identifiers, four evaluation attributes, annual FDC quantiles, monthly allocation profiles, monthly conditional quantiles, donor/calibration/test indices, and archived transport distances. It contains no raw meteorological time series.

## Companion analyses

```bash
python reproduction/benchmark_controls/reproduce.py
python reproduction/sensitivity_checks/reproduce_sensitivity.py
python reproduction/robustness_checks/reproduce_extension.py
python reproduction/robustness_checks/reproduce_epsilon.py
python reproduction/climate_process/reproduce.py
```

Each directory includes the input archive, protocol, result files, and the script used for replay.

## Method keys

`monthly_quantiles`, `monthly_32`, and `conditional_W2` refer to the monthly conditional representation. `FDC`, `FDC_TIMING`, and `allocation` refer to the annual FDC and allocation comparators. `original_OT` identifies the archived joint-transport comparator. The distance construction and the calibrated transfer predictor are separate stages of the analysis.

## Raw-data route

The repository-level `raw_pipeline/` directory supplies the Caravan import, feature construction, transport reconstruction, and plotted-data export workflow. Raw observations must be downloaded separately.
