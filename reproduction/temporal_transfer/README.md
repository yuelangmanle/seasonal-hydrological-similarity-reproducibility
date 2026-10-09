# Temporal transfer and prediction controls

This directory tests hydrological similarity when donor descriptors are computed from an early period and evaluation labels are taken from a later period. It also contains direct-target persistence, calibrated donor blends, snow-regime diagnostics, and prediction controls.

Early target discharge provides annual log-FDC, conditional-monthly, and monthly-allocation descriptors. Donor labels from 2001–2010 provide analog predictions and donor-label interquartile ranges. Calibration records select the distance mixtures and persistence-blend weights. Later labels at physical test sites are withheld from those choices. The availability rule defines the records that can be evaluated in the later period.

## Install

From the repository root:

```bash
python3 -m pip install -r reproduction/temporal_transfer/requirements.txt
```

Raw daily observations are not included. Their source and required transformations are documented in `DATA_PROVENANCE.md`.

## Temporal transfer replay

```bash
python3 reproduction/temporal_transfer/replay_temporal_mask.py \
  --inputs reproduction/temporal_transfer/short_record_inputs.npz \
  --out-dir reproduction/temporal_transfer/rerun/temporal \
  --bootstrap-replicates 5000 --write-target-errors

python3 reproduction/temporal_transfer/regime_snow_false_twin.py \
  --out-dir reproduction/temporal_transfer/rerun/snow \
  --bootstrap-replicates 2000
```

The supplied `results/` directory contains aggregate scores, physical-site bootstrap intervals, and per-target errors. For the ten-year comparison, the FDC, monthly conditional W2, and allocation scores are `0.4542937530`, `0.2947122539`, and `0.3518329145`, respectively. The snow-regime diagnostic identifies 1,951 of 3,528 evaluated pairs (`55.3%`) as cross-regime phase mismatches; its bootstrap interval is `[50.1%, 60.5%]`.

Temporal allocation uses the circular cumulative difference across twelve months. The complete-record benchmark uses the corresponding 365-day expansion; both implementations are retained because they answer different transfer questions.

## Direct-target persistence and donor blends

The `prediction_controls/` directory supplies early-period BFI, flashiness, runoff-ratio, and snow-proxy labels together with the following replays:

```bash
python3 reproduction/temporal_transfer/prediction_controls/test_signature_persistence.py \
  --inputs reproduction/temporal_transfer/short_record_inputs.npz \
  --early-labels reproduction/temporal_transfer/prediction_controls/early_signature_labels.npz \
  --out-dir reproduction/temporal_transfer/rerun/prediction

python3 reproduction/temporal_transfer/prediction_controls/test_persistence_donor_blend.py \
  --inputs reproduction/temporal_transfer/short_record_inputs.npz \
  --early-labels reproduction/temporal_transfer/prediction_controls/early_signature_labels.npz \
  --out-dir reproduction/temporal_transfer/rerun/prediction

python3 reproduction/temporal_transfer/prediction_controls/test_blend_controls.py \
  --inputs reproduction/temporal_transfer/short_record_inputs.npz \
  --early-labels reproduction/temporal_transfer/prediction_controls/early_signature_labels.npz \
  --out-dir reproduction/temporal_transfer/rerun/prediction

python3 reproduction/temporal_transfer/prediction_controls/test_early_mismatch_rule.py \
  --inputs reproduction/temporal_transfer/short_record_inputs.npz \
  --out-dir reproduction/temporal_transfer/rerun/prediction

python3 reproduction/temporal_transfer/prediction_controls/test_signature_change_rule.py \
  --inputs reproduction/temporal_transfer/short_record_inputs.npz \
  --early-labels reproduction/temporal_transfer/prediction_controls/early_signature_labels.npz \
  --rule-dir reproduction/temporal_transfer/rerun/prediction \
  --out-dir reproduction/temporal_transfer/rerun/prediction
```

The persistence model transfers a target’s early signature to its later period. The blend combines this direct prediction with donor estimates, with the blend weight selected on calibration records. The three-family control repeats the calculation for annual FDC, monthly allocation, and conditional monthly W2. All per-target predictions and paired intervals are included in the directory.

The early-mismatch rule evaluates whether descriptor disagreement predicts later donor advantage. It is fitted on calibration records and tested on held-out physical sites. Its results are reported for every domain and cohort in the accompanying CSV and JSON files.

To redraw the prediction-control figures:

```bash
python3 reproduction/temporal_transfer/prediction_controls/plot_prediction_controls.py \
  --data-dir reproduction/temporal_transfer/prediction_controls \
  --out-dir reproduction/temporal_transfer/rerun/figures
```

## Rebuild compact inputs from raw observations

Download the Caravan v1.6 CSV release and prepare the provider tree as `RAW_DIRECTORY/<region>/<gauge_id>.csv` with `date`, `streamflow`, `total_precipitation_sum`, and `temperature_2m_mean` columns. `record_index.csv` records the input cohort and row order.

```bash
python3 reproduction/temporal_transfer/build_compact_short_record_inputs.py \
  --raw RAW_DIRECTORY \
  --scalars reproduction/temporal_transfer/record_index.csv \
  --out reproduction/temporal_transfer/rerun/short_record_inputs.npz

python3 reproduction/temporal_transfer/prediction_controls/test_signature_persistence.py \
  --inputs reproduction/temporal_transfer/rerun/short_record_inputs.npz \
  --raw RAW_DIRECTORY \
  --label-source reproduction/temporal_transfer/short_record_transfer.py \
  --out-dir reproduction/temporal_transfer/rerun/raw_prediction
```

Cached replays require no download. Generated files are written under `rerun/`, which is ignored by Git.
