# Rebuild from Caravan observations

This workflow imports Caravan observations, constructs the study cohort, derives hydrological signatures and monthly distributions, reconstructs the joint-transport distances, reruns the analyses, and exports the numerical tables used by the figures. The compact replay in `../reproduction/` is available when the raw archive is not present.

## 1. Download and import Caravan

Download **Caravan-csv.tar.gz** from the official v1.6 record at [10.5281/zenodo.15530022](https://doi.org/10.5281/zenodo.15530022). The archive is large; retain the provider metadata and verify the archive before extraction.

```bash
python raw_pipeline/import_caravan.py \
  --archive /path/to/Caravan-csv.tar.gz \
  --out /path/to/caravan-study
```

The importer reads the archive sequentially and retains the five study datasets: CAMELS-US, CAMELS-BR, LamaH-CE, HYSETS, and CAMELS-CL. Required daily columns are date, streamflow, precipitation, and temperature. Do not add the raw archive to the Git repository.

If an already extracted provider tree is available, use the layout below:

```text
caravan-study/
  camels/<gauge_id>.csv
  camelsbr/<gauge_id>.csv
  camelscl/<gauge_id>.csv
  hysets/<gauge_id>.csv
  lamah/<gauge_id>.csv
  _attributes/<region>__attributes_other_<region>.csv
```

## 2. Build the retained cohort

```bash
python raw_pipeline/build_raw_features.py \
  --raw-root /path/to/caravan-study \
  --jobs 4
```

The feature builder applies the study screening rules, constructs annual and monthly representations, and writes derived arrays under `raw_pipeline/data/processed/`. It checks the retained record identifiers and the finite core signatures against the distributed reference.

## 3. Reconstruct transport distances

For the full calculation:

```bash
python raw_pipeline/rebuild_transport.py --jobs 4
python raw_pipeline/run_analysis.py
```

The full matrix contains roughly 2.9 million unique pairs and may require several hours. It uses the study settings for the 400-iteration transport solver and epsilon `0.05`.

For a shorter verification run that samples the reconstructed matrix and reuses the archived distances for the downstream analysis:

```bash
python raw_pipeline/rebuild_transport.py --verify-pairs 40
python raw_pipeline/run_analysis.py --reuse-archived-transport
```

## 4. Export inputs and plotted data

```bash
python raw_pipeline/export_rebuilt.py
python raw_pipeline/results/rebuilt_figures/export_widths.py
```

The exporter writes rebuilt inputs under `raw_pipeline/results/rebuilt_reproduction/`, regenerates the numerical CSV tables used by the figures, and renders the figure outputs under `raw_pipeline/results/rebuilt_figures/`.

## Outputs

- `data/processed/raw_verification.json` — retained-record counts and feature comparisons;
- `data/processed/reconstructed_transport_summary.json` — transport reconstruction summary;
- `results/rebuilt_reproduction/` — derived inputs and replay-ready outputs;
- `results/rebuilt_figures/` — regenerated figure tables and graphics.
