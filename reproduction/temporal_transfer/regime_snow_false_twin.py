"""Reproduce a snow-regime diagnostic for annual-FDC false twins.

This audit is deliberately separate from the primary donor-transfer analysis.
It uses the frozen pooled pair sample and the released benchmark attributes to
ask whether annual-FDC-near pairs with contrasting precipitation-phase labels
are more likely to have a large seasonal-timing distance.

The bootstrap is a physical-site multiplier bootstrap conditional on the fixed
pair sample and fixed annual-FDC top-5% membership. It is therefore a
site-dependence interval, not a new pair-sampling or causal uncertainty
interval.
"""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PAIR_PATH = PACKAGE_DIR / "diagnostics_inputs" / "blindspot_pairs.npz"
DEFAULT_BENCHMARK_PATH = PACKAGE_DIR / "diagnostics_inputs" / "benchmark_inputs.npz"
DEFAULT_OUT_DIR = PACKAGE_DIR / "diagnostic_replay"
PAIR_PATH = DEFAULT_PAIR_PATH
BENCHMARK_PATH = DEFAULT_BENCHMARK_PATH
OUT_DIR = DEFAULT_OUT_DIR
CSV_PATH = OUT_DIR / "snow_regime_false_twin.csv"
JSON_PATH = OUT_DIR / "snow_regime_false_twin.json"
PARAGRAPH_PATH = OUT_DIR / "snow_regime_false_twin_paragraph.md"

SEED = 20261008
BOOTSTRAP_REPLICATES = 2000
TIMING_THRESHOLD_DAYS = 30.0
TOP_FDC_FRACTION = 0.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return float(centre - half), float(centre + half)


def weighted_rate(mask: np.ndarray, false_mask: np.ndarray, weights: np.ndarray) -> float:
    den = float(weights[mask].sum())
    return float(weights[mask & false_mask].sum() / den) if den > 0 else float("nan")


def cluster_interval(
    mask: np.ndarray,
    false_mask: np.ndarray,
    endpoint_a: np.ndarray,
    endpoint_b: np.ndarray,
    endpoint_site_index: np.ndarray,
    n_sites: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return estimate, 2.5th and 97.5th percentile of weighted rates."""
    observed = weighted_rate(mask, false_mask, np.ones(mask.size, dtype=float))
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for rep in range(BOOTSTRAP_REPLICATES):
        site_weights = rng.multinomial(n_sites, np.full(n_sites, 1.0 / n_sites))
        pair_weights = site_weights[endpoint_a] * site_weights[endpoint_b]
        same_site = endpoint_a == endpoint_b
        pair_weights[same_site] = site_weights[endpoint_a[same_site]]
        values[rep] = weighted_rate(mask, false_mask, pair_weights)
    values = values[np.isfinite(values)]
    q = np.percentile(values, [2.5, 97.5])
    return observed, float(q[0]), float(q[1])


def main() -> None:
    global PAIR_PATH, BENCHMARK_PATH, OUT_DIR, CSV_PATH, JSON_PATH, PARAGRAPH_PATH, BOOTSTRAP_REPLICATES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIR_PATH)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = ap.parse_args()
    PAIR_PATH, BENCHMARK_PATH, OUT_DIR = args.pairs, args.benchmark, args.out_dir
    BOOTSTRAP_REPLICATES = args.bootstrap_replicates
    CSV_PATH = OUT_DIR / "snow_regime_false_twin.csv"
    JSON_PATH = OUT_DIR / "snow_regime_false_twin.json"
    PARAGRAPH_PATH = OUT_DIR / "snow_regime_false_twin_paragraph.md"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = np.load(PAIR_PATH, allow_pickle=False)
    benchmark = np.load(BENCHMARK_PATH, allow_pickle=False)

    required_pairs = {"all_i", "all_j", "all_d_fdc", "all_d_time"}
    required_benchmark = {"attributes", "site_ids", "ids"}
    missing = required_pairs - set(pairs.files)
    if missing:
        raise KeyError(f"Missing pair arrays: {sorted(missing)}")
    missing = required_benchmark - set(benchmark.files)
    if missing:
        raise KeyError(f"Missing benchmark arrays: {sorted(missing)}")

    endpoint_a = np.asarray(pairs["all_i"], dtype=int)
    endpoint_b = np.asarray(pairs["all_j"], dtype=int)
    d_fdc = np.asarray(pairs["all_d_fdc"], dtype=float)
    d_time = np.asarray(pairs["all_d_time"], dtype=float)
    attributes = np.asarray(benchmark["attributes"], dtype=float)
    ids = np.asarray(benchmark["ids"], dtype=str)
    site_ids = np.asarray(benchmark["site_ids"], dtype=str)
    if attributes.ndim != 2 or attributes.shape[1] < 4:
        raise ValueError("Expected benchmark attributes with snow_fraction in column 4")
    if endpoint_a.size != endpoint_b.size or endpoint_a.size != d_fdc.size or endpoint_a.size != d_time.size:
        raise ValueError("Pair arrays have inconsistent lengths")
    if endpoint_a.max(initial=-1) >= len(attributes) or endpoint_b.max(initial=-1) >= len(attributes):
        raise IndexError("Pair endpoint exceeds benchmark attribute rows")

    snow = attributes[:, 3]
    finite_snow = np.isfinite(snow)
    if not finite_snow.all():
        raise ValueError("Snow fractions must be finite for the released benchmark universe")
    q1, q2, q3 = np.percentile(snow, [25.0, 50.0, 75.0])
    low = snow <= q1
    high = snow >= q3

    # The pair sample and top-5% cut are frozen before any regime labels are used.
    finite_pairs = np.isfinite(d_fdc) & np.isfinite(d_time)
    fdc_cut = float(np.percentile(d_fdc[finite_pairs], 100.0 * TOP_FDC_FRACTION))
    top = finite_pairs & (d_fdc <= fdc_cut)
    false_twin = d_time > TIMING_THRESHOLD_DAYS

    mean_snow = (snow[endpoint_a] + snow[endpoint_b]) / 2.0
    mean_bins = [
        ("mean_snow_q1", mean_snow <= q1),
        ("mean_snow_q2", (mean_snow > q1) & (mean_snow <= q2)),
        ("mean_snow_q3", (mean_snow > q2) & (mean_snow <= q3)),
        ("mean_snow_q4", mean_snow > q3),
    ]
    categories = [
        ("all_top5", top),
        ("both_low_snow", top & low[endpoint_a] & low[endpoint_b]),
        ("both_high_snow", top & high[endpoint_a] & high[endpoint_b]),
        ("cross_low_high_snow", top & ((low[endpoint_a] & high[endpoint_b]) | (high[endpoint_a] & low[endpoint_b]))),
    ] + [(name, top & condition) for name, condition in mean_bins]

    unique_sites, endpoint_site_index = np.unique(site_ids, return_inverse=True)
    site_a = endpoint_site_index[endpoint_a]
    site_b = endpoint_site_index[endpoint_b]
    same_site_pairs = site_a == site_b
    rng = np.random.default_rng(SEED)

    rows: list[dict[str, object]] = []
    for name, mask in categories:
        n = int(mask.sum())
        successes = int((mask & false_twin).sum())
        rate = successes / n if n else float("nan")
        wilson_low, wilson_high = wilson_interval(successes, n)
        estimate, cluster_low, cluster_high = cluster_interval(
            mask, false_twin, site_a, site_b, endpoint_site_index, len(unique_sites), rng
        )
        rows.append(
            {
                "category": name,
                "n_top5_pairs": n,
                "n_timing_gt_30d": successes,
                "false_twin_rate": rate,
                "pair_wilson_low": wilson_low,
                "pair_wilson_high": wilson_high,
                "site_cluster_bootstrap_low": cluster_low,
                "site_cluster_bootstrap_high": cluster_high,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(CSV_PATH, index=False, float_format="%.8f")

    cross_row = table.loc[table["category"] == "cross_low_high_snow"].iloc[0].to_dict()
    metadata = {
        "analysis": "annual_fdc_false_twins_by_snow_regime",
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "timing_threshold_days": TIMING_THRESHOLD_DAYS,
        "top_fdc_fraction": TOP_FDC_FRACTION,
        "pair_sample": {
            "path": str(PAIR_PATH),
            "sha256": sha256(PAIR_PATH),
            "n_pairs": int(len(endpoint_a)),
            "n_same_physical_site_pairs": int(same_site_pairs.sum()),
            "n_self_record_pairs": int((endpoint_a == endpoint_b).sum()),
            "arrays": ["all_i", "all_j", "all_d_fdc", "all_d_time"],
        },
        "benchmark_input": {
            "path": str(BENCHMARK_PATH),
            "sha256": sha256(BENCHMARK_PATH),
            "n_records": int(len(ids)),
            "attributes_column": 3,
            "attributes_column_note": "zero-based column 3 is snow_fraction in the released four-label matrix",
        },
        "thresholds": {
            "snow_q1": float(q1),
            "snow_q2_median": float(q2),
            "snow_q3": float(q3),
            "annual_fdc_top5_cut": fdc_cut,
            "timing_gt_30d": True,
        },
        "definitions": {
            "low_snow": "endpoint snow_fraction <= global Q1",
            "high_snow": "endpoint snow_fraction >= global Q3",
            "cross_low_high_snow": "one endpoint low-snow and the other high-snow",
            "false_twin": "circular timing distance > 30 days among annual-FDC top-5% pairs",
            "bootstrap": "physical-site multiplier bootstrap with fixed pair sample and fixed top-5% membership; same-site pairs receive one cluster weight",
        },
        "cross_low_high_result": cross_row,
        "outputs": {"csv": str(CSV_PATH), "paragraph": str(PARAGRAPH_PATH)},
    }
    JSON_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    paragraph = f"""### Snow-regime false twins

As an exploratory regime diagnostic, we stratified the frozen pooled sample of {len(endpoint_a):,} **record pairs** by the snow-fraction quartiles of the released benchmark attributes. The pair file contains {int(same_site_pairs.sum())} cross-dataset pairs with the same physical site identifier and no self-record pair; these are retained to reproduce the frozen pooled diagnostic. Among the {int(top.sum()):,} pairs in the closest 5% by annual FDC distance (FDC cut-off {fdc_cut:.6f}), {int((top & false_twin).sum()):,} ({float((top & false_twin).mean() / top.mean()):.1%}) had a circular timing distance above 30 d. When one endpoint was in the low-snow quartile (snow fraction <= {q1:.6f}) and the other in the high-snow quartile (>= {q3:.6f}), {int(cross_row['n_top5_pairs']):,} pairs were retained and {int(cross_row['n_timing_gt_30d']):,} ({float(cross_row['false_twin_rate']):.1%}; site-cluster bootstrap 95% interval {float(cross_row['site_cluster_bootstrap_low']):.1%}–{float(cross_row['site_cluster_bootstrap_high']):.1%}) exceeded 30 d. The corresponding rate was {float(table.loc[table['category'] == 'both_low_snow', 'false_twin_rate'].iloc[0]):.1%} for pairs with both endpoints in the low-snow quartile and {float(table.loc[table['category'] == 'both_high_snow', 'false_twin_rate'].iloc[0]):.1%} for pairs with both endpoints in the high-snow quartile. This pattern is consistent with an annual marginal distribution losing precipitation-phase and seasonal timing information; it is a representation-level association, not evidence that snowmelt alone causes the mismatch. The intervals condition on the fixed pair sample and top-5% selection and therefore do not represent uncertainty from regenerating pairs.

The regime labels in this diagnostic are used only after annual-FDC ranking and are not used for donor retrieval or calibration.
"""
    PARAGRAPH_PATH.write_text(paragraph, encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {PARAGRAPH_PATH}")


if __name__ == "__main__":
    main()
