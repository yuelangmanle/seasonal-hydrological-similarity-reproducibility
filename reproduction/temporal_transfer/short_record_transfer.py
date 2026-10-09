"""Short-record donor retrieval and future-attribute transfer experiment.

The experiment is deliberately independent of the all-period feature cache.  A
window (1981 onward) supplies retrieval features. Labels in 2001--2010 at
donor sites supply analog values and donor IQRs; calibration-site labels select
mixture weights. Future labels at test sites are held out until scoring. Site
identifiers are split as physical groups,
so duplicated records from the same gauge cannot cross donor/calibration/test
partitions.

Outputs are written below ``results/short_record_transfer``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
METHOD = PACKAGE_DIR if (PACKAGE_DIR / "signatures.py").exists() else ROOT / "code" / "02_method"
sys.path.insert(0, str(METHOD))
from signatures import baseflow_index, rb_flashiness, snow_fraction  # noqa: E402

OUT = PACKAGE_DIR / "rerun" / "raw_transfer"
RAW = ROOT / "data" / "raw" / "caravan"
SCALARS = PACKAGE_DIR / "record_index.csv"

# The first three windows are the primary length sensitivity.  The shifted
# five-year window is a start-date sensitivity and is reported separately.
WINDOWS = {
    "3yr": (1981, 1983),
    "5yr": (1981, 1985),
    "10yr": (1981, 1990),
    "5yr_shifted": (1986, 1990),
}
FUTURE = (2001, 2010)
ATTRIBUTES = ["bfi", "rb_flashiness", "runoff_ratio", "snow_fraction"]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 20261007
K = 10
MIN_COVERAGE = 0.68


def _date_window(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    dt = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    m = (np.asarray(dt.year) >= start) & (np.asarray(dt.year) <= end)
    out = df.loc[m].copy()
    out["date"] = dt[m]
    return out


def _valid_days(frame: pd.DataFrame, include_climate: bool = False) -> int:
    q = pd.to_numeric(frame["streamflow"], errors="coerce").to_numpy(float)
    if include_climate:
        p = pd.to_numeric(frame["total_precipitation_sum"], errors="coerce").to_numpy(float)
        t = pd.to_numeric(frame["temperature_2m_mean"], errors="coerce").to_numpy(float)
        return int(np.isfinite(q).sum() if not include_climate else (np.isfinite(q) & np.isfinite(p) & np.isfinite(t)).sum())
    return int(np.isfinite(q).sum())


def _coverage_ok(frame: pd.DataFrame, start: int, end: int, include_climate: bool = False) -> bool:
    expected = len(pd.date_range(f"{start}-01-01", f"{end}-12-31", freq="D"))
    return _valid_days(frame, include_climate) / expected >= MIN_COVERAGE


def _annual_fdc(frame: pd.DataFrame, n: int = 199) -> np.ndarray:
    q = pd.to_numeric(frame["streamflow"], errors="coerce").to_numpy(float)
    q = q[np.isfinite(q) & (q >= 0)]
    if q.size == 0 or q.mean() <= 0:
        return np.full(n, np.nan)
    x = np.log1p(q / q.mean())
    return np.quantile(x, (np.arange(n) + 0.5) / n)


def _monthly_conditional(frame: pd.DataFrame, n: int = 32) -> np.ndarray:
    q = pd.to_numeric(frame["streamflow"], errors="coerce").to_numpy(float)
    m = pd.DatetimeIndex(frame["date"]).month.to_numpy()
    valid = np.isfinite(q) & (q >= 0)
    if not valid.any() or q[valid].mean() <= 0:
        return np.full(12 * n, np.nan)
    x = np.log1p(q[valid] / q[valid].mean())
    months = m[valid]
    p = (np.arange(n) + 0.5) / n
    rows = []
    for month in range(1, 13):
        z = x[months == month]
        if z.size < 2:
            return np.full(12 * n, np.nan)
        rows.append(np.quantile(z, p))
    return np.asarray(rows).ravel()


def _monthly_profile(frame: pd.DataFrame) -> np.ndarray:
    q = pd.to_numeric(frame["streamflow"], errors="coerce").to_numpy(float)
    m = pd.DatetimeIndex(frame["date"]).month.to_numpy()
    out = np.array([np.nansum(q[m == month]) for month in range(1, 13)], dtype=float)
    return out / out.sum() if np.isfinite(out).all() and out.sum() > 0 else np.full(12, np.nan)


def _future_attributes(frame: pd.DataFrame) -> np.ndarray:
    q = pd.to_numeric(frame["streamflow"], errors="coerce")
    p = pd.to_numeric(frame["total_precipitation_sum"], errors="coerce")
    t = pd.to_numeric(frame["temperature_2m_mean"], errors="coerce")
    # Attribute-specific valid rows.  A common 68% climate coverage gate is
    # applied before this function is called, then each ratio uses its own
    # finite values to avoid silently treating missing precipitation as zero.
    qser = pd.Series(q.to_numpy(float), index=pd.DatetimeIndex(frame["date"]))
    bfi = baseflow_index(qser)
    rb = rb_flashiness(qser)
    qv, pv = q.to_numpy(float), p.to_numpy(float)
    mp = np.isfinite(qv) & np.isfinite(pv) & (qv >= 0) & (pv >= 0)
    runoff = float(qv[mp].mean() / pv[mp].mean()) if mp.sum() >= 100 and pv[mp].mean() > 0 else np.nan
    tv = t.to_numpy(float)
    mt = np.isfinite(pv) & np.isfinite(tv) & (pv >= 0)
    snow_mask = mt & (tv < 0)
    snow = float(pv[snow_mask].sum() / pv[mt].sum()) if mt.sum() >= 100 and pv[mt].sum() > 0 else np.nan
    return np.asarray([bfi, rb, runoff, snow], dtype=float)


def _allocation_distance(profiles: np.ndarray) -> np.ndarray:
    # Circular W1 distance between cumulative monthly allocations, with the
    # median cyclic shift removed as in the established allocation baseline.
    c = np.cumsum(profiles, axis=1)
    n = len(c)
    out = np.empty((n, n), dtype=float)
    for i in range(n):
        delta = c - c[i]
        out[i] = np.abs(delta - np.median(delta, axis=1, keepdims=True)).sum(axis=1) ** 2
    return out


def _site_split(site_ids: np.ndarray) -> dict[str, np.ndarray]:
    sites = np.asarray(sorted(set(site_ids.tolist())))
    rng = np.random.default_rng(SEED)
    rng.shuffle(sites)
    n = len(sites)
    cuts = [int(0.6 * n), int(0.8 * n)]
    labels = {"donor": sites[:cuts[0]], "calibration": sites[cuts[0]:cuts[1]], "test": sites[cuts[1]:]}
    return {k: np.flatnonzero(np.isin(site_ids, v)) for k, v in labels.items()}


def _median_prediction(distance: np.ndarray, donor: np.ndarray, target: np.ndarray,
                       y: np.ndarray, site: np.ndarray, k: int = K) -> np.ndarray:
    d = np.asarray(distance[np.ix_(target, donor)], dtype=float).copy()
    d[site[target, None] == site[donor][None, :]] = np.inf
    kk = min(k, len(donor))
    nn = np.argsort(d, axis=1, kind="stable")[:, :kk]
    return np.nanmedian(y[donor[nn]], axis=1)


def _score(y_true: np.ndarray, pred: np.ndarray, donor_y: np.ndarray) -> tuple[float, np.ndarray]:
    err = np.abs(pred - y_true)
    iqr = np.nanpercentile(donor_y, 75, axis=0) - np.nanpercentile(donor_y, 25, axis=0)
    iqr = np.where(iqr > 0, iqr, 1.0)
    return float(np.nanmean(err / iqr)), np.nanmean(err, axis=0)


def _normalised_errors(y_true: np.ndarray, pred: np.ndarray,
                       donor_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-target, per-attribute absolute errors and donor IQRs."""
    iqr = np.nanpercentile(donor_y, 75, axis=0) - np.nanpercentile(donor_y, 25, axis=0)
    iqr = np.where(iqr > 0, iqr, 1.0)
    return np.abs(pred - y_true) / iqr, iqr


def _cluster_bootstrap(errors: np.ndarray, sites: np.ndarray, seed: int,
                       reps: int = 5000) -> dict[str, list[float] | float]:
    """Bootstrap mean aggregate error by physical site, preserving duplicates."""
    sites = np.asarray(sites, dtype=str)
    unique = np.asarray(sorted(set(sites.tolist())), dtype=str)
    groups = [np.flatnonzero(sites == s) for s in unique]
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=float)
    for b in range(reps):
        picked = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([groups[i] for i in picked])
        values[b] = float(np.nanmean(errors[rows]))
    q = np.nanpercentile(values, [2.5, 97.5])
    return {"estimate": float(np.nanmean(errors)), "lower": float(q[0]),
            "upper": float(q[1]), "n_sites": int(len(unique)), "replicates": reps}


def _distance_matrices(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    fdc = cdist(features["fdc"], features["fdc"], metric="sqeuclidean") / features["fdc"].shape[1]
    monthly = cdist(features["monthly"], features["monthly"], metric="sqeuclidean") / features["monthly"].shape[1]
    allocation = _allocation_distance(features["profile"])
    # Distances are normalized by the donor-pool median in evaluation.  Keep
    # raw matrices here so each split can use its own scale without leakage.
    return {"annual_fdc": fdc, "monthly_conditional_w2": monthly, "monthly_allocation": allocation}


def run_window(name: str, feature_start: int, feature_end: int,
               records: list[dict], split: dict[str, np.ndarray]) -> dict:
    ids = np.asarray([r["gauge_id"] for r in records], dtype=str)
    sites = np.asarray([r["site_id"] for r in records], dtype=str)
    regions = np.asarray([r["region"] for r in records], dtype=str)
    feat_ok = np.asarray([r["feature_ok"] for r in records], bool)
    label_ok = np.asarray([r["label_ok"] for r in records], bool)
    finite_features = np.asarray([
        np.isfinite(r["fdc"]).all() and np.isfinite(r["monthly"]).all() and np.isfinite(r["profile"]).all()
        for r in records
    ], dtype=bool)
    # Keep the early-feature cohort separate from the final evaluable cohort.
    # The future-label gate is required for a common four-label estimand, but
    # it is not a retrieval feature and must be reported as an observability
    # condition rather than described as future-data leakage.
    early_feature_eligible = feat_ok & finite_features
    eligible = early_feature_eligible & label_ok
    donor = split["donor"][eligible[split["donor"]]]
    calibration = split["calibration"][eligible[split["calibration"]]]
    test = split["test"][eligible[split["test"]]]
    # Reindex records to the full cache order.  Distance matrices are built
    # only for eligible rows. Future test-site labels are held out from all selections;
    # future donor and calibration labels are permitted training information.
    ix = np.flatnonzero(eligible)
    pos = {old: j for j, old in enumerate(ix)}
    donor_l = np.asarray([pos[i] for i in donor], dtype=int)
    cal_l = np.asarray([pos[i] for i in calibration], dtype=int)
    test_l = np.asarray([pos[i] for i in test], dtype=int)
    feature = {k: np.asarray([records[i][k] for i in ix]) for k in ["fdc", "monthly", "profile"]}
    y = np.asarray([records[i]["future_attrs"] for i in ix], dtype=float)
    site = sites[ix]
    region = regions[ix]
    dist = _distance_matrices(feature)
    # Normalize all candidate distances by the donor-pool median, following
    # the paper's donor-scale convention and avoiding test-set information.
    med = {k: float(np.nanmedian(v[np.ix_(donor_l, donor_l)])) for k, v in dist.items()}
    dnorm = {k: v / (med[k] if med[k] > 0 else 1.0) for k, v in dist.items()}
    reference = dnorm["annual_fdc"]
    methods = {}
    per_method_errors: dict[str, np.ndarray] = {}
    per_method_raw_errors: dict[str, np.ndarray] = {}
    for method, raw in dnorm.items():
        candidates = [0.0] if method == "annual_fdc" else WEIGHTS
        calibration_scores = []
        for w in candidates:
            mix = (1.0 - w) * raw + w * reference
            pred = _median_prediction(mix, donor_l, cal_l, y, site)
            score, _ = _score(y[cal_l], pred, y[donor_l])
            calibration_scores.append(score)
        best = int(np.argmin(calibration_scores))
        w = float(candidates[best])
        mix = (1.0 - w) * raw + w * reference
        pred = _median_prediction(mix, donor_l, test_l, y, site)
        score, mae = _score(y[test_l], pred, y[donor_l])
        norm_attr_errors, _ = _normalised_errors(y[test_l], pred, y[donor_l])
        per_method_errors[method] = norm_attr_errors
        per_method_raw_errors[method] = np.abs(pred - y[test_l])
        methods[method] = {
            "weight_fdc": w,
            "candidate_weights": [float(x) for x in candidates],
            "calibration_scores": [float(x) for x in calibration_scores],
            "calibration_score": float(calibration_scores[best]),
            "test_score": score,
            "test_mae": {a: float(v) for a, v in zip(ATTRIBUTES, mae)},
        }
    # Station-cluster intervals are calculated after calibration and donor
    # selection are frozen.  The bootstrap therefore quantifies test sampling
    # uncertainty rather than re-tuning the method on each replicate.
    bootstrap = {}
    bootstrap_attributes_raw = {}
    test_sites = site[test_l]
    for method, err in per_method_errors.items():
        aggregate = np.nanmean(err, axis=1)
        bootstrap[method] = _cluster_bootstrap(aggregate, test_sites, SEED + 1000 + len(bootstrap))
        raw = per_method_raw_errors[method]
        bootstrap_attributes_raw[method] = {
            attr: _cluster_bootstrap(raw[:, ai], test_sites, SEED + 3000 + len(bootstrap_attributes_raw) * 10 + ai)
            for ai, attr in enumerate(ATTRIBUTES)
        }
    fdc_err = per_method_errors["annual_fdc"]
    paired_bootstrap = {}
    for method_index, (method, err) in enumerate(per_method_errors.items()):
        if method == "annual_fdc":
            continue
        aggregate_diff = np.nanmean(err, axis=1) - np.nanmean(fdc_err, axis=1)
        b = _cluster_bootstrap(aggregate_diff, test_sites, SEED + 2000 + method_index)
        paired_bootstrap[method] = b
    paired_w2_vs_allocation = _cluster_bootstrap(
        np.nanmean(per_method_errors["monthly_conditional_w2"], axis=1)
        - np.nanmean(per_method_errors["monthly_allocation"], axis=1),
        test_sites, SEED + 2100)
    by_region = {}
    for reg in sorted(set(region[test_l].tolist())):
        ti = test_l[region[test_l] == reg]
        # Regions are descriptive strata only; donors remain the common donor
        # pool, preserving the fixed physical-site split.
        by_region[reg] = {"n_test": int(len(ti)), "methods": {}}
        for method, raw in dnorm.items():
            w = methods[method]["weight_fdc"]
            mix = (1.0 - w) * raw + w * reference
            pred = _median_prediction(mix, donor_l, ti, y, site)
            score, mae = _score(y[ti], pred, y[donor_l])
            by_region[reg]["methods"][method] = {"test_score": score, "test_mae": {a: float(v) for a, v in zip(ATTRIBUTES, mae)}}
    result = {
        "window": name, "feature_years": [feature_start, feature_end], "future_years": list(FUTURE),
        "n_total_records": int(len(records)),
        "n_feature_eligible": int(early_feature_eligible.sum()),
        "n_label_eligible": int(label_ok.sum()),
        "n_eligible": int(len(ix)),
        "n_donor": int(len(donor_l)), "n_calibration": int(len(cal_l)), "n_test": int(len(test_l)),
        "feature_valid_days_median": float(np.median([r["feature_valid_days"] for r in records if r["feature_ok"]])) if feat_ok.any() else None,
        "future_valid_days_median": float(np.median([r["future_valid_days"] for r in records if r["label_ok"]])) if label_ok.any() else None,
        "distance_medians_donor": med, "methods": methods, "by_region": by_region,
        "bootstrap_aggregate": bootstrap, "bootstrap_attributes_raw": bootstrap_attributes_raw,
        "paired_bootstrap_vs_fdc": paired_bootstrap,
        "paired_bootstrap_w2_vs_allocation": {"monthly_conditional_w2": paired_w2_vs_allocation},
    }
    # Save target-level errors so every aggregate score can be audited without
    # rerunning the station-level feature extraction.
    error_rows = []
    for j, original in enumerate(test_l):
        row = {"target_index": int(original), "gauge_id": str(ids[ix[original]]),
               "site_id": str(site[original]), "region": str(region[original])}
        for method, err in per_method_errors.items():
            row[f"{method}_aggregate"] = float(np.nanmean(err[j]))
            for ai, attr in enumerate(ATTRIBUTES):
                row[f"{method}_{attr}"] = float(err[j, ai])
        error_rows.append(row)
    pd.DataFrame(error_rows).to_csv(OUT / f"per_target_errors_{name}.csv", index=False)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    global RAW, SCALARS, OUT
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--scalars", type=Path, default=SCALARS)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true", help="recompute even when protocol/results exist")
    args = ap.parse_args()
    RAW, SCALARS, OUT = args.raw, args.scalars, args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    scal = pd.read_csv(SCALARS)
    # Use exactly the all-period QC universe; rows without a source CSV are
    # recorded in protocol metadata rather than silently substituted.
    records: list[dict] = []
    t0 = time.time()
    for i, row in scal.iterrows():
        gid, region, site = str(row.gauge_id), str(row.region), str(row.site_id)
        path = RAW / region / f"{gid}.csv"
        rec = {"gauge_id": gid, "region": region, "site_id": site, "feature_ok": False, "label_ok": False,
               "feature_valid_days": 0, "future_valid_days": 0, "future_attrs": [np.nan] * 4}
        if not path.exists():
            records.append(rec)
            continue
        try:
            frame = pd.read_csv(path, usecols=["date", "streamflow", "temperature_2m_mean", "total_precipitation_sum"])
            fwin = {n: _date_window(frame, *years) for n, years in WINDOWS.items()}
            # Store all windows in a single record; the selected window fields
            # are attached below to keep one raw read per station.
            future = _date_window(frame, *FUTURE)
            rec["future_valid_days"] = _valid_days(future, include_climate=True)
            expected_future = len(pd.date_range(f"{FUTURE[0]}-01-01", f"{FUTURE[1]}-12-31", freq="D"))
            rec["label_ok"] = rec["future_valid_days"] / expected_future >= MIN_COVERAGE and np.isfinite(_future_attributes(future)).all()
            rec["future_attrs"] = _future_attributes(future).tolist()
            for n, fw in fwin.items():
                rec[f"{n}_feature_ok"] = _coverage_ok(fw, *WINDOWS[n])
                rec[f"{n}_feature_valid_days"] = _valid_days(fw)
                rec[f"{n}_fdc"] = _annual_fdc(fw)
                rec[f"{n}_monthly"] = _monthly_conditional(fw)
                rec[f"{n}_profile"] = _monthly_profile(fw)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        records.append(rec)
        if (i + 1) % 200 == 0:
            print(f"[short-record] {i + 1}/{len(scal)} stations, {time.time() - t0:.1f}s", flush=True)

    site_ids = np.asarray([r["site_id"] for r in records], dtype=str)
    split = _site_split(site_ids)
    protocol = {
        "analysis": "short_record_transfer",
        "seed": SEED, "k": K, "min_coverage": MIN_COVERAGE,
        "windows": {k: list(v) for k, v in WINDOWS.items()}, "future_years": list(FUTURE),
        "split_unit": "physical site_id", "split_fraction": {"donor": 0.6, "calibration": 0.2, "test": 0.2},
        "n_records": len(records), "n_unique_site_id": int(len(set(site_ids))),
        "split_counts_records": {k: int(len(v)) for k, v in split.items()},
        "split_site_ids": {k: sorted(set(site_ids[v].tolist())) for k, v in split.items()},
        "weights": WEIGHTS, "methods": ["annual_fdc", "monthly_conditional_w2", "monthly_allocation"],
        "normalization": "each distance divided by its donor-donor median; candidate weight mixes method distance with annual FDC",
        "attributes": ATTRIBUTES,
        "eligibility": {
            "early_feature_gate": "at least 68% finite daily discharge in the selected early window and finite feature vectors",
            "future_label_gate": "at least 68% finite daily discharge, precipitation and temperature in 2001-2010 and finite values for all four labels",
            "estimand": "final evaluable cohort conditional on the prespecified future-label availability gate",
            "future_use_boundary": "future test-target labels are excluded from all retrieval and training decisions; future donor labels supply analog values and IQRs, and future calibration labels select weights",
        },
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    all_results = {}
    for n, years in WINDOWS.items():
        selected = []
        for rec in records:
            selected.append({**rec, "feature_ok": rec.get(f"{n}_feature_ok", False), "feature_valid_days": rec.get(f"{n}_feature_valid_days", 0),
                             "fdc": rec.get(f"{n}_fdc", np.full(199, np.nan)), "monthly": rec.get(f"{n}_monthly", np.full(384, np.nan)),
                             "profile": rec.get(f"{n}_profile", np.full(12, np.nan))})
        all_results[n] = run_window(n, *years, selected, split)
        print(f"[short-record] {n}: eligible={all_results[n]['n_eligible']} test={all_results[n]['n_test']}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    # Per-window compact CSV for tables and manuscript extraction.
    rows = []
    for wn, result in all_results.items():
        for method, m in result["methods"].items():
            rows.append({"window": wn, "method": method, **{k: v for k, v in m.items() if k not in ("candidate_weights", "calibration_scores", "test_mae")},
                         "test_score": m["test_score"], **{f"mae_{a}": m["test_mae"][a] for a in ATTRIBUTES},
                         "n_donor": result["n_donor"], "n_calibration": result["n_calibration"], "n_test": result["n_test"]})
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    # Detailed record-level metadata is useful for audit but excludes arrays.
    compact = []
    for r in records:
        compact.append({k: v for k, v in r.items() if not isinstance(v, np.ndarray) and k not in {"future_attrs"}} | {"future_attrs": r["future_attrs"]})
    def _json_default(value):
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)
    (OUT / "record_coverage.json").write_text(json.dumps(compact, default=_json_default, ensure_ascii=False), encoding="utf-8")
    readme = f"""# Short-record donor transfer\n\nFeature windows are 1981 onward ({', '.join(f'{k}: {v[0]}-{v[1]}' for k, v in WINDOWS.items())}); future labels are {FUTURE[0]}-{FUTURE[1]}. The donor/calibration/test split is a deterministic {SEED} split of unique physical `site_id` values at 60/20/20. A record enters a retrieval window when at least {MIN_COVERAGE:.0%} of expected daily discharge observations are finite. Future flow, precipitation and temperature coverage is a prespecified held-out-label availability gate for the evaluation cohort. Future donor labels supply analog predictions and label-normalizing IQRs; future calibration labels select mixture weights. Future test-target labels are used only for the availability gate and final scoring. Early discharge alone supplies similarity features and donor-distance scaling.\n\nDistances are annual log1p mean-normalized FDC, monthly conditional W2 (12 x 32 quantiles), and monthly allocation. Distances are normalized by donor-pool medians. Candidate weights mix each distance with annual FDC and are selected from {WEIGHTS} using calibration labels only. Predictions are medians of K={K} donors. `summary.json` includes 5,000-replicate physical-site bootstrap intervals, and `per_target_errors_<window>.csv` contains target-level errors.\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({k: {m: {"weight": v["weight_fdc"], "test_score": v["test_score"], "mae": v["test_mae"]} for m, v in r["methods"].items()} for k, r in all_results.items()}, indent=2))


if __name__ == "__main__":
    main()
