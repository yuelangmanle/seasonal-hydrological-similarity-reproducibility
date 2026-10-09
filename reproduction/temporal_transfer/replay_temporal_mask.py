"""Replay the current temporal-mask donor-transfer analysis from compact inputs.

The script does not need raw Caravan files. It recomputes distance matrices,
donor-pool scaling, calibration weights, ten-donor median predictions and
future-label errors from ``short_record_inputs.npz``. Donor future labels supply
analog predictions and IQRs, calibration future labels select weights, and test
future labels are withheld until scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


WINDOWS = {
    "3yr": (1981, 1983),
    "5yr": (1981, 1985),
    "10yr": (1981, 1990),
    "5yr_shifted": (1986, 1990),
}
ATTRIBUTES = ["bfi", "rb_flashiness", "runoff_ratio", "snow_fraction"]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
K = 10
SEED = 20261007


def allocation_distance(profiles: np.ndarray) -> np.ndarray:
    c = np.cumsum(profiles, axis=1)
    out = np.empty((len(c), len(c)), dtype=float)
    for i in range(len(c)):
        delta = c - c[i]
        out[i] = np.abs(delta - np.median(delta, axis=1, keepdims=True)).sum(axis=1) ** 2
    return out


def distances(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "annual_fdc": cdist(features["fdc"], features["fdc"], metric="sqeuclidean") / features["fdc"].shape[1],
        "monthly_conditional_w2": cdist(features["monthly"], features["monthly"], metric="sqeuclidean") / features["monthly"].shape[1],
        "monthly_allocation": allocation_distance(features["profile"]),
    }


def median_prediction(distance: np.ndarray, donor: np.ndarray, target: np.ndarray, y: np.ndarray, site: np.ndarray) -> np.ndarray:
    d = distance[np.ix_(target, donor)].copy()
    d[site[target, None] == site[donor][None, :]] = np.inf
    nn = np.argsort(d, axis=1, kind="stable")[:, : min(K, len(donor))]
    return np.nanmedian(y[donor[nn]], axis=1)


def score(y_true: np.ndarray, pred: np.ndarray, donor_y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    raw = np.abs(pred - y_true)
    iqr = np.nanpercentile(donor_y, 75, axis=0) - np.nanpercentile(donor_y, 25, axis=0)
    iqr = np.where(iqr > 0, iqr, 1.0)
    norm = raw / iqr
    return float(np.nanmean(norm)), np.nanmean(raw, axis=0), norm


def cluster_bootstrap(values: np.ndarray, sites: np.ndarray, seed: int, reps: int) -> dict[str, float | int]:
    unique = np.asarray(sorted(set(sites.tolist())), dtype=str)
    groups = [np.flatnonzero(sites == s) for s in unique]
    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=float)
    for b in range(reps):
        picked = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([groups[i] for i in picked])
        boot[b] = float(np.nanmean(values[rows]))
    q = np.nanpercentile(boot, [2.5, 97.5])
    return {"estimate": float(np.nanmean(values)), "lower": float(q[0]), "upper": float(q[1]), "n_sites": int(len(unique)), "replicates": int(reps)}


def run_window(data: dict[str, np.ndarray], name: str, reps: int, write_targets: bool, out_dir: Path) -> dict:
    fdc = data[f"{name}_fdc"]
    monthly = data[f"{name}_monthly"]
    profile = data[f"{name}_profile"]
    feature_ok = data[f"{name}_feature_ok"].astype(bool)
    finite = np.isfinite(fdc).all(axis=1) & np.isfinite(monthly).all(axis=1) & np.isfinite(profile).all(axis=1)
    label_ok = data["label_ok"].astype(bool)
    early_eligible = feature_ok & finite
    eligible = early_eligible & label_ok
    ix = np.flatnonzero(eligible)
    donor = np.intersect1d(ix, data["split_donor"], assume_unique=False)
    calibration = np.intersect1d(ix, data["split_calibration"], assume_unique=False)
    test = np.intersect1d(ix, data["split_test"], assume_unique=False)
    pos = {old: j for j, old in enumerate(ix)}
    donor_l = np.asarray([pos[i] for i in donor], dtype=int)
    cal_l = np.asarray([pos[i] for i in calibration], dtype=int)
    test_l = np.asarray([pos[i] for i in test], dtype=int)
    features = {"fdc": fdc[ix], "monthly": monthly[ix], "profile": profile[ix]}
    y = data["future_attrs"][ix]
    site = data["site_id"][ix].astype(str)
    dist = distances(features)
    donor_medians = {k: float(np.nanmedian(v[np.ix_(donor_l, donor_l)])) for k, v in dist.items()}
    dnorm = {k: v / (donor_medians[k] if donor_medians[k] > 0 else 1.0) for k, v in dist.items()}
    reference = dnorm["annual_fdc"]
    methods = {}
    errors = {}
    raw_errors = {}
    for method, raw_dist in dnorm.items():
        candidates = [0.0] if method == "annual_fdc" else WEIGHTS
        cal_scores = []
        for w in candidates:
            mix = (1 - w) * raw_dist + w * reference
            pred = median_prediction(mix, donor_l, cal_l, y, site)
            cal_scores.append(score(y[cal_l], pred, y[donor_l])[0])
        best = int(np.argmin(cal_scores))
        w = float(candidates[best])
        mix = (1 - w) * raw_dist + w * reference
        pred = median_prediction(mix, donor_l, test_l, y, site)
        aggregate, mae, norm = score(y[test_l], pred, y[donor_l])
        methods[method] = {
            "weight_fdc": w,
            "candidate_weights": [float(x) for x in candidates],
            "calibration_scores": [float(x) for x in cal_scores],
            "calibration_score": float(cal_scores[best]),
            "test_score": aggregate,
            "test_mae": {a: float(v) for a, v in zip(ATTRIBUTES, mae)},
        }
        errors[method] = norm
        raw_errors[method] = np.abs(pred - y[test_l])
    boot = {}
    attr_boot = {}
    for j, (method, err) in enumerate(errors.items()):
        boot[method] = cluster_bootstrap(np.nanmean(err, axis=1), site[test_l], SEED + 1000 + j, reps)
        attr_boot[method] = {a: cluster_bootstrap(raw_errors[method][:, ai], site[test_l], SEED + 3000 + j * 10 + ai, reps) for ai, a in enumerate(ATTRIBUTES)}
    fdc_err = errors["annual_fdc"]
    paired = {}
    for j, (method, err) in enumerate(errors.items()):
        if method != "annual_fdc":
            paired[method] = cluster_bootstrap(np.nanmean(err, axis=1) - np.nanmean(fdc_err, axis=1), site[test_l], SEED + 2000 + j, reps)
    # Paired W2 versus allocation interval is computed from target-level aggregate errors.
    paired_w2_vs_allocation = cluster_bootstrap(
        np.nanmean(errors["monthly_conditional_w2"], axis=1) - np.nanmean(errors["monthly_allocation"], axis=1),
        site[test_l], SEED + 2100, reps)
    if write_targets:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for j, original in enumerate(test_l):
            row = {"target_index": int(ix[original]), "gauge_id": str(data["gauge_id"][ix[original]]), "site_id": str(site[original]), "region": str(data["region"][ix[original]])}
            for method, err in errors.items():
                row[f"{method}_aggregate"] = float(np.nanmean(err[j]))
                for ai, a in enumerate(ATTRIBUTES):
                    row[f"{method}_{a}"] = float(err[j, ai])
            rows.append(row)
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_dir / f"per_target_errors_{name}.csv", index=False)
    return {
        "window": name,
        "feature_years": list(WINDOWS[name]),
        "future_years": [2001, 2010],
        "n_total_records": int(len(data["gauge_id"])),
        "n_feature_eligible": int(early_eligible.sum()),
        "n_label_eligible": int(label_ok.sum()),
        "n_eligible": int(len(ix)),
        "n_donor": int(len(donor_l)),
        "n_calibration": int(len(cal_l)),
        "n_test": int(len(test_l)),
        "distance_medians_donor": donor_medians,
        "methods": methods,
        "bootstrap_aggregate": boot,
        "bootstrap_attributes_raw": attr_boot,
        "paired_bootstrap_vs_fdc": paired,
        "paired_bootstrap_w2_vs_allocation": {"monthly_conditional_w2": paired_w2_vs_allocation},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=Path("short_record_inputs.npz"))
    ap.add_argument("--out-dir", type=Path, default=Path("replay_output"))
    ap.add_argument("--bootstrap-replicates", type=int, default=5000)
    ap.add_argument("--write-target-errors", action="store_true")
    args = ap.parse_args()
    z = np.load(args.inputs, allow_pickle=False)
    data = {k: z[k] for k in z.files}
    results = {name: run_window(data, name, args.bootstrap_replicates, args.write_target_errors, args.out_dir) for name in WINDOWS}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({w: {m: round(v["methods"][m]["test_score"], 6) for m in v["methods"]} for w, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
