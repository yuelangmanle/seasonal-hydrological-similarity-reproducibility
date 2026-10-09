"""Export compact inputs for the current temporal-mask reproduction.

This exporter reads the locally available Caravan daily files once and writes
all derived early-window features, future labels, eligibility flags and the
fixed physical-site split into one NPZ file. The compact file is intended for
derived-input replay; raw Caravan files remain subject to the dataset licence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
ANALYSIS = PACKAGE_DIR / "short_record_transfer.py"
RAW_DEFAULT = ROOT / "data" / "raw" / "caravan"
SCALARS_DEFAULT = PACKAGE_DIR / "record_index.csv"
OUT_DEFAULT = PACKAGE_DIR / "rerun" / "short_record_inputs.npz"


def load_analysis():
    spec = importlib.util.spec_from_file_location("short_record_transfer", ANALYSIS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ANALYSIS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--scalars", type=Path, default=SCALARS_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    a = load_analysis()
    scalars = pd.read_csv(args.scalars)
    n = len(scalars)
    windows = list(a.WINDOWS)
    fdc = {w: np.full((n, 199), np.nan, dtype=np.float64) for w in windows}
    monthly = {w: np.full((n, 384), np.nan, dtype=np.float64) for w in windows}
    profile = {w: np.full((n, 12), np.nan, dtype=np.float64) for w in windows}
    feature_ok = {w: np.zeros(n, dtype=bool) for w in windows}
    feature_valid_days = {w: np.zeros(n, dtype=np.int32) for w in windows}
    future_attrs = np.full((n, 4), np.nan, dtype=np.float64)
    label_ok = np.zeros(n, dtype=bool)
    future_valid_days = np.zeros(n, dtype=np.int32)

    for i, row in scalars.iterrows():
        region = str(row.region)
        gauge = str(row.gauge_id)
        path = args.raw / region / f"{gauge}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["date", "streamflow", "temperature_2m_mean", "total_precipitation_sum"])
        future = a._date_window(frame, *a.FUTURE)
        future_valid_days[i] = a._valid_days(future, include_climate=True)
        expected_future = len(pd.date_range(f"{a.FUTURE[0]}-01-01", f"{a.FUTURE[1]}-12-31", freq="D"))
        vals = a._future_attributes(future)
        future_attrs[i] = vals
        label_ok[i] = future_valid_days[i] / expected_future >= a.MIN_COVERAGE and np.isfinite(vals).all()
        for w, years in a.WINDOWS.items():
            fw = a._date_window(frame, *years)
            feature_ok[w][i] = a._coverage_ok(fw, *years)
            feature_valid_days[w][i] = a._valid_days(fw)
            fdc[w][i] = a._annual_fdc(fw)
            monthly[w][i] = a._monthly_conditional(fw)
            profile[w][i] = a._monthly_profile(fw)

    site_ids = scalars["site_id"].astype(str).to_numpy()
    rng = np.random.default_rng(a.SEED)
    sites = np.asarray(sorted(set(site_ids.tolist())))
    rng.shuffle(sites)
    cuts = [int(0.6 * len(sites)), int(0.8 * len(sites))]
    split_sites = {
        "donor": sites[:cuts[0]],
        "calibration": sites[cuts[0]:cuts[1]],
        "test": sites[cuts[1]:],
    }
    split = {k: np.flatnonzero(np.isin(site_ids, v)).astype(np.int32) for k, v in split_sites.items()}
    payload = {
        "gauge_id": np.asarray(scalars["gauge_id"].astype(str).to_numpy(), dtype="U64"),
        "site_id": np.asarray(site_ids, dtype="U64"),
        "region": np.asarray(scalars["region"].astype(str).to_numpy(), dtype="U64"),
        "future_attrs": future_attrs,
        "label_ok": label_ok,
        "future_valid_days": future_valid_days,
        "seed": np.asarray([a.SEED], dtype=np.int64),
        "min_coverage": np.asarray([a.MIN_COVERAGE], dtype=np.float64),
    }
    for w in windows:
        payload[f"{w}_fdc"] = fdc[w]
        payload[f"{w}_monthly"] = monthly[w]
        payload[f"{w}_profile"] = profile[w]
        payload[f"{w}_feature_ok"] = feature_ok[w]
        payload[f"{w}_feature_valid_days"] = feature_valid_days[w]
    for k, v in split.items():
        payload[f"split_{k}"] = np.asarray(v, dtype=np.int32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    protocol = {
        "n_records": int(n),
        "windows": {k: list(v) for k, v in a.WINDOWS.items()},
        "future_years": list(a.FUTURE),
        "feature_shapes": {w: {"fdc": list(fdc[w].shape), "monthly": list(monthly[w].shape), "profile": list(profile[w].shape)} for w in windows},
        "split_unit": "physical site_id",
        "split_fraction": {k: float(len(v) / len(sites)) for k, v in split_sites.items()},
        "split_counts_records": {k: int(len(v)) for k, v in split.items()},
        "future_gate": "68% finite discharge, precipitation and temperature in 2001-2010 plus finite four labels",
        "source_raw": str(args.raw),
    }
    args.out.with_suffix(".protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
