"""Benchmark resolution and seasonal-target checks."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/benchmark_controls'
sys.path.insert(0, str(ROOT / 'code/02_method'))
from features import load_cache
from ot import expand_to_days

scalars, ids, fdc, profiles, _ = load_cache()
frozen = json.loads((ROOT / 'results/method_development/frozen_protocol.json').read_text())
d, c, t = [np.flatnonzero(np.isin(ids, frozen[k + '_ids'])) for k in ['donor', 'calibration', 'test']]
keys = scalars.site_id.astype(str).to_numpy()
assert not set(keys[d]) & (set(keys[c]) | set(keys[t]))
assert not set(keys[c]) & set(keys[t])
annual, seasonal = [], []
days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
phase = np.exp(2j * np.pi * np.arange(365) / 365)
for i, gid in enumerate(ids):
    raw = pd.read_csv(ROOT / 'data/raw/caravan' / scalars.iloc[i].region / (str(gid) + '.csv'), usecols=['date', 'streamflow'])
    dates = pd.DatetimeIndex(pd.to_datetime(raw.date))
    q = raw.streamflow.to_numpy(float)
    valid = (dates.year >= 1981) & (dates.year <= 2010) & np.isfinite(q) & (q >= 0)
    q, months = q[valid], dates.month.to_numpy()[valid]
    annual.append(np.quantile(np.log1p(q / q.mean()), (np.arange(384) + .5) / 384))
    monthly = np.array([q[months == m].mean() for m in range(1, 13)])
    daily = np.repeat(monthly, days)
    concentration = abs(np.sum(daily * phase) / daily.sum())
    amplitude = (monthly.max() - monthly.min()) / monthly.mean()
    seasonal.append([concentration, amplitude])
    if (i + 1) % 400 == 0:
        print('Raw records read:', i + 1, flush=True)

annual, seasonal = np.asarray(annual), np.asarray(seasonal)
assert np.isfinite(annual).all() and np.isfinite(seasonal).all()
Y = scalars[['bfi', 'rb_flashiness', 'runoff_ratio', 'snow_fraction']].to_numpy(float)
iqr = np.diff(np.percentile(Y[d], [25, 75], axis=0), axis=0)[0]
distance = cdist(annual, annual[d], 'sqeuclidean') / 384
nearest = np.argsort(distance, axis=1, kind='stable')[:, :10]
error = abs(np.median(Y[d[nearest]], axis=1) - Y)
result = {'annual_384': {name: {'score': float(np.mean(error[ix] / iqr)), 'mae': error[ix].mean(0).tolist()} for name, ix in [('calibration', c), ('test', t)]}}

ref = cdist(fdc, fdc[d], 'sqeuclidean') / fdc.shape[1]
ref /= np.median(ref[d])
monthly = np.load(ROOT / 'results/method_development/projected_features.npz')['monthly_quantiles']
cond = cdist(monthly, monthly[d], 'sqeuclidean') / 384
cond /= np.median(cond[d])
ot = np.asarray(np.load(ROOT / 'data/processed/dmatrix_beta1.npy', mmap_mode='r')[:, d], float) ** 2
ot /= np.median(ot[d])
F = np.cumsum(np.vstack([expand_to_days(p) for p in profiles]), axis=1)
allocation = np.empty_like(ref)
for i in range(len(ids)):
    delta = F[d] - F[i]
    allocation[i] = np.abs(delta - np.median(delta, axis=1, keepdims=True)).sum(1) ** 2
allocation /= np.median(allocation[d])
inp = np.load(ROOT / 'data/processed/raw_features.npz')
assert np.array_equal(inp['ids'], ids)
lat, lon = np.deg2rad(inp['coordinates']).T
h = np.sin((lat[:, None] - lat[d]) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[d]) * np.sin((lon[:, None] - lon[d]) / 2) ** 2
geo = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
distances = {'Monthly conditional W2': cond, 'FDC': ref, 'FDC + allocation': .75 * allocation + .25 * ref, 'Original OT + FDC': .75 * ot + .25 * ref, 'Geographic proximity': geo}
errors, neighbors = {}, {}
for name, matrix in distances.items():
    nn = np.argsort(matrix[t], axis=1, kind='stable')[:, :10]
    assert not np.any(keys[d[nn]] == keys[t, None])
    neighbors[name] = d[nn]
    errors[name] = abs(np.median(seasonal[d[nn]], axis=1) - seasonal[t])
result['seasonal_mae'] = {name: e.mean(0).tolist() for name, e in errors.items()}
rng = np.random.default_rng(2026091751)
groups = [np.flatnonzero(keys[t] == key) for key in np.unique(keys[t])]
contrasts = {name: [] for name in errors if name != 'Monthly conditional W2'}
for _ in range(5000):
    ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
    for name in contrasts:
        contrasts[name].append((errors['Monthly conditional W2'][ix] - errors[name][ix]).mean(0))
result['seasonal_paired'] = {name: {'difference': (errors['Monthly conditional W2'] - errors[name]).mean(0).tolist(), 'ci95': np.percentile(v, [2.5, 97.5], axis=0).T.tolist()} for name, v in contrasts.items()}
np.savez_compressed(OUT / 'inputs.npz', ids=np.asarray(ids, dtype=str), site_ids=np.asarray(keys, dtype=str), donor=d, calibration=c, test=t, annual_384=annual, attributes=Y, seasonal_targets=seasonal, **{'neighbors_' + str(i): v for i, v in enumerate(neighbors.values())})
(OUT / 'results.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
