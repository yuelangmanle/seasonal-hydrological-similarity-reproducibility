"""Reproduce supplementary scores and bootstrap intervals from derived inputs.

Run beside inputs.npz and results.json. Raw-flow extraction is documented in
benchmark_controls.py; this replay does not regenerate the derived features.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
z = np.load(ROOT / 'inputs.npz')
expected = json.loads((ROOT / 'results.json').read_text())
d, c, t = [z[name] for name in ['donor', 'calibration', 'test']]
y = z['attributes']
scale = np.diff(np.percentile(y[d], [25, 75], axis=0), axis=0)[0]
annual = z['annual_384']
for label, ix in [('calibration', c), ('test', t)]:
    predicted = []
    for target in ix:
        distance = np.mean((annual[d] - annual[target]) ** 2, axis=1)
        neighbors = np.argsort(distance, kind='stable')[:10]
        predicted.append(np.median(y[d[neighbors]], axis=0))
    error = abs(np.asarray(predicted) - y[ix])
    np.testing.assert_allclose(np.mean(error / scale), expected['annual_384'][label]['score'], atol=1e-12)
names = list(expected['seasonal_mae'])
errors = {}
for i, name in enumerate(names):
    prediction = np.median(z['seasonal_targets'][z['neighbors_' + str(i)]], axis=1)
    errors[name] = abs(prediction - z['seasonal_targets'][t])
    np.testing.assert_allclose(errors[name].mean(0), expected['seasonal_mae'][name], atol=1e-12)
keys = z['site_ids'][t]
groups = [np.flatnonzero(keys == key) for key in np.unique(keys)]
rng = np.random.default_rng(2026091751)
contrasts = {name: [] for name in names[1:]}
for _ in range(5000):
    ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
    for name in contrasts:
        contrasts[name].append((errors[names[0]][ix] - errors[name][ix]).mean(0))
for name, v in contrasts.items():
    np.testing.assert_allclose(np.percentile(v, [2.5, 97.5], axis=0).T, expected['seasonal_paired'][name]['ci95'], atol=1e-12)
print('PASS: annual 384 scores, five seasonal predictors and all eight paired bootstrap intervals')
