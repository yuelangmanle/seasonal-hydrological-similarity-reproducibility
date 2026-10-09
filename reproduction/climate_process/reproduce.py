"""Climate-restricted validation and exact monthly location/shape ablation.

Run from any working directory. Inputs are the adjacent released reproduction
arrays and the archived climate membership table. No external download is needed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
z = np.load(BASE / 'benchmark_inputs.npz')
x = np.load(BASE / 'sensitivity_checks/monthly_controls.npz')
a = np.load(BASE / 'benchmark_controls/inputs.npz')
g = np.load(BASE / 'robustness_checks/extension_inputs.npz')
climate = pd.read_csv(ROOT / 'climate_membership.csv', dtype={'gauge_id': str, 'site_id': str})
np.testing.assert_array_equal(climate.gauge_id.to_numpy(), z['ids'])
np.testing.assert_array_equal(g['ids'], z['ids'])
np.testing.assert_array_equal(a['ids'], z['ids'])
np.testing.assert_array_equal(x['ids'], z['ids'])
ids, keys, y = z['ids'], z['site_ids'], z['attributes']
d0, c0, t0 = [z[k] for k in ['donor', 'calibration', 'test']]
zone = climate.clz_cl_smj.to_numpy(int)
is_temperate = np.isin(zone, PROTOCOL['temperate_zone_ids'])
assert np.isin(zone, np.arange(1, 19)).all()
for left, right in [(d0, c0), (d0, t0), (c0, t0)]:
    assert not set(keys[left]) & set(keys[right])

# The decomposition is performed in float64; the original features are retained.
q = z['monthly_quantiles'].astype(float).reshape(-1, 12, 32)
location = q.mean(axis=2)
shape = (q - location[:, :, None]).reshape(len(q), -1)
features = {
    'FDC': z['fdc'], 'monthly': z['monthly_quantiles'],
    'annual_384': a['annual_384'],
    'monthly_regime': x['monthly_mean'] / x['monthly_mean'].mean(axis=1, keepdims=True),
    'quantile_location': location, 'quantile_shape': shape,
}
distances = {k: cdist(v, v[d0], 'sqeuclidean') / v.shape[1] for k, v in features.items()}
decomposition_error = float(np.max(abs(distances['monthly'] - distances['quantile_location'] - distances['quantile_shape'])))
assert decomposition_error < 1e-12
assert np.max(abs(shape.reshape(-1, 12, 32).mean(axis=2))) < 1e-12
# Direct pairwise calculation checks the algebra independently of cdist.
for i, j in [(0, 1), (27, 891), (100, 2415)]:
    full = np.mean((q[i] - q[j]) ** 2)
    parts = np.mean((location[i] - location[j]) ** 2) + np.mean((shape[i] - shape[j]) ** 2)
    np.testing.assert_allclose(full, parts, rtol=0, atol=1e-12)

distances['original_OT'] = z['original_ot_to_donors'] ** 2
days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
p = z['profiles'].astype(float)
p /= p.sum(axis=1, keepdims=True)
cumulative = np.cumsum(np.repeat(p / days, days, axis=1), axis=1)
timing = np.empty((len(y), len(d0)))
for i in range(len(y)):
    delta = cumulative[d0] - cumulative[i]
    timing[i] = np.abs(delta - np.median(delta, axis=1, keepdims=True)).sum(axis=1) ** 2
distances['allocation'] = timing
lat, lon = np.deg2rad(g['coordinates']).T
h = np.sin((lat[:, None] - lat[d0]) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[d0]) * np.sin((lon[:, None] - lon[d0]) / 2) ** 2
distances['geographic'] = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
methods = PROTOCOL['methods']
original_scale = np.subtract(*np.percentile(y[d0], [75, 25], axis=0))
results, group_rows, record_rows, attribute_rows = [], [], [], []
counts = []
for part, ix in [('donor', d0), ('calibration', c0), ('test', t0)]:
    for dataset in sorted(climate.dataset.unique()):
        for band, mask in [('cool', np.isin(zone, [8, 9, 10])), ('warm', np.isin(zone, [11, 12])), ('other', ~is_temperate)]:
            use = ix[(climate.dataset.to_numpy()[ix] == dataset) & mask[ix]]
            counts.append({'partition': part, 'dataset': dataset, 'band': band, 'records': len(use), 'stations': len(np.unique(keys[use]))})


def report_groups(design, test, errors, scale, donor):
    """Paired station-cluster intervals with duplicate records kept together."""
    subgroup = {'all': np.ones(len(test), bool), 'cool_temperate': np.isin(zone[test], [8, 9, 10]), 'warm_temperate': np.isin(zone[test], [11, 12])}
    for dataset in sorted(climate.dataset.unique()):
        subgroup['dataset_' + dataset] = climate.dataset.to_numpy()[test] == dataset
    snow_cut = float(np.median(y[donor, 3]))
    subgroup['lower_snow_fraction'] = y[test, 3] <= snow_cut
    subgroup['higher_snow_fraction'] = y[test, 3] > snow_cut
    err = np.stack([errors[n] for n in methods], axis=1)
    for label, mask in subgroup.items():
        ind = np.flatnonzero(mask)
        if not len(ind):
            continue
        _, inv = np.unique(keys[test[ind]], return_inverse=True)
        n_group = int(inv.max() + 1)
        vals = (err[ind] / scale).mean(axis=2)
        means = vals.mean(axis=0)
        contrast = vals[:, methods.index('monthly'), None] - vals
        lo = hi = np.full(len(methods), np.nan)
        attr_lo = attr_hi = np.full((len(methods), 4), np.nan)
        if n_group >= 10:
            rng = np.random.default_rng(PROTOCOL['bootstrap']['seed'])
            weight = rng.multinomial(n_group, np.full(n_group, 1 / n_group), size=PROTOCOL['bootstrap']['replicates'])
            sizes = np.bincount(inv)
            denom = weight @ sizes
            sums = np.zeros((n_group, len(methods)))
            np.add.at(sums, inv, contrast)
            valid = denom > 0
            draws = np.full((len(denom), len(methods)), np.nan)
            with np.errstate(all="ignore"):
                draws[valid] = (weight[valid] @ sums) / denom[valid, None]
            lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
            ad = err[ind, methods.index('monthly'), None, :] - err[ind]
            ag = np.zeros((n_group, len(methods), 4))
            np.add.at(ag, inv, ad)
            adraw = np.full((len(denom), len(methods), 4), np.nan)
            with np.errstate(all="ignore"):
                adraw[valid] = (weight[valid] @ ag.reshape(n_group, -1) / denom[valid, None]).reshape(-1, len(methods), 4)
            attr_lo, attr_hi = np.nanpercentile(adraw, [2.5, 97.5], axis=0)
        for j, name in enumerate(methods):
            group_rows.append({'design': design, 'subgroup': label, 'method': name, 'n_records': len(ind), 'n_stations': n_group, 'score': means[j], 'monthly_minus_method': contrast[:, j].mean(), 'ci_low': lo[j], 'ci_high': hi[j], 'snow_cut': snow_cut})
            for k, attr in enumerate(['BFI', 'flashiness', 'runoff_ratio', 'snow_fraction']):
                attribute_rows.append({'design': design, 'subgroup': label, 'method': name, 'attribute': attr, 'n_records': len(ind), 'mae': err[ind, j, k].mean(), 'monthly_minus_method': (err[ind, methods.index('monthly'), k] - err[ind, j, k]).mean(), 'ci_low': attr_lo[j, k], 'ci_high': attr_hi[j, k]})


def evaluate(design, donor, calibration, test, radius=0, frozen=None):
    col = np.flatnonzero(np.isin(d0, donor))
    np.testing.assert_array_equal(d0[col], donor)
    for left, right in [(donor, calibration), (donor, test), (calibration, test)]:
        assert not set(keys[left]) & set(keys[right])
    scale = np.subtract(*np.percentile(y[donor], [75, 25], axis=0))
    assert (scale > 0).all()
    norm = {n: distances[n][:, col] / np.median(distances[n][np.ix_(donor, col)]) for n in methods}
    excluded = (keys[:, None] == keys[donor]) | (distances['geographic'][:, col] < radius)
    assert ((~excluded[calibration]).sum(axis=1) >= 10).all()
    assert ((~excluded[test]).sum(axis=1) >= 10).all()

    def predict(name, weight, targets):
        d = ((1 - weight) * norm[name][targets] + weight * norm['FDC'][targets]).copy()
        d[excluded[targets]] = np.inf
        order = np.argsort(d, axis=1, kind='stable')[:, :10]
        assert np.isfinite(np.take_along_axis(d, order, axis=1)).all()
        neighbor = donor[order]
        error = abs(np.median(y[neighbor], axis=1) - y[targets])
        return error, neighbor

    errors, pure_errors, selected = {}, {}, {}
    for name in methods:
        weights = [0.] if name in PROTOCOL['fixed_unmixed'] else PROTOCOL['weights']
        cal_scores = [float(np.mean(predict(name, w, calibration)[0] / scale)) for w in weights]
        w = frozen[name] if frozen is not None else weights[int(np.argmin(cal_scores))]
        error, neighbor = predict(name, w, test)
        errors[name] = error
        selected[name] = w
        pure_error, _ = predict(name, 0, test)
        pure_errors[name] = pure_error
        results.append({'design': design, 'method': name, 'n_donor': len(donor), 'n_calibration': len(calibration), 'n_test': len(test), 'n_test_stations': len(np.unique(keys[test])), 'fdc_weight': w, 'score': float(np.mean(error / scale)), 'score_original_scale': float(np.mean(error / original_scale)), 'unmixed_score': float(np.mean(pure_error / scale)), 'calibration_score': cal_scores[weights.index(w)], 'attribute_iqr': scale.tolist()})
        for row, target in enumerate(test):
            record = {'design': design, 'method': name, 'gauge_id': ids[target], 'site_id': keys[target], 'zone': int(zone[target]), 'score': float(np.mean(error[row] / scale)), 'score_original_scale': float(np.mean(error[row] / original_scale)), 'donor_indices': ';'.join(map(str, neighbor[row]))}
            record.update(dict(zip(['BFI_error', 'flashiness_error', 'runoff_ratio_error', 'snow_fraction_error'], error[row])))
            record_rows.append(record)
    report_groups(design, test, errors, scale, donor)
    report_groups(design + '_unmixed', test, pure_errors, scale, donor)
    print(design, {n: round(float(np.mean(errors[n] / scale)), 5) for n in methods}, flush=True)
    return selected, errors

weights, pooled_errors = evaluate('pooled', d0, c0, t0)
# Reproduce the existing headline results before interpreting any new analysis.
old = json.loads((BASE / 'development_summary.json').read_text())['models']
for name, oldname in [('FDC', 'FDC'), ('allocation', 'TIMING'), ('original_OT', 'original_OT'), ('monthly', 'monthly_quantiles')]:
    np.testing.assert_allclose(np.mean(pooled_errors[name] / original_scale), old[oldname]['test_score'], rtol=0, atol=1e-10)
    assert weights[name] == old[oldname]['fdc_weight']
dt, ct, tt = [ix[is_temperate[ix]] for ix in [d0, c0, t0]]
evaluate('temperate_global_donors', d0, c0, tt, frozen=weights)
evaluate('temperate_only', dt, ct, tt)
evaluate('temperate_200km', dt, ct, tt, radius=200)
for name, rows in [('scores', results), ('paired_subgroups', group_rows), ('attribute_contrasts', attribute_rows), ('record_errors', record_rows), ('sample_counts', counts)]:
    pd.DataFrame(rows).to_csv(ROOT / (name + '.csv'), index=False)
summary = {'decomposition_max_abs_error': decomposition_error, 'temperate_records': int(is_temperate.sum()), 'temperate_stations': len(np.unique(keys[is_temperate])), 'temperate_donor': len(dt), 'temperate_calibration': len(ct), 'temperate_test': len(tt), 'temperate_test_stations': len(np.unique(keys[tt])), 'climate_zone_disagreement_stations': int(climate.groupby('site_id').clz_cl_smj.nunique().gt(1).sum()), 'headline_reproduction': 'passed', 'all_targets_have_ten_eligible_donors': True}
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
