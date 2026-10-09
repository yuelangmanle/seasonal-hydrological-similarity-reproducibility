"""Rebuild the retained feature cohort from Caravan daily CSV observations."""
from pathlib import Path
import argparse, json, sys, hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code/02_method'))
from signatures import compute_signatures, CORE_SCALAR_NAMES
REGIONS = ['camels', 'lamah', 'camelscl', 'camelsbr', 'hysets']
WINDOW = pd.date_range('1981-01-01', '2010-12-31')
COLS = ['date', 'streamflow', 'total_precipitation_sum', 'temperature_2m_mean']

def process(path):
    path = Path(path)
    raw = pd.read_csv(path, usecols=COLS)
    dates = pd.DatetimeIndex(pd.to_datetime(raw.date))
    if dates.has_duplicates:
        raise ValueError(f'Duplicate dates: {path.name}')
    raw.index = dates
    df = raw.reindex(WINDOW)
    q = df.streamflow.astype(float)
    v = q.to_numpy(); good = np.isfinite(v) & (v >= 0)
    n = int(good.sum()); zero = float(np.mean(v[good] == 0)) if n else 1.
    qc = dict(gauge_id=path.stem, n_valid=n, missing_fraction=float(q.isna().mean()),
              zero_fraction=zero, n_negative=int((np.isfinite(v)&(v<0)).sum()),
              n_inf=int(np.isinf(v).sum()), status='excluded_length_or_zero')
    if n < 9000 or zero > .30:
        return qc, None
    # Match the manuscript extraction conventions; fail on unsupported values.
    if qc['n_negative'] or qc['n_inf']:
        raise ValueError(f'Unexpected negative/infinite flow: {path.name}')
    region = path.parent.name
    sig = compute_signatures(path.stem, region, q, precip=df.total_precipitation_sum,
                             temp=df.temperature_2m_mean)
    ok = (all(np.isfinite(sig.scalars[k]) for k in CORE_SCALAR_NAMES)
          and sig.fdc is not None and np.isfinite(sig.fdc).all()
          and sig.regime_hist is not None and np.isfinite(sig.regime_hist).all()
          and sig.regime_hist.sum() > 0 and np.isfinite(sig.monthly_profile).all())
    if not ok:
        qc['status'] = 'excluded_nonfinite_core'
        return qc, None
    attrs = [sig.scalars[k] for k in ['bfi','rb_flashiness','runoff_ratio','snow_fraction']]
    if not np.isfinite(attrs).all():
        raise ValueError(f'Evaluation attributes missing; do not silently impute: {path.name}')
    months = WINDOW.month.to_numpy(); logflow = np.log1p(v[good]/v[good].mean())
    monthly = np.concatenate([np.quantile(logflow[months[good]==m],(np.arange(32)+.5)/32) for m in range(1,13)])
    means = np.array([v[good & (months==m)].mean() for m in range(1,13)])
    logmeans = np.array([logflow[months[good]==m].mean() for m in range(1,13)])
    row = dict(gauge_id=path.stem,region=region,site_id=path.stem.split('_',1)[1],n_valid=n,**sig.scalars)
    row.update(imputed_runoff_ratio=False,imputed_snow_fraction=False)
    qc['status'] = 'retained'
    return qc, (row,np.asarray(sig.fdc,np.float32),np.asarray(sig.monthly_profile,np.float32),
                np.asarray(sig.regime_hist,np.float32),np.asarray(monthly,np.float32),attrs,means,logmeans)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-root',type=Path,required=True)
    ap.add_argument('--jobs',type=int,default=4)
    ap.add_argument('--limit',type=int,default=0,help='Debug only; incomplete cohorts fail verification.')
    args=ap.parse_args(); raw=args.raw_root.resolve()
    paths=[p for r in REGIONS for p in sorted((raw/r).glob('*.csv'))]
    if not paths:raise FileNotFoundError('Expected regional CSV directories under --raw-root')
    if args.limit:paths=paths[:args.limit]
    out=ROOT/'data/processed';out.mkdir(parents=True,exist_ok=True)
    link=ROOT/'data/raw/caravan';link.parent.mkdir(parents=True,exist_ok=True)
    if not link.exists():link.symlink_to(raw,target_is_directory=True)
    if link.resolve()!=raw:raise ValueError('Raw-data link already points to another source')
    rows=[];kept=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i,(qc,data) in enumerate(ex.map(process,paths,chunksize=8),1):
            rows.append(qc)
            if data is not None:kept.append(data)
            if i%200==0:print('Raw records processed:',i,'retained:',len(kept),flush=True)
    scal=pd.DataFrame([x[0] for x in kept]);ids=scal.gauge_id.to_numpy(dtype=str)
    attrs=sorted((raw/'_attributes').glob('*__attributes_other_*.csv'))
    if not attrs:raise FileNotFoundError('Missing _attributes/*__attributes_other_*.csv')
    coord=pd.concat([pd.read_csv(p) for p in attrs]).set_index('gauge_id').loc[ids,['gauge_lat','gauge_lon']].to_numpy(float)
    if not np.isfinite(coord).all():raise ValueError('Missing coordinates')
    np.savez_compressed(out/'raw_features.npz',ids=ids,site_ids=scal.site_id.to_numpy(dtype=str),
        fdc=np.stack([x[1] for x in kept]),profiles=np.stack([x[2] for x in kept]),
        histograms=np.stack([x[3] for x in kept]),monthly_quantiles=np.stack([x[4] for x in kept]),
        attributes=np.array([x[5] for x in kept]),coordinates=coord,
        monthly_mean=np.array([x[6] for x in kept]),monthly_log_mean=np.array([x[7] for x in kept]))
    scal.to_csv(out/'raw_scalars.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'record_coverage.csv',index=False)
    expected=np.load(ROOT.parent/'reproduction/benchmark_inputs.npz')
    z=np.load(out/'raw_features.npz');checks={}
    np.testing.assert_array_equal(z['ids'],expected['ids'])
    np.testing.assert_array_equal(z['site_ids'],expected['site_ids'])
    for key in ['fdc','profiles','monthly_quantiles','attributes']:
        diff=float(np.max(abs(z[key]-expected[key])));checks[key]=diff
        np.testing.assert_allclose(z[key],expected[key],rtol=0,atol=2e-7 if key!='attributes' else 1e-10)
    for name,key in [('robustness_checks/epsilon_histograms.npz','histograms'),('robustness_checks/extension_inputs.npz','coordinates')]:
        e=np.load(ROOT.parent/'reproduction'/name);np.testing.assert_allclose(z[key],e[key],rtol=0,atol=2e-7)
    report=dict(records=len(ids),source_files=len(paths),retained_by_region=scal.region.value_counts().to_dict(),
        max_absolute_difference=checks,scope='Raw extraction verified against archived inputs; transport matrix not yet recomputed')
    (out/'raw_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
