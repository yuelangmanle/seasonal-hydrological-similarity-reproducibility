"""Fixed-split resolution checks and a transformation-matched annual baseline.

These are follow-up sensitivity analyses, not fresh held-out confirmation.
No setting is selected on the test outcomes.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/02_method'))
from features import load_cache
from seasonal_transport import monthly_flow_quantiles
OUT=ROOT/'results/method_development'
scal,ids,*_=load_cache()
protocol=json.loads((OUT/'frozen_protocol.json').read_text())
parts={p:np.flatnonzero(np.isin(ids,protocol[p+'_ids'])) for p in ['donor','calibration','test']}
features={f'monthly_{n}':[] for n in [32,64,128]}
features.update({f'annual_{n}':[] for n in [32,128]})
for i,gid in enumerate(ids):
    row=scal.iloc[i]
    raw=pd.read_csv(ROOT/'data/raw/caravan'/row.region/(str(gid)+'.csv'),usecols=['date','streamflow'])
    dt=pd.DatetimeIndex(pd.to_datetime(raw.date));keep=(dt.year>=1981)&(dt.year<=2010)
    q=raw.streamflow.to_numpy(float)[keep];months=dt.month.to_numpy()[keep]
    for n in [32,64,128]:features[f'monthly_{n}'].append(monthly_flow_quantiles(q,months,n).ravel())
    good=np.isfinite(q)&(q>=0);x=np.log1p(q[good]/q[good].mean())
    for n in [32,128]:features[f'annual_{n}'].append(np.quantile(x,(np.arange(n)+.5)/n))
    if (i+1)%400==0:print('read',i+1,flush=True)
Y=scal[['bfi','rb_flashiness','runoff_ratio','snow_fraction']].to_numpy(float)
d=parts['donor'];scale=np.diff(np.percentile(Y[d],[25,75],axis=0),axis=0)[0]
out={};errors={}
for name,values in features.items():
    x=np.asarray(values);D=cdist(x,x[d],'sqeuclidean')/x.shape[1]
    nn=np.argsort(D,axis=1,kind='stable')[:,:10]
    err=np.abs(np.median(Y[d[nn]],axis=1)-Y)
    out[name]={p:{'score':float(np.mean(err[parts[p]]/scale)),
                  'mae':err[parts[p]].mean(axis=0).tolist()} for p in ['calibration','test']}
    errors[name]=err[parts['test']]
    features[name]=x.astype(np.float32)
old=np.load(OUT/'projected_features.npz')['monthly_quantiles']
out['cached_representation_max_difference']=float(np.max(np.abs(features['monthly_32']-old)))
np.savez_compressed(OUT/'conditional_resolution_features.npz',ids=ids,**features)
np.savez_compressed(OUT/'conditional_resolution_errors.npz',**errors)
(OUT/'conditional_resolution.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
