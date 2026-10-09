"""Paired uncertainty and region-held-out checks for the frozen method comparison."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/02_method'))
from features import load_cache
from ot import expand_to_days
OUT=ROOT/'results/method_development'
ATTR=['bfi','rb_flashiness','runoff_ratio','snow_fraction']
scal,ids,fdc,profiles,H=load_cache()
keys=scal.site_id.astype(str).to_numpy();Y=scal[ATTR].to_numpy(float)
summary=json.loads((OUT/'development_summary.json').read_text())
errors=dict(np.load(OUT/'test_errors.npz'));test=errors['test_indices'];groups=np.unique(keys[test])
scale=np.array(summary['training_iqr']);rng=np.random.default_rng(1907)
indices=[np.flatnonzero(keys[test]==g) for g in groups]
contrasts={}
for baseline in ['FDC','TIMING','original_OT','levels_1','transitions_1']:
    delta=errors['monthly_quantiles']-errors[baseline]
    # Resample sites, retaining all records and the record-weighted estimand.
    cluster=np.array([delta[i].sum(axis=0) for i in indices])
    sizes=np.array([len(i) for i in indices])
    draws=rng.integers(0,len(cluster),(5000,len(cluster)))
    bootstrap=cluster[draws].sum(axis=1)/sizes[draws].sum(axis=1)[:,None]
    contrast={'mean_error_difference':delta.mean(axis=0).tolist(),
              'cluster_bootstrap_ci95':np.percentile(bootstrap,[2.5,97.5],axis=0).T.tolist(),
              'aggregate_difference':float(np.mean(delta/scale)),
              'aggregate_ci95':np.percentile((bootstrap/scale).mean(axis=1),[2.5,97.5]).tolist(),
              'win_fraction':(delta<0).mean(axis=0).tolist(),
              'ties':(delta==0).mean(axis=0).tolist()}
    contrasts[baseline]=contrast
feature=np.load(OUT/'projected_features.npz')
F=np.cumsum(np.vstack([expand_to_days(p) for p in profiles]),axis=1)
fd=cdist(fdc,fdc,'sqeuclidean')/fdc.shape[1]
tm=np.empty_like(fd)
for i in range(len(ids)):
 d=F-F[i];tm[i]=np.abs(d-np.median(d,axis=1,keepdims=True)).sum(axis=1)**2
monthly=cdist(feature['monthly_quantiles'],feature['monthly_quantiles'],'sqeuclidean')/384
old=np.asarray(np.load(ROOT/'data/processed/dmatrix_beta1.npy'),float)**2
families={'FDC':fd,'FDC_TIMING':tm,'original_OT':old,'conditional_W2':monthly}
regions=scal.region.to_numpy();region_results={}
def evaluate(D,targets,donors):
 d=D[np.ix_(targets,donors)].copy();d[keys[targets,None]==keys[None,donors]]=np.inf
 nearest=np.argsort(d,axis=1,kind='stable')[:,:10]
 return np.abs(np.median(Y[donors[nearest]],axis=1)-Y[targets])
def compare(donors,calibration,test):
 scale=np.percentile(Y[donors],75,axis=0)-np.percentile(Y[donors],25,axis=0)
 scale=np.where(scale>0,scale,1)
 reference=fd/np.median(fd[np.ix_(donors,donors)])
 out={}
 for name,raw in families.items():
  normalized=raw/np.median(raw[np.ix_(donors,donors)])
  weights=[0.] if name in ['FDC','conditional_W2'] else [0.,.25,.5,.75,1.]
  scores=[]
  for w in weights:
   err=evaluate((1-w)*normalized+w*reference,calibration,donors)
   scores.append(float(np.mean(err/scale)))
  w=weights[int(np.argmin(scores))]
  err=evaluate((1-w)*normalized+w*reference,test,donors)
  out[name]={'fdc_weight':w,'calibration_score':min(scores),'test_score':float(np.mean(err/scale)),
             'test_mae':err.mean(axis=0).tolist(),'median_error':np.median(err,axis=0).tolist()}
 return out
for region in np.unique(regions):
 test=np.flatnonzero(regions==region)
 candidate=np.flatnonzero((regions!=region)&~np.isin(keys,keys[test]))
 gs=np.unique(keys[candidate]);np.random.default_rng(20260917).shuffle(gs)
 cut=int(.8*len(gs));donors=candidate[np.isin(keys[candidate],gs[:cut])]
 calibration=candidate[np.isin(keys[candidate],gs[cut:])]
 region_results[region]={'n_donors':len(donors),'n_calibration':len(calibration),'n_test':len(test),
                         'models':compare(donors,calibration,test)}
 print(region,region_results[region],flush=True)
protocol=json.loads((OUT/'frozen_protocol.json').read_text())
cov=pd.read_csv(ROOT/'data/processed/record_coverage.csv').set_index('gauge_id')
eligible=cov.loc[ids,'missing_fraction'].to_numpy()<=.05
parts={part:np.flatnonzero(np.isin(ids,protocol[part+'_ids'])&eligible) for part in ['donor','calibration','test']}
strict={'n_donors':len(parts['donor']),'n_calibration':len(parts['calibration']),'n_test':len(parts['test']),
        'models':compare(parts['donor'],parts['calibration'],parts['test'])}
out={'paired_test_contrasts':contrasts,'leave_one_region_out':region_results,'strict_coverage':strict}
(OUT/'robustness.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'paired_test_contrasts':contrasts,'strict_coverage':strict},indent=2))
