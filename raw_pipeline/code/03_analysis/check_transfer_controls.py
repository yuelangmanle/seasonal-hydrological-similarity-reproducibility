"""Follow-up controls on the existing split; no test-driven parameter selection."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/02_method'))
from features import load_cache
from ot import expand_to_days
O=ROOT/'results/method_development'
s,ids,fdc,profiles,_=load_cache();cfg=json.loads((O/'frozen_protocol.json').read_text())
p={k:np.flatnonzero(np.isin(ids,cfg[k+'_ids'])) for k in ['donor','calibration','test']}
d=p['donor'];t=p['test'];Y=s[['bfi','rb_flashiness','runoff_ratio','snow_fraction']].to_numpy(float)
scale=np.diff(np.percentile(Y[d],[25,75],axis=0),axis=0)[0]
f=np.load(O/'conditional_resolution_features.npz')
D={k:cdist(f[k],f[k][d],'sqeuclidean')/f[k].shape[1] for k in ['monthly_32','annual_32']}
shape=s[['cv','fdc_slope','q95_q50','q90_q50','q50_q10','q05_q50']].to_numpy(float)
mu=shape[d].mean(0);sd=shape[d].std(0);z=(shape-mu)/np.where(sd>0,sd,1)
D['six_shape_signatures']=cdist(z,z[d],'sqeuclidean')/6
D['FDC']=cdist(fdc,fdc[d],'sqeuclidean')/fdc.shape[1]
D['original_OT']=np.asarray(np.load(ROOT/'data/processed/dmatrix_beta1.npy',mmap_mode='r')[:,d],float)**2
F=np.cumsum(np.vstack([expand_to_days(x) for x in profiles]),axis=1)
D['timing']=np.empty_like(D['FDC'])
for i in range(len(ids)):
 dif=F[d]-F[i];D['timing'][i]=np.abs(dif-np.median(dif,axis=1,keepdims=True)).sum(1)**2
for name in D:D[name]/=np.median(D[name][d])
D['original_OT']=.75*D['original_OT']+.25*D['FDC']
D['FDC_TIMING']=.75*D['timing']+.25*D['FDC']
out={};errors={}
for name,dist in D.items():
 nn=np.argsort(dist,axis=1,kind='stable')[:,:20];out[name]={}
 for k in [5,10,20]:
  err=np.abs(np.median(Y[d[nn[:,:k]]],axis=1)-Y)
  out[name][str(k)]={part:{'score':float(np.mean(err[p[part]]/scale)),'mae':err[p[part]].mean(0).tolist()} for part in ['calibration','test']}
  if k==10:errors[name]=err[t]
keys=s.site_id.astype(str).to_numpy()[t];groups=np.unique(keys);ix=[np.flatnonzero(keys==g) for g in groups]
rng=np.random.default_rng(3290);draws=rng.integers(len(ix),size=(5000,len(ix)));sizes=np.array([len(i) for i in ix])
contrasts={}
for name in ['annual_32','six_shape_signatures']:
 delta=errors['monthly_32']-errors[name];totals=np.array([delta[i].sum(0) for i in ix])
 b=totals[draws].sum(1)/sizes[draws].sum(1)[:,None]
 contrasts[name]={'aggregate_difference':float(np.mean(delta/scale)),'ci95':np.percentile((b/scale).mean(1),[2.5,97.5]).tolist()}
out['paired_controls']=contrasts
(O/'transfer_controls.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
