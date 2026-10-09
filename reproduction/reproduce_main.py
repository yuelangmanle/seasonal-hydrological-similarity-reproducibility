"""Reproduce the four primary internal-transfer results from archived inputs."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist
R=Path(__file__).resolve().parent
z=np.load(R/'benchmark_inputs.npz');donors=z['donor'];test=z['test'];cal=z['calibration'];Y=z['attributes']
keys=z['site_ids']
assert not set(keys[donors])&set(keys[test])
assert not set(keys[donors])&set(keys[cal])
assert not set(keys[test])&set(keys[cal])
scale=np.diff(np.percentile(Y[donors],[25,75],axis=0),axis=0)[0]
D={name:cdist(z[key],z[key][donors],'sqeuclidean')/z[key].shape[1] for name,key in [('FDC','fdc'),('monthly_quantiles','monthly_quantiles')]}
D['original_OT']=z['original_ot_to_donors']**2
month_days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
p=z['profiles'].astype(float);p/=p.sum(1,keepdims=True)
F=np.cumsum(np.repeat(p/month_days,month_days,axis=1),axis=1)
D['TIMING']=np.empty((len(Y),len(donors)))
for i in range(len(Y)):
 diff=F[donors]-F[i];D['TIMING'][i]=np.abs(diff-np.median(diff,axis=1,keepdims=True)).sum(1)**2
for k in D:D[k]/=np.median(D[k][donors])
expected=json.loads((R/'development_summary.json').read_text());out={}
for name,dist in D.items():
 weights=[0.] if name=='FDC' else [0.,.25,.5,.75,1.]
 candidates=[]
 for w in weights:
  mixed=(1-w)*dist+w*D['FDC'];nn=np.argsort(mixed[cal],axis=1,kind='stable')[:,:10]
  err=np.abs(np.median(Y[donors[nn]],axis=1)-Y[cal]);candidates.append(float(np.mean(err/scale)))
 w=weights[int(np.argmin(candidates))];nn=np.argsort(((1-w)*dist+w*D['FDC'])[test],axis=1,kind='stable')[:,:10]
 err=np.abs(np.median(Y[donors[nn]],axis=1)-Y[test]);score=float(np.mean(err/scale))
 assert w==expected['models'][name]['fdc_weight']
 np.testing.assert_allclose(score,expected['models'][name]['test_score'],rtol=0,atol=1e-10)
 out[name]={'fdc_weight':w,'test_score':score,'test_mae':err.mean(0).tolist()}
print(json.dumps(out,indent=2));(R/'reproduced_main.json').write_text(json.dumps(out,indent=2))
