"""Sensitivity checks from archived derived inputs.

Place this directory beside benchmark_inputs.npz in the reproduction archive.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist
R=Path(__file__).resolve().parent
candidate=R.parent/'benchmark_inputs.npz'
if not candidate.exists():
    raise FileNotFoundError("Expected reproduction/benchmark_inputs.npz")
z=np.load(candidate);x=np.load(R/'monthly_controls.npz')
d,c,t=[z[k] for k in ['donor','calibration','test']];y=z['attributes'];keys=z['site_ids']
scale=np.diff(np.percentile(y[d],[25,75],axis=0),axis=0)[0]
D={n:cdist(z[k],z[k][d],'sqeuclidean')/z[k].shape[1] for n,k in [('FDC','fdc'),('conditional','monthly_quantiles')]}
D['OT']=z['original_ot_to_donors']**2
days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
def timing(profile):
 p=profile/profile.sum(1,keepdims=True);F=np.cumsum(np.repeat(p/days,days,axis=1),axis=1)
 out=np.empty((len(p),len(d)))
 for i in range(len(p)):
  diff=F[d]-F[i];out[i]=np.abs(diff-np.median(diff,axis=1,keepdims=True)).sum(1)**2
 return out
D['allocation']=timing(z['profiles'])
D['coverage_adjusted_allocation']=timing(x['monthly_mean']*days)
for name,vals in [('monthly_log_mean',x['monthly_log_mean']),('monthly_regime',x['monthly_mean']/x['monthly_mean'].mean(1,keepdims=True))]:D[name]=cdist(vals,vals[d],'sqeuclidean')/12
D={k:v/np.median(v[d]) for k,v in D.items()}
def nearest(distance,idx,unique=False):
 order=np.argsort(distance[idx],axis=1,kind='stable')
 if not unique:return d[order[:,:10]]
 picked=[]
 for row in order:
  seen=set();chosen=[]
  for j in row:
   if keys[d[j]] not in seen:chosen.append(d[j]);seen.add(keys[d[j]])
   if len(chosen)==10:break
  picked.append(chosen)
 return np.array(picked)
result={};errs={}
for mode,unique,names in [('primary',False,list(D)),('unique_donors',True,['FDC','conditional','allocation','OT'])]:
 result[mode]={}
 for name in names:
  weights=[0.] if name=='FDC' else [0.,.25,.5,.75,1.]
  scores=[]
  for w in weights:
   nn=nearest((1-w)*D[name]+w*D['FDC'],c,unique);scores.append(float(np.mean(abs(np.median(y[nn],axis=1)-y[c])/scale)))
  w=weights[int(np.argmin(scores))];nn=nearest((1-w)*D[name]+w*D['FDC'],t,unique)
  error=abs(np.median(y[nn],axis=1)-y[t]);errs[mode+'/'+name]=np.mean(error/scale,axis=1)
  cnt=np.array([len(set(keys[row])) for row in nn]);values,counts=np.unique(cnt,return_counts=True)
  result[mode][name]={'fdc_weight':w,'calibration_score':min(scores),'test_score':float(np.mean(error/scale)),'unique_donor_counts':dict(zip(map(str,values),map(int,counts)))}
  print(mode,name,result[mode][name],flush=True)
groups=[np.flatnonzero(keys[t]==k) for k in np.unique(keys[t])];rng=np.random.default_rng(2026091761)
contrasts={name:errs['primary/conditional']-errs['primary/'+name] for name in ['monthly_log_mean','monthly_regime','coverage_adjusted_allocation']};samples={name:[] for name in contrasts}
for _ in range(5000):
 ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
 for name,diff in contrasts.items():samples[name].append(float(diff[ix].mean()))
result['paired']={name:{'difference':float(diff.mean()),'ci95':np.percentile(samples[name],[2.5,97.5]).tolist()} for name,diff in contrasts.items()}
out=R/'sensitivity_results.json'
if out.exists():
 expected=json.loads(out.read_text())
 for mode in ['primary','unique_donors']:
  for name,item in result[mode].items():np.testing.assert_allclose(item['test_score'],expected[mode][name]['test_score'],atol=1e-12)
 print('PASS: sensitivity scores reproduce saved results')
else:out.write_text(json.dumps(result,indent=2))
print(result['paired'])
