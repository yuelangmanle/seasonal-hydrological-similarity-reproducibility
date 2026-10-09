"""Resource-bounded epsilon experiment; does not replace full-cohort results."""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.spatial.distance import cdist
R=Path(__file__).resolve().parent;O=R/'epsilon_rerun';O.mkdir(exist_ok=True)
from ot import RegimeMetric
z=np.load(R/'extension_inputs.npz');ids=z['ids'];Y=z['attributes'];fdc=z['fdc'];f=z['monthly_quantiles']
H=np.asarray(np.load(R/'epsilon_histograms.npz')['histograms'],float)+1e-12;H/=H.sum((1,2),keepdims=True)
orig=json.loads((R/'frozen_protocol.json').read_text())
rng=np.random.default_rng(2026091741)
d,c,t=[np.sort(rng.choice(np.flatnonzero(np.isin(ids,orig[name+'_ids'])),n,replace=False)) for name,n in [('donor',128),('calibration',64),('test',64)]]
ix=np.r_[d,c,t]
protocol=dict(seed=2026091741,donor_ids=ids[d].tolist(),calibration_ids=ids[c].tolist(),test_ids=ids[t].tolist(),epsilon=[.01,.05,.1],max_iter=100000,marginal_tolerance=1e-7,scope='Reduced fixed cohort: computational sensitivity, not full-cohort reranking',selection='calibration only')
(O/'epsilon_protocol.json').write_text(json.dumps(protocol,indent=2))
scale=np.diff(np.percentile(Y[d],[25,75],axis=0),axis=0)[0]
fd=cdist(fdc[ix],fdc[d],'sqeuclidean')/fdc.shape[1];fd/=np.median(fd[:len(d)])
monthly=cdist(f[ix],f[d],'sqeuclidean')/384
res=[]
def assess(raw,name):
 raw=raw/np.median(raw[:len(d)]);weights=[0.] if name=='FDC' else [0.,.25,.5,.75,1.]
 def err(dist,lo,hi,target):
  nn=np.argsort(dist[lo:hi],axis=1,kind='stable')[:,:10]
  return np.abs(np.median(Y[d[nn]],axis=1)-Y[target])
 cs=[float(np.mean(err((1-w)*raw+w*fd,128,192,c)/scale)) for w in weights]
 w=weights[int(np.argmin(cs))];e=err((1-w)*raw+w*fd,192,256,t)
 rec=dict(method=name,weight=w,calibration_score=min(cs),score=float(np.mean(e/scale)),mae=e.mean(0).tolist());res.append(rec);print(rec,flush=True)
assess(fd,'FDC');assess(monthly,'monthly')
def costs(m,A,B):
 # Batch updates, checked by marginal constraints rather than dual changes.
 u=np.ones_like(A);v=np.ones_like(B)
 for it in range(100000):
  v=B/np.maximum(m.kd@u@m.kf.T,1e-300);u=A/np.maximum(m.kd@v@m.kf.T,1e-300)
  if (it+1)%100==0:
   err=np.max(np.abs(v*(m.kd@u@m.kf.T)-B),axis=(1,2))
   if np.max(err)<1e-7:break
 if not np.isfinite(u).all() or np.max(err)>=1e-7:raise RuntimeError(f'nonconverged {m.reg} {np.max(err)}')
 vals=m.beta*np.einsum('ij,mij->m',m.cd*m.kd,(u@m.kf)@v.transpose(0,2,1))+m.alpha*np.einsum('ij,mij->m',m.cf*m.kf,(u.transpose(0,2,1)@m.kd)@v)
 return vals,it+1,float(np.max(err))
checks=[]
for eps in [.05,.1]:
 m=RegimeMetric.for_regime_grid(reg=eps);selfs,ni,re=costs(m,H[ix],H[ix]);matrix=np.empty((len(ix),len(d)));start=time.time();maxiter=ni;maxres=re
 for j,idx in enumerate(ix):
  v,ni,re=costs(m,np.broadcast_to(H[idx],H[d].shape),H[d]);matrix[j]=np.maximum(v-.5*selfs[j]-.5*selfs[:len(d)],0);maxiter=max(maxiter,ni);maxres=max(maxres,re)
  if j%32==0:print('epsilon',eps,'row',j,'seconds',round(time.time()-start,1),flush=True)
 np.save(O/f'epsilon_{eps}_squared.npy',matrix)
 assess(matrix,f'OT_eps{eps}')
 checks.append(dict(epsilon=eps,max_iterations=maxiter,max_marginal_residual=maxres,seconds=time.time()-start))
 (O/'epsilon_results.json').write_text(json.dumps(dict(protocol=protocol,results=res,convergence=checks),indent=2))
