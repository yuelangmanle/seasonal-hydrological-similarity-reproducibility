"""Frozen exploratory split and distance-constrained donor sensitivity."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
R=Path(__file__).resolve().parent;O=R/'rerun';O.mkdir(exist_ok=True)
z=np.load(R/'extension_inputs.npz');ids=z['ids'];keys=z['site_ids'];Y=z['attributes'];fdc=z['fdc'];p=z['profiles'];f=z
from ot import expand_to_days
D={'FDC':cdist(fdc,fdc,'sqeuclidean')/fdc.shape[1],'monthly':cdist(f['monthly_quantiles'],f['monthly_quantiles'],'sqeuclidean')/384,'OT':z['original_ot'].astype(float)**2}
F=np.cumsum(np.vstack([expand_to_days(x) for x in p]),axis=1);tim=np.empty_like(D['FDC'])
for i in range(len(ids)):
 x=F-F[i];tim[i]=np.abs(x-np.median(x,axis=1,keepdims=True)).sum(1)**2
D['allocation']=tim
coords=z['coordinates']
assert np.isfinite(coords).all()
lat,lon=np.deg2rad(coords).T
h=np.sin((lat[:,None]-lat)/2)**2+np.cos(lat[:,None])*np.cos(lat)*np.sin((lon[:,None]-lon)/2)**2
geo=6371.0088*2*np.arcsin(np.sqrt(np.clip(h,0,1)));D['geographic']=geo
np.savez_compressed(O/'geography.npz',ids=ids,coordinates=coords,distance_km=geo)
protocol=json.loads((R/'protocol.json').read_text());records=[]
def compare(d,c,t,label,radius=0):
 assert not set(keys[d])&set(keys[c]) and not set(keys[d])&set(keys[t]) and not set(keys[c])&set(keys[t])
 scale=np.percentile(Y[d],75,axis=0)-np.percentile(Y[d],25,axis=0);assert (scale>0).all()
 mask=(keys[:,None]==keys[d][None,:]) | (geo[:,d]<radius)
 c0,t0=len(c),len(t);c=c[(~mask[c]).sum(1)>=10];t=t[(~mask[t]).sum(1)>=10]
 assert len(c) and len(t)
 ref=D['FDC'][:,d]/np.median(D['FDC'][np.ix_(d,d)])
 def error(dist,ix):
  m=dist[ix].copy();m[mask[ix]]=np.inf
  nn=np.argsort(m,axis=1,kind='stable')[:,:10]
  assert np.isfinite(np.take_along_axis(m,nn,axis=1)).all()
  return np.abs(np.median(Y[d[nn]],axis=1)-Y[ix])
 for name,raw in D.items():
  dist=raw[:,d]/np.median(raw[np.ix_(d,d)])
  weights=[0.] if name in ['FDC','geographic'] else protocol['weights']
  scores=[np.mean(error((1-w)*dist+w*ref,c)/scale) for w in weights]
  w=weights[int(np.argmin(scores))];e=error((1-w)*dist+w*ref,t)
  records.append(dict(design=label,radius_km=radius,method=name,weight=w,calibration_score=float(min(scores)),score=float(np.mean(e/scale)),n_cal=len(c),n_test=len(t),excluded_cal=c0-len(c),excluded_test=t0-len(t),mae=e.mean(0).tolist()))
 print(label,radius,{x['method']:round(x['score'],5) for x in records[-5:]},flush=True)
 pd.DataFrame(records).to_json(O/'results.json',orient='records',indent=2)
orig=json.loads((R/'frozen_protocol.json').read_text())
d,c,t=[np.flatnonzero(np.isin(ids,orig[x+'_ids'])) for x in ['donor','calibration','test']]
compare(d,c,t,'primary')
for rad in protocol['buffer_km']:compare(d,c,t,'buffer',rad)
for seed in protocol['repeat_seeds']:
 g=np.unique(keys);np.random.default_rng(seed).shuffle(g);n=len(g)
 d,c,t=[np.flatnonzero(np.isin(keys,x)) for x in [g[:int(.6*n)],g[int(.6*n):int(.8*n)],g[int(.8*n):]]]
 compare(d,c,t,str(seed))
pd.DataFrame(records).to_csv(O/'results.csv',index=False)
