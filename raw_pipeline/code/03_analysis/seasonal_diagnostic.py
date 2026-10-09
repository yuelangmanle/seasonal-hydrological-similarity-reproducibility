"""Rebuild the station-disjoint seasonal diagnostic from raw-derived features."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/02_method'))
from features import load_cache
from signatures import MAGNITUDE_NAMES
from ot import expand_to_days
from diagnostic_helpers import sample_pairs,sample_pairs_within,timing_from_cum,curve_rms
scal,ids,fdc,prof,H=load_cache()
F=np.cumsum(np.vstack([expand_to_days(x) for x in prof]),axis=1)
(ROOT/'results').mkdir(exist_ok=True)
for mode in ['pooled','within']:
 a,b=sample_pairs(len(ids),150000,20260911) if mode=='pooled' else sample_pairs_within(len(ids),scal.region.to_numpy(),150000,20260911)
 np.savez_compressed(ROOT/f'results/gap_pairs_{mode}.npz',ia=a,ib=b,d_time=timing_from_cum(F,a,b).astype(np.float32),d_fdc=curve_rms(fdc.astype(float),a,b).astype(np.float32))
out={'catchment_disjoint':{}}
for mode in ['pooled','within']:
 z=dict(np.load(ROOT/f'results/gap_pairs_{mode}.npz'))
 keys=scal.site_id.astype(str).to_numpy();unique=np.unique(keys)
 rng=np.random.default_rng(20260916);rng.shuffle(unique)
 train_sites=np.isin(keys,unique[:len(unique)//2]);test_sites=~train_sites
 a,b=z['ia'],z['ib'];tr=train_sites[a]&train_sites[b];te=test_sites[a]&test_sites[b]
 train_scal=scal[MAGNITUDE_NAMES].to_numpy(float)[train_sites]
 mu,sd=train_scal.mean(axis=0),train_scal.std(axis=0)
 Xs=(scal[MAGNITUDE_NAMES].to_numpy(float)-mu)/np.where(sd>0,sd,1)
 dmag=np.sqrt(((Xs[a]-Xs[b])**2).sum(axis=1))
 block={'train_pairs':int(tr.sum()),'test_pairs':int(te.sum()),'shared_site_ids':0}
 for name,x in [('FDC',z['d_fdc']),('composite',dmag),('individual_signatures',np.abs(Xs[a]-Xs[b]))]:
  x=np.asarray(x,float).reshape(len(a),-1);y=z['d_time'].astype(float)
  xt,xv=x[tr],x[te];yt,yv=y[tr],y[te]
  coef=np.linalg.lstsq(np.column_stack([np.ones(len(xt)),xt]),yt,rcond=None)[0]
  pred=np.column_stack([np.ones(len(xv)),xv])@coef
  mean,sd=xt.mean(axis=0),xt.std(axis=0);sd=np.where(sd>0,sd,1)
  nn=cKDTree((xt-mean)/sd).query((xv-mean)/sd,k=25,workers=2)[1]
  knn=yt[nn].mean(axis=1)
  den=((yv-yv.mean())**2).sum()
  block[name]={'linear_r2':float(1-((yv-pred)**2).sum()/den),
               'knn_r2':float(1-((yv-knn)**2).sum()/den)}
 out['catchment_disjoint'][mode]=block
(ROOT/'results/seasonal_diagnostic.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

rows=[]
for scope,values in out['catchment_disjoint'].items():
 for key in ['FDC','composite','individual_signatures']:
  rows.append(dict(scope=scope,predictor=key,linear_test_R2=values[key]['linear_r2'],knn_test_R2=values[key]['knn_r2'],train_pairs=values['train_pairs'],test_pairs=values['test_pairs']))
frame=pd.DataFrame(rows)
reference=pd.read_csv(ROOT.parent/'figures/figure_data/figure1_seasonal_diagnostics.csv')
for row in rows:
 ref=reference[(reference.scope==row['scope'])&(reference.predictor==row['predictor'])].iloc[0]
 np.testing.assert_allclose([row['linear_test_R2'],row['knn_test_R2']],[ref.linear_test_R2,ref.knn_test_R2],rtol=0,atol=1e-8)
frame.to_csv(ROOT/'results/figure1_seasonal_diagnostics.csv',index=False)
print('PASS: seasonal diagnostic matches Figure 1 source table.')
