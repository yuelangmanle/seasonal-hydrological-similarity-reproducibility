"""Target early-signature persistence vs calibrated monthly-W2 donor transfer.

Uses exactly the existing label functions and compact cohort/splits. BFI/flashiness
need early discharge only; runoff ratio/snow proxy additionally need early forcing.
This is a locked-usefulness comparator, not a tuned persistence model.
"""
from __future__ import annotations
import argparse,importlib.util,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
OUT=Path(__file__).resolve().parent
WINDOWS={'3yr':(1981,1983),'5yr':(1981,1985),'10yr':(1981,1990),'5yr_shifted':(1986,1990)}
ATTRS=['bfi','rb_flashiness','runoff_ratio','snow_fraction']
SPEC={'analysis':'direct early target signature persistence predicts 2001-2010 labels; compared with current calibrated W2 candidate','prediction':'predict each later label by the same label calculated from the early target window; no fit or tuning','information':'BFI and flashiness require early target discharge; runoff ratio and temperature-threshold snow proxy require additional early precipitation/temperature, unlike flow-only W2 retrieval','cohort':'original temporal-mask test targets; per-label finite persistence intersection; four-label score needs all early labels finite','scaling':'same eligible donor future-label IQR as temporal-mask comparator','inference':'5000 fixed-prediction physical-site bootstrap; paired W2 minus persistence error (negative favors W2)','selection':'W2 FDC weights fixed to published window selections, not retuned on persistence results'}
def boot(v,site,reps=5000,seed=2026100907):
 groups=[np.flatnonzero(site==s) for s in sorted(set(site))]; rng=np.random.default_rng(seed); vals=[]
 for b in range(reps):
  ix=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))]); vals.append(float(np.mean(v[ix])))
 return {'estimate':float(np.mean(v)),'ci95':[float(x) for x in np.percentile(vals,[2.5,97.5])],'n_sites':len(groups),'n_records':len(v)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--inputs',type=Path,required=True); ap.add_argument('--early-labels',type=Path); ap.add_argument('--raw',type=Path); ap.add_argument('--label-source',type=Path); ap.add_argument('--out-dir',type=Path,default=OUT); args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True); (args.out_dir/'persistence_protocol.json').write_text(json.dumps(SPEC,indent=2))
 with np.load(args.inputs,allow_pickle=False) as loaded: data={k:loaded[k] for k in loaded.files}
 n=len(data['gauge_id']); early={w:np.full((n,4),np.nan) for w in WINDOWS}
 if args.early_labels:
  with np.load(args.early_labels,allow_pickle=False) as cached:
   assert np.array_equal(cached['gauge_id'],data['gauge_id'])
   assert np.array_equal(cached['site_id'],data['site_id'])
   early={w:cached[w] for w in WINDOWS}
 else:
  if not args.raw or not args.label_source: ap.error('provide --early-labels, or both --raw and --label-source')
  spec=importlib.util.spec_from_file_location('label_source',args.label_source); a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
  def one(i):
   path=args.raw/str(data['region'][i])/(str(data['gauge_id'][i])+'.csv')
   if not path.exists():return i,None
   frame=pd.read_csv(path,usecols=['date','streamflow','total_precipitation_sum','temperature_2m_mean'])
   return i,{w:a._future_attributes(a._date_window(frame,*years)) for w,years in WINDOWS.items()}
  with ThreadPoolExecutor(max_workers=8) as pool:
   for i,vals in pool.map(one,range(n)):
    if vals is not None:
     for w in WINDOWS:early[w][i]=vals[w]
 np.savez_compressed(args.out_dir/'early_signature_labels.npz',**{w:v for w,v in early.items()},gauge_id=data['gauge_id'],site_id=data['site_id'])
 results={}; weights={'3yr':.25,'5yr':0.,'10yr':0.,'5yr_shifted':0.}
 for w in WINDOWS:
  ok=data[w+'_feature_ok']&data['label_ok']; ok&=np.isfinite(data[w+'_fdc']).all(1)&np.isfinite(data[w+'_monthly']).all(1)&np.isfinite(data[w+'_profile']).all(1); ix=np.flatnonzero(ok); pos={int(old):j for j,old in enumerate(ix)}
  d=np.array([pos[int(i)] for i in data['split_donor'] if int(i) in pos]); t=np.array([pos[int(i)] for i in data['split_test'] if int(i) in pos]); y=data['future_attrs'][ix]; site=data['site_id'][ix].astype(str)
  fdc=data[w+'_fdc'][ix]; mon=data[w+'_monthly'][ix]; df=cdist(fdc,fdc,'sqeuclidean')/fdc.shape[1]; dm=cdist(mon,mon,'sqeuclidean')/mon.shape[1]
  df/=np.median(df[np.ix_(d,d)]); dm/=np.median(dm[np.ix_(d,d)]); distance=(1-weights[w])*dm+weights[w]*df
  dt=distance[np.ix_(t,d)].copy();dt[site[t,None]==site[d][None,:]]=np.inf; nn=np.argsort(dt,axis=1,kind='stable')[:,:10]; wp=np.median(y[d[nn]],axis=1); pp=early[w][ix[t]]
  ew=abs(wp-y[t]); ep=abs(pp-y[t]); iqr=np.percentile(y[d],75,axis=0)-np.percentile(y[d],25,axis=0); iqr=np.where(iqr>0,iqr,1.)
  rows=[]
  for j,ti in enumerate(t):
   row={'window':w,'gauge_id':data['gauge_id'][ix[ti]],'site_id':site[ti],'region':data['region'][ix[ti]]}
   for k,attr in enumerate(ATTRS):row.update({f'{attr}_persistence_prediction':pp[j,k],f'{attr}_future_label':y[ti,k],f'{attr}_w2_prediction':wp[j,k],f'{attr}_persistence_raw_error':ep[j,k],f'{attr}_w2_raw_error':ew[j,k]})
   rows.append(row)
  pd.DataFrame(rows).to_csv(args.out_dir/f'persistence_per_target_{w}.csv',index=False)
  r={'n_original_test':len(t),'w2_fdc_weight':weights[w],'donor_iqrs':iqr.tolist(),'attributes':{}}
  for k,attr in enumerate(ATTRS):
   mask=np.isfinite(ep[:,k]); s=site[t][mask]; r['attributes'][attr]={'n_common':int(mask.sum()),'persistence_raw_mae':float(ep[mask,k].mean()),'w2_raw_mae':float(ew[mask,k].mean()),'w2_minus_persistence_raw':boot(ew[mask,k]-ep[mask,k],s),'w2_minus_persistence_normalized':boot((ew[mask,k]-ep[mask,k])/iqr[k],s)}
  for label,inds in [('four_labels',[0,1,2,3]),('discharge_only_bfi_flashiness',[0,1])]:
   mask=np.isfinite(ep[:,inds]).all(1); enp=(ep[mask][:,inds]/iqr[inds]).mean(1); enw=(ew[mask][:,inds]/iqr[inds]).mean(1)
   r[label]={'n_common':int(mask.sum()),'persistence_error':float(enp.mean()),'w2_error':float(enw.mean()),'w2_minus_persistence':boot(enw-enp,site[t][mask])}
  results[w]=r
 (args.out_dir/'persistence_results.json').write_text(json.dumps(results,indent=2,allow_nan=False)); print(json.dumps(results,indent=2))
if __name__=='__main__':main()
