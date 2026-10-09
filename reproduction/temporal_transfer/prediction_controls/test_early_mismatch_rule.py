"""Locked prospective-feature test of an early-FDC donor mismatch rule.

One predictor and a fixed ridge alpha=1; no test-set tuning. Inputs are frozen
1981-1990 discharge features and 2001-2010 labels. Calibration sites fit the
rule; test sites only score it. Pure W2/FDC are fixed before any rule evaluation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

OUT=Path(__file__).resolve().parent
PROTOCOL={
 'analysis':'early annual-FDC donor monthly mismatch predicts future signature-transfer gain',
 'locked_specification':'one-variable standardized ridge with intercept, alpha=1; no variable/threshold/hyperparameter selection',
 'feature_window':[1981,1990],'label_window':[2001,2010],
 'feature':'mean pure monthly conditional squared distance from target to ten early annual-log-FDC nearest donor records, divided by eligible donor-donor median monthly distance',
 'outcome':'mean four donor-IQR-normalized absolute FDC errors minus pure monthly-W2 errors; positive favors monthly W2',
 'base_methods':'pure annual log-FDC and pure monthly conditional W2; 10 donor-record median; no FDC mixture tuning',
 'training':'eligible original calibration physical sites only; standardization on training rows only',
 'evaluation':'original test physical sites; never used for feature/model/threshold selection',
 'policy':'select W2 if predicted gain > 0, otherwise select FDC; zero fixed before scoring',
 'domain_extrapolation':'for each held-out source dataset, exclude all its physical sites from donors/calibration; fit on remaining calibration records and test only original test records of held-out dataset; no retuning',
 'uncertainty':'5000 physical-site cluster bootstrap replicates on frozen predictions; record-weighted estimand',
 'interpretation':'revision-stage hypothesis test on existing public data, not independent prospective cohort or preregistered confirmation',
}

def predictions(d,donor,target,y,sites):
 dt=d[np.ix_(target,donor)].copy(); dt[sites[target,None]==sites[donor][None,:]]=np.inf
 if not (np.isfinite(dt).sum(1)>=10).all(): raise ValueError('insufficient donors')
 nn=np.argsort(dt,axis=1,kind='stable')[:,:10]
 return np.median(y[donor[nn]],axis=1),donor[nn]

def ridge_fit(x,y):
 mu=float(x.mean()); sd=float(x.std()); sd=sd if sd>0 else 1.
 z=(x-mu)/sd; intercept=float(y.mean()); beta=float(z@(y-intercept)/(z@z+1.))
 return {'x_mean':mu,'x_sd':sd,'intercept':intercept,'standardized_slope':beta,'alpha':1.}

def metric(y,p,ef,ew):
 selected=np.where(p>0,ew,ef); variance=float(np.sum((y-y.mean())**2))
 return {'n_records':len(y),'gain_mean':float(y.mean()),'predicted_gain_mean':float(p.mean()),
 'r2':float(1-np.sum((p-y)**2)/variance) if variance>0 else None,
 'spearman_predicted_vs_observed':float(spearmanr(p,y).statistic),
 'mae_gain_prediction':float(np.mean(abs(p-y))),
 'fdc_error':float(ef.mean()),'w2_error':float(ew.mean()),'selected_error':float(selected.mean()),
 'fraction_select_w2':float(np.mean(p>0)),
 'policy_gain_vs_all_fdc':float(np.mean(ef-selected)),
 'policy_gain_vs_all_w2':float(np.mean(ew-selected)),
 'sign_accuracy':float(np.mean((p>0)==(y>0))),
 'all_w2_sign_accuracy':float(np.mean(y>0)),
 'oracle_error':float(np.minimum(ef,ew).mean()),
 'policy_regret_vs_oracle':float(np.mean(selected-np.minimum(ef,ew)))}

def bootstrap(df,seed=2026100903,reps=5000):
 groups=[np.flatnonzero(df.site_id.to_numpy()==s) for s in sorted(df.site_id.unique())]
 rng=np.random.default_rng(seed); records=[]
 for b in range(reps):
  ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]); r=df.iloc[ix]
  # R2 and utility are fixed-prediction out-of-sample metrics.
  records.append(metric(r.gain.to_numpy(),r.predicted_gain.to_numpy(),r.fdc_error.to_numpy(),r.w2_error.to_numpy()))
 out={}
 for k in ['gain_mean','r2','spearman_predicted_vs_observed','mae_gain_prediction','policy_gain_vs_all_fdc','policy_gain_vs_all_w2','policy_regret_vs_oracle']:
  vals=np.asarray([r[k] for r in records],float)
  out[k]=[float(v) for v in np.nanpercentile(vals,[2.5,97.5])]
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--inputs',type=Path,required=True); ap.add_argument('--out-dir',type=Path,default=OUT); args=ap.parse_args()
 args.out_dir.mkdir(parents=True,exist_ok=True)
 (args.out_dir/'locked_protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
 z=np.load(args.inputs,allow_pickle=False); data={k:z[k] for k in z.files}; name='10yr'
 ok=data[name+'_feature_ok']&data['label_ok']; ok&=np.isfinite(data[name+'_fdc']).all(1)&np.isfinite(data[name+'_monthly']).all(1)&np.isfinite(data[name+'_profile']).all(1)
 ix=np.flatnonzero(ok); pos={int(old):j for j,old in enumerate(ix)}
 parts={p:np.array([pos[int(i)] for i in data['split_'+p] if int(i) in pos],int) for p in ['donor','calibration','test']}
 y=data['future_attrs'][ix]; sites=data['site_id'][ix].astype(str); regions=data['region'][ix].astype(str); gauges=data['gauge_id'][ix].astype(str)
 assert not (set(sites[parts['test']])&set(sites[parts['calibration']]) or set(sites[parts['test']])&set(sites[parts['donor']]) or set(sites[parts['donor']])&set(sites[parts['calibration']]))
 f=data[name+'_fdc'][ix]; m=data[name+'_monthly'][ix]; df=cdist(f,f,'sqeuclidean')/f.shape[1]; dm=cdist(m,m,'sqeuclidean')/m.shape[1]
 def fold(d,c,t,label):
  iqr=np.percentile(y[d],75,axis=0)-np.percentile(y[d],25,axis=0); iqr=np.where(iqr>0,iqr,1.)
  scale=float(np.median(dm[np.ix_(d,d)])); scale=scale if scale>0 else 1.
  rows=[]
  for tar,role in [(c,'calibration'),(t,'test')]:
   pf,nn=predictions(df,d,tar,y,sites); pm,_=predictions(dm,d,tar,y,sites)
   ef=abs(pf-y[tar])/iqr; ew=abs(pm-y[tar])/iqr
   x=np.mean(dm[tar[:,None],nn],axis=1)/scale
   for j,ti in enumerate(tar):
    rows.append({'fold':label,'role':role,'input_row':int(ix[ti]),'gauge_id':gauges[ti],'site_id':sites[ti],'region':regions[ti], 'mismatch':float(x[j]),'gain':float(np.mean(ef[j]-ew[j])),'fdc_error':float(ef[j].mean()),'w2_error':float(ew[j].mean())})
  frame=pd.DataFrame(rows); train=frame[frame.role=='calibration']; test=frame[frame.role=='test'].copy()
  model=ridge_fit(train.mismatch.to_numpy(),train.gain.to_numpy()); test['predicted_gain']=model['intercept']+model['standardized_slope']*(test.mismatch-model['x_mean'])/model['x_sd']
  trainbase=float(train.gain.mean()); result=metric(test.gain.to_numpy(),test.predicted_gain.to_numpy(),test.fdc_error.to_numpy(),test.w2_error.to_numpy())
  result.update({'model':model,'n_train_records':len(train),'n_train_sites':train.site_id.nunique(),'n_test_sites':test.site_id.nunique(), 'n_donors':len(d),'training_mean_baseline_mae':float(np.mean(abs(test.gain-trainbase))), 'training_mean_baseline_r2':float(1-np.sum((test.gain-trainbase)**2)/np.sum((test.gain-test.gain.mean())**2)), 'bootstrap_95_ci':bootstrap(test), 'donor_iqrs':iqr.tolist(),'monthly_donor_scale':scale})
  test['selected_method']=np.where(test.predicted_gain>0,'W2','FDC'); frame.to_csv(args.out_dir/f'inputs_{label}.csv',index=False); test.to_csv(args.out_dir/f'test_{label}.csv',index=False)
  return result,test
 result={}; r,t=fold(parts['donor'],parts['calibration'],parts['test'],'primary'); result['primary']=r
 domain_test=[]
 for reg in sorted(set(regions[parts['test']])):
  heldsites=set(sites[regions==reg]); eligible=np.array([s not in heldsites for s in sites]); d=parts['donor'][eligible[parts['donor']]]; c=parts['calibration'][eligible[parts['calibration']]]; t=parts['test'][regions[parts['test']]==reg]
  r,tt=fold(d,c,t,'heldout_'+reg); result[reg]=r; domain_test.append(tt)
 combined=pd.concat(domain_test,ignore_index=True); combined.to_csv(args.out_dir/'test_domain_holdouts_combined.csv',index=False)
 result['domain_holdouts_pooled']=metric(combined.gain.to_numpy(),combined.predicted_gain.to_numpy(),combined.fdc_error.to_numpy(),combined.w2_error.to_numpy()); result['domain_holdouts_pooled']['bootstrap_95_ci']=bootstrap(combined)
 (args.out_dir/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
 print(json.dumps(result,indent=2))
if __name__=='__main__': main()
