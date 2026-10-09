"""Summarize calendar-month sample support from available raw local Caravan files."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser()
ap.add_argument('--inputs',type=Path,required=True)
ap.add_argument('--raw',type=Path,required=True)
ap.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent)
args=ap.parse_args()
OUT=args.out_dir
OUT.mkdir(parents=True,exist_ok=True)
with np.load(args.inputs,allow_pickle=False) as loaded:
 z={k:loaded[k] for k in loaded.files}
WINDOWS={'3yr':(1981,1983),'5yr':(1981,1985),'10yr':(1981,1990),'5yr_shifted':(1986,1990)}
def one(i):
 p=args.raw/str(z['region'][i])/(str(z['gauge_id'][i])+'.csv')
 if not p.exists(): return i,None
 frame=pd.read_csv(p,usecols=['date','streamflow']); dt=pd.DatetimeIndex(pd.to_datetime(frame.date)); q=pd.to_numeric(frame.streamflow,errors='coerce').to_numpy(float); valid=np.isfinite(q)&(q>=0)
 return i,{w:np.bincount(dt.month[valid&(dt.year>=a)&(dt.year<=b)],minlength=13)[1:] for w,(a,b) in WINDOWS.items()}
counts={w:np.zeros((len(z['gauge_id']),12),int) for w in WINDOWS}; missing=[]
with ThreadPoolExecutor(max_workers=8) as pool:
 for i,r in pool.map(one,range(len(z['gauge_id']))):
  if r is None: missing.append(i)
  else:
   for w in WINDOWS: counts[w][i]=r[w]
summary={}; rows=[]; mrows=[]
for w,(a,b) in WINDOWS.items():
 finite=np.isfinite(z[w+'_fdc']).all(1)&np.isfinite(z[w+'_monthly']).all(1)&np.isfinite(z[w+'_profile']).all(1)
 early=z[w+'_feature_ok']&finite; final=early&z['label_ok']; expected=np.bincount(pd.date_range(f'{a}-01-01',f'{b}-12-31').month,minlength=13)[1:]
 for i in np.flatnonzero(early):
  rows.append({'window':w,'gauge_id':z['gauge_id'][i],'site_id':z['site_id'][i],'region':z['region'][i],'final_evaluable':bool(final[i]),'minimum_month_n':int(counts[w][i].min()),'minimum_month_coverage':float((counts[w][i]/expected).min()), **{f'n_month_{m+1}':int(counts[w][i,m]) for m in range(12)}})
 summary[w]={}
 for cohort,mask in [('early_features',early),('final_evaluable',final)]:
  x=counts[w][mask]; mins=x.min(1); covers=(x/expected).min(1)
  summary[w][cohort]={'n_records':int(mask.sum()),'minimum_month_n_quantiles':{str(p):float(np.percentile(mins,p)) for p in [0,1,5,25,50,75,95,99,100]},'minimum_month_coverage_quantiles':{str(p):float(np.percentile(covers,p)) for p in [0,1,5,25,50,75,95,99,100]},'n_records_any_month_lt_2':int((mins<2).sum()),'n_records_any_month_lt_16':int((mins<16).sum()),'n_records_any_month_lt_32':int((mins<32).sum()),'n_records_any_month_lt_50pct':int((covers<.5).sum())}
  for m in range(12):
   mrows.append({'window':w,'cohort':cohort,'month':m+1,'n_records':int(mask.sum()),'expected_days':int(expected[m]), **{f'n_q{p}':float(np.percentile(x[:,m],p)) for p in [0,5,25,50,75,95,100]},'n_month_lt_32':int((x[:,m]<32).sum())})
pd.DataFrame(rows).to_csv(OUT/'monthly_support_per_record.csv',index=False)
pd.DataFrame(mrows).to_csv(OUT/'monthly_support_summary.csv',index=False)
summary['source']={'raw_directory':'user-supplied Caravan v1.6 CSV directory','n_records_inputs':len(z['gauge_id']),'missing_files':missing,'definition':'finite nonnegative daily discharge by observed calendar month; no imputation; counts cover every early eligible row'}
(OUT/'monthly_support.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
