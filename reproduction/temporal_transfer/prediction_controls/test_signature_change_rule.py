"""Fixed early flow-mismatch predictor of cross-period signature change.

Uses the already frozen early mismatch inputs. One ridge predictor, alpha=1,
trained only on calibration sites; later test labels only score predictions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from test_early_mismatch_rule import ridge_fit
from test_signature_persistence import boot

OUT=Path(__file__).resolve().parent
PROTOCOL={
 'feature':'same locked early annual-FDC-donor monthly mismatch as early gain rule',
 'target':'mean absolute early-target versus 2001-2010 target signature difference, normalized by eligible donor later-label IQR',
 'model':'one standardized predictor ridge, intercept, alpha=1, no feature or threshold selection',
 'training':'original calibration physical sites only, or non-heldout calibration sites in dataset holdouts',
 'test':'original test sites only; labels used only for scoring',
 'uncertainty':'5000 physical-site paired bootstrap of fixed predictions',
 'baseline':'constant calibration mean signature-change magnitude',
 'meaning':'predictability of magnitude of cross-period target signature change; not a validated mechanism or a selected failure threshold',
}

def metric(frame):
 y=frame.change.to_numpy(); p=frame.prediction.to_numpy(); b=frame.baseline.to_numpy()
 total=np.sum((y-y.mean())**2)
 return {'n_records':len(y),'n_sites':frame.site_id.nunique(), 'observed_change_mean':float(y.mean()),'predicted_change_mean':float(p.mean()), 'r2':float(1-np.sum((p-y)**2)/total),'constant_calibration_mean_r2':float(1-np.sum((b-y)**2)/total),'spearman':float(spearmanr(y,p).statistic),'mae':float(abs(p-y).mean()),'baseline_mae':float(abs(b-y).mean())}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--inputs',type=Path,required=True)
 ap.add_argument('--early-labels',type=Path,required=True)
 ap.add_argument('--rule-dir',type=Path,required=True)
 ap.add_argument('--out-dir',type=Path,default=OUT)
 args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
 with np.load(args.inputs,allow_pickle=False) as z: data={k:z[k] for k in z.files}
 with np.load(args.early_labels,allow_pickle=False) as z: early=z['10yr']
 gains=json.loads((args.rule_dir/'results.json').read_text())
 results={};test_frames=[]
 files=sorted(args.rule_dir.glob('inputs_*.csv'))
 for file in files:
  label=file.stem.removeprefix('inputs_')
  key='primary' if label=='primary' else label.removeprefix('heldout_')
  iqr=np.array(gains[key]['donor_iqrs'])
  f=pd.read_csv(file,dtype={'site_id':str,'gauge_id':str})
  inds=f.input_row.to_numpy(int)
  f['change']=(abs(early[inds]-data['future_attrs'][inds])/iqr).mean(1)
  f=f[np.isfinite(f.change)].copy()
  train=f[f.role=='calibration']; test=f[f.role=='test'].copy()
  model=ridge_fit(train.mismatch.to_numpy(),train.change.to_numpy())
  test['prediction']=model['intercept']+model['standardized_slope']*(test.mismatch-model['x_mean'])/model['x_sd']
  test['baseline']=float(train.change.mean())
  r=metric(test);r['model']=model;r['n_calibration']=len(train)
  r['mae_difference_vs_constant']=boot(abs(test.prediction.to_numpy()-test.change.to_numpy())-abs(test.baseline.to_numpy()-test.change.to_numpy()),test.site_id.to_numpy())
  test.to_csv(args.out_dir/f'change_test_{label}.csv',index=False)
  results[key]=r
  if key!='primary':test_frames.append(test)
 combined=pd.concat(test_frames,ignore_index=True)
 results['domain_holdouts_pooled']=metric(combined)
 results['domain_holdouts_pooled']['mae_difference_vs_constant']=boot(abs(combined.prediction.to_numpy()-combined.change.to_numpy())-abs(combined.baseline.to_numpy()-combined.change.to_numpy()),combined.site_id.to_numpy())
 combined.to_csv(args.out_dir/'change_domain_holdouts_combined.csv',index=False)
 (args.out_dir/'change_protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
 (args.out_dir/'change_results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
 print(json.dumps(results,indent=2))

if __name__=='__main__':main()
