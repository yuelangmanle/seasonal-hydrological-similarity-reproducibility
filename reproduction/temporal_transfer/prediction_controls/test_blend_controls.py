"""Calibration-locked persistence blends for annual FDC, allocation and W2.

The donor-distance mixture choices are fixed from the original experiment.
Each family chooses a persistence-blend alpha only on calibration sites.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from test_signature_persistence import ATTRS, WINDOWS
from test_persistence_donor_blend import ALPHAS, WEIGHTS

OUT=Path(__file__).resolve().parent
ALLOCATION_WEIGHTS={'3yr':.5,'5yr':.25,'10yr':.25,'5yr_shifted':.5}

def boot(values, sites, reps=5000, seed=2026100907):
    unique, inverse=np.unique(sites,return_inverse=True)
    sums=np.bincount(inverse,weights=values,minlength=len(unique))
    counts=np.bincount(inverse,minlength=len(unique))
    sample=np.random.default_rng(seed).integers(0,len(unique),size=(reps,len(unique)))
    means=sums[sample].sum(1)/counts[sample].sum(1)
    return {'estimate':float(np.mean(values)),'ci95':np.percentile(means,[2.5,97.5]).tolist(),'n_sites':len(unique),'n_records':len(values)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--early-labels',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,default=OUT)
    args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    with np.load(args.inputs,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(args.early_labels,allow_pickle=False) as z:early={k:z[k] for k in z.files}
    assert np.array_equal(early['gauge_id'],data['gauge_id'])
    assert np.array_equal(early['site_id'],data['site_id'])
    protocol={'alpha_grid':ALPHAS,'selection':'independently minimize four-label calibration mean absolute error for each family; smaller alpha wins ties','donor_weights_w2':WEIGHTS,'donor_weights_allocation':ALLOCATION_WEIGHTS,'annual_fdc':'unmixed annual log-FDC distance','allocation':'same circular W1 allocation distance squared as original experiment','scaling':'eligible later donor-label IQR; early donor-donor median distances','test':'same finite early-target-label test cohort; no retuning','inference':'5000 physical-site fixed-prediction paired bootstrap; method minus persistence and method minus W2-blend error; negative favors method','role':'control whether small persistence-blend gain is specific to full monthly distributions or generic donor shrinkage'}
    (args.out_dir/'blend_controls_protocol.json').write_text(json.dumps(protocol,indent=2))
    results={}
    for window in WINDOWS:
        ok=data[window+'_feature_ok']&data['label_ok']
        for kind in ['fdc','monthly','profile']:ok&=np.isfinite(data[window+'_'+kind]).all(1)
        ix=np.flatnonzero(ok);position={int(i):j for j,i in enumerate(ix)}
        parts={p:np.array([position[int(i)] for i in data['split_'+p] if int(i) in position],int) for p in ['donor','calibration','test']}
        donor=parts['donor']; cal=parts['calibration'];test=parts['test']
        target=early[window][ix];y=data['future_attrs'][ix];sites=data['site_id'][ix].astype(str)
        cal=cal[np.isfinite(target[cal]).all(1)];test=test[np.isfinite(target[test]).all(1)]
        iqr=np.percentile(y[donor],75,axis=0)-np.percentile(y[donor],25,axis=0);iqr=np.where(iqr>0,iqr,1.)
        fdc=data[window+'_fdc'][ix]; monthly=data[window+'_monthly'][ix];profile=data[window+'_profile'][ix]
        df=cdist(fdc,fdc,'sqeuclidean')/fdc.shape[1];dm=cdist(monthly,monthly,'sqeuclidean')/monthly.shape[1]
        cumulative=np.cumsum(profile,axis=1);da=np.empty((len(ix),len(ix)),float)
        for j in range(len(ix)):
            delta=cumulative-cumulative[j]
            da[j]=np.abs(delta-np.median(delta,axis=1,keepdims=True)).sum(1)**2
        for distance in [df,dm,da]:distance/=np.median(distance[np.ix_(donor,donor)])
        distances={'annual_fdc':df,'allocation':ALLOCATION_WEIGHTS[window]*df+(1-ALLOCATION_WEIGHTS[window])*da,'monthly_w2':WEIGHTS[window]*df+(1-WEIGHTS[window])*dm}
        e_p=abs(target[test]-y[test])/iqr
        window_result={'n_calibration':len(cal),'n_test':len(test),'methods':{}};errors={};predictions={}
        for name,distance in distances.items():
            def donor_prediction(indices):
                dt=distance[np.ix_(indices,donor)].copy();dt[sites[indices,None]==sites[donor][None,:]]=np.inf
                assert (np.isfinite(dt).sum(1)>=10).all()
                nn=np.argsort(dt,axis=1,kind='stable')[:,:10]
                return np.median(y[donor[nn]],axis=1)
            pc=donor_prediction(cal);pt=donor_prediction(test)
            scores=[float((abs((1-alpha)*target[cal]+alpha*pc-y[cal])/iqr).mean()) for alpha in ALPHAS]
            alpha=ALPHAS[int(np.argmin(scores))]
            pred=(1-alpha)*target[test]+alpha*pt;error=abs(pred-y[test])/iqr
            errors[name]=error;predictions[name]=pred
            window_result['methods'][name]={'alpha_selected':alpha,'calibration_grid':[{'alpha':a,'error':s} for a,s in zip(ALPHAS,scores)],'four_label_error':float(error.mean()),'discharge_only_error':float(error[:,:2].mean()),'donor_only_error':float((abs(pt-y[test])/iqr).mean()),'attributes':{}}
        for name,error in errors.items():
            record=window_result['methods'][name]
            for key,indices in [('four_labels',[0,1,2,3]),('discharge_only_bfi_flashiness',[0,1])]:
                record[key]={'versus_persistence':boot((error[:,indices]-e_p[:,indices]).mean(1),sites[test]),'versus_w2_blend':boot((error[:,indices]-errors['monthly_w2'][:,indices]).mean(1),sites[test])}
            for k,attr in enumerate(ATTRS):
                record['attributes'][attr]={'normalized_mae':float(error[:,k].mean()),'versus_persistence':boot(error[:,k]-e_p[:,k],sites[test]),'versus_w2_blend':boot(error[:,k]-errors['monthly_w2'][:,k],sites[test])}
        rows=[]
        for j,ti in enumerate(test):
            row={'window':window,'input_row':int(ix[ti]),'gauge_id':str(data['gauge_id'][ix[ti]]),'site_id':sites[ti],'region':str(data['region'][ix[ti]])}
            for k,attr in enumerate(ATTRS):
                row[f'{attr}_future_label']=y[ti,k];row[f'{attr}_persistence_prediction']=target[ti,k]
                for name,pred in predictions.items():row[f'{name}_{attr}_blend_prediction']=pred[j,k]
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.out_dir/f'blend_controls_per_target_{window}.csv',index=False)
        results[window]=window_result
    (args.out_dir/'blend_controls_results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
