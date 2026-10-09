"""Calibration-locked blend of early target signatures and monthly donor labels.

Cached early labels avoid rereading raw records. Calibration chooses one common
alpha for all four labels. Tests retain that alpha and the original site split.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from test_signature_persistence import ATTRS, WINDOWS, boot

ALPHAS = [0., .05, .1, .2, .3, .5, .75, 1.]
WEIGHTS = {'3yr': .25, '5yr': 0., '10yr': 0., '5yr_shifted': 0.}
OUT = Path(__file__).resolve().parent
PROTOCOL = {
    'prediction': '(1-alpha)*early target signature + alpha*monthly donor signature prediction',
    'donor_prediction': 'current window-specific calibrated monthly-W2 candidate; weights fixed from prior experiment',
    'alpha_grid': ALPHAS,
    'selection': 'one common alpha per window, minimize calibration four-label mean donor-IQR-normalized absolute error; ties select smaller alpha',
    'scale': 'eligible donor 2001-2010 label IQR, exactly matching the published comparator',
    'test': 'original test physical sites, no alpha retuning; common complete early-label cohort',
    'intervals': '5000 fixed-prediction physical-site paired bootstrap; blend minus persistence error, negative favors blend',
    'information': 'BFI and flashiness use early discharge; runoff ratio and snow proxy also require early precipitation and temperature',
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--early-labels', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/'blend_protocol.json').write_text(json.dumps(PROTOCOL, indent=2))
    with np.load(args.inputs, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    with np.load(args.early_labels, allow_pickle=False) as z:
        early = {k: z[k] for k in z.files}
    assert np.array_equal(early['gauge_id'], data['gauge_id'])
    assert np.array_equal(early['site_id'], data['site_id'])
    results = {}
    for window in WINDOWS:
        ok = data[window+'_feature_ok'] & data['label_ok']
        for kind in ['fdc', 'monthly', 'profile']:
            ok &= np.isfinite(data[window+'_'+kind]).all(1)
        ix = np.flatnonzero(ok)
        position = {int(i): j for j, i in enumerate(ix)}
        parts = {p: np.array([position[int(i)] for i in data['split_'+p] if int(i) in position], int) for p in ['donor','calibration','test']}
        donor, calibration, test = [parts[p] for p in ['donor','calibration','test']]
        labels = data['future_attrs'][ix]
        persistence = early[window][ix]
        sites = data['site_id'][ix].astype(str)
        iqr = np.percentile(labels[donor],75,axis=0)-np.percentile(labels[donor],25,axis=0)
        iqr = np.where(iqr>0,iqr,1.)
        fdc = data[window+'_fdc'][ix]
        monthly = data[window+'_monthly'][ix]
        df = cdist(fdc,fdc,'sqeuclidean')/fdc.shape[1]
        dm = cdist(monthly,monthly,'sqeuclidean')/monthly.shape[1]
        df /= np.median(df[np.ix_(donor,donor)])
        dm /= np.median(dm[np.ix_(donor,donor)])
        distance = WEIGHTS[window]*df+(1-WEIGHTS[window])*dm
        def prediction(target):
            dt=distance[np.ix_(target,donor)].copy()
            dt[sites[target,None]==sites[donor][None,:]]=np.inf
            assert (np.isfinite(dt).sum(1)>=10).all()
            nn=np.argsort(dt,axis=1,kind='stable')[:,:10]
            return np.median(labels[donor[nn]],axis=1)
        calibration = calibration[np.isfinite(persistence[calibration]).all(1)]
        test = test[np.isfinite(persistence[test]).all(1)]
        cp, tp = prediction(calibration), prediction(test)
        scores = [float((abs((1-alpha)*persistence[calibration]+alpha*cp-labels[calibration])/iqr).mean()) for alpha in ALPHAS]
        alpha = ALPHAS[int(np.argmin(scores))]
        blend = (1-alpha)*persistence[test]+alpha*tp
        ep=abs(persistence[test]-labels[test])/iqr
        eb=abs(blend-labels[test])/iqr
        ed=abs(tp-labels[test])/iqr
        record = {'alpha_selected':alpha, 'w2_fdc_weight':WEIGHTS[window], 'n_calibration':len(calibration), 'n_test':len(test), 'calibration_grid':[{'alpha':a,'error':s} for a,s in zip(ALPHAS,scores)], 'attributes':{}}
        for k, attr in enumerate(ATTRS):
            record['attributes'][attr] = {'persistence_normalized_mae':float(ep[:,k].mean()),'blend_normalized_mae':float(eb[:,k].mean()),'donor_normalized_mae':float(ed[:,k].mean()),'blend_minus_persistence':boot(eb[:,k]-ep[:,k],sites[test])}
        for name, inds in [('four_labels',[0,1,2,3]),('discharge_only_bfi_flashiness',[0,1])]:
            pe=ep[:,inds].mean(1); be=eb[:,inds].mean(1); de=ed[:,inds].mean(1)
            record[name]={'persistence_error':float(pe.mean()),'blend_error':float(be.mean()),'donor_error':float(de.mean()),'blend_minus_persistence':boot(be-pe,sites[test])}
        rows=[]
        for j, ti in enumerate(test):
            row={'window':window,'input_row':int(ix[ti]),'gauge_id':str(data['gauge_id'][ix[ti]]),'site_id':sites[ti],'region':str(data['region'][ix[ti]]),'alpha_selected':alpha}
            for k, attr in enumerate(ATTRS):
                row.update({f'{attr}_future_label':labels[ti,k],f'{attr}_persistence_prediction':persistence[ti,k],f'{attr}_donor_prediction':tp[j,k],f'{attr}_blend_prediction':blend[j,k],f'{attr}_persistence_normalized_error':ep[j,k],f'{attr}_blend_normalized_error':eb[j,k]})
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.out_dir/f'blend_per_target_{window}.csv',index=False)
        results[window]=record
    (args.out_dir/'blend_results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
