"""Prespecified development comparison of seasonal transport representations.

Sites are grouped across datasets before a 60/20/20 donor/calibration/test split.
Every configuration uses the same donor pool; all selection uses calibration only.
This is exploratory method development, not a prospective independent study.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/02_method'))
from features import load_cache
from signatures import day_of_year_365
from seasonal_transport import projected_quantiles
from ot import expand_to_days

OUT=ROOT/'results/method_development'
OUT.mkdir(exist_ok=True)
ATTR=['bfi','rb_flashiness','runoff_ratio','snow_fraction']
RADII=[0.,.25,1.,4.]
MIX=[0.,.25,.5,.75,1.]
SEED=2026091601

def squared_distance(x,donors):
    x=np.asarray(x,float)
    return cdist(x,x[donors],metric='sqeuclidean')/x.shape[1]

def main():
    scal,ids,fdc,profiles,_=load_cache()
    keys=scal.site_id.astype(str).to_numpy()
    rng=np.random.default_rng(SEED)
    groups=np.unique(keys);rng.shuffle(groups)
    n=len(groups);donors=np.flatnonzero(np.isin(keys,groups[:int(.6*n)]))
    calibration=np.flatnonzero(np.isin(keys,groups[int(.6*n):int(.8*n)]))
    test=np.flatnonzero(np.isin(keys,groups[int(.8*n):]))
    config={'seed':SEED,'split_group':'site_id','donor_ids':ids[donors].tolist(),
            'calibration_ids':ids[calibration].tolist(),'test_ids':ids[test].tolist(),
            'selection':'mean across four training-IQR-normalized MAEs on calibration targets',
            'radii':RADII,'mixture_weights':MIX,'projections':32,'quantiles':128,
            'daily_coordinate':'log1p(q/mean(q)), clipped at 4, 32 equal-width bins',
            'season_bins':24,'k_donors':10,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'frozen_protocol.json').write_text(json.dumps(config,indent=2))
    feature_path=OUT/'projected_features.npz'
    if feature_path.exists():
        feats=dict(np.load(feature_path));assert list(feats.pop('ids'))==list(ids)
    else:
        feats={f'{kind}_{radius:g}':[] for kind in ['levels','transitions'] for radius in RADII}
        feats['monthly_quantiles']=[]
        directions={}
        for kind,d in [('levels',3),('transitions',4)]:
            v=np.random.default_rng(2701+d).normal(size=(32,d));directions[kind]=v/np.linalg.norm(v,axis=1)[:,None]
        t0=time.time()
        for i,gid in enumerate(ids):
            row=scal.iloc[i]
            path=ROOT/'data/raw/caravan'/row.region/(str(gid)+'.csv')
            raw=pd.read_csv(path,usecols=['date','streamflow'])
            dates=pd.DatetimeIndex(pd.to_datetime(raw.date));mask=(dates.year>=1981)&(dates.year<=2010)
            q=raw.streamflow.to_numpy(float)[mask];dates=dates[mask]
            valid=np.isfinite(q)&(q>=0);mean=q[valid].mean()
            flow=np.log1p(np.maximum(q,0)/mean)
            day=day_of_year_365(dates)
            db=np.minimum(((day-1)/365*24).astype(int),23)
            fb=np.clip((flow/.125),0,31)
            fb=np.where(valid,fb,0).astype(int)
            # Exact aggregation on the declared discretization, equal mass per day.
            index=db[valid]*32+fb[valid]
            unique,w=np.unique(index,return_counts=True)
            d=unique//32;f=unique%32
            angle=2*np.pi*(d+.5)/24
            support=np.column_stack([np.cos(angle),np.sin(angle),(f+.5)*.125])
            for radius in RADII:
                p=support.copy();p[:,:2]*=radius
                feats[f'levels_{radius:g}'].append(projected_quantiles(p,w,directions['levels']).ravel())
            adjacent=valid[1:]&valid[:-1]&(np.diff(dates.values)==np.timedelta64(1,'D'))
            index=db[1:][adjacent]*1024+fb[1:][adjacent]*32+fb[:-1][adjacent]
            unique,w=np.unique(index,return_counts=True)
            d=unique//1024;f=(unique%1024)//32;lag=unique%32
            angle=2*np.pi*(d+.5)/24
            support=np.column_stack([np.cos(angle),np.sin(angle),(f+.5)*.125,(lag+.5)*.125])
            for radius in RADII:
                p=support.copy();p[:,:2]*=radius
                feats[f'transitions_{radius:g}'].append(projected_quantiles(p,w,directions['transitions']).ravel())
            monthly=np.concatenate([np.quantile(flow[valid&(dates.month==m)],(np.arange(32)+.5)/32) for m in range(1,13)])
            feats['monthly_quantiles'].append(monthly)
            if (i+1)%200==0:print('features',i+1,len(ids),'seconds',round(time.time()-t0),flush=True)
        feats={k:np.asarray(v,np.float32) for k,v in feats.items()}
        np.savez_compressed(feature_path,ids=ids,**feats)
    Y=scal[ATTR].to_numpy(float)
    scale=np.percentile(Y[donors],75,axis=0)-np.percentile(Y[donors],25,axis=0)
    assert np.all(scale>0)
    F=np.cumsum(np.vstack([expand_to_days(p) for p in profiles]),axis=1)
    timing=np.empty((len(ids),len(donors)))
    for i in range(len(ids)):
        dif=F[donors]-F[i];timing[i]=np.abs(dif-np.median(dif,axis=1,keepdims=True)).sum(axis=1)**2
    fdcD=squared_distance(fdc,donors)
    def normalize(D):
        return D/np.median(D[np.ix_(donors,np.arange(len(donors)))])
    fdcD=normalize(fdcD);timing=normalize(timing)
    distances={'FDC':fdcD,'TIMING':timing}
    for name,x in feats.items():distances[name]=normalize(squared_distance(x,donors))
    old=np.load(ROOT/'data/processed/dmatrix_beta1.npy',mmap_mode='r')
    distances['original_OT']=normalize(np.asarray(old[:,donors],float)**2)
    candidates={}
    for family,D in distances.items():
        weights=MIX if family!='FDC' else [0.]
        for weight in weights:
            key=f'{family}_fdc{weight:g}'
            combined=(1-weight)*D+weight*fdcD
            nearest=np.argsort(combined,axis=1,kind='stable')[:,:10]
            pred=np.median(Y[donors[nearest]],axis=1)
            error=np.abs(pred-Y)
            score=float(np.mean(error[calibration]/scale))
            candidates[key]={'family':family,'fdc_weight':weight,'calibration_score':score,
                             'calibration_mae':error[calibration].mean(axis=0).tolist()}
    (OUT/'calibration_all.json').write_text(json.dumps(candidates,indent=2))
    winners={family:min((k for k,v in candidates.items() if v['family']==family),key=lambda k:candidates[k]['calibration_score']) for family in distances}
    # Freeze winners before calculating or writing test performance.
    (OUT/'frozen_selections.json').write_text(json.dumps(winners,indent=2))
    result={'attributes':ATTR,'n_donors':len(donors),'n_calibration':len(calibration),'n_test':len(test),
            'training_iqr':scale.tolist(),'models':{}}
    per_target={'test_indices':test,'test_ids':ids[test]}
    for family,key in winners.items():
        rec=candidates[key];w=rec['fdc_weight'];D=(1-w)*distances[family]+w*fdcD
        nearest=np.argsort(D[test],axis=1,kind='stable')[:,:10]
        pred=np.median(Y[donors[nearest]],axis=1);error=np.abs(pred-Y[test])
        result['models'][family]={**rec,'selected_configuration':key,'test_score':float(np.mean(error/scale)),
            'test_mae':error.mean(axis=0).tolist(),'test_median_error':np.median(error,axis=0).tolist()}
        per_target[family]=error
    np.savez_compressed(OUT/'test_errors.npz',**per_target)
    (OUT/'development_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
