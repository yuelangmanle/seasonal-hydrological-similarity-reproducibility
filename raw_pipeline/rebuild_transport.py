"""Recompute the archived 400-iteration joint-transport comparison from raw histograms."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
from pathlib import Path
import argparse,json,sys,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code/02_method'))
from ot import RegimeMetric
H=None
MET=None
SELF=None

def initialise(path):
    global H,MET,SELF
    H=np.asarray(np.load(path)['histograms'],float)+1e-12
    H/=H.sum(axis=(1,2),keepdims=True)
    MET=RegimeMetric.for_regime_grid(H.shape[1],H.shape[2],alpha=1.,beta=1.,reg=.05,debiased=True)
    SELF=MET.self_terms_all(H,n_iter=400)

def row(i):
    values=np.zeros(len(H),np.float32)
    for lo in range(i+1,len(H),500):
        hi=min(lo+500,len(H))
        cost=MET.ot_many(H[i],H[lo:hi],n_iter=400)-.5*SELF[i]-.5*SELF[lo:hi]
        values[lo:hi]=np.sqrt(np.maximum(cost,0)).astype(np.float32)
    return i,values

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs',type=int,default=4)
    ap.add_argument('--verify-pairs',type=int,default=0,help='Only compare sampled rebuilt distances to the archive; no matrix is produced.')
    args=ap.parse_args();src=ROOT/'data/processed/raw_features.npz';start=time.time()
    initialise(src)
    archive=np.load(ROOT.parent/'reproduction/robustness_checks/extension_inputs.npz')
    np.testing.assert_array_equal(np.load(src)['ids'],archive['ids'])
    if args.verify_pairs:
        rng=np.random.default_rng(20260917);pairs=rng.integers(len(H),size=(args.verify_pairs,2))
        diffs=[abs(MET.distance(H[i],H[j],n_iter=400)-float(archive['original_ot'][i,j])) for i,j in pairs]
        assert max(diffs)<1e-4
        print(json.dumps({'pairs':len(pairs),'max_abs_difference':max(diffs),'full_matrix_rebuilt':False},indent=2));return
    dest=ROOT/'data/processed/dmatrix_beta1.npy'
    matrix=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(len(H),len(H)))
    with ProcessPoolExecutor(max_workers=args.jobs,initializer=initialise,initargs=(src,)) as ex:
        for i,values in ex.map(row,range(len(H)),chunksize=1):
            matrix[i,i+1:]=values[i+1:];matrix[i+1:,i]=values[i+1:];matrix[i,i]=0
            if i%50==0:print('Rows:',i,'elapsed seconds:',round(time.time()-start),flush=True)
    matrix.flush();difference=float(np.max(abs(matrix-archive['original_ot'])))
    if difference>=1e-4:raise AssertionError(f'Transport mismatch: {difference}')
    (dest.parent/'transport_verification.json').write_text(json.dumps({'max_abs_difference':difference,'n_iter':400,'epsilon':.05,'seconds':time.time()-start},indent=2))
if __name__=='__main__':main()
