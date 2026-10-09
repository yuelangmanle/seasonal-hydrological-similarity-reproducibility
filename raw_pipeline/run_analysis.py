"""Run the current method comparison and follow-up analyses in dependency order."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reuse-archived-transport',action='store_true',help='Explicit shortcut; otherwise require a newly rebuilt full matrix.')
    args=ap.parse_args()
    z=np.load(ROOT/'data/processed/raw_features.npz')
    if args.reuse_archived_transport:
        old=np.load(ROOT.parent/'reproduction/robustness_checks/extension_inputs.npz')
        np.testing.assert_array_equal(z['ids'],old['ids'])
        np.save(ROOT/'data/processed/dmatrix_beta1.npy',old['original_ot'])
    elif not (ROOT/'data/processed/transport_verification.json').exists():
        raise RuntimeError('Run rebuild_transport.py first, or explicitly choose the archived-matrix shortcut.')
    for name in ['method_development','robustness_checks','benchmark_controls']:
        (ROOT/'results'/name).mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'protocols/robustness_protocol.json',ROOT/'results/robustness_checks/protocol.json')
    scripts=['develop_transport_method.py','validate_method_development.py','check_conditional_resolution.py','check_transfer_controls.py','robustness_checks.py','benchmark_controls.py','seasonal_diagnostic.py']
    for name in scripts:
        print('Running:',name,flush=True)
        subprocess.run([sys.executable,str(ROOT/'code/03_analysis'/name)],check=True,cwd=ROOT)
    frozen=json.loads((ROOT/'protocols/frozen_protocol.json').read_text())
    result=json.loads((ROOT/'results/method_development/frozen_protocol.json').read_text())
    for key in ['donor_ids','calibration_ids','test_ids']:
        assert frozen[key]==result[key],key
    expected=json.loads((ROOT.parent/'reproduction/development_summary.json').read_text())
    actual=json.loads((ROOT/'results/method_development/development_summary.json').read_text())
    for key in expected['models']:
        np.testing.assert_allclose(actual['models'][key]['test_score'],expected['models'][key]['test_score'],atol=1e-9,rtol=0)
    (ROOT/'results/rebuild_status.json').write_text(json.dumps({'raw_features_rebuilt':True,'transport_reused':args.reuse_archived_transport,'primary_scores_match':True},indent=2))
    print('PASS: raw-derived method scores and fixed partitions match the archive.')
if __name__=='__main__':main()
