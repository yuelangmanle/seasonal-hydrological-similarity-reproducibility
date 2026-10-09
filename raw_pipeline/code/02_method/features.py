"""Load freshly constructed, retained raw-data features for analysis scripts."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]

def load_cache(subsample=None, seed=0, stratify_by=None, verbose=True):
    if subsample or stratify_by:
        raise ValueError('The raw rebuild uses the full retained cohort.')
    s = pd.read_csv(ROOT/'data/processed/raw_scalars.csv', dtype={'site_id': str})
    z = np.load(ROOT/'data/processed/raw_features.npz', allow_pickle=False)
    np.testing.assert_array_equal(s.gauge_id.to_numpy(), z['ids'])
    if verbose:
        print('Retained raw-data records:', len(s), flush=True)
    return s, z['ids'], z['fdc'], z['profiles'], z['histograms']
