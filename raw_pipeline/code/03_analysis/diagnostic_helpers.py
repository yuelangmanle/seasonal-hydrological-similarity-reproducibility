"""Pair sampling and distances used by the seasonal diagnostic."""

import numpy as np

def sample_pairs(n: int, k: int, seed: int=0) -> tuple[np.ndarray, np.ndarray]:
    """Sample unique unordered pairs without replacement."""
    rng = np.random.default_rng(seed)
    total = n * (n - 1) // 2
    k = min(k, total)
    picks = np.sort(rng.choice(total, size=k, replace=False))
    starts = np.arange(n - 1, dtype=np.int64)
    rs = starts * n - starts * (starts + 1) // 2
    r = np.searchsorted(rs, picks, side='right') - 1
    return (r.astype(np.int64), (picks - rs[r] + r + 1).astype(np.int64))

def sample_pairs_within(n: int, regions: np.ndarray, k: int, seed: int=0):
    """Sample unique within-dataset pairs without replacement."""
    rng = np.random.default_rng(seed)
    regions = np.asarray(regions)
    labs, members_list, starts = ([], [], [])
    total = 0
    for lab in np.unique(regions):
        mem = np.flatnonzero(regions == lab)
        m = mem.size
        if m < 2:
            continue
        tot = m * (m - 1) // 2
        labs.append(lab)
        members_list.append(mem)
        starts.append(total)
        total += tot
    total = int(total)
    k = min(k, total)
    picks = np.sort(rng.choice(total, size=k, replace=False))
    starts_arr = np.asarray(starts, dtype=np.int64)
    r = np.searchsorted(starts_arr, picks, side='right') - 1
    ia = np.empty(k, dtype=np.int64)
    ib = np.empty(k, dtype=np.int64)
    for ri in range(len(labs)):
        mask = r == ri
        if not mask.any():
            continue
        mem = members_list[ri]
        m = mem.size
        lp = picks[mask] - starts[ri]
        s = np.arange(m - 1, dtype=np.int64)
        rs = s * m - s * (s + 1) // 2
        rr = np.searchsorted(rs, lp, side='right') - 1
        ia[mask] = mem[rr]
        ib[mask] = mem[lp - rs[rr] + rr + 1]
    order = np.lexsort((ib, ia))
    return (ia[order], ib[order])

def timing_from_cum(F: np.ndarray, ia: np.ndarray, ib: np.ndarray, chunk: int=20000) -> np.ndarray:
    """Compute circular allocation distances."""
    out = np.empty(ia.size)
    for s in range(0, ia.size, chunk):
        e = min(s + chunk, ia.size)
        D = F[ia[s:e]] - F[ib[s:e]]
        out[s:e] = np.abs(D - np.median(D, axis=1, keepdims=True)).mean(axis=1) * 365.0
    return out

def curve_rms(C: np.ndarray, ia: np.ndarray, ib: np.ndarray, chunk: int=40000) -> np.ndarray:
    """Compute unweighted quantile RMS distances."""
    out = np.empty(ia.size)
    for s in range(0, ia.size, chunk):
        e = min(s + chunk, ia.size)
        out[s:e] = np.sqrt(np.mean((C[ia[s:e]] - C[ib[s:e]]) ** 2, axis=1))
    return out
