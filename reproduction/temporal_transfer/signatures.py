"""Hydrological signatures for distributional and seasonal-flow comparisons.

The module computes scalar descriptors, normalized flow-duration curves, monthly
allocation profiles, and a two-dimensional day-of-year by log-flow measure.
All definitions are dimensionless or normalized before cross-catchment comparison.
The calendar coordinate uses a 365-day circular representation, mapping 29 February
to 28 February so that the year boundary remains continuous.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Quantile grid for the flow-duration curve; endpoints are excluded to avoid infinities.
TAU = np.linspace(0.0, 1.0, 201)[1:-1]

#: Lyne–Hollick filter parameter (Nathan & McMahon, 1990).
LH_ALPHA = 0.925

#: Two-dimensional regime grid: day of year by log flow.
REGIME_DOY_BINS = 36
REGIME_FLOW_BINS = 24
#: Clipping limit for standardized log flow.
REGIME_LO_Z_CLIP = 3.0

SCALAR_NAMES = [
    "cv",
    "fdc_slope",
    "q95_q50",
    "q90_q50",
    "q50_q10",
    "q05_q50",
    "bfi",
    "rb_flashiness",
    "peak_month_sin",
    "peak_month_cos",
    "circ_conc_R",
    "centroid_doy_sin",
    "centroid_doy_cos",
    "half_flow_doy_sin",
    "half_flow_doy_cos",
    "parde_amp",
    "runoff_ratio",
    "aridity_index",
    "snow_fraction",
]

# Diagnostic predictors for pairwise seasonal timing distance.
# The donor-transfer shape control uses the first six entries,
# and excludes the four attributes used as outcomes.
MAGNITUDE_NAMES = [
    "cv",
    "fdc_slope",
    "q95_q50",
    "q90_q50",
    "q50_q10",
    "q05_q50",
    "bfi",
    "rb_flashiness",
    "runoff_ratio",
    "snow_fraction",
]

#: Signatures computed from streamflow alone. Runoff ratio, aridity index, and
#: snow fraction additionally require meteorological inputs.
CORE_SCALAR_NAMES = [
    "cv",
    "fdc_slope",
    "q95_q50",
    "q90_q50",
    "q50_q10",
    "q05_q50",
    "bfi",
    "rb_flashiness",
    "peak_month_sin",
    "peak_month_cos",
    "circ_conc_R",
    "centroid_doy_sin",
    "centroid_doy_cos",
    "half_flow_doy_sin",
    "half_flow_doy_cos",
    "parde_amp",
]

#: Signatures that require meteorological variables.
CLIMATE_SCALAR_NAMES = ["runoff_ratio", "aridity_index", "snow_fraction"]


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------
def _valid(s: pd.Series) -> np.ndarray:
    v = s.to_numpy(dtype=float)
    return v[~np.isnan(v)]


def day_of_year_365(index: pd.DatetimeIndex) -> np.ndarray:
    """Map dates to a circular 1--365 day-of-year coordinate.

    ``dayofyear`` and ``is_leap_year`` may already be NumPy arrays,
    so both are normalized with ``np.asarray``.
    """
    doy = np.asarray(index.dayofyear).astype(int)
    leap = np.asarray(index.is_leap_year).astype(bool)
    out = doy.copy()
    out[leap & (doy > 59)] -= 1     # shift dates after 29 February
    out[leap & (doy == 60)] = 59    # map 29 February to 28 February
    return out


# ---------------------------------------------------------------------------
# A. Scalar signatures
# ---------------------------------------------------------------------------
def flow_duration_curve(s: pd.Series, tau: np.ndarray = TAU) -> np.ndarray:
    """Return the normalized flow-duration curve ``Q(tau) / mean(Q)``."""
    v = _valid(s)
    if v.size == 0:
        return np.full(tau.size, np.nan)
    m = v.mean()
    if not np.isfinite(m) or m <= 0:
        return np.full(tau.size, np.nan)
    return np.quantile(v / m, tau)


def baseflow_index(s: pd.Series, alpha: float = LH_ALPHA) -> float:
    """Return the baseflow index from a Lyne--Hollick digital filter.

    The index is baseflow volume divided by total runoff volume.
    """
    v = s.to_numpy(dtype=float)
    valid = np.isfinite(v) & (v >= 0)
    if valid.sum() < 100 or v[valid].sum() <= 0:
        return np.nan
    quick = np.zeros_like(v)
    previous_flow = None
    previous_quick = 0.0
    for i, q in enumerate(v):
        if not valid[i]:
            previous_flow = None
            previous_quick = 0.0
            continue
        if previous_flow is not None:
            quick[i] = np.clip(alpha * previous_quick +
                               (1 + alpha) / 2 * (q - previous_flow), 0, q)
        previous_flow, previous_quick = q, quick[i]
    return float(np.sum(v[valid] - quick[valid]) / np.sum(v[valid]))


def rb_flashiness(s: pd.Series) -> float:
    """Return the Richards--Baker flashiness index."""
    v = s.to_numpy(dtype=float)
    valid = np.isfinite(v) & (v >= 0)
    if valid.sum() < 2 or v[valid].sum() <= 0:
        return np.nan
    adjacent = valid[1:] & valid[:-1]
    if isinstance(s.index, pd.DatetimeIndex):
        adjacent &= np.diff(s.index.values) == np.timedelta64(1, 'D')
    return float(np.abs(np.diff(v))[adjacent].sum() / v[valid].sum())


def circular_moments(profile: np.ndarray) -> tuple[float, float, float]:
    """Compute circular statistics from the twelve monthly components.

    Returns ``(R, centroid_doy, peak_month)``:
      R              -- circular concentration (0 uniform, 1 concentrated)
      centroid_doy   -- day of year of the flow centroid
      peak_month     -- month with the largest component (1--12)
    """
    p = np.asarray(profile, dtype=float)
    if p.size != 12 or not np.isfinite(p).all() or p.sum() <= 0:
        return np.nan, np.nan, np.nan
    p = p / p.sum()
    mid_doy = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)
    theta = 2.0 * np.pi * mid_doy / 365.0
    c = float(np.sum(p * np.cos(theta)))
    s_ = float(np.sum(p * np.sin(theta)))
    r = float(np.hypot(c, s_))
    phi = float(np.arctan2(s_, c))
    if phi < 0:
        phi += 2.0 * np.pi
    centroid_doy = phi * 365.0 / (2.0 * np.pi)
    return r, centroid_doy, float(np.argmax(p) + 1)


def parde_amplitude(profile: np.ndarray) -> float:
    """Return the Pardé amplitude (maximum monthly share / minimum share)."""
    p = np.asarray(profile, dtype=float)
    if p.size != 12 or p.min() <= 0:
        return np.nan
    return float(p.max() / p.min())


def half_flow_doy(s: pd.Series) -> float:
    """Return the day on which cumulative annual flow first reaches 50%."""
    idx = s.index
    v = s.to_numpy(dtype=float)
    mask = ~np.isnan(v)
    if mask.sum() < 300:
        return np.nan
    doy = day_of_year_365(idx)[mask]
    vals = v[mask]
    order = np.argsort(doy, kind="stable")
    doy, vals = doy[order], vals[order]
    # Aggregate by day of year; multi-year records contribute repeated dates.
    uniq, inv = np.unique(doy, return_inverse=True)
    daily = np.zeros(uniq.size)
    np.add.at(daily, inv, vals)
    cum = np.cumsum(daily)
    if cum[-1] <= 0:
        return np.nan
    k = int(np.searchsorted(cum, 0.5 * cum[-1]))
    return float(uniq[min(k, uniq.size - 1)])


def snow_fraction(precip: pd.Series, temp: pd.Series, thresh: float = 0.0) -> float:
    """Return precipitation falling on days below the snow-temperature threshold."""
    p = precip.to_numpy(dtype=float)
    t = temp.to_numpy(dtype=float)
    m = ~np.isnan(p) & ~np.isnan(t)
    if m.sum() < 300 or p[m].sum() <= 0:
        return np.nan
    return float(p[m & (t < thresh)].sum() / p[m].sum())


@dataclass
class SignatureSet:
    """Container for the complete descriptor set of one catchment."""

    gauge_id: str
    region: str
    scalars: dict[str, float] = field(default_factory=dict)
    fdc: np.ndarray | None = None
    monthly_profile: np.ndarray | None = None   # monthly allocation profile
    regime_hist: np.ndarray | None = None       # (REGIME_DOY_BINS, REGIME_FLOW_BINS)
    n_valid: int = 0


def compute_signatures(
    gauge_id: str,
    region: str,
    s: pd.Series,
    precip: pd.Series | None = None,
    temp: pd.Series | None = None,
    pet: pd.Series | None = None,
) -> SignatureSet:
    """Compute the complete descriptor set for one daily flow series."""
    v = _valid(s)
    out = SignatureSet(gauge_id=gauge_id, region=region, n_valid=int(v.size))
    if v.size < 300:
        return out

    mean = float(v.mean())
    q = np.quantile(v, [0.05, 0.10, 0.50, 0.90, 0.95])
    q05, q10, q50, q90, q95 = q.tolist()

    fdc = flow_duration_curve(s)
    ln = np.log(np.clip(fdc, 1e-6, None))
    t = np.asarray(TAU)
    # FDC slope over the 33--66% exceedance-probability interval.
    m33 = (t >= 0.33) & (t <= 0.66)
    slope = float(np.polyfit(t[m33], ln[m33], 1)[0]) if m33.sum() > 5 else np.nan

    prof = month_profile(s)
    R, cdoy, peak = circular_moments(prof)
    hf = half_flow_doy(s)

    sc: dict[str, float] = {
        "cv": float(v.std() / mean) if mean > 0 else np.nan,
        "fdc_slope": slope,
        "q95_q50": float(q95 / q50) if q50 > 0 else np.nan,
        "q90_q50": float(q90 / q50) if q50 > 0 else np.nan,
        "q50_q10": float(q50 / q10) if q10 > 0 else np.nan,
        "q05_q50": float(q05 / q50) if q50 > 0 else np.nan,
        "bfi": baseflow_index(s),
        "rb_flashiness": rb_flashiness(s),
        "peak_month_sin": float(np.sin(2 * np.pi * (peak - 0.5) / 12)) if np.isfinite(peak) else np.nan,
        "peak_month_cos": float(np.cos(2 * np.pi * (peak - 0.5) / 12)) if np.isfinite(peak) else np.nan,
        "circ_conc_R": R,
        "centroid_doy_sin": float(np.sin(2 * np.pi * cdoy / 365)) if np.isfinite(cdoy) else np.nan,
        "centroid_doy_cos": float(np.cos(2 * np.pi * cdoy / 365)) if np.isfinite(cdoy) else np.nan,
        "half_flow_doy_sin": float(np.sin(2 * np.pi * hf / 365)) if np.isfinite(hf) else np.nan,
        "half_flow_doy_cos": float(np.cos(2 * np.pi * hf / 365)) if np.isfinite(hf) else np.nan,
        "parde_amp": parde_amplitude(prof),
        "runoff_ratio": np.nan,
        "aridity_index": np.nan,
        "snow_fraction": np.nan,
    }

    # Climate descriptors: runoff ratio is mean flow divided by mean precipitation.
    if precip is not None:
        p = _valid(precip)
        if p.size > 300 and p.mean() > 0:
            sc["runoff_ratio"] = float(mean / p.mean())
    if pet is not None and precip is not None:
        # Aridity index is mean PET divided by mean precipitation.
        # Use precipitation in the denominator; PET/flow is a different quantity.
        # This follows the conventional PET/precipitation definition.
        e, p = _valid(pet), _valid(precip)
        if e.size > 300 and e.mean() > 0 and p.mean() > 0:
            sc["aridity_index"] = float(e.mean() / p.mean())
    if precip is not None and temp is not None:
        sc["snow_fraction"] = snow_fraction(precip, temp)

    out.scalars = sc
    out.fdc = fdc
    out.monthly_profile = prof
    out.regime_hist = flow_regime_histogram(s)
    return out


def month_profile(s: pd.Series) -> np.ndarray:
    """Return the twelve monthly fractions of annual flow."""
    v = s.to_numpy(dtype=float)
    m = np.asarray(s.index.month).astype(int)
    mask = ~np.isnan(v)
    if mask.sum() < 300:
        return np.full(12, np.nan)
    out = np.zeros(12)
    for k in range(1, 13):
        sel = mask & (m == k)
        out[k - 1] = np.nansum(v[sel])
    if out.sum() <= 0:
        return np.full(12, np.nan)
    return out / out.sum()


# ---------------------------------------------------------------------------
# C. Two-dimensional regime measure
# ---------------------------------------------------------------------------
def flow_regime_histogram(s: pd.Series, doy_bins: int = REGIME_DOY_BINS,
                          flow_bins: int = REGIME_FLOW_BINS) -> np.ndarray:
    """Represent a catchment as a probability histogram of day and log flow.

    Each valid day contributes equal mass to one day and flow bin.
    The measure retains both flow magnitude and within-year timing,
    whereas an annual FDC retains only the magnitude marginal.
    """
    v = s.to_numpy(dtype=float)
    mask = ~np.isnan(v) & (v > 0)
    if mask.sum() < 300:
        return np.zeros((doy_bins, flow_bins))

    doy = day_of_year_365(s.index)[mask]
    vals = v[mask]
    logz = np.log(vals) - np.log(vals).mean()          # centered log flow
    sd = logz.std()
    if sd > 0:
        logz = logz / sd
    logz = np.clip(logz, -REGIME_LO_Z_CLIP, REGIME_LO_Z_CLIP)

    db = np.clip(((doy - 1) / 365.0 * doy_bins).astype(int), 0, doy_bins - 1)
    fb = np.clip(((logz + REGIME_LO_Z_CLIP) / (2 * REGIME_LO_Z_CLIP) * flow_bins).astype(int),
                 0, flow_bins - 1)

    hist = np.zeros((doy_bins, flow_bins))
    np.add.at(hist, (db, fb), 1.0)
    return hist / hist.sum()


def regime_bin_centers(doy_bins: int = REGIME_DOY_BINS,
                       flow_bins: int = REGIME_FLOW_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Return bin centers for the day and standardized-log-flow axes."""
    doy_center = (np.arange(doy_bins) + 0.5) / doy_bins * 365.0
    z_center = (np.arange(flow_bins) + 0.5) / flow_bins * (2 * REGIME_LO_Z_CLIP) - REGIME_LO_Z_CLIP
    return doy_center, z_center


def flatten_measure(hist: np.ndarray) -> np.ndarray:
    """Flatten a two-dimensional histogram while preserving total mass."""
    p = hist.ravel()
    tot = p.sum()
    return p / tot if tot > 0 else p
