"""Hydrological descriptors from daily discharge and optional climate series.

Scalar descriptors cover flow variability, quantile ratios, baseflow,
flashiness, seasonal allocation and climate attributes. Normalised quantiles
provide a flow-duration representation. Joint day-of-year and standardised
log-flow histograms support the season-structured transport comparator.
Calendar coordinates use 365 days, merging leap day into 28 February.
Descriptors retain their documented units; callers control comparison scaling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# FDC quantile grid, excluding endpoints 0 and 1.
TAU = np.linspace(0.0, 1.0, 201)[1:-1]

# Lyne-Hollick filter parameter (Nathan & McMahon, 1990).
LH_ALPHA = 0.925

# Joint histogram dimensions: day of year and log flow.
REGIME_DOY_BINS = 36
REGIME_FLOW_BINS = 24
# Absolute clipping threshold for standardised log flow.
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

# Ten diagnostic predictors used to explain pairwise seasonal timing distance.
# The separate donor-transfer shape control uses only the first six entries,
# excluding the four attributes it predicts. 
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

# Descriptors requiring only daily flow; climate descriptors also need
# precipitation, temperature or potential evapotranspiration.
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

# Descriptors requiring external climate variables.
CLIMATE_SCALAR_NAMES = ["runoff_ratio", "aridity_index", "snow_fraction"]


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------
def _valid(s: pd.Series) -> np.ndarray:
    v = s.to_numpy(dtype=float)
    return v[~np.isnan(v)]


def day_of_year_365(index: pd.DatetimeIndex) -> np.ndarray:
    """Map dates to 1..365, merging leap day into 28 February.
    Use np.asarray because pandas date properties may already return ndarrays."""
    doy = np.asarray(index.dayofyear).astype(int)
    leap = np.asarray(index.is_leap_year).astype(bool)
    out = doy.copy()
    out[leap & (doy > 59)] -= 1     # Subtract one after February in leap years
    out[leap & (doy == 60)] = 59    # Merge leap day into 28 February
    return out


# ---------------------------------------------------------------------------
# Scalar descriptors
# ---------------------------------------------------------------------------
def flow_duration_curve(s: pd.Series, tau: np.ndarray = TAU) -> np.ndarray:
    """Return Q(tau)/mean(Q), with tau interpreted as non-exceedance probability."""
    v = _valid(s)
    if v.size == 0:
        return np.full(tau.size, np.nan)
    m = v.mean()
    if not np.isfinite(m) or m <= 0:
        return np.full(tau.size, np.nan)
    return np.quantile(v / m, tau)


def baseflow_index(s: pd.Series, alpha: float = LH_ALPHA) -> float:
    """Return baseflow volume divided by total flow using a Lyne-Hollick filter.
    The forward pass resets at invalid observations and clips quickflow to [0,Q]."""
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
    """Return the Richards-Baker index using valid adjacent-day differences."""
    v = s.to_numpy(dtype=float)
    valid = np.isfinite(v) & (v >= 0)
    if valid.sum() < 2 or v[valid].sum() <= 0:
        return np.nan
    adjacent = valid[1:] & valid[:-1]
    if isinstance(s.index, pd.DatetimeIndex):
        adjacent &= np.diff(s.index.values) == np.timedelta64(1, 'D')
    return float(np.abs(np.diff(v))[adjacent].sum() / v[valid].sum())


def circular_moments(profile: np.ndarray) -> tuple[float, float, float]:
    """Return circular concentration R, centroid day, and peak month (1..12)
    from twelve monthly allocation fractions."""
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
    """Return the maximum/minimum monthly profile ratio (legacy scalar definition)."""
    p = np.asarray(profile, dtype=float)
    if p.size != 12 or p.min() <= 0:
        return np.nan
    return float(p.max() / p.min())


def half_flow_doy(s: pd.Series) -> float:
    """Return the first day of year reaching 50% of aggregated annual flow."""
    idx = s.index
    v = s.to_numpy(dtype=float)
    mask = ~np.isnan(v)
    if mask.sum() < 300:
        return np.nan
    doy = day_of_year_365(idx)[mask]
    vals = v[mask]
    order = np.argsort(doy, kind="stable")
    doy, vals = doy[order], vals[order]
    # Aggregate observations from multiple years by day of year.
    uniq, inv = np.unique(doy, return_inverse=True)
    daily = np.zeros(uniq.size)
    np.add.at(daily, inv, vals)
    cum = np.cumsum(daily)
    if cum[-1] <= 0:
        return np.nan
    k = int(np.searchsorted(cum, 0.5 * cum[-1]))
    return float(uniq[min(k, uniq.size - 1)])


def snow_fraction(precip: pd.Series, temp: pd.Series, thresh: float = 0.0) -> float:
    """Return the precipitation fraction on days below the temperature threshold."""
    p = precip.to_numpy(dtype=float)
    t = temp.to_numpy(dtype=float)
    m = ~np.isnan(p) & ~np.isnan(t)
    if m.sum() < 300 or p[m].sum() <= 0:
        return np.nan
    return float(p[m & (t < thresh)].sum() / p[m].sum())


@dataclass
class SignatureSet:
    """Descriptors and metadata for one catchment."""

    gauge_id: str
    region: str
    scalars: dict[str, float] = field(default_factory=dict)
    fdc: np.ndarray | None = None
    monthly_profile: np.ndarray | None = None   # Twelve monthly fractions of available flow
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
    """Compute catchment descriptors from daily flow and optional climate series."""
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
    # Log-flow slope over non-exceedance probabilities 0.33 to 0.66.
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

    # Runoff ratio = mean Q / mean P, with compatible units.
    if precip is not None:
        p = _valid(precip)
        if p.size > 300 and p.mean() > 0:
            sc["runoff_ratio"] = float(mean / p.mean())
    if pet is not None and precip is not None:
        # Aridity index = mean PET / mean precipitation.
        # Use PET/P, not PET/Q; these represent different quantities.
        # The calculation below implements PET/P.
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
    """Return monthly fractions of summed available daily flow."""
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
# Joint day-of-year and log-flow histogram
# ---------------------------------------------------------------------------
def flow_regime_histogram(s: pd.Series, doy_bins: int = REGIME_DOY_BINS,
                          flow_bins: int = REGIME_FLOW_BINS) -> np.ndarray:
    """Return a joint day-of-year and standardised log-flow probability histogram.
    Each valid positive-flow observation has equal mass. Log flow is centred,
    scaled by its standard deviation when positive, and clipped."""
    v = s.to_numpy(dtype=float)
    mask = ~np.isnan(v) & (v > 0)
    if mask.sum() < 300:
        return np.zeros((doy_bins, flow_bins))

    doy = day_of_year_365(s.index)[mask]
    vals = v[mask]
    logz = np.log(vals) - np.log(vals).mean()          # Dimensionless centred log flow
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
    """Return day-of-year and standardised log-flow histogram bin centres."""
    doy_center = (np.arange(doy_bins) + 0.5) / doy_bins * 365.0
    z_center = (np.arange(flow_bins) + 0.5) / flow_bins * (2 * REGIME_LO_Z_CLIP) - REGIME_LO_Z_CLIP
    return doy_center, z_center


def flatten_measure(hist: np.ndarray) -> np.ndarray:
    """Flatten and normalise histogram mass when the total is positive."""
    p = hist.ravel()
    tot = p.sum()
    return p / tot if tot > 0 else p
