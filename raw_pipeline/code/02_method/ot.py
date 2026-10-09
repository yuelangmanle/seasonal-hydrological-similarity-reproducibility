"""Transport utilities for flow quantiles and seasonal allocation.

One-dimensional W2 compares quantile functions. Circular W1 compares annual
allocation profiles in days. RegimeMetric compares joint day-of-year and
standardised log-flow histograms using factorised Sinkhorn iterations.
The ground cost combines squared flow differences and squared circular day
differences. Entropy determines the plan but is excluded from reported costs.
Self-cost subtraction therefore differs from full-objective Sinkhorn divergence.
The Kronecker kernel reduces matrix-vector work from O((nd*nf)^2) to
O(nd*nf*(nd+nf)); self-tests compare factorised and dense calculations."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# One-dimensional W2
# ---------------------------------------------------------------------------
def w2_1d(u: np.ndarray, v: np.ndarray, tau: np.ndarray | None = None) -> float:
    """Compute W2 between quantile vectors on a common probability grid."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.isfinite(u) & np.isfinite(v)
    if m.sum() < 3:
        return float("nan")
    u, v = u[m], v[m]
    if tau is None:
        w = np.full(u.size, 1.0 / u.size)
    else:
        t = np.asarray(tau, dtype=float)[m]
        order = np.argsort(t)
        t = t[order]
        u = u[order]
        v = v[order]
        w = np.gradient(t)
    return float(np.sqrt(np.sum(w * (u - v) ** 2) / np.sum(w)))


# ---------------------------------------------------------------------------
# Circular W1 timing distance, in days
# ---------------------------------------------------------------------------
_MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)


def expand_to_days(profile12: np.ndarray) -> np.ndarray:
    """Expand monthly fractions to 365 daily masses, uniformly within each month."""
    p = np.asarray(profile12, dtype=float)
    if p.size != 12 or not np.isfinite(p).all() or p.sum() <= 0:
        return np.full(365, np.nan)
    p = p / p.sum()
    return np.repeat(p / _MONTH_DAYS, _MONTH_DAYS.astype(int))


def timing_distance_days(p: np.ndarray, q: np.ndarray) -> float:
    """Compute circular W1 in days: minimise the absolute cumulative difference
    over its median offset. Zero indicates identical annual allocation."""
    pd_ = expand_to_days(p)
    qd_ = expand_to_days(q)
    if not np.isfinite(pd_).all() or not np.isfinite(qd_).all():
        return float("nan")
    d = np.cumsum(pd_) - np.cumsum(qd_)
    return float(np.abs(d - np.median(d)).mean() * 365.0)


# ---------------------------------------------------------------------------
# Season-structured transport
# ---------------------------------------------------------------------------
def cost_factors(doy_center: np.ndarray, z_center: np.ndarray,
                 doy_norm: float = 365.0) -> tuple[np.ndarray, np.ndarray]:
    """Return seasonal and flow ground-cost factors for the product grid."""
    d = np.abs(doy_center[:, None] - doy_center[None, :])
    d = np.minimum(d, doy_norm - d) / doy_norm          # Circular distance normalised to [0,0.5]
    cd = d ** 2
    dz = z_center[:, None] - z_center[None, :]
    cf = dz ** 2
    return cd, cf


def kron_apply(v: np.ndarray, kd: np.ndarray, kf: np.ndarray) -> np.ndarray:
    """Apply (Kd x Kf) to C-order vec(V) as Kd @ V @ Kf.T."""
    return kd @ v @ kf.T


def _pair_cost(u: np.ndarray, v: np.ndarray, kd: np.ndarray, kf: np.ndarray,
               cd: np.ndarray, cf: np.ndarray, alpha: float, beta: float) -> float:
    """Evaluate <P,C> from Sinkhorn scaling factors without forming the full plan.
    P(d,f;d2,f2) = U[d,f]*Kd[d,d2]*Kf[f,f2]*V[d2,f2]."""
    t1 = np.sum((cd * kd) * (u @ kf @ v.T))          # Seasonal contribution
    t2 = np.sum((cf * kf) * (u.T @ kd @ v))          # Flow contribution
    return beta * float(t1) + alpha * float(t2)


def sinkhorn_cost(a: np.ndarray, b: np.ndarray, kd: np.ndarray, kf: np.ndarray,
                  cd: np.ndarray, cf: np.ndarray, alpha: float, beta: float,
                  n_iter: int = 400, tol: float = 1e-10) -> float:
    """Return transport cost from factorised Sinkhorn iterations, excluding entropy."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    a = a / sa
    b = b / sb

    u = np.ones_like(a)
    v = np.ones_like(b)
    tiny = 1e-300
    for _ in range(n_iter):
        u_prev = u
        v = b / np.maximum(kron_apply(u, kd, kf), tiny)
        u = a / np.maximum(kron_apply(v, kd, kf), tiny)
        # Reject unstable scaling factors for degenerate inputs.
        # This guard returns NaN rather than clipping.
        um = u.max()
        if not np.isfinite(um) or um > 1e200:
            return float("nan")
        if np.max(np.abs(u - u_prev)) < tol:
            break
    return _pair_cost(u, v, kd, kf, cd, cf, alpha, beta)


class RegimeMetric:
    """Season-structured transport with optional self-cost subtraction.
    
    alpha and beta weight flow and seasonal costs; beta=0 removes seasonal cost.
    reg is the entropic regularisation strength. debiased subtracts half of each
    self-cost before clipping and taking a square root. Reported costs exclude
    entropy and do not define the standard full-objective Sinkhorn divergence."""

    def __init__(self, n_doy: int, n_flow: int, alpha: float = 1.0,
                 beta: float = 1.0, reg: float = 0.02, debiased: bool = True,
                 doy_center: np.ndarray | None = None,
                 z_center: np.ndarray | None = None):
        self.n_doy = n_doy
        self.n_flow = n_flow
        self.alpha = alpha
        self.beta = beta
        self.reg = reg
        self.debiased = debiased

        # Bin centres must match histogram construction
        # to preserve the intended alpha:beta weighting.
        # Use signatures.regime_bin_centers(); defaults
        # are intended for self-test grids.
        self.doy_center = ((np.arange(n_doy) + 0.5) / n_doy * 365.0
                           if doy_center is None else np.asarray(doy_center, float))
        self.z_center = ((np.arange(n_flow) + 0.5) / n_flow * 4.0 - 2.0
                         if z_center is None else np.asarray(z_center, float))
        self.cd, self.cf = cost_factors(self.doy_center, self.z_center)
        self.kd = np.exp(-self.beta * self.cd / reg)
        self.kf = np.exp(-self.alpha * self.cf / reg)
        self._self_cache: dict[tuple, float] = {}

    @classmethod
    def for_regime_grid(cls, n_doy: int | None = None, n_flow: int | None = None,
                        **kw) -> "RegimeMetric":
        """Construct using the histogram bin centres defined in signatures.
        Use this entry point for analysis histograms; default constructor centres
        are intended for self-test grids."""
        from signatures import REGIME_DOY_BINS, REGIME_FLOW_BINS, regime_bin_centers
        nd = REGIME_DOY_BINS if n_doy is None else n_doy
        nf = REGIME_FLOW_BINS if n_flow is None else n_flow
        doy_c, z_c = regime_bin_centers(nd, nf)
        return cls(nd, nf, doy_center=doy_c, z_center=z_c, **kw)

    # Single transport calculation
    def ot(self, a: np.ndarray, b: np.ndarray, n_iter: int = 400) -> float:
        return sinkhorn_cost(a, b, self.kd, self.kf, self.cd, self.cf,
                             self.alpha, self.beta, n_iter=n_iter)

    def _uv(self, a: np.ndarray, b: np.ndarray, n_iter: int = 400,
            tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute Sinkhorn scaling factors for reconstructing the transport plan."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        sa, sb = a.sum(), b.sum()
        if not (np.isfinite(sa) and np.isfinite(sb) and sa > 0 and sb > 0):
            return None
        a, b = a / sa, b / sb
        u = np.ones_like(a)
        v = np.ones_like(b)
        tiny = 1e-300
        for _ in range(n_iter):
            u_prev = u
            v = b / np.maximum(kron_apply(u, self.kd, self.kf), tiny)
            u = a / np.maximum(kron_apply(v, self.kd, self.kf), tiny)
            um = u.max()
            if not np.isfinite(um) or um > 1e200:
                return None
            if np.max(np.abs(u - u_prev)) < tol:
                break
        return u, v

    def day_transport(self, a: np.ndarray, b: np.ndarray,
                      n_iter: int = 400) -> np.ndarray | None:
        """Marginalise the plan over flow bins to a (n_doy, n_doy) matrix.
        M[d,d2] is the mass transported from day bin d to day bin d2."""
        uv = self._uv(a, b, n_iter=n_iter)
        if uv is None:
            return None
        u, v = uv
        P = (u[:, None, :, None] * self.kd[:, :, None, None]
             * self.kf[None, None, :, :] * v[None, :, None, :])
        return P.sum(axis=(2, 3))

    def _self_term(self, a: np.ndarray, n_iter: int = 400) -> float:
        # Include n_iter in the cache key because iterative
        # self-costs depend on the iteration budget.
        # Mixing budgets would make centred costs inconsistent.
        key = (int(n_iter), np.ascontiguousarray(a, dtype=np.float64).tobytes())
        if key not in self._self_cache:
            self._self_cache[key] = self.ot(a, a, n_iter=n_iter)
        return self._self_cache[key]

    # Distances
    def distance(self, a: np.ndarray, b: np.ndarray, n_iter: int = 400) -> float:
        """Return the square root of the optionally self-centred transport cost.
        Match n_iter to the precomputed matrix metadata to keep costs comparable."""
        ab = self.ot(a, b, n_iter=n_iter)
        if not np.isfinite(ab):
            return float("nan")
        if self.debiased:
            # Self-centred cost: OT(a,b) - 0.5*OT(a,a) - 0.5*OT(b,b).
            # Clip negative values before the square root; entropy is excluded.
            m = (ab - 0.5 * self._self_term(a, n_iter)
                 - 0.5 * self._self_term(b, n_iter))
            return float(np.sqrt(max(m, 0.0)))
        return float(np.sqrt(max(ab, 0.0)))

    # Batched transport from a to B
    # Use a shared fixed kernel for target-to-donor comparisons.
    # Batching reduces repeated Python/NumPy dispatch overhead.
    # Stack B as (m,n_doy,n_flow); matrix multiplication
    # broadcasts over the first dimension.
    def ot_many(self, a: np.ndarray, B: np.ndarray, n_iter: int = 300,
                tol: float = 1e-9) -> np.ndarray:
        """Return batched transport costs from a to each measure in B, excluding entropy."""
        a = np.asarray(a, dtype=float)
        B = np.asarray(B, dtype=float)
        m = B.shape[0]
        sa = a.sum()
        if sa <= 0 or not np.isfinite(sa):
            return np.full(m, np.nan)
        a = a / sa
        sb = B.sum(axis=(1, 2), keepdims=True)
        good = np.isfinite(sb[:, 0, 0]) & (sb[:, 0, 0] > 0)
        Bn = B / np.maximum(sb, 1e-300)

        U = np.ones_like(Bn)
        V = np.ones_like(Bn)
        A = np.broadcast_to(a, Bn.shape)
        tiny = 1e-300
        for _ in range(n_iter):
            U_prev = U
            V = Bn / np.maximum(self.kd @ U @ self.kf.T, tiny)
            U = A / np.maximum(self.kd @ V @ self.kf.T, tiny)
            um = float(np.max(U))
            if not np.isfinite(um) or um > 1e200:
                return np.full(m, np.nan)
            if np.max(np.abs(U - U_prev)) < tol:
                break

        cd_kd = self.cd * self.kd
        cf_kf = self.cf * self.kf
        P = U @ self.kf                                # (m, nd, nf)
        e1 = np.einsum("ij,mij->m", cd_kd, P @ V.transpose(0, 2, 1))
        Q = U.transpose(0, 2, 1) @ self.kd             # (m, nf, nd)
        e2 = np.einsum("ij,mij->m", cf_kf, Q @ V)
        out = self.beta * e1 + self.alpha * e2
        out[~good] = np.nan
        return out

    def _self_terms_many(self, B: np.ndarray, n_iter: int = 400) -> np.ndarray:
        out = np.empty(len(B), dtype=float)
        for i in range(len(B)):
            out[i] = self._self_term(B[i], n_iter)
        return out

    def self_terms_all(self, H: np.ndarray, n_iter: int = 400) -> np.ndarray:
        """Precompute self-costs for every measure using the same n_iter as later distances."""
        return self._self_terms_many(H, n_iter)

    # Paired batches A_i, B_i
    def ot_pairs(self, A: np.ndarray, B: np.ndarray, n_iter: int = 300,
                 tol: float = 1e-9, chunk: int = 2000) -> np.ndarray:
        """Return costs for pairs A_i and B_i, both shaped (p, n_doy, n_flow)."""
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        p = A.shape[0]
        out = np.empty(p, dtype=float)
        tiny = 1e-300
        cd_kd, cf_kf = self.cd * self.kd, self.cf * self.kf
        for s in range(0, p, chunk):
            e = min(s + chunk, p)
            a, b = A[s:e], B[s:e]
            sa = a.sum(axis=(1, 2), keepdims=True)
            sb = b.sum(axis=(1, 2), keepdims=True)
            good = (np.isfinite(sa[:, 0, 0]) & (sa[:, 0, 0] > 0) &
                    np.isfinite(sb[:, 0, 0]) & (sb[:, 0, 0] > 0))
            an = a / np.maximum(sa, tiny)
            bn = b / np.maximum(sb, tiny)
            U = np.ones_like(an)
            V = np.ones_like(bn)
            for _ in range(n_iter):
                U_prev = U
                V = bn / np.maximum(self.kd @ U @ self.kf.T, tiny)
                U = an / np.maximum(self.kd @ V @ self.kf.T, tiny)
                um = float(np.max(U))
                if not np.isfinite(um) or um > 1e200:
                    break
                if np.max(np.abs(U - U_prev)) < tol:
                    break
            P = U @ self.kf
            e1 = np.einsum("ij,mij->m", cd_kd, P @ V.transpose(0, 2, 1))
            Q = U.transpose(0, 2, 1) @ self.kd
            e2 = np.einsum("ij,mij->m", cf_kf, Q @ V)
            val = self.beta * e1 + self.alpha * e2
            val[~good] = np.nan
            out[s:e] = val
        return out

    def distance_pairs(self, H: np.ndarray, ia: np.ndarray, ib: np.ndarray,
                       self_terms: np.ndarray | None = None,
                       n_iter: int = 300) -> np.ndarray:
        """Return distances for H[ia] and H[ib]; optionally reuse precomputed self-costs."""
        H = np.asarray(H, dtype=float)
        ia = np.asarray(ia, dtype=np.int64)
        ib = np.asarray(ib, dtype=np.int64)
        ab = self.ot_pairs(H[ia], H[ib], n_iter=n_iter)
        if not self.debiased:
            return np.sqrt(np.maximum(ab, 0.0))
        st = (self.self_terms_all(H, n_iter) if self_terms is None
              else np.asarray(self_terms, float))
        m = ab - 0.5 * st[ia] - 0.5 * st[ib]
        return np.sqrt(np.maximum(m, 0.0))

    def distance_many(self, a: np.ndarray, B: np.ndarray,
                      n_iter: int = 300) -> np.ndarray:
        """Return batched distances from a to each measure in B."""
        ab = self.ot_many(a, B, n_iter=n_iter)
        if not self.debiased:
            return np.sqrt(np.maximum(ab, 0.0))
        m = (ab - 0.5 * self._self_term(a, n_iter)
             - 0.5 * self._self_terms_many(B, n_iter))
        return np.sqrt(np.maximum(m, 0.0))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
def self_test(verbose: bool = True) -> bool:
    """Check factorisation, Gaussian W2, timing distance, and transport consistency."""
    rng = np.random.default_rng(0)
    ok = True

    nd, nf = 6, 5
    m = rng.random((nd, nd))
    cd = m + m.T
    m = rng.random((nf, nf))
    cf = m + m.T
    kd, kf = np.exp(-cd), np.exp(-cf)
    V = rng.random((nd, nf))
    fast = kron_apply(V, kd, kf)
    dense = (np.kron(kd, kf) @ V.ravel(order="C")).reshape(nd, nf)
    e1 = float(np.max(np.abs(fast - dense)))
    ok &= e1 < 1e-12
    if verbose:
        print(f"[self_test] Kronecker identity      max. difference {e1:.2e}   {'OK' if e1 < 1e-12 else 'FAIL'}")

    from math import sqrt
    from statistics import NormalDist
    # Use the cosine probability map to refine normal-quantile tails.
    # The sine Jacobian reduces endpoint contributions in trapezoidal integration.
    s = np.linspace(0.0, 1.0, 20001)
    t = 0.5 * (1.0 - np.cos(np.pi * s))
    t = np.clip(t, 1e-12, 1 - 1e-12)
    w = 0.5 * np.pi * np.sin(np.pi * s)
    q1 = np.array([NormalDist(0.0, 1.0).inv_cdf(x) for x in t])
    q2 = np.array([NormalDist(2.0, 3.0).inv_cdf(x) for x in t])
    integ = w * (q1 - q2) ** 2
    got = float(np.sqrt(np.trapezoid(integ, s) / np.trapezoid(w, s)))
    want = sqrt((0 - 2) ** 2 + (1 - 3) ** 2)
    e2 = abs(got - want) / want
    ok &= e2 < 2e-4
    if verbose:
        print(f"[self_test] Gaussian W2 reference        {got:.6f} vs {want:.6f}   relative error {e2:.2e}   "
              f"{'OK' if e2 < 2e-4 else 'FAIL'}")

    peak = np.ones(12)
    peak[3] = 10.0                       # April peak, accounting for 10/21 of annual allocation
    d_same = timing_distance_days(peak, peak)
    d1 = timing_distance_days(peak, np.roll(peak, 1))
    d3 = timing_distance_days(peak, np.roll(peak, 3))
    # For this profile, three-month displacement should give about three times
    # the circular W1 of a one-month displacement; check the expected magnitude.
    ratio = d3 / d1 if d1 > 0 else np.nan
    good = (d_same < 1e-9) and (2.5 < ratio < 3.5) and (35 < d3 < 50)
    ok &= good
    if verbose:
        print(f"[self_test] Timing distance (days)        self {d_same:.6f}, shift 1 month {d1:.2f}, "
              f"3 months {d3:.2f}(ratio {ratio:.2f})   {'OK' if good else 'FAIL'}")

    # Self-distance should be near zero; seasonal redistribution should increase it.
    rng2 = np.random.default_rng(7)
    A = rng2.random((nd, nf)) + 0.05
    A[1, 0:3] += 3.0                     # Add mass to day bin 1
    B = A.copy()
    B[1, 0:3] -= 3.0
    B[4, 0:3] += 3.0                     # Move the mass to day bin 4
    met = RegimeMetric(nd, nf, alpha=1.0, beta=1.0, reg=0.05)
    d_aa = met.distance(A, A)
    d_ab = met.distance(A, B)
    good2 = np.isfinite(d_aa) and np.isfinite(d_ab) and abs(d_aa) < 0.05 and d_ab > 3 * max(abs(d_aa), 1e-6)
    ok &= good2
    if verbose:
        print(f"[self_test] Centred transport         self {d_aa:.4f}, seasonal redistribution {d_ab:.4f}   "
              f"{'OK' if good2 else 'FAIL'}")

    # Batched and scalar calculations should agree at the same iteration budget.
    Bsz = rng2.random((7, nd, nf)) + 0.05
    dm = met.distance_many(A, Bsz, n_iter=400)
    ds = np.array([met.distance(A, Bsz[i]) for i in range(7)])
    e4 = float(np.nanmax(np.abs(dm - ds)))
    ok &= e4 < 1e-9
    if verbose:
        print(f"[self_test] Batch vs scalar          max. difference {e4:.2e}   {'OK' if e4 < 1e-9 else 'FAIL'}")

    # Paired-batch and scalar calculations should also agree.
    ia_p = np.array([0, 1, 2, 3, 4, 5, 6])
    ib_p = np.array([3, 6, 1, 5, 0, 2, 4])
    dp = met.distance_pairs(Bsz, ia_p, ib_p, n_iter=400)
    dref = np.array([met.distance(Bsz[i], Bsz[j]) for i, j in zip(ia_p, ib_p)])
    e5 = float(np.max(np.abs(dp - dref)))
    ok &= e5 < 1e-9
    if verbose:
        print(f"[self_test] Paired batch vs scalar      max. difference {e5:.2e}   {'OK' if e5 < 1e-9 else 'FAIL'}")

    # The day-marginal plan should recover the source day marginal.
    M = met.day_transport(A, B, n_iter=2000)
    e6 = float(np.max(np.abs(M.sum(axis=1) - A.sum(axis=1) / A.sum())))
    ok &= e6 < 1e-6
    if verbose:
        print(f"[self_test] Transport marginal conservation      max. difference {e6:.2e}   {'OK' if e6 < 1e-6 else 'FAIL'}")

    # With beta=0, ground cost has no seasonal component.
    met0 = RegimeMetric(nd, nf, alpha=1.0, beta=0.0, reg=0.05)
    d_b0_a = met0.distance(A, A)
    d_b0_b = met0.distance(A, B)
    good7 = abs(d_b0_a) < 1e-4 and np.isfinite(d_b0_b) and d_b0_b < d_ab * 1.5
    ok &= good7
    if verbose:
        print(f"[self_test] beta=0 check        self {d_b0_a:.2e}, seasonal redistribution "
              f"{d_b0_b:.4f} < beta=1 value {d_ab:.4f}   "
              f"{'OK' if good7 else 'FAIL'}")

    # Check bin centres against histogram construction.
    try:
        from signatures import regime_bin_centers
        dc, zc = regime_bin_centers(36, 24)
        m2 = RegimeMetric.for_regime_grid(36, 24, reg=0.05)
        e7 = max(float(np.max(np.abs(m2.doy_center - dc))),
                 float(np.max(np.abs(m2.z_center - zc))))
        lo = 0.5 / 24 * (2 * 3.0) - 3.0      # -2.875: 24 bins over +/-3 standard deviations
        hi = 23.5 / 24 * (2 * 3.0) - 3.0     # +2.875
        span_ok = (abs(m2.z_center[0] - lo) < 1e-9 and
                   abs(m2.z_center[-1] - hi) < 1e-9)
        ok &= (e7 < 1e-12) and span_ok
        if verbose:
            print(f"[self_test] Histogram bin alignment      max. difference {e7:.2e};"
                  f"z centres {m2.z_center[0]:.2f}..{m2.z_center[-1]:.2f}   "
                  f"{'OK' if (e7 < 1e-12 and span_ok) else 'FAIL'}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        if verbose:
            print(f"[self_test] Histogram bin alignment      FAIL: {exc}")

    return bool(ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
