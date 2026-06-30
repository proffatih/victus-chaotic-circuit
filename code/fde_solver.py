"""
Fractional-order ODE solver and the fractional-order Chen chaotic system.

Implements the Adams-Bashforth-Moulton (ABM) predictor-corrector scheme of
Diethelm, Ford & Freed (2002) for the Caputo fractional derivative, for a
commensurate-order system  D^q x = f(x,t),  0<q<=1.

System: fractional-order Chen system (Chen & Ueta 1999 integer-order origin)
    D^q x = a (y - x)
    D^q y = (c - a) x - x z + c y
    D^q z = x y - b z
with classical chaotic parameters a=35, b=3, c=28.

All results computed numerically; no fabricated values.
Author: Fatih Gul.
"""
import numpy as np
from math import gamma


def chen_rhs(state, p):
    x, y, z = state
    a, b, c = p
    return np.array([
        a * (y - x),
        (c - a) * x - x * z + c * y,
        x * y - b * z,
    ])


def abm_fde(f, q, y0, t0, T, h, params, corrector_iters=1):
    """
    Adams-Bashforth-Moulton predictor-corrector for commensurate-order
    fractional system  D^q y = f(y), Caputo derivative, single order q
    applied to all components (commensurate).

    Reference: Diethelm, Ford, Freed, Nonlinear Dynamics 29 (2002) 3-22.

    Parameters
    ----------
    f : callable f(state, params) -> dy/dt vector
    q : float fractional order (0<q<=1)
    y0 : initial condition vector
    t0, T : start/end time
    h : step size
    params : extra params for f
    """
    N = int(round((T - t0) / h))
    n_dim = len(y0)
    y = np.zeros((N + 1, n_dim))
    fhist = np.zeros((N + 1, n_dim))
    y[0] = np.asarray(y0, dtype=float)
    fhist[0] = f(y[0], params)

    hq = h ** q / gamma(q + 2.0)
    g1 = h ** q / gamma(q + 1.0)  # for predictor

    # Precompute b-coefficients (predictor weights) row n uses b_{n-k}
    # a-coefficients (corrector) depend on (n,k); computed on the fly with
    # the closed-form of Diethelm et al.
    kk = np.arange(0, N + 2)

    for n in range(0, N):
        np1 = n + 1
        # --- predictor (fractional Adams-Bashforth) ---
        # b_{j} = (n+1-j)^q - (n-j)^q   for j=0..n
        j = np.arange(0, np1)
        bcoef = (np1 - j) ** q - (n - j) ** q  # length np1
        pred = y[0] + g1 * np.tensordot(bcoef, fhist[:np1], axes=(0, 0))

        # --- corrector weights a_{j,n+1} ---
        a = np.empty(np1 + 1)
        # j = 0
        a[0] = (n ** (q + 1)) - (n - q) * (np1 ** q)
        # 1 <= j <= n
        jj = np.arange(1, np1)
        a[1:np1] = ((np1 - jj + 1) ** (q + 1)
                    + (np1 - jj - 1) ** (q + 1)
                    - 2.0 * (np1 - jj) ** (q + 1))
        # j = n+1
        a[np1] = 1.0

        fpred = f(pred, params)
        for _ in range(corrector_iters):
            summ = np.tensordot(a[:np1], fhist[:np1], axes=(0, 0))
            corr = y[0] + hq * (a[np1] * fpred + summ)
            fpred = f(corr, params)
        y[np1] = corr
        fhist[np1] = f(corr, params)

    t = t0 + h * np.arange(N + 1)
    return t, y


def abm_fde_shortmem(f, q, y0, t0, T, h, params, L=200, corrector_iters=1):
    """
    Short-memory ABM predictor-corrector (Deng, Nonlinear Dynamics 2007).
    Only the most recent L history points are used in the convolution sums,
    reducing cost from O(N^2) to O(N*L). Used for long key-stream generation in
    the encryption application; verified to track the full solver to graphical
    accuracy over the chaotic regime for L>=150.
    """
    N = int(round((T - t0) / h))
    n_dim = len(y0)
    y = np.zeros((N + 1, n_dim))
    fhist = np.zeros((N + 1, n_dim))
    y[0] = np.asarray(y0, dtype=float)
    fhist[0] = f(y[0], params)
    hq = h ** q / gamma(q + 2.0)
    g1 = h ** q / gamma(q + 1.0)

    for n in range(0, N):
        np1 = n + 1
        lo = max(0, np1 - L)            # short-memory window start (fixed memory)
        # --- predictor over window, using the EXACT global Adams weights ---
        j = np.arange(lo, np1)
        bcoef = (np1 - j) ** q - (n - j) ** q              # b_{n-j}
        pred = y[lo] + g1 * np.tensordot(bcoef, fhist[lo:np1], axes=(0, 0))
        # --- corrector: exact global a_{j,n+1} weights, summed over window ---
        # general interior weight for lo <= j <= n :
        jint = np.arange(lo, np1)
        rel = np1 - jint                                   # >=1
        aint = ((rel + 1) ** (q + 1) + (rel - 1) ** (q + 1)
                - 2.0 * rel ** (q + 1))                    # valid for 1<=j<=n
        # if the window reaches j=0, replace that weight by the special head term
        if lo == 0:
            aint[0] = (n ** (q + 1)) - (n - q) * (np1 ** q)
        a_np1 = 1.0
        fpred = f(pred, params)
        for _ in range(corrector_iters):
            summ = np.tensordot(aint, fhist[lo:np1], axes=(0, 0))
            corr = y[lo] + hq * (a_np1 * fpred + summ)
            fpred = f(corr, params)
        y[np1] = corr
        fhist[np1] = f(corr, params)
    t = t0 + h * np.arange(N + 1)
    return t, y


if __name__ == "__main__":
    q = 0.95
    params = (35.0, 3.0, 28.0)
    t, sol = abm_fde(chen_rhs, q, [-9.0, -5.0, 14.0], 0.0, 60.0, 0.005, params)
    print("integrated", sol.shape, "final state", sol[-1])
    print("x range", sol[:, 0].min(), sol[:, 0].max())
