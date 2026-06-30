"""
Nonlinear-dynamics analysis of the fractional-order Chen system.
Produces:
  results/trajectory_q095.npz       (reference attractor for phase portraits)
  results/bifurcation_c.csv         (local maxima of x vs parameter c)
  results/bifurcation_q.csv         (local maxima of x vs fractional order q)
  results/lyapunov_q.csv            (largest Lyapunov exponent vs q, Benettin)
  results/test01_q.csv              (0-1 test K-statistic vs q)
All numbers from real computation.
"""
import json
import numpy as np
from fde_solver import abm_fde, chen_rhs

A, B = 35.0, 3.0
Q_REF = 0.95
C_REF = 28.0
IC = np.array([-9.0, -5.0, 14.0])


def integrate(q, c, T, h, ic=IC):
    return abm_fde(chen_rhs, q, ic, 0.0, T, h, (A, B, c))


def local_maxima(x):
    idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    return x[idx]


# ---------- 1. Reference trajectory ----------
def reference_trajectory():
    t, sol = integrate(Q_REF, C_REF, 100.0, 0.005)
    # discard transient (first 30 t.u.)
    keep = t >= 30.0
    np.savez("../results/trajectory_q095.npz", t=t[keep], x=sol[keep, 0],
             y=sol[keep, 1], z=sol[keep, 2])
    print("reference trajectory saved:", sol[keep].shape)


# ---------- 2. Bifurcation vs c ----------
def bifurcation_c(cmin=20.0, cmax=35.0, ncols=160, h=0.01, T=70.0):
    cs = np.linspace(cmin, cmax, ncols)
    rows = []
    for i, c in enumerate(cs):
        t, sol = integrate(Q_REF, c, T, h)
        x = sol[t >= 40.0, 0]   # discard transient
        lm = local_maxima(x)
        lm = lm[-40:] if len(lm) > 40 else lm  # cap points per column
        for v in lm:
            rows.append((c, v))
        if i % 20 == 0:
            print(f"bif_c {i}/{ncols} c={c:.2f} maxima={len(lm)}")
    arr = np.array(rows)
    np.savetxt("../results/bifurcation_c.csv", arr, delimiter=",",
               header="c,xmax", comments="")
    print("bifurcation_c done", arr.shape)


# ---------- 3. Bifurcation vs q ----------
def bifurcation_q(qmin=0.80, qmax=1.0, ncols=160, h=0.01, T=70.0):
    qs = np.linspace(qmin, qmax, ncols)
    rows = []
    for i, q in enumerate(qs):
        t, sol = integrate(q, C_REF, T, h)
        x = sol[t >= 40.0, 0]
        lm = local_maxima(x)
        lm = lm[-40:] if len(lm) > 40 else lm
        for v in lm:
            rows.append((q, v))
        if i % 20 == 0:
            print(f"bif_q {i}/{ncols} q={q:.3f} maxima={len(lm)}")
    arr = np.array(rows)
    np.savetxt("../results/bifurcation_q.csv", arr, delimiter=",",
               header="q,xmax", comments="")
    print("bifurcation_q done", arr.shape)


# ---------- 4. Largest Lyapunov exponent (Benettin) vs q ----------
def largest_lyapunov(q, c, T=120.0, h=0.01, d0=1e-7, renorm_every=20):
    """Benettin two-trajectory method adapted to the ABM solver.
    Re-integrates two nearby trajectories in short segments, renormalising
    the separation; LLE = mean log-growth / segment time."""
    seg_t = renorm_every * h
    n_seg = int(T / seg_t)
    x1 = IC.copy()
    x2 = IC.copy() + np.array([d0, 0.0, 0.0])
    logsum = 0.0
    count = 0
    for s in range(n_seg):
        _, s1 = abm_fde(chen_rhs, q, x1, 0.0, seg_t, h, (A, B, c))
        _, s2 = abm_fde(chen_rhs, q, x2, 0.0, seg_t, h, (A, B, c))
        x1 = s1[-1].copy()
        x2e = s2[-1].copy()
        diff = x2e - x1
        d = np.linalg.norm(diff)
        if s >= 3 and d > 0:        # skip transient segments
            logsum += np.log(d / d0)
            count += 1
        # renormalise
        x2 = x1 + (d0 / d) * diff
    return logsum / (count * seg_t) if count else float("nan")


def lyapunov_vs_q(qmin=0.85, qmax=1.0, n=24):
    qs = np.linspace(qmin, qmax, n)
    rows = []
    for i, q in enumerate(qs):
        lle = largest_lyapunov(q, C_REF)
        rows.append((q, lle))
        print(f"LLE q={q:.4f} -> {lle:.5f}")
    arr = np.array(rows)
    np.savetxt("../results/lyapunov_q.csv", arr, delimiter=",",
               header="q,LLE", comments="")
    print("lyapunov_vs_q done")


# ---------- 5. 0-1 test for chaos vs q ----------
def test_0_1(phi, c_count=100):
    """Gottwald-Melbourne 0-1 test. Returns median K over random c in (0,pi)."""
    phi = phi - np.mean(phi)
    N = len(phi)
    n = np.arange(1, N + 1)
    Ks = []
    rng = np.random.default_rng(0)
    cs = rng.uniform(np.pi / 5, 4 * np.pi / 5, c_count)
    ncut = N // 10
    for cc in cs:
        p = np.cumsum(phi * np.cos(n * cc))
        qy = np.cumsum(phi * np.sin(n * cc))
        M = np.empty(ncut)
        for nn in range(1, ncut + 1):
            M[nn - 1] = np.mean((p[nn:] - p[:-nn]) ** 2
                                + (qy[nn:] - qy[:-nn]) ** 2)
        Dt = M - (np.mean(phi) ** 2) * (1 - np.cos(n[:ncut] * cc)) / (1 - np.cos(cc))
        xi = np.arange(1, ncut + 1)
        # correlation between time and mean-square displacement
        Kc = np.corrcoef(xi, Dt)[0, 1]
        Ks.append(Kc)
    return float(np.median(Ks))


def test01_vs_q(qmin=0.85, qmax=1.0, n=24, h=0.02, T=200.0):
    qs = np.linspace(qmin, qmax, n)
    rows = []
    for q in qs:
        t, sol = integrate(q, C_REF, T, h)
        phi = sol[t >= 50.0, 0]
        phi = phi[::5]  # subsample to reduce oversampling correlation
        K = test_0_1(phi)
        rows.append((q, K))
        print(f"0-1 test q={q:.4f} -> K={K:.4f}")
    arr = np.array(rows)
    np.savetxt("../results/test01_q.csv", arr, delimiter=",",
               header="q,K", comments="")
    print("test01_vs_q done")


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    if task in ("ref", "all"):
        reference_trajectory()
    if task in ("bifc", "all"):
        bifurcation_c()
    if task in ("bifq", "all"):
        bifurcation_q()
    if task in ("lle", "all"):
        lyapunov_vs_q()
    if task in ("t01", "all"):
        test01_vs_q()
