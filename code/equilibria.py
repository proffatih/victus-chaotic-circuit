"""
Equilibria and stability analysis for the fractional-order Chen system.
Computes equilibrium points, Jacobian eigenvalues, and the fractional
stability condition |arg(lambda)| > q*pi/2 (Tavazoei & Haeri 2008).
Outputs results/equilibria.json
"""
import json
import numpy as np
import sympy as sp

a, b, c = 35.0, 3.0, 28.0
q = 0.95

# Equilibria: solve f(x,y,z)=0
xs, ys, zs = sp.symbols('x y z', real=True)
eqs = [a*(ys-xs), (c-a)*xs - xs*zs + c*ys, xs*ys - b*zs]
sols = sp.solve(eqs, [xs, ys, zs], dict=True)

def jac(x, y, z):
    return np.array([
        [-a,        a,    0.0],
        [c - a - z, c,   -x  ],
        [y,         x,   -b  ],
    ])

results = {"system": "fractional Chen", "params": {"a": a, "b": b, "c": c},
           "q": q, "equilibria": []}

for s in sols:
    xv = float(sp.re(s[xs])); yv = float(sp.re(s[ys])); zv = float(sp.re(s[zs]))
    J = jac(xv, yv, zv)
    ev = np.linalg.eigvals(J)
    # fractional stability: stable iff all |arg(lambda)| > q*pi/2
    args = np.abs(np.angle(ev))
    thresh = q * np.pi / 2.0
    stable = bool(np.all(args > thresh))
    # saddle index / whether it can support chaos (unstable focus-node)
    results["equilibria"].append({
        "point": [round(xv, 6), round(yv, 6), round(zv, 6)],
        "eigenvalues_real": [round(float(e.real), 5) for e in ev],
        "eigenvalues_imag": [round(float(e.imag), 5) for e in ev],
        "min_abs_arg_deg": round(float(np.min(args) * 180 / np.pi), 4),
        "frac_stability_threshold_deg": round(thresh * 180 / np.pi, 4),
        "fractionally_stable": stable,
    })

with open("../results/equilibria.json", "w") as fh:
    json.dump(results, fh, indent=2)

print(json.dumps(results, indent=2))
