"""
Fractional-order capacitor (FOC) realised as a Foster-I RC ladder.

We synthesise a band-limited constant-phase element (CPE) of order q whose
impedance approximates  Z_cpe(jw) = 1 / (C_q (jw)^q)  over the working band of
the analog oscillator. The network is a Foster-I structure:

    Z(s) = R_s + sum_{i=1}^{n}  R_i / (1 + s R_i C_i)

with logarithmically spaced relaxation frequencies w_i = 1/(R_i C_i). The
branch resistances R_i are obtained by a non-negative least-squares (NNLS) fit
of the network impedance to the target CPE impedance over the band; the C_i are
then C_i = 1/(R_i w_i). This is the classical RC-ladder CPE design (Valsa &
Vlach 2013; Oustaloup 2000) and yields strictly positive, buildable components.

Output:
  results/foc_ladder.json   component values + phase accuracy
  results/foc_response.npz  frequency response for plotting
"""
import json
import numpy as np
from scipy.optimize import nnls


def design_foc(q=0.9, Cq=1e-6, fb=10.0, fh=10e3, n_branch=9):
    """Fit Foster-I RC ladder to CPE of order q over [fb, fh]."""
    wb, wh = 2 * np.pi * fb, 2 * np.pi * fh
    # relaxation frequencies of the branches, log-spaced, slightly outside band
    w_relax = np.logspace(np.log10(wb) - 0.3, np.log10(wh) + 0.3, n_branch)

    # frequency grid for fitting
    f = np.logspace(np.log10(fb), np.log10(fh), 200)
    w = 2 * np.pi * f
    s = 1j * w
    Ztar = 1.0 / (Cq * (s) ** q)        # target CPE impedance

    # Basis impedances: each branch i contributes R_i * b_i(s),
    #   b_i(s) = 1/(1 + s/w_i)  (since C_i=1/(R_i w_i) -> R_i/(1+sR_iC_i)=R_i/(1+s/w_i))
    # plus a series R_s term (basis = 1).
    B = np.zeros((len(w), n_branch + 1), dtype=complex)
    B[:, 0] = 1.0
    for i, wi in enumerate(w_relax):
        B[:, i + 1] = 1.0 / (1.0 + s / wi)

    # NNLS on stacked real/imag, weighted by 1/|Ztar| so the fit minimises
    # RELATIVE error uniformly across the 3-decade band (otherwise the
    # low-frequency, high-impedance points dominate and collapse the network).
    Wt = 1.0 / np.abs(Ztar)
    Bw = B * Wt[:, None]
    Ztarw = Ztar * Wt
    A = np.vstack([Bw.real, Bw.imag])
    bvec = np.concatenate([Ztarw.real, Ztarw.imag])
    coef, _ = nnls(A, bvec, maxiter=20000)

    Rs = coef[0]
    R = coef[1:]
    # drop negligible branches
    mask = R > 1e-9 * R.max()
    R = R[mask]
    w_relax = w_relax[mask]
    C = 1.0 / (R * w_relax)

    # evaluate full response over wide band
    fp = np.logspace(np.log10(fb) - 1, np.log10(fh) + 1, 500)
    sp = 1j * 2 * np.pi * fp
    Z = np.full_like(sp, Rs, dtype=complex)
    for Ri, Ci in zip(R, C):
        Z = Z + Ri / (1.0 + sp * Ri * Ci)
    phase = np.angle(Z, deg=True)
    target = -q * 90.0
    band = (fp >= fb) & (fp <= fh)
    mean_phase = float(np.mean(phase[band]))
    ripple = float(np.std(phase[band]))
    # magnitude relative error vs ideal CPE in band
    Ztar_band = 1.0 / (Cq * (1j * 2 * np.pi * fp[band]) ** q)
    mag_relerr = float(np.mean(np.abs(np.abs(Z[band]) - np.abs(Ztar_band))
                               / np.abs(Ztar_band)))
    return dict(Rs=Rs, R=R, C=C, w_relax=w_relax, fp=fp, Z=Z, phase=phase,
                target=target, mean_phase=mean_phase, ripple=ripple,
                mag_relerr=mag_relerr, q=q, Cq=Cq, fb=fb, fh=fh)


def main():
    d = design_foc(q=0.9, Cq=1e-6, fb=10.0, fh=10e3, n_branch=9)
    out = {
        "order_q": d["q"],
        "pseudo_capacitance_F_s^(q-1)": d["Cq"],
        "approx_band_Hz": [d["fb"], d["fh"]],
        "topology": "Foster-I: series R_s + parallel (R_i || C_i) branches",
        "Rs_ohm": round(float(d["Rs"]), 4),
        "n_branches": int(len(d["R"])),
        "R_ohm": [round(float(r), 4) for r in d["R"]],
        "C_farad": [float(c) for c in d["C"]],
        "relaxation_freq_Hz": [round(float(wr / (2 * np.pi)), 4) for wr in d["w_relax"]],
        "target_phase_deg": d["target"],
        "mean_phase_in_band_deg": round(d["mean_phase"], 3),
        "phase_ripple_std_deg": round(d["ripple"], 3),
        "magnitude_mean_rel_error": round(d["mag_relerr"], 4),
    }
    with open("../results/foc_ladder.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez("../results/foc_response.npz", f=d["fp"], mag=np.abs(d["Z"]),
             phase=d["phase"], Rs=d["Rs"], R=d["R"], C=d["C"],
             target=d["target"], q=d["q"], Cq=d["Cq"])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
