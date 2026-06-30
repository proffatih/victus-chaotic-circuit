"""
Generate all publication-grade figures (vector PDF + 300-dpi PNG).
Colourblind-safe palette (Wong 2011 / Okabe-Ito).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 0.9,
})
# Okabe-Ito colourblind-safe palette
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "yellow": "#F0E442", "black": "#000000"}

FIG = "../figures/"
RES = "../results/"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG + name + ".pdf", bbox_inches="tight")
    fig.savefig(FIG + name + ".png", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------- Fig 1: phase portraits (q=0.9) ----------
def fig_phase():
    d = np.load(RES + "trajectory_q090.npz")
    x, y, z = d["x"], d["y"], d["z"]
    fig = plt.figure(figsize=(9, 3.0))
    ax1 = fig.add_subplot(131); ax1.plot(x, y, color=CB["blue"], lw=0.4)
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$"); ax1.set_title("(a) $x$–$y$")
    ax2 = fig.add_subplot(132); ax2.plot(x, z, color=CB["red"], lw=0.4)
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$z$"); ax2.set_title("(b) $x$–$z$")
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot(x, y, z, color=CB["green"], lw=0.3)
    ax3.set_xlabel("$x$"); ax3.set_ylabel("$y$"); ax3.set_zlabel("$z$")
    ax3.set_title("(c) 3D attractor"); ax3.grid(False)
    save(fig, "fig_phase_portraits")


# ---------- Fig 2: time series ----------
def fig_timeseries():
    d = np.load(RES + "trajectory_q090.npz")
    t = d["t"]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(t, d["x"], color=CB["blue"], label="$x(t)$")
    ax.plot(t, d["z"], color=CB["red"], label="$z(t)$", alpha=0.8)
    ax.set_xlabel("time $t$ (a.u.)"); ax.set_ylabel("state")
    ax.set_xlim(t.min(), t.min() + 30)
    ax.legend(ncol=2, loc="upper right")
    save(fig, "fig_timeseries")


# ---------- Fig 3: bifurcation vs c and vs q ----------
def fig_bifurcation():
    bc = np.loadtxt("../data/bifurcation_c.csv", delimiter=",", skiprows=1)
    bq = np.loadtxt("../data/bifurcation_q.csv", delimiter=",", skiprows=1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.plot(bc[:, 0], bc[:, 1], ".", ms=0.4, color=CB["blue"], alpha=0.5)
    a1.set_xlabel("parameter $c$"); a1.set_ylabel("$x_{\\max}$")
    a1.set_title("(a) Bifurcation vs $c$ ($q=0.95$)")
    a2.plot(bq[:, 0], bq[:, 1], ".", ms=0.4, color=CB["red"], alpha=0.5)
    a2.set_xlabel("fractional order $q$"); a2.set_ylabel("$x_{\\max}$")
    a2.set_title("(b) Bifurcation vs $q$ ($c=28$)")
    save(fig, "fig_bifurcation")


# ---------- Fig 4: Lyapunov + 0-1 test ----------
def fig_lyap_01():
    ly = np.loadtxt(RES + "lyapunov_q.csv", delimiter=",", skiprows=1)
    t1 = np.loadtxt(RES + "test01_q.csv", delimiter=",", skiprows=1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.0))
    a1.plot(ly[:, 0], ly[:, 1], "o-", ms=3, color=CB["blue"])
    a1.axhline(0, color=CB["black"], lw=0.6, ls="--")
    a1.set_xlabel("fractional order $q$")
    a1.set_ylabel("largest Lyapunov exponent $\\lambda_1$")
    a1.set_title("(a) LLE (Benettin)")
    a2.plot(t1[:, 0], t1[:, 1], "s-", ms=3, color=CB["green"])
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("fractional order $q$"); a2.set_ylabel("0–1 test $K$")
    a2.set_title("(b) 0–1 test for chaos")
    save(fig, "fig_lyapunov_test01")


# ---------- Fig 5: FOC frequency response ----------
def fig_foc():
    d = np.load(RES + "foc_response.npz")
    f = d["f"]; mag = d["mag"]; ph = d["phase"]; target = float(d["target"])
    q = float(d["q"]); Cq = float(d["Cq"])
    ideal = 1.0 / (Cq * (2 * np.pi * f) ** q)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.0))
    a1.loglog(f, mag, color=CB["blue"], label="RC ladder")
    a1.loglog(f, ideal, "--", color=CB["red"], label="ideal CPE $\\omega^{-q}$")
    a1.set_xlabel("frequency (Hz)"); a1.set_ylabel("|Z| ($\\Omega$)")
    a1.set_title("(a) FOC magnitude"); a1.legend()
    a2.semilogx(f, ph, color=CB["blue"], label="RC ladder")
    a2.axhline(target, ls="--", color=CB["red"],
               label="target $-q\\cdot90^\\circ$")
    a2.set_ylim(-95, -55)
    a2.set_xlabel("frequency (Hz)"); a2.set_ylabel("phase (deg)")
    a2.set_title("(b) FOC phase"); a2.legend()
    save(fig, "fig_foc_response")


# ---------- Fig 6: circuit attractor vs numerical ----------
def fig_circuit():
    cd = np.loadtxt("../data/ngspice_chen.csv")
    SX = 5.0
    t = cd[:, 0]; m = t > 0.02
    X = cd[m, 1] * SX; Z = cd[m, 3] * SX
    nd = np.load(RES + "trajectory_q090.npz")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.plot(nd["x"], nd["z"], color=CB["blue"], lw=0.4)
    a1.set_xlabel("$x$"); a1.set_ylabel("$z$")
    a1.set_title("(a) Numerical (ABM, $q=0.9$)")
    a2.plot(X, Z, color=CB["red"], lw=0.4)
    a2.set_xlabel("$x$ (V$\\times$%g)" % SX); a2.set_ylabel("$z$ (V$\\times$%g)" % SX)
    a2.set_title("(b) ngspice circuit")
    save(fig, "fig_circuit_attractor")


# ---------- Fig 7: encryption pipeline (images + histograms) ----------
def fig_encryption():
    name = "camera"
    d = np.load(RES + "enc_%s.npz" % name)
    plain, cipher, dec = d["plain"], d["cipher"], d["dec"]
    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    ax[0, 0].imshow(plain, cmap="gray"); ax[0, 0].set_title("(a) plaintext")
    ax[0, 1].imshow(cipher, cmap="gray"); ax[0, 1].set_title("(b) ciphertext")
    ax[0, 2].imshow(dec, cmap="gray"); ax[0, 2].set_title("(c) decrypted")
    for a in ax[0]:
        a.axis("off")
    ax[1, 0].hist(plain.flatten(), bins=256, color=CB["blue"])
    ax[1, 0].set_title("(d) plaintext histogram")
    ax[1, 1].hist(cipher.flatten(), bins=256, color=CB["red"])
    ax[1, 1].set_title("(e) ciphertext histogram")
    # correlation scatter
    rng = np.random.default_rng(0)
    H, W = plain.shape
    ys = rng.integers(0, H - 1, 2000); xs = rng.integers(0, W - 1, 2000)
    ax[1, 2].plot(plain[ys, xs], plain[ys, xs + 1], ".", ms=1,
                  color=CB["blue"], label="plain", alpha=0.5)
    ax[1, 2].plot(cipher[ys, xs], cipher[ys, xs + 1], ".", ms=1,
                  color=CB["red"], label="cipher", alpha=0.5)
    ax[1, 2].set_title("(f) adjacent-pixel corr. (H)")
    ax[1, 2].set_xlabel("pixel $(i,j)$"); ax[1, 2].set_ylabel("pixel $(i,j{+}1)$")
    ax[1, 2].legend(markerscale=6)
    for a in ax[1, :2]:
        a.set_xlabel("intensity"); a.set_ylabel("count")
    save(fig, "fig_encryption")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["phase", "ts", "bif", "lyap", "foc", "circuit", "enc"]
    if "phase" in which: fig_phase()
    if "ts" in which: fig_timeseries()
    if "bif" in which: fig_bifurcation()
    if "lyap" in which: fig_lyap_01()
    if "foc" in which: fig_foc()
    if "circuit" in which: fig_circuit()
    if "enc" in which: fig_encryption()
