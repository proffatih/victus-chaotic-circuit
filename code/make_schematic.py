"""Block-level schematic of the analog fractional-Chen oscillator and the
Foster-I fractional-order capacitor. Vector PDF + PNG."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "grey": "#666666"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.6),
                               gridspec_kw={"width_ratios": [1.7, 1]})

# ---------------- (a) signal-flow block diagram ----------------
axL.set_xlim(0, 10); axL.set_ylim(0, 6); axL.axis("off")
axL.set_title("(a) Analog signal-flow realisation", fontsize=11)


def block(ax, x, y, w, h, text, color):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       fc=color, ec="black", alpha=0.85, lw=1.0)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=8.5, color="white", weight="bold")


def arrow(ax, p1, p2, text=""):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                        color=CB["grey"], lw=1.0)
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.18, text, fontsize=8, ha="center")


# three fractional integrators
block(axL, 1.0, 4.3, 1.8, 0.9, r"$\int^{q}$  $\to X$", CB["blue"])
block(axL, 1.0, 2.6, 1.8, 0.9, r"$\int^{q}$  $\to Y$", CB["green"])
block(axL, 1.0, 0.9, 1.8, 0.9, r"$\int^{q}$  $\to Z$", CB["red"])
# summers
block(axL, 4.2, 4.3, 1.5, 0.9, r"$\Sigma$  $a(Y{-}X)$", CB["grey"])
block(axL, 4.2, 2.6, 1.5, 0.9, r"$\Sigma$ (c-a)X" + "\n" + r"$-XZ+cY$", CB["grey"])
block(axL, 4.2, 0.9, 1.5, 0.9, r"$\Sigma$  $XY{-}bZ$", CB["grey"])
# multipliers
block(axL, 7.1, 1.7, 1.5, 0.8, r"$\times$  $XZ$", CB["orange"])
block(axL, 7.1, 0.4, 1.5, 0.8, r"$\times$  $XY$", CB["orange"])

# feedback arrows
arrow(axL, (5.7, 4.75), (8.9, 4.75))
arrow(axL, (8.9, 4.75), (8.9, 5.5)); arrow(axL, (8.9, 5.5), (0.4, 5.5))
arrow(axL, (0.4, 5.5), (0.4, 4.75)); arrow(axL, (0.4, 4.75), (1.0, 4.75), "")
arrow(axL, (2.8, 4.75), (4.2, 4.75), "$X$")
arrow(axL, (2.8, 3.05), (4.2, 3.05), "$Y$")
arrow(axL, (2.8, 1.35), (4.2, 1.35), "$Z$")
arrow(axL, (5.7, 3.05), (1.0, 3.05))   # back to Y integrator
arrow(axL, (5.7, 1.35), (1.0, 1.35))   # back to Z integrator
arrow(axL, (8.6, 2.1), (5.7, 2.6))     # XZ into Y summer
arrow(axL, (8.6, 0.8), (5.7, 1.1))     # XY into Z summer
axL.text(5.0, 5.75, "fractional integrators use FOC feedback (panel b)",
         fontsize=7.5, ha="center", style="italic", color=CB["grey"])

# ---------------- (b) Foster-I FOC ladder ----------------
with open("../results/foc_ladder.json") as fh:
    foc = json.load(fh)
axR.set_xlim(0, 10); axR.set_ylim(0, 6); axR.axis("off")
axR.set_title("(b) Foster-I fractional capacitor\n($q=%.2f$, %d branches)"
              % (foc["order_q"], foc["n_branches"]), fontsize=11)
# draw series node with parallel R||C branches
n = min(foc["n_branches"], 5)
axR.plot([0.5, 9.5], [5.2, 5.2], "k-", lw=1.2)  # top rail
axR.plot([0.5, 9.5], [0.6, 0.6], "k-", lw=1.2)  # bottom rail
axR.text(0.2, 5.4, "a", fontsize=10); axR.text(9.4, 5.4, "b", fontsize=10)
xs = np.linspace(1.4, 8.6, n)
for i, x in enumerate(xs):
    axR.plot([x, x], [3.2, 5.2], "k-", lw=1.0)
    axR.plot([x, x], [0.6, 2.6], "k-", lw=1.0)
    # resistor box
    axR.add_patch(plt.Rectangle((x - 0.22, 2.7), 0.44, 0.55, fc="white",
                                ec=CB["blue"], lw=1.2))
    axR.text(x, 2.95, "$R_%d$" % (i + 1), fontsize=7, ha="center")
    # capacitor plates
    axR.plot([x - 0.25, x + 0.25], [2.45, 2.45], color=CB["red"], lw=2)
    axR.plot([x - 0.25, x + 0.25], [2.25, 2.25], color=CB["red"], lw=2)
    axR.plot([x, x], [2.6, 2.45], "k-", lw=1.0)
    axR.plot([x, x], [2.25, 0.6], "k-", lw=1.0)
    axR.text(x + 0.3, 2.3, "$C_%d$" % (i + 1), fontsize=7)
axR.text(5.0, 0.0,
         "$Z(s)\\approx 1/(C_q\\,(j\\omega)^{q})$, phase $\\approx %.1f^\\circ$"
         % foc["mean_phase_in_band_deg"], fontsize=8, ha="center")

fig.tight_layout()
fig.savefig("../figures/fig_schematic.pdf", bbox_inches="tight")
fig.savefig("../figures/fig_schematic.png", bbox_inches="tight", dpi=300)
print("saved fig_schematic")
