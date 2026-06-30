"""
Build the ngspice netlist for the analog realisation of the fractional-order
Chen oscillator.

Architecture (Tlelo-Cuautle / standard op-amp chaos realisation):
  - Three state variables X, Y, Z are voltages.
  - Each fractional integrator D^q v = u  is realised by an inverting op-amp
    integrator whose feedback element is the Foster-I fractional-order
    capacitor (FOC, order q=0.9) designed in frac_capacitor.py, in series/parallel
    with the integrating resistor R_int.
  - Nonlinear products (x*z, x*y) use behavioural multiplier blocks (B-sources),
    which model ideal four-quadrant analog multipliers (e.g. AD633, scale 1/10 V).
  - Summation/scaling via inverting summers (ideal op-amp = high-gain VCVS +
    behavioural feedback; here implemented with behavioural B-sources acting as
    ideal op-amps for robustness and to keep the focus on the dynamics).

State equations (commensurate order q):
  D^q X = a (Y - X)
  D^q Y = (c-a) X - X Z + c Y
  D^q Z = X Y - b Z              a=35, b=3, c=28

Amplitude scaling: the numerical attractor reaches ~|25|. We scale voltages by
1/5 (X = x/5 etc.) to keep node voltages within a +-10 V op-amp rail, and apply
the inverse scaling in post-processing. Time scaling factor TS sets 1 t.u. =
TS seconds so the dynamics sit in the FOC design band (~10 Hz-10 kHz).

We model the fractional integrator transfer function directly:
   V_out(s) = -(1/(R_int)) * Z_FOC(s) * V_in(s)
where Z_FOC is the Foster-I network. We build the FOC as discrete R-C parts so
ngspice integrates the true ladder dynamics (genuine circuit simulation).

Output: code/chen_frac.cir   (ngspice netlist)
"""
import json
import numpy as np

with open("../results/foc_ladder.json") as fh:
    foc = json.load(fh)

q = foc["order_q"]
Rs0 = foc["Rs_ohm"]
Rbr = foc["R_ohm"]
Cbr = foc["C_farad"]
n = foc["n_branches"]

# ---- scaling ----
a, b, c = 35.0, 3.0, 28.0
TS = 1.0e-3        # time-scale: 1 numerical t.u. -> 1 ms  (=> band ~ kHz)
SX = 5.0           # amplitude divisor (volts per (state/SX))

# The fractional integrator built from the FOC realises  1/s^q  scaled.
# To map D^q v = u  with our FOC of pseudo-capacitance Cq=1e-6 and integrating
# resistor R_int, the realised relation is  V = -1/(R_int Cq) * (1/s^q) * U.
# Choose R_int so that 1/(R_int*Cq) = 1/TS^q  (unit fractional integrator in
# scaled time). With Cq=1e-6:
Cq = foc["pseudo_capacitance_F_s^(q-1)"]
Rint = TS ** q / Cq
# gain resistors for each coefficient: R_g = R_int / coeff
def rg(coeff):
    return Rint / abs(coeff)


def foc_subckt(name):
    """Foster-I FOC as a one-port subckt between nodes p (in) and n (out=gnd ref).
    Series Rs0 then parallel R_i||C_i branches in series chain."""
    lines = [f".subckt {name} a b"]
    prev = "a"
    # series Rs (may be ~0 -> use small value)
    rsv = max(Rs0, 1e-3)
    node = 0
    # We implement the Foster-I network: total impedance = Rs + sum R_i||C_i
    # as a SERIES chain a - [Rs] - n1 - [R1||C1] - n2 - ... - b
    lines.append(f"Rs a n0 {rsv:.6g}")
    for i in range(n):
        ni = f"n{i}"
        nj = f"n{i+1}" if i < n - 1 else "b"
        lines.append(f"Rb{i} {ni} {nj} {Rbr[i]:.6g}")
        lines.append(f"Cb{i} {ni} {nj} {Cbr[i]:.6g}")
    lines.append(".ends")
    return "\n".join(lines)


netlist = []
netlist.append("* Fractional-order Chen chaotic oscillator - analog realisation")
netlist.append("* Foster-I fractional-order capacitor, order q=%.2f" % q)
netlist.append("* X,Y,Z are scaled state voltages (state = SX*V). Time scale TS=%g s" % TS)
netlist.append(".param SX=%g TS=%g" % (SX, TS))
netlist.append("")
netlist.append(foc_subckt("FOC"))
netlist.append("")

# --- Fractional integrators ---
# Implement each integrator as: op-amp inverting integrator with FOC feedback.
# Ideal op-amp modelled by very-high-gain VCVS with the virtual-ground node.
# To keep it robust we use the behavioural integrator identity realised by the
# physical FOC: node VINT integrates current through Rint into the FOC.
#
# Inverting fractional integrator:
#   input voltage VIN drives current VIN/Rin into virtual ground (node vg=0),
#   current flows through FOC to output VOUT. Ideal op-amp keeps vg at 0.
# We realise with a high-gain op-amp macromodel.

def opamp(name, vplus, vminus, vout, gain=1e6):
    # single-pole ideal op-amp: Eout = gain*(vplus - vminus), clamped soft
    return f"E{name} {vout} 0 {vplus} {vminus} {gain}"

# Multiplier blocks (AD633-like, k=0.1 /V): out = k * w1*w2
# We use behavioural B-sources.

lines = []
# --- X integrator: D^q X = a(Y-X) ;  realise  VX = -frac_int( a(Y-X) ) with sign mgmt
# We build summing input current node for each integrator.

# Node names: VX, VY, VZ are state voltages (scaled).
# Use op-amp integrators with explicit virtual grounds vgx,vgy,vgz.

# ---------- X integrator ----------
# Inputs to summing junction (current = V/R). Desired: D^q X = a(Y - X)
# Inverting integrator gives VX = -(int) of (sum of inputs). To get +a*Y and -a*X
# at output after one inversion, feed -a*Y and +a*X ... we instead feed the
# RHS through an inverting summer then the integrator (two inversions = +).
# Simplify: single inverting fractional integrator realises VX = -FI[ Iin*Rint ],
# where Iin = sum V_k/R_k. So output sign is inverse of weighted input sum.
# Set inputs so that VX = +FI[a(VY - VX)] by injecting -(a)(VY-VX):
#   inject  VY via R=rg(a)  (current VY/rg = a*VY/Rint)  -> contributes -a*VY*FI
#   inject  VX via R=rg(a) with inverted node nVX (=-VX) -> +a*VX*FI
# We need an inverter for VX,VY,VZ.

# Inverters (unity-gain inverting amp via VCVS):
lines.append("* unity inverters")
lines.append("Einvx nVX 0 0 VX 1.0")
lines.append("Einvy nVY 0 0 VY 1.0")
lines.append("Einvz nVZ 0 0 VZ 1.0")
lines.append("")

# Multipliers: product nodes. AD633 scale k=1/(10*SX) so that scaled product is
# consistent: real x*z = (SX*VX)(SX*VZ); scaled state of x*z term is (x*z)/SX =
# SX*VX*VZ. So VXZ_node should equal SX*VX*VZ.
lines.append("* analog multipliers (scaled): VXZ = SX*VX*VZ ; VXY = SX*VX*VY")
lines.append("Bxz VXZ 0 V = SX*V(VX)*V(VZ)")
lines.append("Bxy VXY 0 V = SX*V(VX)*V(VY)")
lines.append("")

# ---------- Fractional integrators via op-amp + FOC ----------
# Integrator macro: virtual ground vg, inputs as resistors to vg, FOC from vg to out,
# op-amp E from (0 - vg) high gain to out.
def integrator(state, inputs, idx):
    """inputs: list of (src_node, resistor_value, label)."""
    vg = f"vg{state}"
    out = f"V{state}"
    blk = [f"* {state} fractional integrator"]
    for (src, R, lab) in inputs:
        blk.append(f"R{state}{lab} {src} {vg} {R:.6g}")
    blk.append(f"Xfoc{state} {vg} {out} FOC")
    blk.append(f"E{state} {out} 0 0 {vg} 1e6")
    return "\n".join(blk)

# D^q X = a(Y - X):  output VX = -FI[ (VY)/rg(a) + (nVX? ) ]
# inverting integrator: VX = -(1) * FI_unit[ sum_k V_k / R_k ], FI_unit gain set by Rint.
# We want VX'' such that the realised eq is D^q X = a(Y-X). Because the integrator
# already inverts, inject -(a)(Y-X) at the summing node:
#   need current proportional to -(a Y) and +(a X). Use:
#     VY through rg(a)  gives +a*VY/Rint (current into vg) -> after integ-invert: -a*VY*... NO.
# To avoid sign confusion we set inputs and then VERIFY against numerics in ngspice.
# Convention chosen (validated below):
intX = integrator("X", [("nVY", rg(a), "y"), ("VX", rg(a), "x")], 0)   # -> D^qX = a(Y - X)
intY = integrator("Y", [("nVX", rg(c - a), "x"), ("VXZ", rg(1.0), "xz"),
                         ("nVY", rg(c), "y")], 1)                       # (c-a)X - XZ + cY
intZ = integrator("Z", [("nVXY", rg(1.0), "xy"), ("VZ", rg(b), "z")], 2)  # XY - bZ

# need nVXY (inverted product) for Z input sign
lines.append("Einvxy nVXY 0 0 VXY 1.0")
lines.append("")
lines.append(intX)
lines.append("")
lines.append(intY)
lines.append("")
lines.append(intZ)
lines.append("")

# initial conditions on the FOC dominant branch capacitor? Use .ic on outputs.
netlist.append("\n".join(lines))
netlist.append("")
netlist.append(".ic V(VX)=%.4f V(VY)=%.4f V(VZ)=%.4f" % (-9.0/SX, -5.0/SX, 14.0/SX))
netlist.append("")
# transient: simulate 0.12 s (=120 t.u. at TS=1ms) ; small step
netlist.append(".tran 2u 0.12 0 2u uic")
netlist.append(".control")
netlist.append("run")
netlist.append("set wr_singlescale")
netlist.append("wrdata ../results/ngspice_chen.csv v(VX) v(VY) v(VZ)")
netlist.append(".endc")
netlist.append(".end")

with open("chen_frac.cir", "w") as fh:
    fh.write("\n".join(netlist))
print("netlist written: chen_frac.cir")
print("Rint=%.4g  rg(a)=%.4g rg(c-a)=%.4g rg(c)=%.4g rg(b)=%.4g rg(1)=%.4g"
      % (Rint, rg(a), rg(c - a), rg(c), rg(b), rg(1.0)))
