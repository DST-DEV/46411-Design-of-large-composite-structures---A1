from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from turbine_blade import TurbineBlade
from direct_stiffness import Frame, BeamElement, ManualCrossSection
import classical_laminate_theory as clt
import scivis 

# %% Create blade from input data
data_dir = Path(__file__).parent / "_data"
blade = TurbineBlade()

# Create laminates at blade section specified by the layup
laminates = blade.create_laminates()
span_coords = np.stack([[section.r_start for section in laminates],
                        [section.r_end for section in laminates]], axis=1)
t_spar = np.array([section.thickness for section in laminates])

# Calculate blade thickness & chord and spar width from splines
t_bld = np.average(blade.thickness_abs(span_coords), axis=1)
chord_bld = np.average(blade.chord(span_coords), axis=1)
w_spar = np.average(blade.w_spar(span_coords), axis=1)

x = np.zeros(len(laminates) + 1)
x[:-1] = span_coords[:, 0]
x[-1] = span_coords[-1, 1]
fig, ax, _ = scivis.plot_line(x, blade.w_spar(x)*1e3, ax_labels=["r", "b"],
                              ax_units=["m", "mm"])
import matplotlib.pyplot as plt
plt.show()

fig, ax, _ = scivis.plot_line(
    x, blade.w_spar(x)*1e3,
    ax_labels=["r", "b"],
    ax_units=["m", "mm"]
)
plt.show()

y = np.zeros_like(x)
y[:-1] = chord_bld
y[-1] = chord_bld[-1]

fig, ax, _ = scivis.plot_line(
    x, y,
    ax_labels=["r", "chord"],
    ax_units=["m", "m"]
)
plt.show()

t_rel = t_bld / chord_bld

y = np.zeros_like(x)
y[:-1] = t_rel
y[-1] = t_rel[-1]

fig, ax, _ = scivis.plot_line(
    x, y,
    ax_labels=["r", "t/c"],
    ax_units=["m", "-"]
)
plt.show()

y = np.zeros_like(x)
y[:-1] = t_spar
y[-1] = t_spar[-1]

fig, ax, _ = scivis.plot_line(
    x, y*1e3,
    ax_labels=["r", "h_spar"],
    ax_units=["m", "mm"]
)
plt.show()

# %% Determine stiffness the cross sections (simplified beam)
# I replaced this block code 
#abd_bld = np.stack([np.linalg.inv(section.laminate.ABD_matrix)
#                    for section in laminates])
#E_f = 1 / abd_bld[:, 0, 0]/t_spar
#Since it calculate Ef in the wrong way.
# --- REPLACE ---
A_inv = np.stack([np.linalg.inv(section.laminate.ABD_matrix[:3, :3])
                  for section in laminates])

E_f = 1.0 / (A_inv[:, 0, 0] * t_spar)
# --- END REPLACEMENT ---

#I_bld = t_spar * w_spar * d**2 / 2
#EI_bld = E_f * I_bld

# distanza tra i due spar caps
d = t_bld + t_spar

# momento d'inerzia semplificato della sezione
I_bld = t_spar * w_spar * d**2 / 2

# rigidezza flessionale
EI_bld = E_f * I_bld

# EI_bld: shape (len(laminates),)

x = np.zeros(len(laminates) + 1)
x[:-1] = span_coords[:, 0]     # r_start
x[-1]  = span_coords[-1, 1]    # last r_end

y = np.zeros_like(x)
y[:-1] = EI_bld
y[-1]  = EI_bld[-1]            #Until the last point

fig, ax, _ = scivis.plot_line(
    x, y,
    ax_labels=["r", "EI_bld"],
    ax_units=["m", "N·m²"]
)

import matplotlib.pyplot as plt
plt.show()

# %% Solve the beam for the given loads
# Set up the frame system
profiles = [ManualCrossSection(A=2*w_spar[i]*d[i],  iy=np.inf, iz=I_bld[i],
                               j=np.inf)
            for i in range(len(I_bld))]

beams = [BeamElement(cross_section=profiles[i], E=E_f[i], G=np.inf,
                     idx=(i+1, i+2),
                     coords=((span_coords[i, 0], 0, 0),
                             (span_coords[i, 1], 0, 0)))
         for i in range(len(span_coords))]

frame = Frame(beams)

# Set up the loading
loads = pd.read_csv(data_dir / "loads.csv")
idx_nearest = [(np.abs(span_coords[:, 0] - coord_load)).argmin()
               for coord_load in loads["Radial position [m]"]]

F = np.zeros((frame.n_nodes, 3))
F[idx_nearest, 1] = loads["Force [N]"]

# Fix dofs of blade root in space
fixed_dofs = ("u_x1", "u_z1", "theta_y1")

# Solve the frame system
U, R = frame.solve_fea_system(F=F.flatten(), fixed_dofs=fixed_dofs,
                              is_3d=False)
U = U.reshape((frame.n_nodes, 3))
R = R.reshape((frame.n_nodes, 3))

import numpy as np
import matplotlib.pyplot as plt

# Node x-positions (should match frame nodes)
x_nodes = x.copy()

# Nodal transverse loads (same direction you used: column 1)
F_tr = np.zeros(frame.n_nodes)
F_tr[idx_nearest] = loads["Force [N]"].to_numpy()

# Root reactions from solver
V0 = R[0, 1]          # root transverse reaction
M0 = R[0, 2]          # root reaction moment (check if this is the correct component)

# Build shear and moment along x using equilibrium
V = np.zeros_like(x_nodes, dtype=float)
M = np.zeros_like(x_nodes, dtype=float)

for i, xi in enumerate(x_nodes):
    mask = x_nodes[idx_nearest] <= xi
    P_sum = F_tr[idx_nearest][mask].sum()

    # Shear: V(x) = V0 - sum(P up to x)
    V[i] = V0 - P_sum

    # Moment: M(x) = M0 + V0*(x-0) - sum(P_j*(x-x_j))
    xj = x_nodes[idx_nearest][mask]
    Pj = F_tr[idx_nearest][mask]
    M[i] = M0 + V0 * xi - np.sum(Pj * (xi - xj))

# Plot V and M
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x_nodes, V)
ax.set_xlabel("x [m]")
ax.set_ylabel("Shear V [N]")
ax.grid()
plt.show()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x_nodes, M)
ax.set_xlabel("x [m]")
ax.set_ylabel("Bending moment M [N·m]")
ax.grid()
plt.show()

# %% Task 1: Tower clearance
U_tip = U[-1, :]
print(f"Tip deflection {U_tip[1]:.2f} m")

# Calculate distance of tip to tower
tilt = np.deg2rad(5)
cone = np.deg2rad(2.5)
tower_clearance = 18.26
U_corrected = U[:, 1] * np.cos(tilt + cone)

print(f"Tower clearance: {tower_clearance-U_corrected[-1]:.2f} m")

# Plot deflection over blade span
fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(x, U[:, 1])
ax.set_xlabel(r"$x$ [m]")
ax.set_ylabel(r"$u$ [m]")
ax.grid()

# EI along nodes (extend last value)
EI_nodes = np.zeros_like(x_nodes, dtype=float)
EI_nodes[:-1] = EI_bld
EI_nodes[-1] = EI_bld[-1]

# Curvature kappa = M / EI
kappa = M / EI_nodes   # [1/m]

# Cumulative trapezoidal integration without scipy
def cumtrapz(y, x):
    out = np.zeros_like(y, dtype=float)
    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out

# Fixed root -> theta(0)=0 and w(0)=0
theta = cumtrapz(kappa, x_nodes)   # rotation [rad]
w_bt  = cumtrapz(theta, x_nodes)   # deflection [m]

# Plots
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x_nodes, kappa)
ax.set_xlabel("x [m]")
ax.set_ylabel("Curvature κ [1/m]")
ax.grid()
plt.show()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x_nodes, theta)
ax.set_xlabel("x [m]")
ax.set_ylabel("Rotation θ [rad]")
ax.grid()
plt.show()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x_nodes, w_bt, label="Beam theory (integrated)")
ax.plot(x_nodes, U[:, 1], "--", label="FEA displacement U[:,1]")
ax.set_xlabel("x [m]")
ax.set_ylabel("Deflection w [m]")
ax.grid()
ax.legend()
plt.show()

# %% Task 2: Buckling and strength
# %%% Task 2a: Buckling
# Calculate cross sectional stress from beam bending
udd = np.gradient(np.gradient(U[:, 1], x), x)
sigma = np.zeros((frame.n_nodes, 2))
sigma[:-1, 0] = udd[:-1] * E_f * (t_bld/2+t_spar)
sigma[1:, 1] = udd[1:] * E_f * (t_bld/2+t_spar)

idx_crit = np.argmax(sigma, axis=1)

# Calculate ABD matrix for cross sectios
ABD_bld = np.stack([section.laminate.ABD_matrix for section in laminates])
D_bld = ABD_bld[:, 3:6, 3:6]

# Calculate critical buckling load
def P_func(m, a, b, D):
    return np.pi**2 * (D[...,0,0]*(m/a)**2
                       + 2*(D[...,0,1] + 2*D[:,2,2])*(1/b)**2
                       + D[...,1,1]*(a/m)**2*(1/b)**4)

l_sections = np.array([section.length for section in laminates])
P_buckling = np.stack([P_func(m=m+1, a=l_sections, b=w_spar, D=D_bld)
                      for m in range(10)],
                      axis = 0)

P_cr_buckling = np.min(P_buckling, axis=0)

fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(x, sigma)
ax.set_xlabel(r"$x$ [m]")
ax.set_ylabel(r"$\sigma$ [N/m^2]")
ax.grid()

fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(span_coords[:, 0], P_cr_buckling)
ax.set_xlabel(r"$x$ [m]")
ax.set_ylabel(r"$P_{cr}$ [N]")
ax.grid()

# %%% Task 2b: Strength
sigma1_max = np.max([sigma[:-1, 0], sigma[1:, 1]], axis=0)

failure_tsaihill = np.zeros(len(laminates))
failure_tsaiwu = np.zeros(len(laminates))

for i in range(len(laminates)):
    failure_tsaihill[i] = \
        np.max([
            clt.failure.TsaiHill.failure_index(
                stress=(sigma1_max[i], 0, 0),
                **laminates[i].laminate.plies[j].material.strength_as_dict())
            for j in range(len(laminates[i].laminate.plies))]
            )

    failure_tsaiwu[i] = \
        np.max([
            clt.failure.TsaiWu.failure_index(
                stress=(sigma1_max[i], 0, 0),
                **laminates[i].laminate.plies[j].material.strength_as_dict())
            for j in range(len(laminates[i].laminate.plies))]
            )

if any(failure_tsaiwu>=1) or any(failure_tsaihill>=1):
    print("Failure detected.")
