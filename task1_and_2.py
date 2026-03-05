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
fig.show()

# %% Determine stiffness the cross sections (simplified beam)
abd_bld = np.stack([np.linalg.inv(section.laminate.ABD_matrix)
                    for section in laminates])

E_f = 1 / abd_bld[:, 0, 0]/t_spar
d = t_bld+t_spar
I_bld = t_spar * w_spar * d**2 / 2
EI_bld = E_f * I_bld

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
