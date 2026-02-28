from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from turbine_blade import TurbineBlade
from direct_stiffness import Frame, BeamElement, ManualCrossSection

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

# %% Determine stiffness the cross sections (simplified beam)
abd_bld = np.stack([np.linalg.inv(section.laminate.ABD_matrix)
                    for section in laminates])

E_f = 1 / abd_bld[:, 0, 0]/t_spar
d = t_bld+t_spar
I_bld = t_spar * w_spar * d**2 / 2
EI_bld = E_f * I_bld

# %% Calculate the bending
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
U = U.reshape((frame.n_nodes, 3)) #TODO: include tilt and precone

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
x = np.zeros(len(U))
x[:-1] = span_coords[:, 0]
x[-1] = span_coords[-1, 1]
ax.plot(x, U[:, 1])
ax.set_xlabel("x [m]")
ax.set_ylabel("u [m]")
ax.grid()
