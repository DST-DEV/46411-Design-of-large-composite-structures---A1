from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from turbine_blade import TurbineBlade
from direct_stiffness import Frame, BeamElement, ManualCrossSection
import classical_laminate_theory as clt
import scivis

# %% User inputs

exp_fld = Path(__file__).parent / "plots"
savefigs = False  # % Selection whether to export the figures as svg files
if savefigs: exp_fld.mkdir(exist_ok=True)

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

# Plot spar cap width vs spanwise position
x = np.zeros(len(laminates) + 1)
x[:-1] = span_coords[:, 0]
x[-1] = span_coords[-1, 1]

x_mid = np.average(span_coords, axis=1)

fig, ax, _ = scivis.plot_line(x, blade.w_spar(x), ax_labels=["r", "b"],
                              ax_units=["m", "m"],
                              exp_fld=exp_fld, fname="spar_width_vs_span",
                              savefig=savefigs)

# Plot chord vs spanwise position
fig, ax, _ = scivis.plot_line(x, blade.chord(x), ax_labels=["r", "c"],
                              ax_units=["m", "m"],
                              exp_fld=exp_fld, fname="chord_vs_span",
                              savefig=savefigs)

# Plot relative thickness vs spanwise position
fig, ax, _ = scivis.plot_line(x, blade.thickness(x), ax_labels=["r", "t/r"],
                              ax_units=["m", None],
                              exp_fld=exp_fld, fname="rel_thickness_vs_span",
                              savefig=savefigs)

# Plot spar cap thickness vs spanwise position
t_spar_plot = np.zeros_like(x)
t_spar_plot[:-1] = t_spar
t_spar_plot[-1] = t_spar[-1]
fig, ax, _ = scivis.plot_line(x, t_spar_plot*1e3, ax_labels=["r", "t_f"],
                              ax_units=["m", "mm"],
                              exp_fld=exp_fld, fname="spar_thickness_vs_span",
                              savefig=savefigs)

# %% Determine stiffness the cross sections (simplified beam)
A_inv = np.stack([np.linalg.inv(section.laminate.ABD_matrix[:3, :3])
                  for section in laminates])

E_f = 1.0 / (A_inv[:, 0, 0] * t_spar)
d = t_bld+t_spar
I_bld = t_spar * w_spar * d**2 / 2
EI_bld = E_f * I_bld

# Plot bending stiffness vs spanwise position
EI_bld_plot = np.zeros_like(x)
EI_bld_plot[:-1] = EI_bld
EI_bld_plot[-1]  = EI_bld[-1]
fig, ax, _ = scivis.plot_line(x, EI_bld_plot, ax_labels=["r", "EI"],
                              ax_units=["m", "Nm^2"],
                              exp_fld=exp_fld, fname="stiffness_vs_span",
                              savefig=savefigs)

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
res_bending = frame.all_beam_fields(U, n_pts=2)

# %% Task 1: Tower clearance
w_tip = res_bending["w"][-1]
print(f"Tip deflection {w_tip:.2f} m")

# Calculate distance of tip to tower
tilt = np.deg2rad(5)
cone = np.deg2rad(2.5)
tower_clearance = 18.26
w_corrected = res_bending["w"] * np.cos(tilt + cone)

print(f"Tower clearance: {tower_clearance-w_corrected[-1]:.2f} m")

# Plot deflection vs spanwise position
fig, ax, _ = scivis.plot_line(res_bending["x"], res_bending["w"],
                              ax_labels=["r", "w"], ax_units=["m", "m"],
                              exp_fld=exp_fld, fname="deflection_vs_span",
                              savefig=savefigs)

# Plot deflection angle vs spanwise position
fig, ax, _ = scivis.plot_line(res_bending["x"], res_bending["theta"],
                              ax_labels=["r", r"\theta"], ax_units=["m", "m"],
                              exp_fld=exp_fld, fname="deflection_angle_vs_span",
                              savefig=savefigs)

# Plot curvature vs spanwise position
fig, ax, _ = scivis.plot_line(res_bending["x"], res_bending["kappa"],
                              ax_labels=["r", r"\kappa"], ax_units=["m", "m"],
                              exp_fld=exp_fld, fname="curvature_vs_span",
                              savefig=savefigs)

# Plot bending moment distribution vs spanwise position
fig, ax, _ = scivis.plot_line(res_bending["x"], res_bending["M"]*1e-6,
                              ax_labels=["r", r"M"], ax_units=["m", "MNm"],
                              exp_fld=exp_fld, fname="bending_moment_vs_span",
                              savefig=savefigs)

# Plot shear force distribution vs spanwise position
fig, ax, _ = scivis.plot_line(res_bending["x"], res_bending["Q"]*1e-6,
                              ax_labels=["r", r"Q"], ax_units=["m", "MN"],
                              exp_fld=exp_fld, fname="shear_force_vs_span",
                              savefig=savefigs)

# %% Task 2: Buckling and strength
# %%% Task 2a: Buckling
# Calculate cross sectional stress from beam bending
sigma = np.zeros(frame.n_nodes)
sigma[:-1] = -res_bending["M"][:-1] / I_bld * (t_bld/2+t_spar)

idx_crit = np.argmax(sigma)

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


# Plot axial stress distribution vs spanwise position
fig, ax, _ = scivis.plot_line(x, sigma*1e-6, ax_labels=["r", r"\sigma"],
                              ax_units=["m", "N/mm^2"],
                              exp_fld=exp_fld, fname="stress_vs_span",
                              savefig=savefigs)

# Plot critical buckling load vs spanwise position
P_cr_buckling_plot = np.zeros_like(x)
P_cr_buckling_plot[:-1] = P_cr_buckling
P_cr_buckling_plot[-1] = P_cr_buckling[-1]
fig, ax, _ = scivis.plot_line(x, P_cr_buckling_plot*1e-6,
                              ax_labels=["r", r"P_{cr}"], ax_units=["m", "MN"],
                              exp_fld=exp_fld, fname="buckling_load_vs_span",
                              savefig=savefigs)

# %%% Task 2b: Strength
failure_tsaihill = np.zeros(len(laminates))
failure_tsaiwu = np.zeros(len(laminates))

for i in range(len(laminates)):
    failure_tsaihill[i] = \
        np.max([
            clt.failure.TsaiHill.failure_index(
                stress=(sigma[i], 0, 0),
                **laminates[i].laminate.plies[j].material.strength_as_dict())
            for j in range(len(laminates[i].laminate.plies))]
            )

    failure_tsaiwu[i] = \
        np.max([
            clt.failure.TsaiWu.failure_index(
                stress=(sigma[i], 0, 0),
                **laminates[i].laminate.plies[j].material.strength_as_dict())
            for j in range(len(laminates[i].laminate.plies))]
            )

if any(failure_tsaiwu>=1) or any(failure_tsaihill>=1):
    print("Failure detected.")

# Plot Tsai-Wu and Tsai-Hill
failure_plot = np.zeros((2, len(x)))
failure_plot[0, :-1] = failure_tsaihill
failure_plot[1, :-1] = failure_tsaiwu
failure_plot[0, -1] = failure_tsaihill[-1]
failure_plot[1, -1] = failure_tsaiwu[-1]
fig, ax, _ = scivis.plot_line(x, failure_plot,
                              plt_labels=["Tsai-Hill", "Tsai-Wu"],
                              ax_labels=["r", r"failure index"],
                              ax_units=["m", None],
                              exp_fld=exp_fld, fname="failure_vs_span",
                              savefig=savefigs)
ax.set_ylabel("Failure index")

plt.show()
