from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from turbine_blade import TurbineBlade
import classical_laminate_theory as clt
import scivis

# %% User inputs

exp_fld = Path(__file__).parent / "plots"
savefigs = True  # % Selection whether to export the figures as svg files
if savefigs: exp_fld.mkdir(exist_ok=True)

# %% Create blade from input data
data_dir = Path(__file__).parent / "_data"
blade = TurbineBlade()

# Create laminates at blade section specified by the layup
laminates = blade.create_laminates()
span_coords = np.stack([[section.r_start for section in laminates],
                        [section.r_end for section in laminates]], axis=1)
l_beams = np.array([section.length for section in laminates])
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

figsize = (16, 10)
plot_profile = "partsize"
plot_scale = .5
rcparams = scivis.rcparams._prepare_rcparams(profile=plot_profile,
                                             scale=plot_scale)
with mpl.rc_context(rcparams):
    fig, ax = scivis.subplots(figsize=figsize)
    fig, ax, _ = scivis.plot_line(x, blade.w_spar(x), ax, ax_labels=["r", "b"],
                                  ax_units=["m", "m"],
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True,
                                  exp_fld=exp_fld,
                                  fname="p1_spar_width_vs_span",
                                  savefig=savefigs)

    # Plot chord vs spanwise position
    fig, ax = scivis.subplots(figsize=figsize)
    fig, ax, _ = scivis.plot_line(x, blade.chord(x), ax, ax_labels=["r", "c"],
                                  ax_units=["m", "m"],
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True,
                                  exp_fld=exp_fld,
                                  fname="p1_chord_vs_span",
                                  savefig=savefigs)

    # Plot relative thickness vs spanwise position
    fig, ax = scivis.subplots(figsize=figsize, profile=plot_profile,
                              scale=plot_scale)
    ax2 = ax.twinx()
    ax2.set_zorder(2)
    ax2_color = "royalblue"
    fig, ax, _ = scivis.plot_line(x, blade.thickness(x), ax,
                                  ax_labels=["r", "t/r"],
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True,
                                  ax_units=["m", None])
    fig, ax2, _ = scivis.plot_line(x, blade.thickness_abs(x), ax2,
                                  ax_labels=["r", "t"],
                                  ax_units=["m", "m"],
                                  colors=ax2_color,
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True)
    ax2.grid(which="both", visible=False)
    ax2.tick_params(axis='y', which = "both", colors=ax2_color)
    ax2.yaxis.label.set_color(ax2_color)
    ax2.spines['right'].set_color(ax2_color)

    # Synchronize font properties
    ax2.yaxis.label.set_fontproperties(ax.yaxis.label.get_fontproperties())
    for t1, t2 in zip(ax.get_yticklabels(), ax2.get_yticklabels()):
        t2.set_fontproperties(t1.get_fontproperties())

    if savefigs:
        fig.savefig(exp_fld / "p1_thickness_vs_span.svg")


    # Plot spar cap thickness vs spanwise position
    fig, ax = scivis.subplots(figsize=figsize)
    t_spar_plot = np.zeros_like(x)
    t_spar_plot[:-1] = t_spar
    t_spar_plot[-1] = t_spar[-1]
    fig, ax, _ = scivis.plot_line(x, t_spar_plot*1e3, ax,
                                  ax_labels=["r", "t_f"],
                                  ax_units=["m", "mm"],
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True,
                                  exp_fld=exp_fld,
                                  fname="p1_spar_thickness_vs_span",
                                  savefig=savefigs)

# %% Determine stiffness the cross sections (simplified beam)
A_inv = np.stack([np.linalg.inv(section.laminate.ABD_matrix[:3, :3])
                  for section in laminates], axis=0)

E_f = 1.0 / (A_inv[:, 0, 0] * t_spar)
d = t_bld - t_spar
I_bld = t_spar * w_spar * d**2 / 2
EI_bld = E_f * I_bld

# Plot stiffness parameters over span
figsize = (16, 10)
plot_profile = "partsize"
plot_scale = .5
rcparams = scivis.rcparams._prepare_rcparams(profile=plot_profile,
                                             scale=plot_scale)
with mpl.rc_context(rcparams):
    # Plot equivalent Young's moduls vs spanwise position
    E_f_plot = np.zeros_like(x)
    E_f_plot[:-1] = E_f
    E_f_plot[-1]  = E_f[-1]
    fig, ax = scivis.subplots(figsize=figsize)
    fig, ax, _ = scivis.plot_line(x, E_f_plot*1e-6, ax,
                                  ax_labels=["r", "E_f"],
                                  ax_units=["m", "N/mm^2"],
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True,
                                  exp_fld=exp_fld,
                                  fname="p1_youngs_mod_vs_span",
                                  savefig=savefigs)

    # Plot area moment of inertia vs spanwise position
    I_bld_plot = np.zeros_like(x)
    I_bld_plot[:-1] = I_bld
    I_bld_plot[-1]  = I_bld[-1]
    fig, ax = scivis.subplots(figsize=figsize)
    fig, ax, _ = scivis.plot_line(x, I_bld_plot, ax,
                                  ax_labels=["r", "I"],
                                  ax_units=["m", "m^4"],
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True,
                                  exp_fld=exp_fld, fname="p1_inertia_vs_span",
                                  savefig=savefigs)

    # Plot bending stiffness vs spanwise position
    EI_bld_plot = np.zeros_like(x)
    EI_bld_plot[:-1] = EI_bld
    EI_bld_plot[-1]  = EI_bld[-1]
    fig, ax, _ = scivis.plot_line(x, EI_bld_plot*1e-6,
                                  ax_labels=["r", "EI"],
                                  ax_units=["m", "Nmm^2"],
                                  profile="partsize", scale=.65,
                                  override_axes_settings=True,
                                  exp_fld=exp_fld, fname="p1_stiffness_vs_span",
                                  savefig=savefigs)

# %% Solve the beam for the given loads
# Set up the loading
loads = pd.read_csv(data_dir / "loads.csv")
idx_nearest = [(np.abs(span_coords[:, 0] - coord_load)).argmin()
               for coord_load in loads["Radial position [m]"]]

Q = np.zeros_like(x)
for i in range(len(idx_nearest)):
    Q[:idx_nearest[i]+1] += loads["Force [N]"][i]

M = np.zeros_like(x)
kappa = np.zeros_like(x)

N_beams = len(laminates)
for i in range(N_beams):
    idx = N_beams - i - 1
    M[idx] = Q[idx]*l_beams[idx] + M[idx+1]

kappa[:-1] = M[:-1]/EI_bld

theta = np.zeros_like(x)
w = np.zeros_like(x)
for idx in range(1, N_beams+1):
    theta[idx] = M[idx]*l_beams[idx-1]/EI_bld[idx-1] \
        +  Q[idx]*l_beams[idx-1]**2/(2*EI_bld[idx-1]) \
        + theta[idx-1]

    w[idx] = theta[idx]*l_beams[idx-1] \
        + M[idx]*l_beams[idx-1]**2/(2*EI_bld[idx-1]) \
        + Q[idx]*l_beams[idx-1]**3/(6*EI_bld[idx-1]) + w[idx-1]

# %% Task 1: Tower clearance
w_tip = w[-1]
print(f"Tip deflection {w_tip:.2f} m")

# Calculate distance of tip to tower
tower_clearance = 18.26
print(f"Tower clearance: {tower_clearance-w_tip:.2f} m")

figsize = (16, 10)
plot_profile = "partsize"
plot_scale = .5
rcparams = scivis.rcparams._prepare_rcparams(profile=plot_profile,
                                             scale=plot_scale)
with mpl.rc_context(rcparams):
    # Plot deflection vs spanwise position
    fig, ax = scivis.subplots(figsize=figsize)
    fig, ax, _ = scivis.plot_line(x, w, ax,
                                  ax_labels=["r", "w"], ax_units=["m", "m"],
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True,
                                  exp_fld=exp_fld,
                                  fname="p1_deflection_vs_span",
                                  savefig=savefigs)

    # Plot deflection angle vs spanwise position
    fig, ax = scivis.subplots(figsize=figsize)
    fig, ax, _ = scivis.plot_line(x,
                                  np.rad2deg(theta), ax,
                                  ax_labels=["r", r"\theta"],
                                  ax_units=["m", r"\degree"],
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True,
                                  exp_fld=exp_fld,
                                  fname="p1_deflection_angle_vs_span",
                                  savefig=savefigs)

    # Plot curvature vs spanwise position
    fig, ax = scivis.subplots(figsize=figsize)
    fig, ax, _ = scivis.plot_line(x, kappa, ax,
                                  ax_labels=["r", r"\kappa"],
                                  ax_units=["m", "m^{-1}"],
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True,
                                  exp_fld=exp_fld,
                                  fname="p1_curvature_vs_span",
                                  savefig=savefigs)

    # Plot bending moment distribution vs spanwise position
    fig, ax = scivis.subplots(figsize=figsize)
    fig, ax, _ = scivis.plot_line(x, M*1e-6, ax,
                                  ax_labels=["r", r"M"],
                                  ax_units=["m", "MNm"],
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True,
                                  exp_fld=exp_fld,
                                  fname="p1_bending_moment_vs_span",
                                  savefig=savefigs)

    # Plot shear force distribution vs spanwise position
    fig, ax = scivis.subplots(figsize=figsize)
    fig, ax, _ = scivis.plot_line(x, Q*1e-6, ax,
                                  ax_labels=["r", r"Q"], ax_units=["m", "MN"],
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True,
                                  exp_fld=exp_fld,
                                  fname="p1_shear_force_vs_span",
                                  savefig=savefigs)

# %% Task 2: Buckling and strength
# %%% Task 2a: Buckling
# Calculate cross sectional stress from beam bending
sigma = np.zeros_like(x)
sigma[:-1] = M[:-1] / I_bld * (t_bld/2+t_spar)

epsilon = np.zeros_like(x)
epsilon[:-1] = -kappa[:-1] * (t_bld/2+t_spar)

idx_crit = np.argmax(sigma)

# Plot axial stress distribution vs spanwise position
fig, ax, _ = scivis.plot_line(x, sigma*1e-6, ax_labels=["r", r"\sigma"],
                              ax_units=["m", "N/mm^2"],
                              profile="partsize", scale=.6,
                              exp_fld=exp_fld, fname="p2_stress_vs_span",
                              savefig=savefigs)

# Plot axial strain distribution vs spanwise position
fig, ax, _ = scivis.plot_line(x, epsilon, ax_labels=["r", r"\varepsilon"],
                              ax_units=["m", None],
                              profile="partsize", scale=.55,
                              exp_fld=exp_fld, fname="p2_strain_vs_span",
                              savefig=savefigs)

# Calculate ABD matrix for cross sectios
ABD_bld = np.stack([section.laminate.ABD_matrix for section in laminates])
D_bld = ABD_bld[:, 3:6, 3:6]

# Calculate critical buckling load
def P_func(m, a, b, D):
    return np.pi**2 * (D[...,0,0]*(m/a)**2
                       + 2*(D[...,0,1] + 2*D[:,2,2])*(1/b)**2
                       + D[...,1,1]*(a/m)**2*(1/b)**4)

m_vec = np.arange(1, 5)
l_sections = np.array([section.length for section in laminates])
P_buckling = np.stack([P_func(m=m, a=l_sections, b=w_spar, D=D_bld)
                      for m in m_vec],
                      axis = 0)

P_cr_buckling = np.min(P_buckling, axis=0)
idx_P_cr = np.argmin(P_buckling, axis=0)

Nx = np.zeros_like(x)
Nx[:-1] = sigma[:-1] * t_spar

buckling_reserve = np.zeros_like(x)
buckling_reserve[:-1] = Nx[:-1] / P_cr_buckling

# Plot critical buckling load vs spanwise position
P_cr_buckling_plot = np.zeros_like(x)
P_cr_buckling_plot[:-1] = P_cr_buckling
P_cr_buckling_plot[-1] = P_cr_buckling[-1]

fig, ax, _ = scivis.plot_line(x, P_cr_buckling_plot*1e-6,
                              ax_labels=["r", r"P_{cr}"], ax_units=["m", "MN"],
                              profile="partsize", scale=.65,
                              exp_fld=exp_fld, fname="p2_buckling_load_vs_span",
                              savefig=savefigs)

P_buckling_plot = np.zeros([len(m_vec), len(x)])
P_buckling_plot[:, :-1] = P_buckling
P_buckling_plot[:, -1] = P_buckling[:, -1]

colors = ["#da881b", "#d02940", "#8f0062", "#1a0079"]
rcparams = scivis.rcparams._prepare_rcparams(profile="partsize", scale=.55)
with mpl.rc_context(rcparams):
    fig, ax, _ = scivis.plot_line(x, P_buckling_plot*1e-6,
                                  ax_labels=["r", r"P"], ax_units=["m", "MN/m"],
                                  plt_labels=[fr"$P\:(m={m})$" for m in m_vec],
                                  linestyles="-", colors=colors,
                                  profile="partsize", scale=.55)
    ax.plot(x, P_cr_buckling_plot*1e-6, ls="-", c="k", label="$P_{cr}$")
    ax.plot(x, Nx*1e-6, ls="-.", c="k", lw=2, label="$N_x$")

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=30)
    if savefigs:
        fig.savefig(exp_fld / "p2_buckling_load_all_vs_span.svg")

    fig, ax, _ = scivis.plot_line(x, buckling_reserve*1e2,
                                  ax_labels=["r", r"N_x/P_{cr}"],
                                  ax_units=["m", r"\%"],
                                  profile="partsize", scale=.55)
    ax.axhline(y=100, ls="-.", c="k", lw=2)

    if savefigs:
        fig.savefig(exp_fld / "p2_buckling_reserve_vs_span.svg")

fig, ax, _ = scivis.plot_line(x, np.vstack([P_cr_buckling_plot, Nx])*1e-6,
                              ax_labels=["r", r"load"], ax_units=["m", "MN/m"],
                              plt_labels=["$P_{cr}$", "$N_x$"],
                              profile="partsize", scale=.55)

# %%% Task 2b: Strength
failure_tsaihill = {"uniax": np.zeros((2, len(laminates))),
                    "triax": np.zeros((2, len(laminates)))}
failure_tsaiwu = {"uniax": np.zeros((2, len(laminates))),
                  "triax": np.zeros((2, len(laminates)))}

for i in range(len(laminates)):
    uniax_found = False
    triax_found = False
    for j in range(2):
        laminate_i = laminates[i]
        if laminate_i.ply_names[j] == "uniax1":
            strength_i = laminate_i.laminate.plies[j].material.strength_as_dict()
            failure_tsaihill["uniax"][0, i] = \
                clt.failure.TsaiHill.failure_index(stress=(sigma[i], 0, 0),
                                                   **strength_i)
            failure_tsaihill["uniax"][1, i] = \
                clt.failure.TsaiHill.failure_index(stress=(-sigma[i], 0, 0),
                                                   **strength_i)
            failure_tsaiwu["uniax"][0, i] = \
                clt.failure.TsaiWu.failure_index(stress=(sigma[i], 0, 0),
                                                   **strength_i)
            failure_tsaiwu["uniax"][1, i] = \
                clt.failure.TsaiWu.failure_index(stress=(-sigma[i], 0, 0),
                                                   **strength_i)
        elif laminate_i.ply_names[j] == "triax1":
            strength_i = laminate_i.laminate.plies[j].material.strength_as_dict()
            failure_tsaihill["triax"][0, i] = \
                clt.failure.TsaiHill.failure_index(stress=(sigma[i], 0, 0),
                                                   **strength_i)
            failure_tsaihill["triax"][1, i] = \
                clt.failure.TsaiHill.failure_index(stress=(-sigma[i], 0, 0),
                                                   **strength_i)
            failure_tsaiwu["triax"][0, i] = \
                clt.failure.TsaiWu.failure_index(stress=(sigma[i], 0, 0),
                                                   **strength_i)
            failure_tsaiwu["triax"][1, i] = \
                clt.failure.TsaiWu.failure_index(stress=(-sigma[i], 0, 0),
                                                   **strength_i)


if any((np.any(failure_tsaiwu["uniax"]>=1), np.any(failure_tsaiwu["triax"]>=1),
       np.any(failure_tsaihill["uniax"]>=1), np.any(failure_tsaihill["triax"]>=1))):
    print("Failure detected.")

# Plot Tsai-Wu and Tsai-Hill
rcparams = scivis.rcparams._prepare_rcparams(profile="partsize", scale=.65)
with mpl.rc_context(rcparams):
    for i, load_case in enumerate(["tensile", "compressive"]):
        failure_plot = np.zeros((4, len(x)))
        failure_plot[0, :-1] = failure_tsaihill["uniax"][i, :]
        failure_plot[1, :-1] = failure_tsaihill["triax"][i, :]
        failure_plot[2, :-1] = failure_tsaiwu["uniax"][i, :]
        failure_plot[3, :-1] = failure_tsaiwu["triax"][i, :]
        failure_plot[0, -1] = failure_tsaihill["uniax"][i, -1]
        failure_plot[1, -1] = failure_tsaihill["triax"][i, -1]
        failure_plot[2, -1] = failure_tsaiwu["uniax"][i, -1]
        failure_plot[3, -1] = failure_tsaiwu["triax"][i, -1]

        # fig, ax = scivis.subplots(figsize=(20,8))
        fig, ax, _ = scivis.plot_line(x, failure_plot,
                                      ax_labels=["r", None], ax_units=["m", None],
                                      colors=["#da881b", "#8f0062"]*2,
                                      linestyles=["-"]*2 + ["--"]*2,
                                      profile="partsize", scale=.6,
                                      override_axes_settings=True,
                                      show_legend=False)

        # # Add two legend manually
        # legend_handles = [mpl.lines.Line2D([0], [0], c='black', ls=ls, label=lbl)
        #                   for ls, lbl in [["-", "Tsai-Hill"], ["--", "Tsai-Wu"]]]
        # ax.legend(handles=legend_handles, loc="upper left",
        #           bbox_to_anchor=(1, 1), fontsize=33)
        # ax.add_artist(ax.get_legend())

        # legend_handles = [mpl.lines.Line2D([0], [0], c=c, ls="-", label=lbl)
        #                   for c, lbl in [["#da881b", "Uniaxial ply"],
        #                                  ["#8f0062", "Triaxial Ply"]]]
        # ax.legend(handles=legend_handles, loc="upper left",
        #           bbox_to_anchor=(1, .7), fontsize=33)

        ax.set_ylabel("Failure index")

        if savefigs:
            fig.savefig(exp_fld / f"p2_failure_vs_span_{load_case}.svg")

    # Create legend as separate figure
    fig_leg, ax_leg = scivis.subplots(figsize=(12, 4))

    # Hide axis
    ax_leg.axis("off")

    # --- First legend (failure criteria) ---
    legend_handles_1 = [mpl.lines.Line2D([0], [0], c='black', ls=ls, label=lbl)
                        for ls, lbl in [["-", "Tsai-Hill"], ["--", "Tsai-Wu"]]]

    legend1 = ax_leg.legend(handles=legend_handles_1, loc="upper left",
                            bbox_to_anchor=(0, -.3), fontsize=33)
    ax_leg.add_artist(legend1)

    # --- Second legend (laminates) ---
    legend_handles_2 = [mpl.lines.Line2D([0], [0], c=c, ls="-", label=lbl)
                      for c, lbl in [["#da881b", "Uniaxial ply"],
                                     ["#8f0062", "Triaxial Ply"]]]
    legend2 = ax_leg.legend(handles=legend_handles_2,
                            loc="upper left", bbox_to_anchor=(0.45, -.3),
                            fontsize=33)
    plt.tight_layout()

    # if savefigs:
    #     fig_leg.savefig(exp_fld / "p2_failure_index_legend.svg")

plt.show()
