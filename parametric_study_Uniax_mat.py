from copy import deepcopy
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from turbine_blade import TurbineBlade
import classical_laminate_theory as clt
import scivis

# %% User inputs
# Select parametric study

exp_fld = Path(__file__).parent / "plots" / "parametric" / "Case a"
savefigs = True  # % Selection whether to export the figures as svg files
if savefigs: exp_fld.mkdir(exist_ok=True)
colors = ["#da881b", "#d02940", "#8f0062", "#1a0079"]

# %% Create blade from input data
data_dir = Path(__file__).parent / "_data"
blade_baseline = TurbineBlade()
blade_EGF = deepcopy(blade_baseline)
blade_CF_HT = deepcopy(blade_baseline)
blade_CF_HM = deepcopy(blade_baseline)

# Adjust uniaxial ply to E-Glass fiber
uniax_ply_EGF = blade_EGF._multiax_plies["uniax"]
uniax_ply_EGF["elastic_properties"]["E1"] = 40e9
uniax_ply_EGF["elastic_properties"]["E2"] = 9.8e9
uniax_ply_EGF["elastic_properties"]["G12"] = 2.8e9
uniax_ply_EGF["elastic_properties"]["nu12"] = .3
uniax_ply_EGF["elastic_properties"]["nu21"] = .3
uniax_ply_EGF["strength_properties"]["sigma_1t"] = 1100e6
uniax_ply_EGF["strength_properties"]["sigma_1c"] = 600e6
uniax_ply_EGF["strength_properties"]["sigma_2t"] = 20e6
uniax_ply_EGF["strength_properties"]["sigma_2c"] = 140e6
uniax_ply_EGF["strength_properties"]["tau_12"] = 70e6

# Adjust uniaxial ply to HT carbon fiber
uniax_ply_CFHT = blade_CF_HT._multiax_plies["uniax"]
uniax_ply_CFHT["elastic_properties"]["E1"] = 136e9
uniax_ply_CFHT["elastic_properties"]["E2"] = 10e9
uniax_ply_CFHT["elastic_properties"]["G12"] = 5.2e9
uniax_ply_CFHT["elastic_properties"]["nu12"] = .3
uniax_ply_CFHT["elastic_properties"]["nu21"] = .3
uniax_ply_CFHT["strength_properties"]["sigma_1t"] = 1800e6
uniax_ply_CFHT["strength_properties"]["sigma_1c"] = 1200e6
uniax_ply_CFHT["strength_properties"]["sigma_2t"] = 40e6
uniax_ply_CFHT["strength_properties"]["sigma_2c"] = 220e6
uniax_ply_CFHT["strength_properties"]["tau_12"] = 80e6

# Adjust uniaxial ply to HM carbon fiber
uniax_ply_CFHM = blade_CF_HM._multiax_plies["uniax"]
uniax_ply_CFHM["elastic_properties"]["E1"] = 181e9
uniax_ply_CFHM["elastic_properties"]["E2"] = 10.3e9
uniax_ply_CFHM["elastic_properties"]["G12"] = 7.17e9
uniax_ply_CFHM["elastic_properties"]["nu12"] = .28
uniax_ply_CFHM["elastic_properties"]["nu21"] = .28
uniax_ply_CFHM["strength_properties"]["sigma_1t"] = 1500e6
uniax_ply_CFHM["strength_properties"]["sigma_1c"] = 1500e6
uniax_ply_CFHM["strength_properties"]["sigma_2t"] = 40e6
uniax_ply_CFHM["strength_properties"]["sigma_2c"] = 246e6
uniax_ply_CFHM["strength_properties"]["tau_12"] = 68e6

# Create the spanwise parameters
laminates = [blade_baseline.create_laminates(),
             blade_EGF.create_laminates(),
             blade_CF_HT.create_laminates(),
             blade_CF_HM.create_laminates()]
laminate_names = ["Original", "E-GF", "HT-CF", "HM-CF"]

span_coords = np.stack([[section.r_start for section in laminates[0]],
                        [section.r_end for section in laminates[0]]], axis=1)  # Same for both blades
l_beams = np.array([section.length for section in laminates[0]])  # Same for both blades
t_spar = np.array([section.thickness for section in laminates[0]])  # Same for both blades

# Calculate blade thickness & chord and spar width from splines
t_bld = np.average(blade_baseline.thickness_abs(span_coords), axis=1)  # Same for both blades
chord_bld = np.average(blade_baseline.chord(span_coords), axis=1)  # Same for both blades
w_spar = np.average(blade_baseline.w_spar(span_coords), axis=1)  # Same for both blades

# Repeat dimensions for convenient handling later (Uniform handling for the
# other parametric study)
N_laminates = len(laminates)
N_beams = len(l_beams)
l_beams = np.tile(l_beams, (N_laminates, 1))
t_spar = np.tile(t_spar, (N_laminates, 1))
t_bld = np.tile(t_bld, (N_laminates, 1))
chord_bld = np.tile(chord_bld, (N_laminates, 1))
w_spar = np.tile(w_spar, (N_laminates, 1))

x = np.zeros(len(laminates[0]) + 1)
x[:-1] = span_coords[:, 0]
x[-1] = span_coords[-1, 1]

# %% Determine stiffness the cross sections (simplified beam)
A_inv = np.stack([[np.linalg.inv(section.laminate.ABD_matrix[:3, :3])
                   for section in laminate] for laminate in laminates], axis=0)

E_f = 1.0 / (A_inv[..., 0, 0] * t_spar)
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
    # Plot bending stiffness vs spanwise position
    EI_bld_plot = np.zeros([N_laminates, len(x)])
    EI_bld_plot[:, :-1] = EI_bld
    EI_bld_plot[:, -1]  = EI_bld[:, -1]

    fig, ax, _ = scivis.plot_line(x, EI_bld_plot*1e-6,
                                  ax_labels=["r", "EI"],
                                  ax_units=["m", "Nmm^2"],
                                  plt_labels=laminate_names,
                                  profile="partsize", scale=.65,
                                  colors=colors, linestyles="-",
                                  override_axes_settings=True,
                                  exp_fld=exp_fld,
                                  fname="p3a_stiffness_vs_span",
                                  savefig=savefigs)

# %% Solve the beam for the given loads
# Set up the loading
loads = pd.read_csv(data_dir / "loads.csv")
idx_nearest = [(np.abs(span_coords[:, 0] - coord_load)).argmin()
               for coord_load in loads["Radial position [m]"]]

Q = np.zeros([N_laminates, len(x)])
for i in range(len(idx_nearest)):
    Q[:, :idx_nearest[i]+1] += loads["Force [N]"][i]

M = np.zeros([N_laminates, len(x)])
kappa = np.zeros([N_laminates, len(x)])

for i in range(N_beams):
    idx = N_beams - i - 1
    M[:, idx] = Q[:, idx]*l_beams[:, idx] + M[:, idx+1]
kappa[:, :-1] = M[:, :-1]/EI_bld

theta = np.zeros([N_laminates, len(x)])
w = np.zeros([N_laminates, len(x)])
for idx in range(1, N_beams+1):
    theta[:, idx] = M[:, idx]*l_beams[:, idx-1]/EI_bld[:, idx-1] \
        +  Q[:, idx]*l_beams[:, idx-1]**2/(2*EI_bld[:, idx-1]) \
        + theta[:, idx-1]

    w[:, idx] = theta[:, idx]*l_beams[:, idx-1] \
        + M[:, idx]*l_beams[:, idx-1]**2/(2*EI_bld[:, idx-1]) \
        + Q[:, idx]*l_beams[:, idx-1]**3/(6*EI_bld[:, idx-1]) + w[:, idx-1]

# %% Task 1: Tower clearance
w_tip = w[:, -1]
print("Tip deflection:\n"
      +"\n".join([f"{name}: {w_tip_i:.2f} m"
                  for name, w_tip_i in zip(laminate_names, w_tip)]))

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
                                  plt_labels=laminate_names,
                                  colors=colors, linestyles="-",
                                  profile=plot_profile, scale=plot_scale,
                                  override_axes_settings=True,
                                  exp_fld=exp_fld,
                                  fname="p3a_deflection_vs_span",
                                  savefig=savefigs)

# %% Task 2: Buckling and strength
# %%% Task 2a: Buckling
# Calculate cross sectional stress from beam bending
sigma = np.zeros([N_laminates, len(x)])
sigma[:, :-1] = M[:, :-1] / I_bld * (t_bld/2+t_spar)

epsilon = np.zeros([N_laminates, len(x)])
epsilon[:, :-1] = -kappa[:, :-1] * (t_bld/2+t_spar)

# Plot axial stress distribution vs spanwise position
fig, ax, _ = scivis.plot_line(x, sigma*1e-6, ax_labels=["r", r"\sigma"],
                              ax_units=["m", "N/mm^2"],
                              plt_labels=laminate_names,
                              colors=colors, linestyles="-",
                              profile="partsize", scale=.6,
                              exp_fld=exp_fld, fname="p3a_stress_vs_span",
                              savefig=savefigs)

# Plot axial strain distribution vs spanwise position
fig, ax, _ = scivis.plot_line(x, epsilon, ax_labels=["r", r"\varepsilon"],
                              ax_units=["m", None],
                              plt_labels=laminate_names,
                              colors=colors, linestyles="-",
                              profile="partsize", scale=.55,
                              exp_fld=exp_fld, fname="p3a_strain_vs_span",
                              savefig=savefigs)

# Calculate ABD matrix for cross sectios
D_bld = np.stack([[section.laminate.ABD_matrix[3:6, 3:6]
                   for section in laminate] for laminate in laminates], axis=0)

# Calculate critical buckling load
def P_func(m, a, b, D):
    return np.pi**2 * (D[...,0,0]*(m/a)**2
                       + 2*(D[...,0,1] + 2*D[...,2,2])*(1/b)**2
                       + D[...,1,1]*(a/m)**2*(1/b)**4)

m_vec = np.arange(1, 5)
P_buckling = np.stack([P_func(m=m, a=l_beams, b=w_spar, D=D_bld)
                      for m in m_vec], axis = -1)

P_cr_buckling = np.min(P_buckling, axis=-1)
idx_P_cr = np.argmin(P_buckling, axis=-1)

Nx = np.zeros([N_laminates, len(x)])
Nx[:, :-1] = sigma[:, :-1] * t_spar

buckling_reserve = np.zeros([N_laminates, len(x)])
buckling_reserve[:, :-1] = Nx[:, :-1] / P_cr_buckling

# Plot critical buckling load vs spanwise position
P_cr_buckling_plot = np.zeros([N_laminates, len(x)])
P_cr_buckling_plot[:, :-1] = P_cr_buckling
P_cr_buckling_plot[:, -1] = P_cr_buckling[:, -1]

fig, ax, _ = scivis.plot_line(x, P_cr_buckling_plot*1e-6,
                              ax_labels=["r", r"P_{cr}"], ax_units=["m", "MN"],
                              plt_labels=laminate_names,
                              colors=colors, linestyles="-",
                              profile="partsize", scale=.65,
                              exp_fld=exp_fld, fname="p3a_buckling_load_vs_span",
                              savefig=savefigs)

fig, ax, _ = scivis.plot_line(x, buckling_reserve*1e2,
                              ax_labels=["r", r"N_x/P_{cr}"],
                              ax_units=["m", r"\%"],
                              plt_labels=laminate_names,
                              colors=colors, linestyles="-",
                              profile="partsize", scale=.55)
ax.axhline(y=100, ls="-.", c="k", lw=2)

if savefigs:
    fig.savefig(exp_fld / "p3a_buckling_reserve_vs_span.svg")

# %%% Task 2b: Strength
failure_tsaihill = {"uniax": np.zeros([N_laminates, 2, N_beams]),
                    "triax": np.zeros([N_laminates, 2, N_beams])}
failure_tsaiwu = {"uniax": np.zeros([N_laminates, 2, N_beams]),
                    "triax": np.zeros([N_laminates, 2, N_beams])}

for i in range(N_beams):
    uniax_found = False
    triax_found = False
    for j in range(2):
        for k in range(N_laminates):
            laminate_i = laminates[k][i]
            if laminate_i.ply_names[j] == "uniax1":
                strength_i = laminate_i.laminate.plies[j].material.strength_as_dict()
                failure_tsaihill["uniax"][k, 0, i] = \
                    clt.failure.TsaiHill.failure_index(stress=(sigma[k, i], 0, 0),
                                                       **strength_i)
                failure_tsaihill["uniax"][k, 1, i] = \
                    clt.failure.TsaiHill.failure_index(stress=(-sigma[k, i], 0, 0),
                                                       **strength_i)
                failure_tsaiwu["uniax"][k, 0, i] = \
                    clt.failure.TsaiWu.failure_index(stress=(sigma[k, i], 0, 0),
                                                       **strength_i)
                failure_tsaiwu["uniax"][k, 1, i] = \
                    clt.failure.TsaiWu.failure_index(stress=(-sigma[k, i], 0, 0),
                                                       **strength_i)
            elif laminate_i.ply_names[j] == "triax1":
                strength_i = laminate_i.laminate.plies[j].material.strength_as_dict()
                failure_tsaihill["triax"][k, 0, i] = \
                    clt.failure.TsaiHill.failure_index(stress=(sigma[k, i], 0, 0),
                                                       **strength_i)
                failure_tsaihill["triax"][k, 1, i] = \
                    clt.failure.TsaiHill.failure_index(stress=(-sigma[k, i], 0, 0),
                                                       **strength_i)
                failure_tsaiwu["triax"][k, 0, i] = \
                    clt.failure.TsaiWu.failure_index(stress=(sigma[k, i], 0, 0),
                                                       **strength_i)
                failure_tsaiwu["triax"][k, 1, i] = \
                    clt.failure.TsaiWu.failure_index(stress=(-sigma[k, i], 0, 0),
                                                       **strength_i)

# Plot Tsai-Wu and Tsai-Hill
rcparams = scivis.rcparams._prepare_rcparams(profile="partsize", scale=.65)
with mpl.rc_context(rcparams):
    for i, load_case in enumerate(["tensile", "compressive"]):
        for ply in ["uniax", "triax"]:
            failure_plot = np.zeros((2*N_laminates, len(x)))
            failure_plot[:N_laminates, :-1] = failure_tsaihill[ply][:, i, :]
            failure_plot[N_laminates:, :-1] = failure_tsaiwu[ply][:, i, :]
            failure_plot[:N_laminates, -1] = failure_tsaihill[ply][:, i, -1]
            failure_plot[N_laminates:, -1] = failure_tsaiwu[ply][:, i, -1]

            # fig, ax = scivis.subplots(figsize=(16,8))
            fig, ax, _ = scivis.plot_line(x, failure_plot[:N_laminates, :],
                                          linestyles="-", colors=colors,
                                          show_legend=False)
            fig, ax, _ = scivis.plot_line(x, failure_plot[N_laminates:, :],
                                          ax=ax,
                                          ax_labels=["r", None],
                                          ax_units=["m", None],
                                          profile=plot_profile,
                                          scale=plot_scale,
                                          override_axes_settings=True,
                                          linestyles="--", colors=colors,
                                          show_legend=False)

            # # Add two legends manually
            # legend_handles = [mpl.lines.Line2D([0], [0], c='black',
            #                                    ls=ls, label=lbl)
            #                   for ls, lbl in [["-", "Tsai-Hill"],
            #                                   ["--", "Tsai-Wu"]]]
            # ax.legend(handles=legend_handles, loc="upper left",
            #           bbox_to_anchor=(1, 1), fontsize=33)
            # ax.add_artist(ax.get_legend())

            # legend_handles = [mpl.lines.Line2D([0], [0], c=c, ls="-",
            #                                    label=lbl)
            #                   for c, lbl in zip(colors, laminate_names)]
            # ax.legend(handles=legend_handles, loc="upper left",
            #           bbox_to_anchor=(1, .7), fontsize=33)

            ax.set_ylabel("Failure index")

            if savefigs:
                fig.savefig(exp_fld / f"p3a_failure_vs_span_{ply}_{load_case}.svg")

    # Create legend as separate figure
    fig_leg, ax_leg = scivis.subplots(figsize=(16, 4))

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
                        for c, lbl in zip(colors, laminate_names) ]
    legend2 = ax_leg.legend(handles=legend_handles_2, ncols=2,
                            loc="upper left", bbox_to_anchor=(0.4, -.3),
                            fontsize=33)
    plt.tight_layout()

    # if savefigs:
        # fig_leg.savefig(exp_fld / "p3a_failure_index_legend.svg")

plt.show()
