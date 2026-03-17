from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from turbine_blade import TurbineBlade
import classical_laminate_theory as clt
import scivis

# %% User inputs
# Select parametric study

exp_fld = Path(__file__).parent / "plots" / "parametric"/ "Case b"
savefigs = True  # % Selection whether to export the figures as svg files
if savefigs: exp_fld.mkdir(exist_ok=True)

# %% Create blade from input data
data_dir = Path(__file__).parent / "_data"
blade = TurbineBlade()

# Create the spanwise parameters
laminates = blade.create_laminates()

span_coords = np.stack([[section.r_start for section in laminates],
                        [section.r_end for section in laminates]], axis=1)  # Same for both blades
l_beams = np.array([section.length for section in laminates])  # Same for both blades
t_spar = np.array([section.thickness for section in laminates])

# Calculate blade thickness & chord and spar width from splines
t_bld = np.average(blade.thickness_abs(span_coords), axis=1)  # Same for both blades
chord_bld = np.average(blade.chord(span_coords), axis=1)  # Same for both blades
w_spar = np.average(blade.w_spar(span_coords), axis=1)  # Same for both blades

# Adjust spar cap thickness in a +-5% range
upper_lim = .1
lower_lim = -.1
t_spar_fac = np.linspace(lower_lim, upper_lim, 11) + 1
t_spar_fac_ticks = np.arange(lower_lim* 100, upper_lim* 100 + 1, 2)

N_cases = len(t_spar_fac)
N_beams = len(l_beams)
t_spar = np.tile(t_spar, (N_cases, 1)) * np.reshape(t_spar_fac, (-1, 1))
l_beams = np.tile(l_beams, (N_cases, 1))
t_bld = np.tile(t_bld, (N_cases, 1))
chord_bld = np.tile(chord_bld, (N_cases, 1))
w_spar = np.tile(w_spar, (N_cases, 1))

x = np.zeros(len(laminates) + 1)
x[:-1] = span_coords[:, 0]
x[-1] = span_coords[-1, 1]

# %% Determine stiffness the cross sections (simplified beam)
A_inv = np.stack([np.linalg.inv(section.laminate.ABD_matrix[:3, :3])
                   for section in laminates], axis=0)

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

def add_colorbar(ax, rcparams, fontsize_factor=.85):
    norm = mpl.colors.Normalize(vmin=lower_lim*100, vmax=upper_lim*100)
    formatter = FuncFormatter(lambda x, pos: f"{x:+}")
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="berlin"),
                        ax=ax, ticks=t_spar_fac_ticks, format=formatter)
    cbar.ax.tick_params(direction="out", length=10, width=1.5)
    # enforce rc font
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(rcparams["font.family"])
        label.set_fontsize(rcparams["font.size"]*fontsize_factor)

    cbar.set_label(r"$\Delta t_{spar}$ [%]",
                   fontsize=mpl.rcParams["axes.labelsize"]*fontsize_factor)

with mpl.rc_context(rcparams):
    # Plot bending stiffness vs spanwise position
    I_bld_plot = np.zeros([N_cases, len(x)])
    I_bld_plot[:, :-1] = I_bld
    I_bld_plot[:, -1]  = I_bld[:, -1]

    fig, ax, _ = scivis.plot_line(x, I_bld_plot,
                                  ax_labels=["r", "I"],
                                  ax_units=["m", "m^2"],
                                  profile="partsize", scale=.65,
                                  cmap="berlin", linestyles="-",
                                  show_legend=False,
                                  override_axes_settings=True)

    add_colorbar(ax=ax, rcparams=rcparams)

    if savefigs:
        fig.savefig(exp_fld / "p3b_inertia_vs_span_vs_span.svg")

    # Plot bending stiffness vs spanwise position
    EI_bld_plot = np.zeros([N_cases, len(x)])
    EI_bld_plot[:, :-1] = EI_bld
    EI_bld_plot[:, -1]  = EI_bld[:, -1]

    fig, ax, _ = scivis.plot_line(x, EI_bld_plot*1e-6,
                                  ax_labels=["r", "EI"],
                                  ax_units=["m", "Nmm^2"],
                                  profile="partsize", scale=.65,
                                  cmap="berlin", linestyles="-",
                                  show_legend=False,
                                  override_axes_settings=True)

    add_colorbar(ax=ax, rcparams=rcparams)

    if savefigs:
        fig.savefig(exp_fld / "p3b_stiffness_vs_span.svg")

# %% Solve the beam for the given loads
# Set up the loading
loads = pd.read_csv(data_dir / "loads.csv")
idx_nearest = [(np.abs(span_coords[:, 0] - coord_load)).argmin()
               for coord_load in loads["Radial position [m]"]]

Q = np.zeros([N_cases, len(x)])
for i in range(len(idx_nearest)):
    Q[:, :idx_nearest[i]+1] += loads["Force [N]"][i]

M = np.zeros([N_cases, len(x)])
kappa = np.zeros([N_cases, len(x)])

for i in range(N_beams):
    idx = N_beams - i - 1
    M[:, idx] = Q[:, idx]*l_beams[:, idx] + M[:, idx+1]
kappa[:, :-1] = M[:, :-1]/EI_bld

theta = np.zeros([N_cases, len(x)])
w = np.zeros([N_cases, len(x)])
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
      +"\n".join([f"{name:+} %: {w_tip_i:.2f} m"
                  for name, w_tip_i in zip(t_spar_fac_ticks, w_tip)]))

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
                                  cmap="berlin", linestyles="-",
                                  override_axes_settings=True,
                                  show_legend=False)
    add_colorbar(ax=ax, rcparams=rcparams)

    if savefigs:
        fig.savefig(exp_fld / "p3b_deflection_vs_span.svg")

# %% Task 2: Buckling and strength
# %%% Task 2a: Buckling
# Calculate cross sectional stress from beam bending
sigma = np.zeros([N_cases, len(x)])
sigma[:, :-1] = M[:, :-1] / I_bld * (t_bld/2+t_spar)

epsilon = np.zeros([N_cases, len(x)])
epsilon[:, :-1] = -kappa[:, :-1] * (t_bld/2+t_spar)

plot_profile = "partsize"
plot_scale = .6
rcparams = scivis.rcparams._prepare_rcparams(profile=plot_profile,
                                             scale=plot_scale)
with mpl.rc_context(rcparams):
    # Plot axial stress distribution vs spanwise position
    fig, ax, _ = scivis.plot_line(x, sigma*1e-6, ax_labels=["r", r"\sigma"],
                                  ax_units=["m", "N/mm^2"],
                                  profile=plot_profile, scale=plot_scale,
                                  cmap="berlin", linestyles="-",
                                  show_legend=False)
    add_colorbar(ax=ax, rcparams=rcparams)

    if savefigs:
        fig.savefig(exp_fld / "p3b_stress_vs_span.svg")

    # Plot axial strain distribution vs spanwise position
    fig, ax, _ = scivis.plot_line(x, epsilon, ax_labels=["r", r"\varepsilon"],
                                  ax_units=["m", None],
                                  profile=plot_profile, scale=plot_scale,
                                  cmap="berlin", linestyles="-",
                                  show_legend=False)
    add_colorbar(ax=ax, rcparams=rcparams)

    if savefigs:
        fig.savefig(exp_fld / "p3b_strain_vs_span.svg")

# Calculate ABD matrix for cross sectios
D_bld = np.stack([section.laminate.ABD_matrix[3:6, 3:6]
                   for section in  laminates], axis=0)

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

Nx = np.zeros([N_cases, len(x)])
Nx[:, :-1] = sigma[:, :-1] * t_spar

buckling_reserve = np.zeros([N_cases, len(x)])
buckling_reserve[:, :-1] = Nx[:, :-1] / P_cr_buckling

# Plot critical buckling load vs spanwise position
P_cr_buckling_plot = np.zeros([N_cases, len(x)])
P_cr_buckling_plot[:, :-1] = P_cr_buckling
P_cr_buckling_plot[:, -1] = P_cr_buckling[:, -1]

plot_profile = "partsize"
plot_scale = .65
rcparams = scivis.rcparams._prepare_rcparams(profile=plot_profile,
                                             scale=plot_scale)
with mpl.rc_context(rcparams):
    fig, ax, _ = scivis.plot_line(x, P_cr_buckling_plot*1e-6,
                                  ax_labels=["r", r"P_{cr}"], ax_units=["m", "MN"],
                                  profile=plot_profile, scale=plot_scale,
                                  cmap="berlin", linestyles="-",
                                  show_legend=False)
    add_colorbar(ax=ax, rcparams=rcparams)

    if savefigs:
        fig.savefig(exp_fld / "p3b_buckling_load_vs_span.svg")

plot_profile = "partsize"
plot_scale = .55
rcparams = scivis.rcparams._prepare_rcparams(profile=plot_profile,
                                             scale=plot_scale)
with mpl.rc_context(rcparams):
    fig, ax, _ = scivis.plot_line(x, buckling_reserve*1e2,
                                  ax_labels=["r", r"N_x/P_{cr}"],
                                  profile=plot_profile, scale=plot_scale,
                                  cmap="berlin", linestyles="-",
                                  show_legend=False)
    ax.axhline(y=100, ls="-.", c="k", lw=2)
    add_colorbar(ax=ax, rcparams=rcparams)

    if savefigs:
        fig.savefig(exp_fld / "p3b_buckling_reserve_vs_span.svg")

# %%% Task 2b: Strength
failure_tsaihill = {"uniax": np.zeros([N_cases, 2, N_beams]),
                    "triax": np.zeros([N_cases, 2, N_beams])}
failure_tsaiwu = {"uniax": np.zeros([N_cases, 2, N_beams]),
                    "triax": np.zeros([N_cases, 2, N_beams])}

for i in range(N_beams):
    uniax_found = False
    triax_found = False
    for j in range(2):
        for k in range(N_cases):
            laminate_i = laminates[i]
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

plot_profile = "partsize"
plot_scale = .65
rcparams = scivis.rcparams._prepare_rcparams(profile=plot_profile,
                                             scale=plot_scale)
with mpl.rc_context(rcparams):
    for i, load_case in enumerate(["tensile", "compressive"]):
        for ply in ["uniax", "triax"]:
            failure_plot = np.zeros((2*N_cases, len(x)))
            failure_plot[:N_cases, :-1] = failure_tsaihill[ply][:, i, :]
            failure_plot[N_cases:, :-1] = failure_tsaiwu[ply][:, i, :]
            failure_plot[:N_cases, -1] = failure_tsaihill[ply][:, i, -1]
            failure_plot[N_cases:, -1] = failure_tsaiwu[ply][:, i, -1]

            # fig, ax = scivis.subplots(figsize=(16,8))
            fig, ax, _ = scivis.plot_line(x, failure_plot[:N_cases, :],
                                          cmap="berlin", linestyles="-",
                                          show_legend=False)
            fig, ax, _ = scivis.plot_line(x, failure_plot[N_cases:, :], ax=ax,
                                          ax_labels=["r", None],
                                          ax_units=["m", None],
                                          profile=plot_profile, scale=plot_scale,
                                          override_axes_settings=True,
                                          cmap="berlin", linestyles="--",
                                          show_legend=False)

            ax.set_ylabel("Failure index")

            # Add legend manually
            legend_handles = [mpl.lines.Line2D([0], [0], c='black',
                                               ls=ls, label=lbl)
                              for ls, lbl in [["-", "Tsai-Hill"],
                                              ["--", "Tsai-Wu"]]]
            ax.legend(handles=legend_handles, loc="upper right", fontsize=33)

            # Add colorbar
            add_colorbar(ax=ax, rcparams=rcparams, fontsize_factor=.9)

            if savefigs:
                fig.savefig(exp_fld / f"p3b_failure_vs_span_{ply}_{load_case}.svg")

plt.show()
