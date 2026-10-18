"""Figures for the MAT-02 manuscript from profiles.json, gate_curves.json, spectra.json and uq_xmax.json.
Writes ../../paper_mat02/figures/*.pdf."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "..", "..", "paper_mat02", "figures"); os.makedirs(outdir, exist_ok=True)
BLUE, ORANGE, AQUA, PURPLE, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a4fbf", "#c4314b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 6.6, "axes.linewidth": 0.6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 1.3, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
J = lambda f: json.load(open(os.path.join(here, f)))
NAC = {5e19: (BLUE, r"compensated ($N_A=5\times10^{19}$)"), 1e18: (ORANGE, r"clean ($N_A=10^{18}$)")}
LEE = [(7.0e13, "10 K"), (1.08e14, "150 K")]


def clean(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def label(ax, s, x=-0.2):
    ax.text(x, 1.03, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom")


def fig1():
    P = J("profiles.json")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, (name, d), tag in zip(axs, P.items(), "abc"):
        z = np.array(d["z"]) / 10; V = np.array(d["V"]) - d["mu"]
        ax.axvspan(z.min(), 0, color="#f1efe9", lw=0)
        ax.plot(z, V, color=INK, lw=1.0)
        ax.axhline(0, color=RED, lw=0.7, ls="--")
        nb = np.array(d["n_bso"]); nc = np.array(d["n_cap"]); sc = 0.6 / max(nb.max(), nc.max(), 1)
        ax.fill_between(z, -1.6, -1.6 + nb * sc, color=BLUE, alpha=0.5, lw=0)
        ax.fill_between(z, -1.6, -1.6 + nc * sc, color=ORANGE, alpha=0.7, lw=0)
        ax.set_ylim(-1.7, 1.6); ax.set_xlim(z.min(), 6)
        ax.set_title({"interface": "at the interface", "donor sheet 10 A": "donor sheet 1 nm inside the cap",
                      "double layer": "on the cap surface"}[name], fontsize=7.8)
        ax.text(0.97, 0.95, f"cap: {100 * d['cap_frac']:.0f}%", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_xlabel("$z$ (nm)"); clean(ax, grid=False); label(ax, tag, x=-0.05 if tag != "a" else -0.25)
    axs[0].set_ylabel(r"$E_c-E_F$ (eV)")
    axs[0].text(-4.6, 1.35, r"LaInO$_3$", fontsize=7, color=INK2); axs[0].text(0.4, 1.35, r"BaSnO$_3$", fontsize=7, color=INK2)
    fig.savefig(os.path.join(outdir, "fig1_profiles.pdf")); plt.close(fig)


def fig2():
    G = J("gate_curves.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.32))
    ax = axs[0]
    for g in G:
        col, lab = NAC[g["NA"]]
        ls = {24.0: "-", 38.0: "--", 50.0: ":"}[g["eps_cap"]]
        ns = np.array([r["ns"] for r in g["rows"]]) / 1e13; x = np.array([r["x_max"] for r in g["rows"]]) / 10
        ax.plot(ns, x, color=col, ls=ls, marker="o", ms=2.2,
                label=(lab if g["eps_cap"] == 24.0 else None))
    for e, ls in ((24, "-"), (38, "--"), (50, ":")):
        ax.plot([], [], color=INK2, ls=ls, label=rf"$\varepsilon_\mathrm{{cap}}={e}$")
    ax.axhline(0.41, color=INK2, lw=0.6); ax.text(1.1, 0.46, "one LaInO$_3$ unit cell", fontsize=6.5, color=INK2)
    for n, t in LEE:
        ax.axvline(n / 1e13, color=RED, lw=0.6, ls="--"); ax.text(n / 1e13 + 0.12, 2.75, t, fontsize=6.5, color=RED, rotation=90, va="center")
    ax.set_xlabel(r"$n_s$ ($10^{13}$ cm$^{-2}$)"); ax.set_ylabel(r"$x_\mathrm{max}$ (nm)"); ax.set_ylim(0, 5.2)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0)); clean(ax); label(ax, "a")
    ax = axs[1]
    for g in G:
        if g["eps_cap"] != 24.0:
            continue
        col, lab = NAC[g["NA"]]
        rr = [r for r in g["rows"] if r.get("cbo_bracketed", False) and r["ns"] >= 3e13]
        ax.plot([r["ns"] / 1e13 for r in rr], [r["cbo_min"] for r in rr], color=col, marker="o", ms=2.2, label=lab)
    ax.axhspan(1.35, 1.70, color=AQUA, alpha=0.25, lw=0); ax.text(1.1, 1.45, "measured / DFT offsets", fontsize=6.5, color=INK2)
    for n, t in LEE:
        ax.axvline(n / 1e13, color=RED, lw=0.6, ls="--")
    ax.set_xlabel(r"$n_s$ ($10^{13}$ cm$^{-2}$)"); ax.set_ylabel("offset needed, double layer (eV)")
    ax.legend(frameon=False, loc="upper left"); clean(ax); label(ax, "b")
    fig.savefig(os.path.join(outdir, "fig2_limits.pdf")); plt.close(fig)


def fig3():
    S = J("spectra.json"); hw = np.array(S["hw"])
    CC = {"FF": (INK, "interface"), "DON": (AQUA, "donor sheet 0.5 nm"), "EDL": (RED, "cap surface"),
          "EDLX": (PURPLE, "cap surface (offset 3.0 eV)")}
    MK = {"FF": "o", "DON": "s", "EDL": "v", "EDLX": "^"}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.42))
    for k, NA in enumerate((5e19, 1e18)):
        ax = axs[k]
        for c, (col, lab) in CC.items():
            rr = sorted([r for r in S["rows"] if r["NA"] == NA and r["case"] == c], key=lambda r: r["ns"])
            ax.plot([r["ns"] / 1e13 for r in rr], [r["mean_coll"] for r in rr], color=col, marker=MK[c], ms=3, label=lab)
        ax.set_xlabel(r"$n_s$ ($10^{13}$ cm$^{-2}$)"); ax.set_ylabel(r"mean absorption energy $\bar E$ (eV)")
        ax.set_title(NAC[NA][1], fontsize=7.8); clean(ax); label(ax, "ab"[k])
        if k == 0:
            ax.legend(frameon=False, loc="lower right", fontsize=6.0)
    ax = axs[2]
    for c, (col, lab) in CC.items():
        r = [r for r in S["rows"] if r["NA"] == 1e18 and r["case"] == c and r["ns"] == 7e13][0]
        ax.plot(hw, r["spec30"], color=col, lw=1.0, label=lab)
    ax.set_xlim(0.1, 1.6); ax.set_xlabel(r"$\hbar\omega$ (eV)"); ax.set_ylabel("absorbed power (norm.)")
    ax.set_title(r"clean, $n_s=7\times10^{13}$, $\gamma=30$ meV", fontsize=7.8); clean(ax); label(ax, "c")
    fig.savefig(os.path.join(outdir, "fig3_isb.pdf")); plt.close(fig)


def fig4():
    if not os.path.exists(os.path.join(here, "uq_xmax.json")):
        return
    U = J("uq_xmax.json")
    good = [r for r in U["rows"] if r.get("ok")]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.3), gridspec_kw=dict(wspace=0.32))
    for ax, key, tag in zip(axs, ("7.00e+13", "1.08e+14"), "ab"):
        x = np.array([r[key]["x_max"] for r in good]) / 10
        na = np.array([r["params"]["NA"] for r in good])
        sc = ax.scatter(np.log10(na), x, c=[r["params"]["cbo"] for r in good], cmap="viridis", s=9, lw=0)
        ax.axhline(0.41, color=INK2, lw=0.6)
        ax.set_xlabel(r"$\log_{10} N_A$ (cm$^{-3}$)"); ax.set_ylabel(r"$x_\mathrm{max}$ (nm)")
        ax.set_title({"7.00e+13": r"$n_s=7.0\times10^{13}$ cm$^{-2}$", "1.08e+14": r"$n_s=1.08\times10^{14}$ cm$^{-2}$"}[key], fontsize=7.8)
        clean(ax); label(ax, tag)
    cb = fig.colorbar(sc, ax=axs, shrink=0.85, pad=0.02); cb.set_label("offset (eV)", fontsize=7.5)
    fig.savefig(os.path.join(outdir, "fig4_uq.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("figures written to", os.path.abspath(outdir))
