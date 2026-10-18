"""Every number quoted in the MAT-02 manuscript, from gate_map, gate_curves, donor_layer, spectra, profiles and
uq_xmax. Prints a flat list; the manuscript is checked against it."""
import json, os
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(here, f)))
NAME = {5e19: "compensated", 1e18: "clean"}

print("== gate_curves (300 K, n0 = 5e12): x_max (A), field-free cap margin (eV) and fraction, double-layer offset (eV)")
for g in J("gate_curves.json"):
    print(f"  {NAME[g['NA']]} eps {g['eps_cap']:.0f} sigma_P {g['sigma_P']:.3e}")
    for r in g["rows"]:
        est = g["eps_cap"] * 8.8541878128e-12 * max(r["margin_FF"], 0) / (1.602176634e-19 * (r["ns"] - 5e12) * 1e4) * 1e10
        print(f"    ns {r['ns']:.2e}: x_max {r['x_max']:.1f} (capacitor estimate eps eps0 margin_FF / (e dn) = {est:.1f}) | FF margin "
              f"{r['margin_FF']:+.3f} frac {r['frac_FF']:.3f}" + (f" | offset {r['cbo_min']:.2f}" if "cbo_min" in r else ""))
print("== gate_map (n0 = 5e12): x_max, surface-decaying donor share (lam 10, 20 A), double-layer offset (>= 4.00 = search limit)")
for g in J("gate_map.json"):
    for a, e1, e2, c in zip(g["sheet"], g["exp"][0::2], g["exp"][1::2], g["cbo"]):
        print(f"  {NAME[g['NA']]} eps {g['eps_cap']:.0f} T {g['T']:.0f} ns {a['ns']:.2e}: x_max {a['x_max']:.1f} | share {e1['share_max']:.2f} / "
              f"{e2['share_max']:.2f} | offset {c['cbo_min']:.2f}")
print("== donor_layer: uniform layer next to the interface")
for o in J("donor_layer.json"):
    print(f"  {NAME[o['NA']]} T {o['T']:.0f} ns {o['ns']:.2e}: w_max {o['w_max']:.1f} A ({o['bracketed']}) N_D {o['N_D_min']:.2e} "
          f"V_O fraction {100 * o['vacancy_site_fraction']:.2f}%")
S = J("spectra.json")["rows"]
print("== spectra (300 K)")
for NA in (5e19, 1e18):
    for ns in sorted(set(r["ns"] for r in S)):
        R = {r["case"]: r for r in S if r["NA"] == NA and r["ns"] == ns}
        ff = R["FF"]
        print(f"  {NAME[NA]} ns {ns:.2e}: " + " | ".join(
            f"{c} cap {R[c]['cap_frac']:.3f} nBSO {R[c]['n_bso']:.2e} Ebar {R[c]['mean_coll']:.3f} ({100 * (R[c]['mean_coll'] / ff['mean_coll'] - 1):+.1f}%) "
            f"med30 {R[c]['median30']:.3f} ({100 * (R[c]['median30'] / ff['median30'] - 1):+.1f}%) E21 {R[c]['E21']:.3f} f/e {R[c]['f_per_electron']:.2f}"
            for c in ("FF", "DON", "EDL", "EDLX")))
empty = [(r, [q for q in S if q["NA"] == r["NA"] and q["ns"] == r["ns"] and q["case"] == "FF"][0]) for r in S
         if r["case"] == "DON" and r["cap_frac"] < 0.01]
print("  DON vs FF with an empty cap: max |dEbar| %", round(max(abs(a["mean_coll"] / b["mean_coll"] - 1) for a, b in empty) * 100, 2),
      "max |dmedian30| %", round(max(abs(a["median30"] / b["median30"] - 1) for a, b in empty) * 100, 2),
      "max |dE21| meV", round(max(abs(a["E21"] - b["E21"]) for a, b in empty) * 1e3, 1))
edl = [r for r in S if r["case"] == "EDL" and r["ns"] >= 5e13]
for NA in (5e19, 1e18):
    e = [r for r in edl if r["NA"] == NA]
    print(f"  EDL {NAME[NA]} ns >= 5e13: n_BSO {min(r['n_bso'] for r in e):.2e}-{max(r['n_bso'] for r in e):.2e}, Ebar "
          f"{min(r['mean_coll'] for r in e):.3f}-{max(r['mean_coll'] for r in e):.3f}, f/e {min(r['f_per_electron'] for r in e):.2f}-{max(r['f_per_electron'] for r in e):.2f}")
    f = [r for r in S if r["NA"] == NA and r["case"] == "FF"]
    print(f"  FF {NAME[NA]}: Ebar {min(r['mean_coll'] for r in f):.3f}-{max(r['mean_coll'] for r in f):.3f}, f/e "
          f"{min(r['f_per_electron'] for r in f):.2f}-{max(r['f_per_electron'] for r in f):.2f}, m_opt/m0 {1/max(r['f_per_electron'] for r in f):.3f}-{1/min(r['f_per_electron'] for r in f):.3f}")
P = J("profiles.json")
print("== profiles (clean, 7e13):", {k: (round(v["cap_frac"], 3), round(v["margin"], 3)) for k, v in P.items()})
if os.path.exists(os.path.join(here, "uq_xmax.json")):
    U = J("uq_xmax.json")
    print("== uq_xmax")
    for k, st in U["stats"].items():
        sp = sorted(st["spearman"].items(), key=lambda kv: -abs(kv[1]))[:4]
        print(f"  {k}: n {st['n']} x_max p5 {st['x_p5']:.1f} p50 {st['x_p50']:.1f} p95 {st['x_p95']:.1f} | reachable with FF "
              f"{st['frac_reachable']:.2f} | x >= 4.1 A {st['frac_x_ge_unit_cell']:.2f} | x >= 25 A {st['frac_x_ge_half_cap']:.2f} | "
              + ", ".join(f"{a} {b:+.2f}" for a, b in sp))
print("== spectra, details")
for r in sorted(S, key=lambda r: (r["NA"], r["case"], r["ns"])):
    if r["case"] in ("DON", "EDLX"):
        print(f"  {NAME[r['NA']]} {r['case']} ns {r['ns']:.2e}: cap {r['cap_frac']:.4f} margin {r['margin_cap']:+.3f}")
for NA in (5e19, 1e18):
    f = sorted([r for r in S if r["NA"] == NA and r["case"] == "FF"], key=lambda r: r["ns"])
    xs = [r["ns"] for r in f]; ys = [r["mean_coll"] for r in f]
    for r in sorted([r for r in S if r["NA"] == NA and r["case"] == "EDL" and r["cap_frac"] > 0.01], key=lambda r: r["ns"]):
        ref = float(np.interp(r["n_bso"], xs, ys))
        print(f"  {NAME[NA]} EDL ns {r['ns']:.2e}: n_BSO {r['n_bso']:.2e} Ebar {r['mean_coll']:.3f} vs field-free Ebar at the same "
              f"BSO density {ref:.3f} ({100 * (r['mean_coll'] / ref - 1):+.1f}%)")
C = J("conv.json")
print("== conv: max |d cap_frac| over nstates and dz",
      max(abs(a["cap_frac"] - b["cap_frac"]) for a in C for b in C if (a["NA"], a["ns"], a["x"]) == (b["NA"], b["ns"], b["x"])))
for o in C:
    print("  ", o)
