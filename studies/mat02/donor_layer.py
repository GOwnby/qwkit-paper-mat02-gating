"""MAT-02 (5): the added positive charge (n_s + N_dep(n_s) - sigma_P) as a uniformly doped layer of thickness w next to
the interface (ionized donors, e.g. oxygen vacancies in the first LaInO3 layers). Largest w with < 1% of the electrons
in cap states and the corresponding smallest ionized-donor (charge) density, and the vacancy density for doubly
ionized vacancies (half of it) as a fraction of the oxygen sites of LaInO3 (3 per pseudocubic cell of 4.1 A,
4.3e22 cm^-3). n0 = 5e12, 300 K and 150 K, cap permittivity 24.
Writes donor_layer.json."""
import json
from common2 import *
from gate_map import frac, bisect

NO = 3 / (4.1e-8) ** 3
out = []
for NA in (5e19, 1e18):
    for T in (300.0, 150.0):
        p = dict(BASE); p["NA"] = NA; p["T"] = T
        sigP = sigma_P(NA, p=p)
        for ns in (5e13, 7e13, 1.08e14):
            add = ns - N0
            w, ok = bisect(lambda w_: frac(ns, p, sigP, ("layer", w_))[0], 1.0, D_CAP, n=8)
            r = solve_placed(ns, p, sigP, ("layer", w)) if ok else None
            nd = r["N_D"] / (w * 1e-8) if ok else float("nan")
            o = dict(NA=NA, T=T, ns=ns, w_max=w, bracketed=ok, N_D_min=nd, vacancy_density=nd / 2, vacancy_site_fraction=nd / 2 / NO)
            out.append(o)
            print(f"NA {NA:.0e} T {T:.0f} ns {ns:.2e}: uniform donor layer w_max {w:.1f} A ({'ok' if ok else 'not bracketed'}) -> "
                  f"N_D >= {nd:.2e} cm^-3 (V_O2+ on {100 * nd / 2 / NO:.1f}% of O sites)", flush=True)
json.dump(out, open(os.path.join(here, "donor_layer.json"), "w"), indent=1)
