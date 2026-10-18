"""MAT-02 (3): curves for the figures. n0 = 5e12 cm^-2, 300 K, n_s from 1e13 to 1.2e14 cm^-2:
  x_max(n_s)   largest distance of the added positive charge (a donor sheet carrying n_s + N_dep(n_s) - sigma_P) from
               the interface with < 1% of the electrons in cap states (cap permittivity 24, 38, 50); 0 if a sheet next
               to the interface (centred 1 A inside the cap) fails;
  cbo_min(n_s) smallest conduction-band offset for which the electrolyte double layer (all added charge on the cap
               surface) keeps < 1% of the electrons out of the cap (cap permittivity 24; searched from 1 to 6 eV);
  margin_FF    cap margin and cap fraction with all positive charge at the interface (field-free cap).
Writes gate_curves.json."""
import json
from multiprocessing import Pool
from common2 import *
from gate_map import frac, bisect, THR

NS_C = [1e13, 2e13, 3e13, 4e13, 5e13, 6e13, 7e13, 8e13, 9e13, 1e14, 1.08e14, 1.2e14]


def job(args):
    NA, eps_cap = args
    p = dict(BASE); p["NA"] = NA; p["eps_cap"] = eps_cap
    sigP = sigma_P(NA, p=p)
    rows = []
    for ns in NS_C:
        fFF, mFF, ndep = frac(ns, p, sigP, ("interface",))
        xmax, okx = bisect(lambda x: frac(ns, p, sigP, ("sheet", x))[0], 1.0, D_CAP, n=8)
        if not okx:
            xmax = 0.0
        row = dict(ns=ns, x_max=xmax, x_bracketed=okx, frac_FF=fFF, margin_FF=mFF, N_dep=ndep,
                   added=ns + ndep - sigP)
        if eps_cap == 24.0:
            tmax, okc = bisect(lambda t: frac(ns, dict(p, cbo=6.0 - t), sigP, ("surface",))[0], 0.0, 5.0, n=9)
            row.update(cbo_min=6.0 - tmax, cbo_bracketed=okc)
        rows.append(row)
        print(f"NA {NA:.0e} eps {eps_cap:.0f} ns {ns:.2e}: x_max {xmax:.1f} A | field-free cap: frac {fFF:.3f} margin {mFF:+.3f}"
              + (f" | double layer offset >= {row['cbo_min']:.2f} eV" if "cbo_min" in row else ""), flush=True)
    return dict(NA=NA, eps_cap=eps_cap, sigma_P=sigP, rows=rows)


if __name__ == "__main__":
    jobs = [(NA, e) for NA in (5e19, 1e18) for e in (24.0, 38.0, 50.0)]
    with Pool(int(os.environ.get("NPROC", "1"))) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(here, "gate_curves.json"), "w"), indent=1)
