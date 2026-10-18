"""MAT-02 (1): where must the added positive charge sit for the interface to reach the measured densities without
electrons entering the cap? n0 = 5e12 cm^-2 (Lee et al. 2024); the interface keeps its polarization sheet sigma_P and
the added positive charge (n_s + N_dep(n_s) - sigma_P, i.e. including the change of the depletion charge) is
  (a) a donor sheet a distance x from the interface inside the cap; reported: the largest x for which cap-localized
      states hold < 1% of the electrons (0 if even a sheet next to the interface, centred 1 A inside, fails), and
      whether the field-free cap (all of it at the interface) passes;
  (b) partly in donors decaying from the surface with length lam (oxygen vacancies created by the electrolyte), the
      rest at the interface: the largest share of n_s - n0 in such donors;
  (c) on the cap surface (electrolyte double layer) with a larger conduction-band offset: the smallest offset
      (searched between 1 and 4 eV; 4.00 = not reached).
Variants: compensated / clean BSO, cap permittivity 24 (baseline), 38 and 50, 300 K and 150 K (the temperature of the
largest measured density). Writes gate_map.json."""
import json
from multiprocessing import Pool
from common2 import *

NS_T = [3e13, 5e13, 7e13, 1.08e14, 1.2e14]
THR = 0.01


def frac(ns, p, sigP, place):
    r = solve_placed(ns, p, sigP, place)
    return cap_state(r) + (r["N_dep"],)


def bisect(fun, lo, hi, n=9):
    """Largest t in [lo, hi] with fun(t) < THR, assuming fun increases with t. Returns (t, bracketed)."""
    if fun(lo) >= THR:
        return lo, False
    if fun(hi) < THR:
        return hi, True
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if fun(mid) < THR:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), True


def job(args):
    NA, eps_cap, T = args
    p = dict(BASE); p["NA"] = NA; p["eps_cap"] = eps_cap; p["T"] = T
    sigP = sigma_P(NA, p=p)
    out = dict(NA=NA, eps_cap=eps_cap, T=T, sigma_P=sigP, sheet=[], exp=[], cbo=[])
    for ns in NS_T:
        add = ns - N0
        fFF, mFF, _ = frac(ns, p, sigP, ("interface",))
        xmax, ok = bisect(lambda x: frac(ns, p, sigP, ("sheet", x))[0], 1.0, D_CAP)
        if not ok:
            xmax = 0.0
        out["sheet"].append(dict(ns=ns, x_max=xmax, bracketed=ok, frac_FF=fFF, margin_FF=mFF, FF_ok=bool(fFF < THR)))
        for lam in (10.0, 20.0):
            smax, oks = bisect(lambda s: frac(ns, p, sigP, ("surface_donors", lam, s * add))[0], 0.0, 1.0)
            out["exp"].append(dict(ns=ns, lam=lam, share_max=smax if oks else 0.0, bracketed=oks))
        tmax, okc = bisect(lambda t: frac(ns, dict(p, cbo=4.0 - t), sigP, ("surface",))[0], 0.0, 3.0)
        out["cbo"].append(dict(ns=ns, cbo_min=4.0 - tmax, bracketed=okc))
        print(f"NA {NA:.0e} eps_cap {eps_cap:.0f} T {T:.0f} ns {ns:.2e}: field-free cap frac {fFF:.4f} margin {mFF:+.3f} | donor "
              f"sheet x_max {xmax:.1f} A | surface-donor share max {out['exp'][-2]['share_max']:.2f} (10 A) "
              f"{out['exp'][-1]['share_max']:.2f} (20 A) | double layer needs offset >= {4.0 - tmax:.2f} eV", flush=True)
    return out


if __name__ == "__main__":
    jobs = [(NA, e, T) for NA in (5e19, 1e18) for e in (24.0, 38.0, 50.0) for T in (300.0, 150.0)]
    with Pool(int(os.environ.get("NPROC", "1"))) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(here, "gate_map.json"), "w"), indent=1)
