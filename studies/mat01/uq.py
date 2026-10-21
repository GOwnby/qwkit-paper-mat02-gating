"""MAT-01 study 5: parameter uncertainty. Latin-hypercube Monte Carlo over the literature ranges of
docs/MAT01_literature.md; for n_s = 3e13 and 1e14 cm^-2 (300 K): E21, E_F - E1, the RPA loss peak, centroid and median (energy below which half the absorbed power lies),
the peak sheet chi(2) (RPA, gamma = 30 meV), the LO-phonon 2->1 lifetime (unscreened) and the margin between E_F and the lowest cap conduction-band energy. Spearman rank
correlations identify the parameters that matter. Writes uq.json."""
import json
from multiprocessing import Pool
from scipy.stats import qmc, spearmanr
from common import *

RANGES = dict(m=(0.17, 0.22, "lin"), beta=(0.25, 0.56, "lin"), eps_inf=(3.3, 4.3, "lin"),
              cbo=(1.35, 1.70, "lin"), m_cap=(0.30, 0.50, "lin"), NA=(1e18, 5e19, "log"), E_pin=(0.5, 1.5, "lin"))
NSAMP = 100
HWL = np.arange(0.10, 2.5001, 0.01)
HW2 = np.arange(0.10, 1.2001, 0.01)
LST = float(np.prod([(l / t) ** 2 for t, l in MODES]))      # eps_s = eps_inf * prod (w_LO/w_TO)^2 (consistent)


def sample():
    s = qmc.LatinHypercube(d=len(RANGES), seed=12).random(NSAMP)
    out = []
    for row in s:
        p = dict(BASE)
        for (k, (a, b, kind)), u in zip(RANGES.items(), row):
            p[k] = float(np.exp(np.log(a) + u * (np.log(b) - np.log(a)))) if kind == "log" else float(a + u * (b - a))
        p["eps_s"] = p["eps_inf"] * LST
        out.append(p)
    return out


def job(args):
    i, p, ns = args
    try:
        r = solve(ns, p)
    except Exception as e:
        return dict(i=i, ns=ns, ok=False, err=str(e))
    E = r["E"]
    cens = {}
    for kern in (False, True):
        k = krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=150, gamma=p["gamma"],
                                    eps_b=eps_b(p, True), kernel=kern, zc=0.0, occ_tol=1e-5)
        L = np.array([k.loss(h) for h in HWL])
        c = np.cumsum(L) / np.sum(L)
        cens[kern] = (float(HWL[int(np.argmax(L))]), float(np.sum(HWL * L) / np.sum(L)), float(np.interp(0.5, c, HWL)))
    # chi(2): collective, gamma = 30 meV (the gamma = 10 meV peaks are accidental double resonances, Sec. 6)
    k = krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=150, gamma=0.030,
                                eps_b=eps_b(p, True), kernel=True, zc=0.0, occ_tol=1e-5)
    c2 = np.abs(np.array([k.sheet_p2(h) for h in HW2])) / 8.8541878128e-12 * 1e18
    import lifetimes as LT
    try:
        lt = LT.job_params(p, ns)
    except Exception:
        lt = dict(pauli_screened=float("nan"))
    return dict(i=i, ns=ns, ok=bool(r["err"] < 1e-3), E21=float(E[1] - E[0]), EF=float(r["mu"] - E[0]),
                peak=cens[True][0], centroid=cens[True][1], centroid_bare=cens[False][1], ratio=cens[True][1] / cens[False][1],
                median=cens[True][2], median_bare=cens[False][2], ratio_med=cens[True][2] / cens[False][2],
                margin_cap=float(np.min(r["V"][r["z"] < 0]) - r["mu"]),
                chi_s=float(c2.max()), chi_hw=float(HW2[int(np.argmax(c2))]), tau21_fs=float(lt["pauli_screened"]),
                margin=float(p["cbo"] - r["mu"]), N_dep=float(r["N_dep"]))


if __name__ == "__main__":
    P = sample()
    jobs = [(i, P[i], ns) for ns in (3e13, 1e14) for i in range(NSAMP)]
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=4)
    out = dict(ranges={k: list(v[:2]) for k, v in RANGES.items()}, samples=P, rows=res, stats={})
    for ns in (3e13, 1e14):
        rows = [r for r in res if r["ns"] == ns and r.get("ok")]
        st = {}
        for q in ("E21", "EF", "peak", "centroid", "centroid_bare", "ratio", "median", "median_bare", "ratio_med", "chi_s", "tau21_fs",
                  "margin", "margin_cap"):
            v = np.array([r[q] for r in rows])
            ok = np.isfinite(v); v = v[ok]
            pct = np.percentile(v, [5, 50, 95])
            rr = [r for r, o in zip(rows, ok) if o]
            rho = {k: float(spearmanr([P[r["i"]][k] for r in rr], v)[0]) for k in RANGES}
            st[q + "_width90"] = float(pct[2] - pct[0])
            st[q] = dict(p5=float(pct[0]), p50=float(pct[1]), p95=float(pct[2]), spearman=rho)
            top = sorted(rho.items(), key=lambda kv: -abs(kv[1]))[:3]
            print(f"ns {ns:.0e} {q:9s}: median {pct[1]:.4g} [{pct[0]:.4g}, {pct[2]:.4g}]  top: " + ", ".join(f"{k} {v:+.2f}" for k, v in top), flush=True)
        st["n_ok"] = len(rows); st["frac_margin_lt_0.2"] = float(np.mean([r["margin_cap"] < 0.2 for r in rows]))
        st["frac_margin_lt_0"] = float(np.mean([r["margin_cap"] < 0.0 for r in rows]))
        out["stats"][f"{ns:.0e}"] = st
    json.dump(out, open(os.path.join(here, "uq.json"), "w"))
