"""MAT-02 (4): parameter uncertainty of the gating limit. The 100 Latin-hypercube samples of the MAT-01 uncertainty
study (studies/mat01/uq.py: BSO mass and nonparabolicity, eps_inf (eps_s by LST), offset, cap mass, acceptor density,
pinning; cap permittivity 24) at 300 K, n0 = 5e12 cm^-2: the largest distance x_max of the added positive charge from
the interface (donor sheet) with < 1% of the electrons in cap states, at n_s = 7e13 and 1.08e14 cm^-2, and the cap
fraction and margin with all positive charge at the interface. Spearman correlations. Writes uq_xmax.json."""
import json
from multiprocessing import Pool
from scipy.stats import spearmanr
from common2 import *
from gate_map import frac, bisect
import uq as U

NS_U = [7e13, 1.08e14]


def job(args):
    i, p = args
    out = dict(i=i, params={k: p[k] for k in U.RANGES})
    try:
        sigP = sigma_P(p["NA"], p=p)
        for ns in NS_U:
            xmax, ok = bisect(lambda x: frac(ns, p, sigP, ("sheet", x))[0], 1.0, D_CAP, n=8)
            if not ok:
                xmax = 0.0
            fFF, mFF, _ = frac(ns, p, sigP, ("interface",))
            out[f"{ns:.2e}"] = dict(x_max=xmax, bracketed=ok, frac_FF=fFF, margin_FF=mFF)
        out["ok"] = True
    except Exception as e:
        out.update(ok=False, err=str(e))
    print(i, {k: v for k, v in out.items() if k not in ("params",)}, flush=True)
    return out


if __name__ == "__main__":
    S = U.sample()
    with Pool(int(os.environ.get("NPROC", "1"))) as pool:
        res = pool.map(job, list(enumerate(S)), chunksize=1)
    stats = {}
    good = [r for r in res if r.get("ok")]
    for ns in NS_U:
        key = f"{ns:.2e}"
        x = np.array([r[key]["x_max"] for r in good]); f = np.array([r[key]["frac_FF"] for r in good])
        st = dict(n=len(x), x_p5=float(np.percentile(x, 5)), x_p50=float(np.median(x)), x_p95=float(np.percentile(x, 95)),
                  frac_reachable=float(np.mean(f < 0.01)), frac_x_ge_unit_cell=float(np.mean(x >= 4.1)),
                  frac_x_ge_half_cap=float(np.mean(x >= 25.0)), spearman={})
        for k in U.RANGES:
            v = np.array([r["params"][k] for r in good])
            st["spearman"][k] = float(spearmanr(v, x).correlation)
        stats[key] = st
        print(key, st, flush=True)
    json.dump(dict(rows=res, stats=stats), open(os.path.join(here, "uq_xmax.json"), "w"), indent=1)
