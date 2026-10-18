"""MAT-02 (2): what the intersubband response sees of where the gate charge sits. For n0 = 5e12 and n_s = 3e13, 5e13,
7e13, 1.08e14 (300 K, compensated and clean), the same electron density with the added positive charge
  FF   at the interface (field-free cap, the MAT-01 baseline),
  DON  as a donor sheet 5 A inside the cap,
  EDL  on the cap surface (electrolyte double layer), baseline offset 1.6 eV (the cap fills at high density),
  EDLX on the cap surface with a 3.0 eV offset (no cap filling; a model device, beyond the measured offsets).
Reported per case: cap fraction and margin, E21, the k = 0 TDH mean absorption energy (bare and collective, as in
studies/mat01/meanmode.py), the integrated z-oscillator strength per electron, f = (m0/n_s) sum N E z^2 2/hbar^2
(the f-sum weight, i.e. m0 over the mean optical mass), the k-resolved RPA median at gamma = 30 meV, and the density
in BSO-localized states (what a Hall measurement dominated by the high-mobility interface channel counts).
Writes spectra.json."""
import json
from multiprocessing import Pool
from common2 import *
from linear import metrics

QE = 1.602176634e-19; EPS0 = 8.8541878128e-12; HBAR = 1.054571817e-34; M0 = 9.1093837015e-31
NS = [3e13, 5e13, 7e13, 1.08e14]
HW = np.arange(0.10, 2.5001, 0.005)


def tdh_k0(r, p):
    z, psi, E = r["z"], r["psi"], r["E"]; dz = z[1] - z[0]
    nn = r["n_n"] * 1e4
    pairs = [(a, b) for a in range(len(E)) for b in range(a + 1, len(E)) if nn[a] - nn[b] > 1e-6 * nn.sum()]
    Ep = np.array([E[b] - E[a] for a, b in pairs]); Np = np.array([nn[a] - nn[b] for a, b in pairs])
    F = np.array([np.cumsum(psi[:, a] * psi[:, b]) * dz for a, b in pairs]).T
    K = (F.T @ F) * dz * 1e-10 * QE / (EPS0 * p["eps_inf"])
    Z = np.array([np.sum(psi[:, a] * z * psi[:, b]) * dz for a, b in pairs])
    s = np.sqrt(Ep * Np)
    w2, X = np.linalg.eigh(np.diag(Ep ** 2) + 2 * s[:, None] * K * s[None, :])
    Om = np.sqrt(np.clip(w2, 0, None)); fb = Ep * Np * Z ** 2; fc = (X.T @ (s * Z)) ** 2
    f_per_e = float(2 * fb.sum() * QE * 1e-20 / HBAR ** 2 * M0 / nn.sum())
    i = int(np.argmax(fc))
    return dict(mean_bare=float(np.sum(fb * Ep) / fb.sum()), mean_coll=float(np.sum(fc * Om) / fc.sum()),
                top_E=float(Om[i]), top_share=float(fc[i] / fc.sum()), f_per_electron=f_per_e)


def job(args):
    NA, ns, case = args
    p = dict(BASE); p["NA"] = NA
    sigP = sigma_P(NA, p=p)
    place = {"FF": ("interface",), "DON": ("sheet", 5.0), "EDL": ("surface",), "EDLX": ("surface",)}[case]
    if case == "EDLX":
        p["cbo"] = 3.0
    r = solve_placed(ns, p, sigP, place, extend_to=3.0)
    cf, mc = cap_state(r)
    z = r["z"]; dzz = z[1] - z[0]
    capw = np.array([np.sum(r["psi"][z < 0, j] ** 2) * dzz for j in range(len(r["E"]))])
    o = dict(NA=NA, ns=ns, case=case, cap_frac=cf, margin_cap=mc, E21=float(r["E"][1] - r["E"][0]),
             n_bso=float(np.sum(r["n_n"][capw <= 0.5])), N_dep=float(r["N_dep"]), N_D=float(r.get("N_D", 0.0)))
    o.update(tdh_k0(r, p))
    k = krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=200, gamma=0.030,
                                eps_b=eps_b(p, True), kernel=True, zc=0.0, occ_tol=1e-5)
    L = np.array([k.loss(h) for h in HW])
    c = np.cumsum(L) / L.sum()
    o["median30"] = float(np.interp(0.5, c, HW)); o["main30"] = metrics(HW, L)["main"]
    o["spec30"] = (L / L.max()).tolist()
    print(f"NA {NA:.0e} ns {ns:.2e} {case:4s}: cap frac {cf:.3f} margin {mc:+.3f} | E21 {o['E21']:.3f} | mean bare {o['mean_bare']:.3f} "
          f"coll {o['mean_coll']:.3f} | top {o['top_E']:.3f} ({o['top_share']:.2f}) | f/e {o['f_per_electron']:.2f} | median30 {o['median30']:.3f} "
          f"| n in BSO states {o['n_bso']:.2e}", flush=True)
    return o


if __name__ == "__main__":
    jobs = [(NA, ns, c) for NA in (5e19, 1e18) for ns in NS for c in ("FF", "DON", "EDL", "EDLX")]
    with Pool(int(os.environ.get("NPROC", "1"))) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(dict(hw=HW.tolist(), rows=res), open(os.path.join(here, "spectra.json"), "w"))
