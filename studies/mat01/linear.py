"""MAT-01 studies 1-2 (revised): subband structure and linear intersubband response of the LaInO3/BaSnO3 2DEG versus
gated sheet density, for N_A = 5e19 (strongly compensated) and 1e18 cm^-3, T = 300 K (and 10 K at four densities).
Subbands: self-consistent with exact in-plane band-edge masses and fitted nonparabolicity; all subbands up to 3 eV
above E_1 (at most 90) enter the response.
Spectra (absorbed power for a z-polarized drive E_d = D/eps_b, 0.10-2.5 eV): single particle and k-resolved RPA with
gamma = 10 meV, RPA with gamma = 30 meV. Metrics in the window: strongest peak, FWHM, centroid, median (energy below which half the
absorbed power lies; insensitive to the Lorentzian tails that bias the centroid), quartiles, fraction within +-50 meV
of the strongest peak (0.88 for a single Lorentzian line with gamma = 10 meV); ratios to the single-particle values
and to E21. Also the margin between the Fermi level and the lowest cap conduction-band energy and the cap weight of
each subband.
Modes (300 K, three densities): k = 0 collective response with gamma = 2 meV; peak energies and the fraction of the
absorption carried by each.
Writes linear.json."""
import json
from multiprocessing import Pool
from common import *

HW = np.arange(0.10, 2.5001, 0.005)
HWM = np.arange(0.10, 2.5001, 0.002)
NSL = [5e12, 1e13, 1.5e13, 2e13, 3e13, 4e13, 5e13, 6e13, 7e13, 8e13, 1e14, 1.2e14]


def metrics(hw, L):
    i = int(np.argmax(L))
    lo = i
    while lo > 0 and L[lo - 1] >= L[i] / 2:
        lo -= 1
    hi = i
    while hi < len(L) - 1 and L[hi + 1] >= L[i] / 2:
        hi += 1
    loc = [j for j in range(1, len(L) - 1) if L[j] > L[j - 1] and L[j] >= L[j + 1] and L[j] > 0.2 * L[i]]
    return dict(main=float(hw[i]), fwhm=float(hw[hi] - hw[lo]), centroid=float(np.sum(hw * L) / np.sum(L)),
                frac50=float(np.sum(L[np.abs(hw - hw[i]) <= 0.05]) / np.sum(L)),
                maxima=[float(hw[j]) for j in loc], weights=[float(L[j] / L[i]) for j in loc])


def modes(r, p):
    """k = 0 collective modes: peaks of the gamma = 2 meV response and their share of the absorption."""
    k = krpa.KRPA(r["z"], r["psi"], r["E"][:, None], (r["n_n"] / r["n_n"].sum())[:, None], np.array([r["n_n"].sum() * 1e4]),
                  gamma=0.002, eps_b=eps_b(p, True), kernel=True, zc=0.0, occ_tol=1e-5)
    L = np.array([k.loss(h) for h in HWM])
    pk = [j for j in range(1, len(L) - 1) if L[j] > L[j - 1] and L[j] >= L[j + 1] and L[j] > 1e-3 * L.max()]
    # assign each grid point to the nearest peak and integrate
    E = HWM[pk]
    owner = np.argmin(np.abs(HWM[:, None] - E[None, :]), axis=1)
    w = np.array([L[owner == i].sum() for i in range(len(pk))]) / L.sum()
    order = np.argsort(-w)[:6]
    return dict(E=[float(E[i]) for i in order], weight=[float(w[i]) for i in order])


def job(args):
    NA, T, ns = args
    p = dict(BASE); p["NA"] = NA; p["T"] = T
    r = solve(ns, p)
    E = r["E"]
    occ = r["n_n"] / r["n_n"].sum()
    Z = (r["psi"].T * r["z"]) @ r["psi"] * (r["z"][1] - r["z"][0])
    out = dict(NA=NA, T=T, ns=ns, n_states=len(E), mu_rel=float(r["mu"] - E[0]), mu_abs=float(r["mu"]),
               E_rel=[float(e - E[0]) for e in E[:8]], occ=occ[:6].tolist(), z1=extent(r), m_n=r["m_n"][:4].tolist(),
               G_n=r["G_n"][:4].tolist(), N_dep=float(r["N_dep"]), w_d=float(r["w_d"]), margin=float(p["cbo"] - r["mu"]),
               iters=int(r["iters"]), err=float(r["err"]), z1n=[float(abs(Z[0, j])) for j in range(1, 6)],
               margin_cap=float(np.min(r["V"][r["z"] < 0]) - r["mu"]), n_basis_emax=float(E[-1] - E[0]),
               capw=[float(np.sum(r["psi"][r["z"] < 0, j] ** 2) * (r["z"][1] - r["z"][0])) for j in range(8)])
    spec = {}
    for tag, kern, g in (("bare", False, 0.010), ("rpa", True, 0.010), ("rpa30", True, 0.030), ("bare30", False, 0.030)):
        k = krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=200, gamma=g,
                                    eps_b=eps_b(p, True), kernel=kern, zc=0.0, occ_tol=1e-5)
        L = np.array([k.loss(h) for h in HW])
        spec[tag] = L
        out[tag] = metrics(HW, L)
    out["ratio_cen_rpa_bare"] = out["rpa"]["centroid"] / out["bare"]["centroid"]
    out["ratio_cen_rpa_E21"] = out["rpa"]["centroid"] / (E[1] - E[0])
    for tag in ("bare", "rpa", "rpa30", "bare30"):                 # median: energy below which half the power lies
        c = np.cumsum(spec[tag]) / np.sum(spec[tag])
        out[tag]["median"] = float(np.interp(0.5, c, HW))
        out[tag]["quartiles"] = [float(np.interp(0.25, c, HW)), float(np.interp(0.75, c, HW))]
    out["ratio_med_rpa_bare"] = out["rpa"]["median"] / out["bare"]["median"]
    if T == 300.0:
        out["spectra"] = {k: (v / v.max()).tolist() for k, v in spec.items() if k in ("bare", "rpa", "rpa30")}
        if ns in (1e13, 3e13, 1e14):
            out["modes"] = modes(r, p)
    return out


if __name__ == "__main__":
    jobs = [(NA, T, ns) for NA in (5e19, 1e18) for T in (300.0, 10.0) for ns in NSL if T == 300.0 or ns in (1e13, 3e13, 5e13, 1e14)]
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=1)
    for o in res:
        print(f"NA {o['NA']:.0e} T {o['T']:3.0f} ns {o['ns']:.1e} (n={o['n_states']}, err {o['err']:.0e}): E21 {o['E_rel'][1]:.3f} "
              f"EF-E1 {o['mu_rel']:.3f} margin {o['margin']:.2f} occ2 {o['occ'][1]:.2f} <z>1 {o['z1']:.1f} m1 {o['m_n'][0]:.3f} | "
              f"bare cen {o['bare']['centroid']:.3f} f50 {o['bare']['frac50']:.2f} | RPA peak {o['rpa']['main']:.3f} cen {o['rpa']['centroid']:.3f} "
              f"f50 {o['rpa']['frac50']:.2f} | g30 cen {o['rpa30']['centroid']:.3f} f50 {o['rpa30']['frac50']:.2f} | "
              f"cen ratio {o['ratio_cen_rpa_bare']:.2f} /E21 {o['ratio_cen_rpa_E21']:.2f} | median bare {o['bare']['median']:.3f} rpa {o['rpa']['median']:.3f} "
              f"ratio {o['ratio_med_rpa_bare']:.2f} | margin_cap {o['margin_cap']:.3f} capw3 {o['capw'][2]:.2f}"
              + (f" | modes {np.round(o['modes']['E'][:4], 3)} w {np.round(o['modes']['weight'][:4], 2)}" if 'modes' in o else ""), flush=True)
    json.dump(dict(hw=HW.tolist(), params=BASE, rows=res), open(os.path.join(here, "linear.json"), "w"))
