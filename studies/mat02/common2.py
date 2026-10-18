"""MAT-02 shared set-up: the MAT-01 interface model (studies/mat01/common.py) with the added positive charge of
electrolyte gating placed on the cap surface, inside the cap (ionized donors, e.g. oxygen vacancies) or at the
interface. Lee et al., Adv. Electron. Mater. 2024, 2400811: 5 nm LaInO3 cap, Hall density about 5e12 cm^-2 without
gate bias, 7.0e13 at 10 K and 1.08e14 at 150 K at V_GS = 8 V."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "mat01")); sys.path.insert(0, os.path.join(here, "..", ".."))
from common import *                     # noqa: F401,F403  (BASE, solve, materials, I, np, ...)
here = os.path.dirname(os.path.abspath(__file__))   # (the star import brings in studies/mat01's `here`)

N0 = 5e12                                # ungated density of the gated samples (Lee 2024, Fig. 3b)
D_CAP = 50.0                             # A


DZ = 0.5                                 # A, the same grid for compensated and clean BSO (common.solve would use 1 A
                                         # for the 808 A box of the clean case)


def sigma_P(NA, n0=N0, p=None, **kw):
    """Interface (polarization) sheet that gives the ungated density n0 with a field-free cap."""
    p = dict(BASE if p is None else p); p["NA"] = NA
    r0 = solve(n0, p, extend_to=0.0, dz=DZ, **kw)
    return n0 + r0["N_dep"]


def solve_placed(ns, p, sigP, place, extend_to=0.0, **kw):
    """Self-consistent interface at n_s with the positive charge beyond the polarization sheet sigP (the added charge,
    n_s + N_dep(n_s) - sigP, which includes the change of the depletion charge) placed as
      ("interface",)            all at the interface: field-free cap (the MAT-01 baseline);
      ("sheet", x)              a donor sheet 2 A wide centred x A inside the cap (x is clipped to >= 1 A);
      ("layer", w)              donors uniform over the inner w A of the cap;
      ("surface",)              on the cap surface: an ideal electrolyte double layer;
      ("surface_donors", lam, total)  `total` cm^-2 in donors decaying from the surface with length lam, the rest
                                at the interface."""
    kind = place[0]
    kw = dict(kw, extend_to=extend_to, dz=DZ)
    if kind == "interface":
        q_ = dict(p); q_["f_int"] = 1.0
        return solve(ns, q_, **kw)
    if kind == "sheet":
        return solve(ns, p, sigma_int=sigP, nd_cap=sheet_profile(place[1], 1.0), rest_to="donors", **kw)
    if kind == "layer":
        return solve(ns, p, sigma_int=sigP, nd_cap=uniform_profile(1.0, max(place[1], 1.0)), rest_to="donors", **kw)
    if kind == "surface":
        return solve(ns, p, sigma_int=sigP, **kw)
    if kind == "surface_donors":
        return solve(ns, p, nd_cap=exp_profile(place[1], place[2]), rest_to="interface", **kw)
    raise ValueError(kind)


def _normalized(mask_fn, total):
    """Profile with the given shape (a function of z returning weights >= 0), normalized ON THE GRID to `total` cm^-2
    within the cap (z < 0), so that the discretization does not change the donor sheet density."""
    def prof(z):
        w = np.where(z < 0, mask_fn(z), 0.0).astype(float)
        dzz = z[1] - z[0]
        return w * total / max(np.sum(w) * dzz * 1e-8, 1e-300)
    return prof


def sheet_profile(x, total, w=2.0):
    """Donor sheet of `total` cm^-2 centred a distance x (A) from the interface inside the cap, width w (A)."""
    xc = float(np.clip(x, w / 2, D_CAP - w / 2))
    return _normalized(lambda z: np.abs(z + xc) <= w / 2 + 1e-9, total)


def exp_profile(lam, total):
    """Donors decaying from the cap surface (z = -D_CAP) into the cap with length lam (A), `total` cm^-2."""
    return _normalized(lambda z: np.exp(-(z + D_CAP) / lam), total)


def uniform_profile(total, depth=D_CAP):
    """Donors uniform over the inner `depth` A of the cap (next to the interface)."""
    return _normalized(lambda z: z >= -depth, total)


def cap_state(r):
    """Fraction of the electrons in cap-localized states and the cap margin (min cap band minus mu, eV)."""
    z = r["z"]; dzz = z[1] - z[0]
    capw = np.array([np.sum(r["psi"][z < 0, j] ** 2) * dzz for j in range(len(r["E"]))])
    return float(np.sum(r["n_n"][capw > 0.5]) / r["ns"]), float(np.min(r["V"][z < 0]) - r["mu"])
