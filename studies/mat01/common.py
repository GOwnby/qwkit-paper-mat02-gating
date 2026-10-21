"""MAT-01 shared parameters and helpers (sources: docs/MAT01_literature.md).
BaSnO3: band-edge mass 0.19 m0, Kane nonparabolicity beta = 0.56 1/eV (Niedermeier et al. 2017; theory 0.25),
eps_s 20 (17-20), eps_inf 4.0 at the intersubband energies (3.3 above the phonons, 4.2-4.3 at 0.75 eV), polar modes
TO/LO 135/154, 244.5/421, 628/723 cm^-1 (Stanislavchuk et al. 2012). LaInO3: CBO 1.6 eV (DFT 1.35-1.7), m 0.46,
eps_s 24. Deep acceptors in BaSnO3 N_A = 5e19 cm^-3 (Kim et al. 2019, PLD films) or 1e18 (cleaner MBE films);
pinning E_pin = 1.0 eV below the bulk conduction band. Linewidth gamma = 10 meV HWHM baseline."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from qwkit import interface as I, krpa, scattering as sc

CM = 1.23984198e-4                                   # eV per cm^-1
MODES = [(135 * CM, 154 * CM), (244.5 * CM, 421 * CM), (628 * CM, 723 * CM)]
BASE = dict(m=0.19, beta=0.56, eps_s=20.0, eps_inf=4.0, cbo=1.6, m_cap=0.46, eps_cap=24.0, NA=5e19, E_pin=1.0,
            gamma=0.010, T=300.0, f_int=1.0)
NS = [1e13, 2e13, 3e13, 5e13, 7e13, 1e14]


def materials(p):
    well = dict(V=0.0, m=p["m"], Eg=1.0 / p["beta"], eps_s=p["eps_s"], eps_inf=p["eps_inf"])
    cap = dict(V=p["cbo"], m=p["m_cap"], Eg=5.0, eps_s=p["eps_cap"], eps_inf=4.4)
    return well, cap


def box_length(p):
    """Well-material length: 1.4 x the depletion width at full pinning (from 300 A to 2500 A)."""
    if p["NA"] <= 0:
        return 600.0
    wd = np.sqrt(2 * p["eps_s"] * 8.854e-12 * p["E_pin"] / (1.602e-19 * p["NA"] * 1e6)) * 1e10
    return float(np.clip(1.4 * wd + 150.0, 300.0, 2500.0))


def solve(ns, p=BASE, nstates=8, Lscale=1.0, extend_to=3.0, nmax=90, dz=None, **kw):
    """Self-consistent interface (the density uses the lowest `nstates` subbands), then the full set of subbands up to
    extend_to eV above E_1 (at most nmax) on the converged potential, with exact in-plane band-edge masses and fitted
    nonparabolicity for every subband (the optical response needs the empty upper states)."""
    well, cap = materials(p)
    L = box_length(p) * Lscale
    dz = (0.5 if L <= 800 else 1.0) if dz is None else dz
    r = I.solve(ns, T=p["T"], NA=p["NA"], E_pin=p["E_pin"], f_int=p["f_int"], well=well, cap=cap, L=L, dz=dz,
                nstates=nstates, **kw)
    if extend_to:
        r = extend(r, extend_to, nmax)
    return r


def extend(r, emax_rel=3.0, nmax=90, buffer=0.6, nbuf=12):
    """All subbands up to emax_rel eV above E_1 (at most nmax). The nonparabolic states belong to different
    energy-dependent Hamiltonians and overlap slightly; they are Loewdin-orthonormalized as a set that extends
    `buffer` eV (and at least nbuf states) beyond the kept range, and only then truncated. Orthonormalizing the kept
    set alone gives its topmost states spurious dipoles (they cannot be orthogonalized against the missing states)."""
    from qwkit import core as q
    z, V, m, Eg = r["z"], r["V"], r["m_edge"], r["Eg"]
    n = len(r["E"])
    while True:
        E, psi = q.solve_states(z, V, m, Eg, nstates=n, E_guess=None, tol=1e-8)
        if E[-1] - E[0] > emax_rel + buffer or n >= nmax + nbuf:
            break
        n = min(nmax + nbuf, n + 8)
    dz = z[1] - z[0]
    psi = q.orthonormalize(z, psi / np.sqrt(np.sum(psi ** 2, axis=0) * dz))
    keep = E - E[0] <= emax_rel
    keep[nmax:] = False
    keep[:len(r["E"])] = True
    E, psi = E[keep], psi[:, keep]
    m_n, G_n = I.inplane_params(z, psi, E, V, m, Eg, emax=max(r["mu"] - E[0], 0.0) + 0.3)
    n_n = np.array([I.subband_density(E[i], m_n[i], G_n[i], r["mu"], r["T"]) for i in range(len(E))])
    r = dict(r); r.update(E=E, psi=psi, m_n=m_n, G_n=G_n, n_n=n_n, n_scf=len(r["E"]))
    return r


def eps_b(p, phonons=True):
    if not phonons:
        return p["eps_inf"]
    return krpa.phonon_eps(p["eps_inf"], MODES)


def response(r, p=BASE, kernel=True, phonons=True, nk=300, gamma=None):
    return krpa.KRPA.from_subbands(r["z"], r["psi"], r["E"], r["m_n"], r["G_n"], r["mu"], r["T"], nk=nk,
                                   gamma=p["gamma"] if gamma is None else gamma, eps_b=eps_b(p, phonons),
                                   kernel=kernel, zc=0.0)


def extent(r, n=0):
    z, psi = r["z"], r["psi"]; dz = z[1] - z[0]
    w = psi[:, n] ** 2
    return float(np.sum(w * z) * dz)
