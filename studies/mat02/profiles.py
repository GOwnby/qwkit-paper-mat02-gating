"""MAT-02 (6): band profiles for Fig. 1. Clean BSO (N_A = 1e18), n0 = 5e12, n_s = 7e13 cm^-2, 300 K: added positive
charge at the interface (field-free cap), as a donor sheet 10 A inside the cap, and on the cap surface (electrolyte
double layer). Saves z, conduction-band profile, Fermi level, and the electron density split into BSO and cap states.
Writes profiles.json."""
import json
from common2 import *

NS = 7e13
p = dict(BASE); p["NA"] = 1e18
sigP = sigma_P(1e18, p=p)
cases = {"interface": ("interface",), "donor sheet 10 A": ("sheet", 10.0), "double layer": ("surface",)}
out = {}
for name, place in cases.items():
    r = solve_placed(NS, p, sigP, place)
    z = r["z"]; dzz = z[1] - z[0]
    capw = np.array([np.sum(r["psi"][z < 0, j] ** 2) * dzz for j in range(len(r["E"]))])
    psi2 = r["psi"] ** 2
    n_bso = psi2[:, capw <= 0.5] @ r["n_n"][capw <= 0.5] * 1e8
    n_cap = psi2[:, capw > 0.5] @ r["n_n"][capw > 0.5] * 1e8
    sel = z < 150
    f, m = cap_state(r)
    out[name] = dict(z=z[sel].tolist(), V=r["V"][sel].tolist(), mu=float(r["mu"]), n_bso=n_bso[sel].tolist(),
                     n_cap=n_cap[sel].tolist(), cap_frac=f, margin=m, E=[float(e) for e in r["E"][:6]])
    print(f"{name}: cap fraction {f:.3f}, margin {m:+.3f} eV, E - mu {np.round(r['E'][:5] - r['mu'], 3)}", flush=True)
json.dump(out, open(os.path.join(here, "profiles.json"), "w"))
