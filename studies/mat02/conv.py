"""MAT-02 (7): numerical checks of the cap fraction near the limits: number of self-consistent subbands (8, 14, 20)
and grid (0.5, 0.25 A), compensated at 7e13 (sheet at 6 A) and clean at 1.08e14 (sheet at 3 A and 5 A), 300 K.
Writes conv.json."""
import json
from common2 import *
import common2

out = []
for NA, ns, x in ((5e19, 7e13, 6.0), (1e18, 1.08e14, 3.0), (1e18, 1.08e14, 5.0)):
    p = dict(BASE); p["NA"] = NA
    for dz in (0.5, 0.25):
        common2.DZ = dz
        sigP = sigma_P(NA, p=p)
        for nst in (8, 14, 20):
            r = solve_placed(ns, p, sigP, ("sheet", x), nstates=nst)
            f, m = cap_state(r)
            o = dict(NA=NA, ns=ns, x=x, dz=dz, nstates=nst, cap_frac=f, margin=m, occupied=int((r["n_n"] > 1e-3 * ns).sum()))
            out.append(o)
            print(o, flush=True)
common2.DZ = 0.5
json.dump(out, open(os.path.join(here, "conv.json"), "w"), indent=1)
