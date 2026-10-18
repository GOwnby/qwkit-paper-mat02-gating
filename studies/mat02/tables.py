"""LaTeX rows of Table 1 of the MAT-02 manuscript (from gate_map.json): x_max (nm), surface-donor share (lam = 1 nm,
2 nm) and the double-layer offset at n_s = 7e13 and 1.08e14 cm^-2. Writes ../../paper_mat02/table_map.tex."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(here, "gate_map.json")))
NAME = {5e19: "compensated", 1e18: "clean"}
rows = []
for NA in (5e19, 1e18):
    for T in (300.0, 150.0):
        for e in (24.0, 38.0, 50.0):
            g = [x for x in G if x["NA"] == NA and x["T"] == T and x["eps_cap"] == e][0]
            cells = []
            for ns in (7e13, 1.08e14):
                a = [s for s in g["sheet"] if s["ns"] == ns][0]
                e1, e2 = [s for s in g["exp"] if s["ns"] == ns]
                c = [s for s in g["cbo"] if s["ns"] == ns][0]
                x = "--" if (a["x_max"] == 0 and not a["bracketed"]) else f"{a['x_max'] / 10:.2f}"
                off = "$>4$" if c["cbo_min"] >= 3.999 else f"{c['cbo_min']:.2f}"
                cells += [x, f"{e1['share_max']:.2f} ({e2['share_max']:.2f})", off]
            lab = f"{NAME[NA]}, {T:.0f} K" if e == 24.0 else ""
            rows.append(f"{lab} & {e:.0f} & " + " & ".join(cells) + r" \\")
        if not (NA == 1e18 and T == 150.0):
            rows.append(r"\addlinespace")
open(os.path.join(here, "..", "..", "paper_mat02", "table_map.tex"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
