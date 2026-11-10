"""ZZZUWK: does preferred orientation explain its intensities? (referee 1, RSC Advances; resubmission D2)

The submitted text (Section 4.4) said the intensity changes "do not follow the angle to the c axis" but quoted three
reflections that do follow it. Here every reflection is tabulated, and March-Dollase texture is FITTED to each of the
20 measured patterns against its own powder-study entry: every low-index axis [uvw] in reciprocal space with
components in {-2..2}, r on a log grid 0.25-4 (plates r < 1, needles r > 1), and a zero offset of +-0.1 deg. The
factor is averaged over each reflection's symmetry-equivalent members (augment._march_dollase with the CIF rotations).
Score: Pearson correlation over the measured range, background-subtracted pattern as in the benchmark.

    PYTHONPATH=src python scripts/42_zzzuwk_texture.py     -> results/zzzuwk_texture.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from importlib import import_module
from math import gcd

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
bench = import_module("19_real_bench")
hubm = import_module("21_hubness")
from xrdclip.data import augment as A          # noqa: E402

CASE = "ZZZUWK"
R_GRID = np.geomspace(0.25, 4.0, 33)
ZEROS = np.arange(-0.1, 0.1001, 0.02)


def axes() -> list:
    out, seen = [], set()
    for v in itertools.product(range(-2, 3), repeat=3):
        if v == (0, 0, 0):
            continue
        g = gcd(gcd(abs(v[0]), abs(v[1])), abs(v[2]))
        v = tuple(x // g for x in v)
        key = max(v, tuple(-x for x in v))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def pearson(q: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
    a, b = q[m] - q[m].mean(), y[m] - y[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main() -> int:
    shards = bench.shard_table(None)
    lib_rc = hubm.library_refcodes(shards)
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, _, Q = bench.build_queries(f"{bench.DATA}/opxrd/opxrd_truth.json",
                                      f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json", gidx)
    truth = {t["id"]: t for t in json.load(open(f"{bench.DATA}/opxrd/opxrd_truth.json"))}
    rows = [i for i, q in enumerate(qinfo) if q["stem"] == CASE]
    own = [next(x["refcode"] for x in truth[qinfo[i]["id"]]["same_form"]
                if x["is_anchor"] and x["is_powder_study"] and x["rmsd"] < 0.05) for i in rows]
    recs = hubm.records_at(shards, [gidx[rc] for rc in own], bench.RENDER_FIELDS + ("ops",))
    AX = axes()
    per, table_rows = [], {}
    for j, i in enumerate(rows):
        q = Q["raw_bkg"][i]
        m = q > 0
        m = (A.GRID >= A.GRID[m][0]) & (A.GRID <= A.GRID[m][-1])
        rec = dict(recs[j])
        rec["rot"] = np.asarray(rec["ops"])[:, :, :3]
        base = max((pearson(q, A.render_sticks(rec, A.sweep_params(zero_offset=z)), m), z) for z in ZEROS)
        z0 = base[1]
        best = (base[0], None, 1.0)
        for ax in AX:
            for r in R_GRID:
                p = pearson(q, A.render_sticks(rec, A.sweep_params(zero_offset=z0, march=float(r), po_axis=ax)), m)
                if p > best[0]:
                    best = (p, ax, float(r))
        per.append({"id": qinfo[i]["id"], "own_entry": own[j], "zero_deg": round(z0, 2),
                    "pearson_no_texture": round(base[0], 4), "pearson_best_texture": round(best[0], 4),
                    "best_axis": best[1], "best_r": round(best[2], 3)})
        print(per[-1], flush=True)
        # per-reflection heights (measured vs simulated without and with the fitted texture), relative to max
        tt = np.asarray(rec["two_theta"]) + z0
        hkl = np.asarray(rec["hkl"])
        recip = np.asarray(rec["recip"])
        y0 = A.render_sticks(rec, A.sweep_params(zero_offset=z0))
        yt = (A.render_sticks(rec, A.sweep_params(zero_offset=z0, march=best[2], po_axis=best[1]))
              if best[1] is not None else y0)
        c_star = np.array([0, 0, 1.0]) @ recip
        for t, h in zip(tt, hkl):
            if not (A.GRID[m][0] <= t <= A.GRID[m][-1]):
                continue
            w = np.abs(A.GRID - t) <= 0.15
            if y0[w].max() < 0.03:
                continue
            g = h @ recip
            phi = float(np.degrees(np.arccos(min(1.0, abs(g @ c_star) / np.linalg.norm(g) / np.linalg.norm(c_star)))))
            k = f"{tuple(int(x) for x in h)}"
            table_rows.setdefault(k, {"two_theta": round(float(t), 2), "angle_to_c_star_deg": round(phi, 1),
                                      "measured": [], "sim": [], "sim_textured": []})
            table_rows[k]["measured"].append(float(q[w].max() / q[m].max()))
            table_rows[k]["sim"].append(float(y0[w].max()))
            table_rows[k]["sim_textured"].append(float(yt[w].max()))
    table = sorted(({"hkl": k, **{a: v[a] for a in ("two_theta", "angle_to_c_star_deg")},
                     "n": len(v["measured"]), "measured_mean": round(float(np.mean(v["measured"])), 3),
                     "sim_mean": round(float(np.mean(v["sim"])), 3),
                     "sim_textured_mean": round(float(np.mean(v["sim_textured"])), 3),
                     "ratio_measured_over_sim": round(float(np.mean(v["measured"]) / max(np.mean(v["sim"]), 1e-6)), 2)}
                    for k, v in table_rows.items()), key=lambda r: r["two_theta"])
    ax_count = {}
    for p in per:
        ax_count[str(p["best_axis"])] = ax_count.get(str(p["best_axis"]), 0) + 1
    out = {"n_patterns": len(per), "axes_tested": len(AX), "r_grid": [float(R_GRID[0]), float(R_GRID[-1]), len(R_GRID)],
           "median_pearson_no_texture": float(np.median([p["pearson_no_texture"] for p in per])),
           "median_pearson_best_texture": float(np.median([p["pearson_best_texture"] for p in per])),
           "best_axis_counts": ax_count, "median_best_r": float(np.median([p["best_r"] for p in per])),
           "per_pattern": per, "reflections": table}
    json.dump(out, open("results/zzzuwk_texture.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_pattern", "reflections")}, indent=1))
    for r in table:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
