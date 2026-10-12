"""Is the learned model's failure discrimination or the measured-to-simulated gap? (referee 10-12)

  PYTHONPATH=src python scripts/26_model_probe_check.py      -> results/model_probe_check.json

The 8,000 hubness probes (21 prep) are library entries rendered either cleanly, exactly like the library
patterns, or with the training-time degradations and then background-subtracted like the measured queries.
For each probe the model scores the probe's pattern against all 309,080 structure embeddings; the correct
answer is the probe's own entry. Entries of the same CSD refcode family (same 6-letter stem) are set aside,
so redeterminations and polymorphs neither count as hits nor as errors. If the model already fails on clean
simulated probes, its problem is discrimination at library scale, not the gap to measured data.
"""
from __future__ import annotations

import json
import os
import sys
from importlib import import_module

import numpy as np
import torch

sys.path.insert(0, "scripts")
bench = import_module("19_real_bench")
hubm = import_module("21_hubness")

from xrdclip.data import augment as A          # noqa: E402
from xrdclip.models.clip import XRDClip        # noqa: E402


def main() -> int:
    work, ckpt = "cache/real_bench", "runs/xrdclip_poly/best.pt"
    torch.set_num_threads(4)
    shards = bench.shard_table(None)
    lib_rc = hubm.library_refcodes(shards)
    stem = hubm.stems_of(lib_rc)
    lib_emb = torch.from_numpy(np.concatenate([np.load(os.path.join(
        work, f"emb_{os.path.basename(os.path.dirname(p))}_{os.path.basename(p)[:-4]}_{n}.npy")) for p, n, _ in shards]
    ).astype(np.float32))
    hub = np.load(os.path.join(work, "hub_ctx.npz"))
    probes, pg = hub["P__raw"], hub["probe_g"]
    P = len(probes) // 2
    model = XRDClip(embed_dim=256)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    model.eval()
    with torch.no_grad():
        x = np.stack([A.standardize(y) for y in probes])
        pe = torch.cat([model.encode_pattern(torch.from_numpy(x[i:i + 256]).unsqueeze(1)) for i in range(0, len(x), 256)])
    out = {"n_library": int(lib_emb.shape[0])}
    for i, name in enumerate(hubm.PROBE_SETS):
        ranks = []
        for j in range(i * P, (i + 1) * P):
            s = (pe[j] @ lib_emb.T).numpy()
            g = int(pg[j])
            other = stem != stem[g]
            ranks.append(int((s[other] > s[g]).sum()))
        r = np.array(ranks)
        out[name] = {"n": int(len(r)), "top1": float((r == 0).mean()), "top10": float((r < 10).mean()),
                     "top100": float((r < 100).mean()), "median_rank": float(np.median(r) + 1)}
        print(name, out[name], flush=True)
    # the 188 benchmark targets: each pattern's own (anchor) entry rendered cleanly, scored with the same
    # answer sets as the measured pattern, so the only difference from the measured run is the query itself
    gidx = {rc: i for i, rc in enumerate(lib_rc)}
    qinfo, _, Q = bench.build_queries(f"{bench.DATA}/opxrd/opxrd_truth.json",
                                      f"{bench.DATA}/opxrd/opxrd_cnrs_organic.json", gidx)
    truth = {t["id"]: t for t in json.load(open(f"{bench.DATA}/opxrd/opxrd_truth.json"))}
    anc = [next(a for a in sorted(truth[q["id"]]["anchors"]) if a in gidx) for q in qinfo]
    sim = bench.render(hubm.records_at(shards, [gidx[a] for a in anc], bench.RENDER_FIELDS))
    with torch.no_grad():
        enc = lambda x: torch.cat([model.encode_pattern(torch.from_numpy(x[i:i + 64]).unsqueeze(1))
                                   for i in range(0, len(x), 64)])
        e_sim = enc(np.stack([A.standardize(y) for y in sim]))
        e_meas = enc(Q["std_bkg"])
    for name, E in (("own_entry_clean", e_sim), ("measured_bkg", e_meas)):
        for td in ("std", "sc"):
            ranks = []
            for qi, q in enumerate(qinfo):
                tr, ig = q["sets"][td]
                if not tr:
                    continue
                s = (E[qi] @ lib_emb.T).numpy()
                thr = s[[gidx[rc] for rc in tr]].max()
                above = s > thr
                excl = [gidx[rc] for rc in set(tr + ig)]
                ranks.append(int(above.sum() - above[excl].sum()))
            r = np.array(ranks)
            out[f"{name}|{td}"] = {"n": int(len(r)), "top1": float((r == 0).mean()), "top10": float((r < 10).mean()),
                                   "median_rank": float(np.median(r) + 1)}
            print(name, td, out[f"{name}|{td}"], flush=True)
    with open("results/model_probe_check.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
