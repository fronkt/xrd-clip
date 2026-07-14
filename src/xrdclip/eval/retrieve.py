"""Full-library retrieval evaluation for a trained XRD-CLIP model (SPEC §4.1, §4.3).

Unlike the in-batch proxy used during training, this embeds the ENTIRE test-split structure library
once, then embeds each (optionally degraded) query pattern and ranks the whole library. This is the
number that goes in the paper.

Two entry points:
  evaluate_clean        top-k over the test library, clean queries
  evaluate_robustness   the headline sweep: one degradation dialed up, model vs both baselines
"""
from __future__ import annotations

import numpy as np
import torch

from xrdclip.data import augment as A
from xrdclip.data.dataset import structure_features
from xrdclip.eval import baselines as B


@torch.no_grad()
def embed_library(model, store, refcodes, device, batch=256) -> np.ndarray:
    """Structure embeddings for the whole library. (n_ref, embed_dim), L2-normalised."""
    model.eval()
    embs = []
    for i in range(0, len(refcodes), batch):
        chunk = refcodes[i:i + batch]
        recs = [store[rc] for rc in chunk]
        A_max = max(len(r["atom_z"]) for r in recs)
        az = torch.zeros(len(recs), A_max, dtype=torch.long)
        af = torch.zeros(len(recs), A_max, 3, dtype=torch.float32)
        mask = torch.zeros(len(recs), A_max, dtype=torch.bool)
        lat = torch.zeros(len(recs), 8, dtype=torch.float32)
        for j, r in enumerate(recs):
            n = len(r["atom_z"])
            az[j, :n] = torch.from_numpy(np.asarray(r["atom_z"], dtype=np.int64))
            af[j, :n] = torch.from_numpy(np.asarray(r["atom_frac"], dtype=np.float32))
            mask[j, :n] = True
            lat[j] = torch.from_numpy(structure_features(r))
        z = model.encode_structure(az.to(device), af.to(device), lat.to(device), mask.to(device))
        embs.append(z.cpu().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def embed_queries(model, store, refcodes, params, device, rng, batch=256) -> np.ndarray:
    """Pattern embeddings for queries rendered under `params`. (n_query, embed_dim)."""
    model.eval()
    embs = []
    for i in range(0, len(refcodes), batch):
        chunk = refcodes[i:i + batch]
        pats = np.stack([
            A.standardize(A.render_sticks(store[rc], params, rng=rng)) for rc in chunk
        ])
        x = torch.from_numpy(pats).unsqueeze(1).to(device)
        embs.append(model.encode_pattern(x).cpu().numpy())
    return np.concatenate(embs, axis=0)


def model_retrieval(model, store, refcodes, params, device, rng, ks=(1, 5, 10)) -> dict:
    lib = embed_library(model, store, refcodes, device)
    q = embed_queries(model, store, refcodes, params, device, rng)
    scores = q @ lib.T                                   # cosine (both normalised)
    return B.retrieval_metrics(scores, np.arange(len(refcodes)), ks)


def robustness_sweep(model, store, refcodes, device, seed=0) -> dict:
    """The headline experiment: model vs both baselines, one degradation at a time.

    Returns nested dict: {axis: {level: {model, correlation, peak_list}}}.
    """
    rng = np.random.default_rng(seed)

    # baselines share a fixed clean reference library / peak set
    refs_z = B.build_reference_patterns([store[rc] for rc in refcodes], A.CLEAN)
    ref_peaks = [B.extract_peaks(A.render_sticks(store[rc], A.CLEAN)) for rc in refcodes]
    lib = embed_library(model, store, refcodes, device)
    truth = np.arange(len(refcodes))

    axes = {
        "zero_offset": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
        "strain": [0.0, 0.002, 0.005, 0.008, 0.012],
        "domain_nm": [1e4, 100, 50, 20, 10],
        "march": [1.0, 0.9, 0.8, 0.7, 0.5],
        "counts": [0, 1e5, 1e4, 5e3, 2e3],
    }

    out: dict = {}
    for axis, levels in axes.items():
        out[axis] = {}
        for lv in levels:
            p = A.sweep_params(**{axis: lv})
            # model
            q = embed_queries(model, store, refcodes, p, device, rng)
            m_model = B.retrieval_metrics(q @ lib.T, truth)
            # correlation
            q_z = np.stack([A.standardize(A.render_sticks(store[rc], p, rng=rng)) for rc in refcodes])
            m_corr = B.retrieval_metrics(B.correlation_search_match(q_z, refs_z), truth)
            # peak-list
            q_pk = [B.extract_peaks(A.render_sticks(store[rc], p, rng=rng)) for rc in refcodes]
            m_peak = B.retrieval_metrics(B.peak_list_match(q_pk, ref_peaks), truth)
            out[axis][str(lv)] = {"model": m_model, "correlation": m_corr, "peak_list": m_peak}
    return out
