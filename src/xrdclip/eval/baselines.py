"""Classical search-match baselines — the methods XRD-CLIP must beat (SPEC §4.2).

Implemented BEFORE the model, on purpose. If the classical baselines already saturate under
realistic degradation, there is no paper here and we want to know that in week 0, not week 3.

Two baselines, because they fail differently:

  correlation_search_match
      Bin the query and every reference onto the same grid; rank by Pearson correlation. This is
      the workhorse. Its weakness is structural: it compares intensities bin-by-bin, so ANY peak
      shift misaligns every peak at once and the score collapses. Strain and zero-offset are
      therefore near-worst-case for it.

  peak_list_match
      Extract the query's strongest peak positions; score each reference by how many of its peaks
      fall within a 2theta tolerance. This is the honest strong baseline: by construction it is far
      more shift-tolerant than correlation, because the tolerance window absorbs small offsets.

Beating (1) is easy and unimpressive. Beating (2) under texture and mixtures is the real claim.
"""
from __future__ import annotations

import numpy as np

from xrdclip.data import augment as A


# ------------------------------------------------------------------------------------------
# baseline 1: full-pattern correlation
# ------------------------------------------------------------------------------------------

def build_reference_patterns(sticks_list: "list[dict]", params: A.Params = A.CLEAN) -> np.ndarray:
    """Render the reference library once. Shape (n_ref, N_POINTS), z-scored per pattern."""
    out = np.zeros((len(sticks_list), A.N_POINTS), dtype=np.float32)
    for i, s in enumerate(sticks_list):
        out[i] = A.standardize(A.render_sticks(s, params))
    return out


def correlation_search_match(queries: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Rank references by Pearson correlation.

    Both `queries` and `refs` must already be z-scored per row, which turns Pearson correlation
    into a plain dot product / N -- so the whole library search is one matmul.

    Returns (n_query, n_ref) similarity scores.
    """
    q = np.ascontiguousarray(queries, dtype=np.float32)
    r = np.ascontiguousarray(refs, dtype=np.float32)
    return (q @ r.T) / q.shape[1]


# ------------------------------------------------------------------------------------------
# baseline 2: peak-list / Hanawalt-style matching
# ------------------------------------------------------------------------------------------

def extract_peaks(y: np.ndarray, grid: np.ndarray = A.GRID, n_peaks: int = 20,
                  min_height: float = 0.05, min_sep_deg: float = 0.15) -> np.ndarray:
    """Top-`n_peaks` peak positions, strongest first, with a minimum separation.

    Operates on a max-normalised (not z-scored) pattern.
    """
    idx = []
    order = np.argsort(-y)
    min_sep = int(min_sep_deg / (grid[1] - grid[0]))
    for i in order:
        if y[i] < min_height:
            break
        if any(abs(i - j) < min_sep for j in idx):
            continue
        idx.append(i)
        if len(idx) >= n_peaks:
            break
    return grid[np.array(sorted(idx), dtype=int)] if idx else np.array([], dtype=np.float32)


def _peak_indicator(peaks_list: "list[np.ndarray]", tol_deg: float,
                    grid: np.ndarray = A.GRID, dilate: bool = False) -> np.ndarray:
    """Binary (n, N_POINTS) peak-presence matrix; optionally dilated by +/- tol_deg.

    This is what turns peak-list matching into a matmul. A naive double loop over
    (n_query x n_ref) pairs is O(10^9) for a realistic library and would make the baseline the
    slowest thing in the project -- which would be an absurd reason to weaken it.
    """
    step = grid[1] - grid[0]
    half = int(round(tol_deg / step))
    M = np.zeros((len(peaks_list), grid.size), dtype=np.float32)
    for i, pk in enumerate(peaks_list):
        if pk.size == 0:
            continue
        idx = np.clip(np.round((pk - grid[0]) / step).astype(int), 0, grid.size - 1)
        if not dilate:
            M[i, idx] = 1.0
        else:
            for o in range(-half, half + 1):
                M[i, np.clip(idx + o, 0, grid.size - 1)] = 1.0
    return M


def peak_list_match(query_peaks: "list[np.ndarray]", ref_peaks: "list[np.ndarray]",
                    tol_deg: float = 0.15) -> np.ndarray:
    """Symmetric (F1) peak-list match score. Returns (n_query, n_ref).

    A one-sided score -- "what fraction of QUERY peaks does this reference cover?" -- is broken,
    and not subtly: a reference with many peaks covers everything and wins every query. Measured
    on clean self-retrieval it loses ~2% of top-1 to exactly those spurious ties. We must not ship
    a baseline crippled in a way our model isn't; beating a strawman proves nothing.

    So we score both directions and combine:
        precision = query peaks covered by a reference peak  / n_query_peaks
        recall    = reference peaks covered by a query peak  / n_ref_peaks
        F1        = 2PR / (P + R)
    Peak-dense references now pay for their extra peaks through recall. Both directions are a
    matmul against a tolerance-dilated indicator, so this stays fast.
    """
    Q_sharp = _peak_indicator(query_peaks, tol_deg, dilate=False)
    Q_dil = _peak_indicator(query_peaks, tol_deg, dilate=True)
    R_sharp = _peak_indicator(ref_peaks, tol_deg, dilate=False)
    R_dil = _peak_indicator(ref_peaks, tol_deg, dilate=True)

    n_q = np.array([max(len(p), 1) for p in query_peaks], dtype=np.float32)[:, None]
    n_r = np.array([max(len(p), 1) for p in ref_peaks], dtype=np.float32)[None, :]

    precision = (Q_sharp @ R_dil.T) / n_q      # query peaks explained by the reference
    recall = (Q_dil @ R_sharp.T) / n_r         # reference peaks explained by the query

    denom = precision + recall
    return np.where(denom > 0, 2.0 * precision * recall / np.maximum(denom, 1e-8), 0.0).astype(np.float32)


# ------------------------------------------------------------------------------------------
# shared retrieval metrics
# ------------------------------------------------------------------------------------------

def retrieval_metrics(scores: np.ndarray, truth: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Top-k accuracy and MRR from a (n_query, n_ref) score matrix.

    `truth[i]` is the index into the reference library of query i's correct structure.
    """
    order = np.argsort(-scores, axis=1)
    ranks = np.array([int(np.where(order[i] == truth[i])[0][0]) for i in range(len(truth))])
    out = {f"top{k}": float((ranks < k).mean()) for k in ks}
    out["mrr"] = float((1.0 / (ranks + 1)).mean())
    out["median_rank"] = float(np.median(ranks) + 1)
    out["n_query"] = int(len(truth))
    out["n_ref"] = int(scores.shape[1])
    return out
