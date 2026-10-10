"""Strong classical baselines for v2 (docs/continuation-plan-2026-10.md, Phase 0).

The URTC baselines (fixed-grid Pearson, naive peak list) cannot absorb a peak shift by construction,
so beating them under shift/strain says little. A powder-diffraction reviewer will ask for the
methods the field actually uses, all of which correct shifts explicitly:

  B1  wcc_scores        de Gelder (2001) weighted cross-correlation, maximised over a 2theta
                        zero-shift scan. With l = 0 it degenerates to "Pearson + zero scan".
  B2  lnq_wcc_scores    the same, on a ln(q) grid. Isotropic strain d -> d(1+e) is an exact uniform
                        translation in ln q (ln q shifts by -ln(1+e)), so a lag scan absorbs it. A
                        2theta zero offset is NOT uniform in ln q (-cot(theta) * delta / 2), so the
                        zero scan is applied first in 2theta. Our construction, from Bragg's law.
  B3  vc_gpwdf_scores   variable-cell Gaussian powder-pattern similarity after Otero-de-la-Roza,
                        J. Appl. Cryst. 57:1401 (2024): an analytic Gaussian-overlap similarity
                        between the query's peak list and the candidate's reflections, maximised
                        over the candidate's six cell parameters (+-10 % lengths, +-5 deg angles,
                        the paper's defaults) plus a zero offset (which the paper applies in
                        pre-processing; we fit it jointly).

All scores are "higher = better match" so they drop into the same rank code as everything else.

B3 is a reimplementation, not critic2. Two deliberate differences: (i) the query peak list comes from
simple peak picking on the rendered pattern (critic2 fits peaks with XRPD FIT; ours is cruder, which
favours the model, so it is the conservative choice for us); (ii) symmetry-equivalent reflections
share one representative hkl, so a strain that breaks the crystal's symmetry does not split them.
Validate against critic2 COMPAREVC before quoting B3 numbers in a paper.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from xrdclip.data import augment as A

STEP = float(A.STEP)


# ------------------------------------------------------------------------------------------
# B1: de Gelder weighted cross-correlation + zero-shift scan
# ------------------------------------------------------------------------------------------

def _xcorr(q: np.ndarray, r: np.ndarray, max_lag: int) -> np.ndarray:
    """c[i, j, k] = sum_x q_i(x) r_j(x + lag_k) for lag_k in [-max_lag, max_lag].

    Zero-padded FFT, so there is no wrap-around. Shape (nq, nr, 2*max_lag + 1).
    """
    n = q.shape[1]
    nfft = 1 << int(math.ceil(math.log2(n + max_lag + 1)))
    Fq = np.fft.rfft(q, nfft)
    Fr = np.fft.rfft(r, nfft)
    full = np.fft.irfft(np.conj(Fq)[:, None, :] * Fr[None, :, :], nfft)   # lag >= 0 at front
    pos = full[..., : max_lag + 1]
    neg = full[..., nfft - max_lag:] if max_lag > 0 else full[..., :0]
    return np.concatenate([neg, pos], axis=-1)


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """a[i, k] = sum_x x_i(x) x_i(x + lag_k), lags -max_lag..max_lag. Shape (n, 2*max_lag + 1)."""
    n = x.shape[1]
    nfft = 1 << int(math.ceil(math.log2(n + max_lag + 1)))
    F = np.fft.rfft(x, nfft)
    full = np.fft.irfft(np.abs(F) ** 2, nfft)
    neg = full[:, nfft - max_lag:] if max_lag > 0 else full[:, :0]
    return np.concatenate([neg, full[:, : max_lag + 1]], axis=1)


def _triangle(l_pts: int) -> np.ndarray:
    if l_pts <= 0:
        return np.ones(1)
    r = np.arange(-l_pts, l_pts + 1)
    return np.maximum(0.0, 1.0 - np.abs(r) / l_pts)


def _wcc_core(q: np.ndarray, r: np.ndarray, l_pts: int, shift_pts: int) -> np.ndarray:
    """max over shift d in [-shift_pts, shift_pts] of WCC(q, r shifted by d). Returns (nq, nr)."""
    w = _triangle(l_pts)
    L = len(w) // 2
    c = _xcorr(q, r, L + shift_pts)                                   # lags -(L+S)..(L+S)
    # numerator for every shift d: sum_k w[k] c[d + k - L]  ->  valid correlation with w
    num = np.stack([c[..., d: d + 2 * L + 1] @ w for d in range(2 * shift_pts + 1)], axis=-1)
    aq = _autocorr(q, L) @ w
    ar = _autocorr(r, L) @ w
    den = np.sqrt(np.maximum(aq[:, None] * ar[None, :], 1e-12))
    return (num.max(axis=-1) / den).astype(np.float32)


def wcc_scores(q_raw: np.ndarray, r_raw: np.ndarray, l_deg: float = 1.0,
               shift_deg: float = 0.5) -> np.ndarray:
    """B1. q_raw/r_raw are max-normalised, non-negative patterns on A.GRID, shape (n, N_POINTS).

    l_deg = 0 gives "Pearson + zero scan": inputs are z-scored and only exact alignment counts.
    """
    if l_deg <= 0:
        q = np.stack([A.standardize(y) for y in q_raw])
        r = np.stack([A.standardize(y) for y in r_raw])
    else:
        q = q_raw - q_raw.min(axis=1, keepdims=True)
        r = r_raw - r_raw.min(axis=1, keepdims=True)
    return _wcc_core(q, r, int(round(l_deg / STEP)), int(round(shift_deg / STEP)))


# ------------------------------------------------------------------------------------------
# B2: ln(q) grid -- isotropic strain becomes a pure lag
# ------------------------------------------------------------------------------------------

_TT_LO, _TT_HI = A.TWO_THETA_MIN + 0.5, A.TWO_THETA_MAX - 0.5   # margin for the zero scan
_LNQ = np.linspace(np.log(np.sin(np.radians(_TT_LO / 2))), np.log(np.sin(np.radians(_TT_HI / 2))),
                   A.N_POINTS)
_LNQ_STEP = float(_LNQ[1] - _LNQ[0])
_TT_OF_LNQ = np.degrees(2 * np.arcsin(np.exp(_LNQ)))


def _to_lnq(y: np.ndarray, delta: float = 0.0) -> np.ndarray:
    """Resample a 2theta pattern onto the ln q grid after removing a zero offset `delta`."""
    return np.interp(_TT_OF_LNQ + delta, A.GRID, y).astype(np.float32)


def lnq_wcc_scores(q_raw: np.ndarray, r_raw: np.ndarray, l_lnq: float = 0.01,
                   max_strain: float = 0.03, shift_deg: float = 0.3,
                   shift_step_deg: float = 0.04) -> np.ndarray:
    """B2. Max over (zero offset delta, ln-q lag) of a triangle-weighted cross-correlation.

    l_lnq = 0.01 is ~0.3 deg of 2theta at 2theta = 25 deg. A lag of ln(1 + max_strain) covers
    isotropic strain up to max_strain exactly; the triangle tolerates the anisotropic residue.
    """
    r = np.stack([_to_lnq(y) for y in r_raw])
    r = r - r.min(axis=1, keepdims=True)
    l_pts = max(1, int(round(l_lnq / _LNQ_STEP)))
    lag_pts = int(round(math.log1p(max_strain) / _LNQ_STEP))
    best = None
    for delta in np.arange(-shift_deg, shift_deg + 1e-9, shift_step_deg):
        q = np.stack([_to_lnq(y, delta) for y in q_raw])
        q = q - q.min(axis=1, keepdims=True)
        s = _wcc_core(q, r, l_pts, lag_pts)
        best = s if best is None else np.maximum(best, s)
    return best


# ------------------------------------------------------------------------------------------
# B3: variable-cell Gaussian powder-pattern similarity (VC-GPWDF-style)
# ------------------------------------------------------------------------------------------

def query_peaks(y: np.ndarray, n_peaks: int = 40, min_height: float = 0.02,
                min_sep_deg: float = 0.06) -> "tuple[np.ndarray, np.ndarray]":
    """Local maxima of a max-normalised pattern: (2theta, height), strongest first, capped."""
    y = np.asarray(y, dtype=np.float32)
    is_max = np.r_[False, (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]), False] & (y >= min_height)
    idx = np.where(is_max)[0]
    idx = idx[np.argsort(-y[idx])]
    sep = max(1, int(round(min_sep_deg / STEP)))
    keep: list[int] = []
    for i in idx:
        if all(abs(i - j) >= sep for j in keep):
            keep.append(i)
        if len(keep) >= n_peaks:
            break
    keep = np.array(sorted(keep), dtype=int)
    return A.GRID[keep].astype(np.float32), y[keep].astype(np.float32)


def _recip_from_cell(cell: torch.Tensor) -> torch.Tensor:
    """(P, 6) cell (a, b, c in A; alpha, beta, gamma in deg) -> (P, 3, 3) reciprocal rows, no 2pi.

    Any orientation works: only |h . B| (the metric) enters d-spacings.
    """
    a, b, c = cell[:, 0], cell[:, 1], cell[:, 2]
    al, be, ga = (torch.deg2rad(cell[:, i]) for i in (3, 4, 5))
    ca, cb, cg, sg = torch.cos(al), torch.cos(be), torch.cos(ga), torch.sin(ga)
    zero = torch.zeros_like(a)
    vol_term = torch.clamp(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg, min=1e-8)
    Av = torch.stack([
        torch.stack([a, zero, zero], -1),
        torch.stack([b * cg, b * sg, zero], -1),
        torch.stack([c * cb, c * (ca - cb * cg) / sg, c * torch.sqrt(vol_term) / sg], -1),
    ], dim=1)                                                          # rows = a, b, c vectors
    return torch.linalg.inv(Av).transpose(1, 2)


def _two_theta_from(hkl: torch.Tensor, recip: torch.Tensor) -> torch.Tensor:
    g = torch.einsum("pkh,phx->pkx", hkl, recip)
    inv_d = torch.linalg.norm(g, dim=-1)
    s = torch.clamp(A.WAVELENGTH * inv_d / 2, max=0.9999)
    return torch.rad2deg(2 * torch.asin(s))


def _overlap(t1, w1, m1, t2, w2, m2, sig):
    """sum_ij w1_i w2_j exp(-(t1_i - t2_j)^2 / (4 sig^2)) with masks. Batched over P."""
    d = t1[:, :, None] - t2[:, None, :]
    k = torch.exp(-(d * d) / (4 * sig * sig))
    return torch.einsum("pi,pij,pj->p", w1 * m1, k, w2 * m2)


def vc_gpwdf_scores(q_peaks: "list[tuple[np.ndarray, np.ndarray]]", cands: "list[dict]",
                    pairs: "list[tuple[int, int]]", max_elong: float = 0.10, max_ang: float = 5.0,
                    max_zero: float = 0.5, sigmas=(0.40, 0.20, 0.10, 0.05), steps: int = 40,
                    zero_init: "np.ndarray | None" = None, max_elems: float = 1.5e8,
                    device: str = "cpu",
                    prior: "tuple[float, float, float] | None" = None,
                    return_params: bool = False):
    """B3 similarity in [0, 1] for each (query index, candidate index) pair.

    Coarse-to-fine: the Gaussian width anneals through `sigmas` (deg 2theta), so the wide early
    stages give a smooth landscape and the final narrow stage is discriminating; the score is the
    final-stage similarity. Parameters are bounded through tanh. `zero_init` (per pair) seeds the
    zero offset, e.g. from B1's best shift.

    `prior = (lam, sigma_elong, sigma_ang_deg)` adds lam * sum((e / sigma_e)^2 + (dang / sigma_a)^2)
    to the objective (the reported score stays the plain similarity). Unregularised, +-10 % / +-5 deg
    lets a SIBLING polymorph's cell be bent onto the query's peaks: on 40 families, clean queries,
    fixed-cell scoring gets 0.99-1.0 within-family top-1 but +-3 % / +-1 deg drops it to ~0.7. A
    Gaussian prior sized to real thermal expansion (~1 % linear) is the physically honest constraint.
    """
    out = np.zeros(len(pairs), dtype=np.float32)
    params_out = np.zeros((len(pairs), 7), dtype=np.float32)
    sizes = np.array([len(cands[j]["two_theta"]) for _, j in pairs])
    order = np.argsort(sizes)                                          # pad similar sizes together
    # Memory-bounded chunks: the candidate self-overlap is P x nc x nc, which for P = 2048 and
    # nc = 1000 reflections is 8 GB. Cap P * nc^2 at max_elems instead of P.
    chunks, s0 = [], 0
    while s0 < len(order):
        s1 = s0 + 1
        while s1 < len(order) and (s1 - s0 + 1) * int(sizes[order[s1]]) ** 2 <= max_elems:
            s1 += 1
        chunks.append(order[s0:s1]); s0 = s1
    for sel in chunks:
        P = len(sel)
        nq = max(len(q_peaks[pairs[k][0]][0]) for k in sel)
        nc = max(len(cands[pairs[k][1]]["two_theta"]) for k in sel)
        tq = torch.zeros(P, nq); wq = torch.zeros(P, nq); mq = torch.zeros(P, nq)
        hkl = torch.zeros(P, nc, 3); wc = torch.zeros(P, nc); mc = torch.zeros(P, nc)
        cell0 = torch.zeros(P, 6)
        for row, k in enumerate(sel):
            i, j = pairs[k]
            t, h = q_peaks[i]
            tq[row, : len(t)] = torch.from_numpy(t); wq[row, : len(t)] = torch.from_numpy(h)
            mq[row, : len(t)] = 1
            c = cands[j]
            n = len(c["two_theta"])
            hkl[row, :n] = torch.from_numpy(c["hkl"].astype(np.float32))
            wc[row, :n] = torch.from_numpy(np.asarray(c["intensity"], dtype=np.float32) / 100.0)
            mc[row, :n] = 1
            cell0[row] = torch.from_numpy(np.asarray(c["cell"], dtype=np.float32))

        tq, wq, mq, hkl, wc, mc, cell0 = (x.to(device) for x in (tq, wq, mq, hkl, wc, mc, cell0))
        raw = torch.zeros(P, 7)
        if zero_init is not None:
            z0 = np.clip(zero_init[sel] / max_zero, -0.95, 0.95)
            raw[:, 6] = torch.from_numpy(np.arctanh(z0).astype(np.float32))
        raw = raw.to(device).requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=0.05)

        def score(sig):
            e = max_elong * torch.tanh(raw[:, :3])
            dang = max_ang * torch.tanh(raw[:, 3:6])
            cell = torch.cat([cell0[:, :3] * (1 + e), cell0[:, 3:] + dang], dim=1)
            tc = _two_theta_from(hkl, _recip_from_cell(cell)) + max_zero * torch.tanh(raw[:, 6:7])
            iqc = _overlap(tq, wq, mq, tc, wc, mc, sig)
            iqq = _overlap(tq, wq, mq, tq, wq, mq, sig)
            icc = _overlap(tc, wc, mc, tc, wc, mc, sig)
            return iqc / torch.sqrt(torch.clamp(iqq * icc, min=1e-12))

        for sig in sigmas:
            for _ in range(steps):
                opt.zero_grad()
                loss = -score(sig).sum()
                if prior is not None:
                    lam, se, sa = prior
                    e = max_elong * torch.tanh(raw[:, :3])
                    dang = max_ang * torch.tanh(raw[:, 3:6])
                    loss = loss + lam * (((e / se) ** 2).sum() + ((dang / sa) ** 2).sum())
                loss.backward()
                opt.step()
        with torch.no_grad():
            out[sel] = score(sigmas[-1]).cpu().numpy()
            params_out[sel, :3] = (max_elong * torch.tanh(raw[:, :3])).cpu().numpy()
            params_out[sel, 3:6] = (max_ang * torch.tanh(raw[:, 3:6])).cpu().numpy()
            params_out[sel, 6] = (max_zero * torch.tanh(raw[:, 6])).cpu().numpy()
    return (out, params_out) if return_params else out


# ------------------------------------------------------------------------------------------
# library-scale B1 / B2 (one query at a time, reference spectra cached)
# ------------------------------------------------------------------------------------------

class WCCLibrary:
    """B1 against a large library without materialising (n_query, n_ref, n_lag) arrays.

    The reference FFTs and triangle-weighted autocorrelations are computed once; each query costs one
    batched complex multiply + irfft over the library, done in reference chunks.
    """

    def __init__(self, r_raw: np.ndarray, l_deg: float = 1.0, shift_deg: float = 0.5,
                 chunk: int = 4000, lnq: bool = False, l_lnq: float = 0.01, max_strain: float = 0.03):
        self.lnq = lnq
        if lnq:
            r_raw = np.stack([_to_lnq(y) for y in r_raw])
            step = _LNQ_STEP
            self.l_pts = max(1, int(round(l_lnq / step)))
            self.s_pts = int(round(math.log1p(max_strain) / step))
        else:
            self.l_pts = int(round(l_deg / STEP))
            self.s_pts = int(round(shift_deg / STEP))
        r = (r_raw - r_raw.min(axis=1, keepdims=True)).astype(np.float32)
        self.w = _triangle(self.l_pts)
        self.L = len(self.w) // 2
        self.max_lag = self.L + self.s_pts
        n = r.shape[1]
        self.nfft = 1 << int(math.ceil(math.log2(n + self.max_lag + 1)))
        self.chunk = chunk
        self.Fr = [np.fft.rfft(r[i:i + chunk], self.nfft).astype(np.complex64) for i in range(0, len(r), chunk)]
        self.ar = (_autocorr(r, self.L) @ self.w).astype(np.float32)

    def _score_one(self, q: np.ndarray) -> np.ndarray:
        q = (q - q.min()).astype(np.float32)
        Fq = np.conj(np.fft.rfft(q, self.nfft)).astype(np.complex64)
        aq = float(_autocorr(q[None], self.L)[0] @ self.w)
        out = []
        for Fr in self.Fr:
            full = np.fft.irfft(Fq[None, :] * Fr, self.nfft).astype(np.float32)
            c = np.concatenate([full[:, self.nfft - self.max_lag:], full[:, : self.max_lag + 1]], axis=1)
            # numerator for every shift = valid correlation of c with the triangle
            win = np.lib.stride_tricks.sliding_window_view(c, len(self.w), axis=1)   # (nr, 2S+1, 2L+1)
            num = (win @ self.w).max(axis=1)
            out.append(num)
        num = np.concatenate(out)
        return (num / np.sqrt(np.maximum(aq * self.ar, 1e-12))).astype(np.float32)

    def scores(self, q_raw: np.ndarray, shift_deg: float = 0.3, shift_step_deg: float = 0.04) -> np.ndarray:
        """(n_query, n_ref). In ln-q mode the 2theta zero offset is scanned outside, as in B2."""
        rows = []
        for q in q_raw:
            if not self.lnq:
                rows.append(self._score_one(q))
            else:
                best = None
                for delta in np.arange(-shift_deg, shift_deg + 1e-9, shift_step_deg):
                    s = self._score_one(_to_lnq(q, delta))
                    best = s if best is None else np.maximum(best, s)
                rows.append(best)
        return np.stack(rows)
