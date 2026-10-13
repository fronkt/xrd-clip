"""Torch port of `strong_baselines.WCCLibrary` and fixed-grid Pearson, for FULL score matrices on a GPU.

WCCLibrary scores one query at a time through FFT cross-correlations. That is fine for ranks, but far too
slow to keep every score of ~8k probe rows against 309k library entries (scripts/21_hubness.py). The
triangle weight can be folded into the reference instead:

    num(d) = sum_k w_k sum_x q(x) r(x + d + k) = sum_x q(x) r_w(x + d),    r_w(y) = sum_k w_k r(y + k),

so each zero-scan shift d is one (rows x n) @ (n x refs) matrix product against a shifted slice of r_w, and
the score is max_d num(d) / sqrt(a_q a_r) with a_x = sum_x x(x) x_w(x). Zero padding matches the FFT
version, so scores agree with WCCLibrary to float32 rounding. TF32 must stay off (`full_precision`): its
10-bit mantissa moves scores by ~1e-3, enough to reorder near-ties.

Shapes: patterns are rows on the 2251-point grid (A.GRID, or the ln q grid for lnq).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from xrdclip.data import augment as A
from xrdclip.eval import strong_baselines as SB


def full_precision() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _smooth(x: torch.Tensor, w: torch.Tensor, ext: int) -> torch.Tensor:
    """x_w(y) = sum_k w_k x(y + k) for y in [-ext, n + ext), x zero outside [0, n). (B, n) -> (B, n + 2 ext)."""
    L = (w.numel() - 1) // 2
    xp = F.pad(x, (L + ext, L + ext))
    return F.conv1d(xp[:, None, :], w.view(1, 1, -1))[:, 0, :]


class TorchWCC:
    """Same parameters and defaults as `SB.WCCLibrary(r_raw, **spec)` + `.scores(q_raw)`.

    prep_rows(q_raw)  -> query-side tensors (one per zero offset in ln q mode)
    prep_refs(r_raw)  -> reference-side tensors for one library block
    scores(rows, refs) -> (n_rows, n_refs) float32 on the device
    For lnq, pass the 2theta patterns; the ln q resampling uses the same numpy code as WCCLibrary.
    """

    def __init__(self, l_deg: float = 1.0, shift_deg: float = 0.5, lnq: bool = False, l_lnq: float = 0.01,
                 max_strain: float = 0.03, device: str = "cpu", version: int = 1):
        self.lnq, self.device, self.version = lnq, device, version
        if lnq:
            self.l_pts = max(1, int(round(l_lnq / SB._LNQ_STEP)))
            self.s_pts = int(round(math.log1p(max_strain) / SB._LNQ_STEP))
        else:
            self.l_pts = int(round(l_deg / SB.STEP))
            self.s_pts = int(round(shift_deg / SB.STEP))
        self.w = torch.tensor(SB._triangle(self.l_pts), dtype=torch.float32, device=device)

    # -- query side ------------------------------------------------------------------------------
    def _rows(self, q: np.ndarray) -> "tuple[torch.Tensor, torch.Tensor]":
        q = (q - q.min(axis=1, keepdims=True)).astype(np.float32)
        t = torch.from_numpy(q).to(self.device)
        a = (t * _smooth(t, self.w, 0)).sum(1)
        return t, a

    def prep_rows(self, q_raw: np.ndarray, shift_deg: float = 0.3,
                  shift_step_deg: float = 0.04) -> "list[tuple[torch.Tensor, torch.Tensor]]":
        if not self.lnq:
            return [self._rows(q_raw)]
        return [self._rows(np.stack([SB._to_lnq(y, delta, self.version) for y in q_raw]))
                for delta in SB.lnq_offsets(self.version, shift_deg, shift_step_deg)]

    # -- reference side --------------------------------------------------------------------------
    def prep_refs(self, r: np.ndarray, already_lnq: bool = False) -> "tuple[torch.Tensor, torch.Tensor]":
        """r: (n_refs, n) 2theta patterns, or ln q patterns when `already_lnq` (saves the resampling)."""
        if self.lnq and not already_lnq:
            r = np.stack([SB._to_lnq(y, 0.0, self.version) for y in r])
        r = (r - r.min(axis=1, keepdims=True)).astype(np.float32)
        t = torch.from_numpy(r).to(self.device)
        rw = _smooth(t, self.w, self.s_pts)                                    # (C, n + 2S)
        n = t.shape[1]
        a = (t * rw[:, self.s_pts: self.s_pts + n]).sum(1)
        return rw, a

    def scores(self, rows: "list[tuple[torch.Tensor, torch.Tensor]]",
               refs: "tuple[torch.Tensor, torch.Tensor]") -> torch.Tensor:
        rw, ar = refs
        best = None
        for q, aq in rows:
            n = q.shape[1]
            num = None
            for d in range(2 * self.s_pts + 1):
                c = q @ rw[:, d: d + n].T
                num = c if num is None else torch.maximum(num, c, out=num)
            s = num / torch.sqrt(torch.clamp(aq[:, None] * ar[None, :], min=1e-12))
            best = s if best is None else torch.maximum(best, s, out=best)
        return best


def pearson(q_std: torch.Tensor, r_std: torch.Tensor) -> torch.Tensor:
    """Fixed-grid Pearson on standardized rows, as in scripts/19_real_bench.score_block."""
    return q_std @ r_std.T / r_std.shape[1]


def standardize_rows(y: np.ndarray) -> np.ndarray:
    return np.stack([A.standardize(v) for v in y])
