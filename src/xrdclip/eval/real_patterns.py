"""Measured-pattern preprocessing shared by the real-data evaluations (scripts/17_..., scripts/19_...).

Convert 2theta to the Cu K-alpha1 scale through d-spacing (synchrotron data included), resample onto
the model grid (5-50 deg), zero outside the measured range, then one of two variants: `raw`
(max-normalised) or `bkg` (rolling-minimum background removed, then normalised). K-alpha2 is left in
lab data, so classical and learned methods see the same input.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from xrdclip.data import augment as A


def to_cu_grid(tt: np.ndarray, y: np.ndarray, wavelength: float,
               method: str = "interp") -> "tuple[np.ndarray, np.ndarray]":
    """Measured (2theta, I) at `wavelength` -> intensity on A.GRID at Cu K-alpha1, plus coverage mask.

    method "interp" point-samples the converted pattern at each grid point (the benchmark's default).
    "binavg" (10-14 referee check) averages it over each grid point's bin (+-STEP/2, clipped to the measured
    range), via the cumulative trapezoid integral of the piecewise-linear pattern, so fine-step synchrotron
    peaks keep their area instead of being sampled at arbitrary phases (the R1 fix of ln q, for every score).
    """
    tt, y = np.asarray(tt, float), np.asarray(y, float)
    d = wavelength / (2 * np.sin(np.radians(tt / 2)))
    s = A.WAVELENGTH / (2 * d)
    ok = (s > 0) & (s < 1)
    tt_cu = np.degrees(2 * np.arcsin(s[ok]))
    order = np.argsort(tt_cu)
    tt_cu, yy = tt_cu[order], y[ok][order]
    mask = (A.GRID >= tt_cu[0]) & (A.GRID <= tt_cu[-1])
    out = np.zeros(A.N_POINTS)
    if method == "interp":
        out[mask] = np.interp(A.GRID[mask], tt_cu, yy)
    elif method == "binavg":
        tt_cu, keep = np.unique(tt_cu, return_index=True)        # strictly increasing for the integral
        yy = yy[keep]
        cum = np.r_[0.0, np.cumsum((yy[1:] + yy[:-1]) * 0.5 * np.diff(tt_cu))]

        def integral(t):                                           # integral from tt_cu[0] to t (inside the range)
            k = np.clip(np.searchsorted(tt_cu, t, side="right") - 1, 0, len(tt_cu) - 2)
            yt = np.interp(t, tt_cu, yy)
            return cum[k] + (t - tt_cu[k]) * (yy[k] + yt) * 0.5
        g = A.GRID[mask]
        lo = np.clip(g - A.STEP / 2, tt_cu[0], tt_cu[-1])
        hi = np.clip(g + A.STEP / 2, tt_cu[0], tt_cu[-1])
        w = hi - lo
        out[mask] = np.where(w > 0, (integral(hi) - integral(lo)) / np.where(w > 0, w, 1), np.interp(g, tt_cu, yy))
    else:
        raise ValueError(method)
    return out.astype(np.float32), mask


def preprocess(y: np.ndarray, mask: np.ndarray, bkg: bool) -> np.ndarray:
    y = y.copy()
    if bkg:
        w = int(round(2.0 / A.STEP))
        lo = np.where(mask, y, np.inf)
        base = minimum_filter1d(np.where(np.isfinite(lo), lo, y[mask].max() if mask.any() else 0), w)
        base = uniform_filter1d(base, w)
        y = np.where(mask, np.clip(y - base, 0, None), 0)
    else:
        y = np.where(mask, y - y[mask].min(), 0)
    m = y.max()
    return (y / m).astype(np.float32) if m > 0 else y


def query_patterns(pattern: dict, wavelength, variant: str) -> np.ndarray:
    """One opXRD pattern record -> preprocessed query on A.GRID."""
    wl = float(wavelength or A.WAVELENGTH)
    y, mask = to_cu_grid(pattern["two_theta"], pattern["intensity"], wl)
    return preprocess(y, mask, bkg=(variant == "bkg"))


def temp_k(s) -> "float | None":
    m = re.search(r"([\d.]+)\s*K", str(s or ""))
    return float(m.group(1)) if m else None
