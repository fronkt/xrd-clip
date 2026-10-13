"""Physics-grounded PXRD degradation: stick pattern -> realistic diffractogram.

This module IS the method. The contrastive model's robustness comes entirely from what it sees
here, so every transform is a real physical effect with a real parameterisation, not a generic
"add noise" nuisance model:

    strain            isotropic cell expansion/contraction -> coherent, theta-dependent peak shift
    zero offset       goniometer misalignment              -> constant peak shift
    Caglioti          instrumental resolution              -> theta-dependent broadening
    Scherrer          finite crystallite size              -> cos(theta)^-1 broadening
    pseudo-Voigt      real peak shape                      -> Gaussian/Lorentzian mixing
    March-Dollase     preferred orientation                -> hkl-dependent intensity redistribution
    Chebyshev + hump  background & amorphous scatter       -> additive baseline
    Poisson           counting statistics                  -> multiplicative-ish noise

The renderer is deliberately split into `sample_params` and `render`, because the robustness
sweep (SPEC §4.3) must hold every effect at its clean value and vary exactly one. A renderer
that only knows how to randomise everything at once cannot produce that figure.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

WAVELENGTH = 1.54056
TWO_THETA_MIN = 5.0
TWO_THETA_MAX = 50.0
STEP = 0.02
GRID = np.arange(TWO_THETA_MIN, TWO_THETA_MAX + STEP / 2, STEP, dtype=np.float32)
N_POINTS = GRID.size  # 2251


@dataclass
class Params:
    """A concrete degradation. Defaults are the CLEAN pattern: no shift, no texture, sharp peaks."""
    strain: float = 0.0            # fractional isotropic cell strain, d -> d(1+strain)
    axis_strain: tuple = (0.0, 0.0, 0.0)  # per-axis a,b,c length strain (thermal expansion); angles kept
    zero_offset: float = 0.0       # deg 2theta, constant
    u: float = 0.002               # Caglioti U (deg^2)
    v: float = -0.001              # Caglioti V (deg^2)
    w: float = 0.004               # Caglioti W (deg^2)
    domain_nm: float = 1e4         # crystallite size; large -> negligible Scherrer broadening
    eta: float = 0.5               # pseudo-Voigt Lorentzian fraction
    march: float = 1.0             # March-Dollase r; 1.0 = no preferred orientation
    po_axis: tuple = (0, 0, 1)     # texture axis (hkl)
    bkg_scale: float = 0.0         # Chebyshev background amplitude (fraction of I_max)
    bkg_coef: tuple = (0.0,) * 6
    hump_scale: float = 0.0        # amorphous hump amplitude
    hump_center: float = 20.0
    hump_width: float = 8.0
    counts: float = 0.0            # peak counts for Poisson noise; 0 -> noiseless


CLEAN = Params()


@dataclass
class Ranges:
    """Sampling ranges for training-time augmentation (SPEC §3.2)."""
    strain: tuple = (-0.005, 0.005)
    zero_offset: tuple = (-0.10, 0.10)
    u: tuple = (0.0, 0.02)
    v: tuple = (-0.02, 0.005)
    w: tuple = (0.001, 0.03)
    domain_nm: tuple = (20.0, 500.0)      # log-uniform
    eta: tuple = (0.1, 0.9)
    march: tuple = (0.7, 1.3)
    bkg_scale: tuple = (0.0, 0.15)
    hump_scale: tuple = (0.0, 0.10)
    counts: tuple = (2e3, 1e6)            # log-uniform; low counts = noisy
    p_texture: float = 0.5                # probability a pattern has any preferred orientation
    p_background: float = 0.8


def sample_params(rng: np.random.Generator, r: Ranges = Ranges()) -> Params:
    """Draw one random degradation. Called fresh every epoch, per sample."""
    axis_choices = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    march = float(rng.uniform(*r.march)) if rng.random() < r.p_texture else 1.0
    has_bkg = rng.random() < r.p_background
    return Params(
        strain=float(rng.uniform(*r.strain)),
        zero_offset=float(rng.uniform(*r.zero_offset)),
        u=float(rng.uniform(*r.u)),
        v=float(rng.uniform(*r.v)),
        w=float(rng.uniform(*r.w)),
        domain_nm=float(np.exp(rng.uniform(np.log(r.domain_nm[0]), np.log(r.domain_nm[1])))),
        eta=float(rng.uniform(*r.eta)),
        march=march,
        po_axis=axis_choices[rng.integers(len(axis_choices))],
        bkg_scale=float(rng.uniform(*r.bkg_scale)) if has_bkg else 0.0,
        bkg_coef=tuple(rng.normal(0, 1, 6)),
        hump_scale=float(rng.uniform(*r.hump_scale)) if has_bkg else 0.0,
        hump_center=float(rng.uniform(12.0, 28.0)),
        hump_width=float(rng.uniform(4.0, 14.0)),
        counts=float(np.exp(rng.uniform(np.log(r.counts[0]), np.log(r.counts[1])))),
    )


# --------------------------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------------------------

def _shift_two_theta(two_theta: np.ndarray, d_hkl: np.ndarray, p: Params) -> np.ndarray:
    """Isotropic strain (theta-dependent) + zero offset (constant).

    Strain acts on d-spacing, so we go back through Bragg's law rather than shifting 2theta
    directly. That distinction matters: strain compresses the low-angle peaks less than the
    high-angle ones, which is exactly the signature a correlation-based search-match cannot
    absorb — and therefore exactly the invariance we want the encoder to learn.
    """
    if p.strain != 0.0:
        sin_theta = WAVELENGTH / (2.0 * d_hkl * (1.0 + p.strain))
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        two_theta = np.degrees(2.0 * np.arcsin(sin_theta))
    return two_theta + p.zero_offset


def _axis_strained_d(hkl: np.ndarray, recip: np.ndarray, axis_strain: tuple) -> np.ndarray:
    """d-spacings after scaling the a, b, c lengths by (1 + e_a), (1 + e_b), (1 + e_c), angles fixed.

    Scaling direct row a_i by (1 + e_i) scales reciprocal row b_i by 1 / (1 + e_i), so
    g = sum_i h_i b_i / (1 + e_i). This is the anisotropic thermal-expansion case (Bond 2021): low-
    temperature CSD structures vs room-temperature patterns, unequal per axis, sometimes negative.
    """
    scale = 1.0 / (1.0 + np.asarray(axis_strain, dtype=np.float64))
    g = (hkl.astype(np.float64) * scale[None, :]) @ recip.astype(np.float64)
    return (1.0 / np.maximum(np.linalg.norm(g, axis=1), 1e-12)).astype(np.float32)


def _fwhm(two_theta: np.ndarray, p: Params) -> np.ndarray:
    """Caglioti instrumental resolution + Scherrer size broadening, added in quadrature."""
    theta = np.radians(two_theta / 2.0)
    tan_t = np.tan(theta)
    inst_sq = p.u * tan_t**2 + p.v * tan_t + p.w
    inst_sq = np.maximum(inst_sq, 1e-6)

    # Scherrer: FWHM(2theta, rad) = K*lambda / (D * cos(theta)); K ~ 0.9
    d_ang = p.domain_nm * 10.0  # nm -> angstrom
    size = np.degrees(0.9 * WAVELENGTH / (d_ang * np.maximum(np.cos(theta), 1e-3)))

    return np.sqrt(inst_sq + size**2)


def _march_dollase(hkl: np.ndarray, recip: np.ndarray, p: Params) -> np.ndarray:
    """Preferred-orientation intensity correction.

    P(alpha) = (r^2 cos^2(alpha) + sin^2(alpha)/r)^(-3/2)

    alpha is the angle between each reflection's plane normal (g = hkl . B, in reciprocal space)
    and the texture axis. r = 1 returns unity everywhere.
    """
    if p.march == 1.0:
        return np.ones(len(hkl), dtype=np.float32)

    g = hkl.astype(np.float32) @ recip                       # plane normals
    axis = np.asarray(p.po_axis, dtype=np.float32) @ recip   # texture axis in the same space
    gn = np.linalg.norm(g, axis=1)
    an = np.linalg.norm(axis)
    if an < 1e-8:
        return np.ones(len(hkl), dtype=np.float32)

    cos_a = np.abs(g @ axis) / (np.maximum(gn, 1e-8) * an)
    cos_a = np.clip(cos_a, 0.0, 1.0)
    sin2, cos2 = 1.0 - cos_a**2, cos_a**2
    r = p.march
    return (r**2 * cos2 + sin2 / r) ** (-1.5)


def _profile(grid: np.ndarray, centers: np.ndarray, inten: np.ndarray,
             fwhm: np.ndarray, eta: float) -> np.ndarray:
    """Sum of pseudo-Voigt peaks onto the grid.

    Only evaluates each peak within +/-6 FWHM of its centre — without that windowing this is an
    O(n_peaks x n_grid) dense outer product and the dataloader becomes the training bottleneck.
    """
    out = np.zeros(grid.size, dtype=np.float32)
    lo_edge, hi_edge, step = grid[0], grid[-1], grid[1] - grid[0]

    for c, i0, f in zip(centers, inten, fwhm):
        if i0 <= 0 or not (lo_edge - 1.0 <= c <= hi_edge + 1.0):
            continue
        half = 6.0 * f
        lo = max(lo_edge, c - half)
        hi = min(hi_edge, c + half)
        if hi <= lo:
            continue
        a = int((lo - lo_edge) / step)
        b = min(grid.size, int((hi - lo_edge) / step) + 2)
        if b <= a:
            continue

        x = grid[a:b] - c
        f = max(f, 1e-4)
        gauss = np.exp(-4.0 * np.log(2.0) * (x / f) ** 2) * (2.0 / f) * np.sqrt(np.log(2.0) / np.pi)
        lorentz = (2.0 / (np.pi * f)) / (1.0 + 4.0 * (x / f) ** 2)
        out[a:b] += i0 * (eta * lorentz + (1.0 - eta) * gauss)

    return out


def _background(grid: np.ndarray, p: Params) -> np.ndarray:
    bkg = np.zeros(grid.size, dtype=np.float32)
    if p.bkg_scale > 0:
        x = 2.0 * (grid - grid[0]) / (grid[-1] - grid[0]) - 1.0  # -> [-1, 1] for Chebyshev
        cheb = np.polynomial.chebyshev.chebval(x, np.asarray(p.bkg_coef, dtype=np.float32))
        cheb = cheb - cheb.min()
        m = cheb.max()
        if m > 0:
            bkg += p.bkg_scale * cheb / m
    if p.hump_scale > 0:
        bkg += p.hump_scale * np.exp(-0.5 * ((grid - p.hump_center) / p.hump_width) ** 2)
    return bkg


def render_sticks(sticks: dict, p: Params = CLEAN,
                  rng: "np.random.Generator | None" = None,
                  grid: np.ndarray = GRID) -> np.ndarray:
    """Stick pattern -> diffractogram on `grid`, normalised to unit max.

    `sticks` needs keys: two_theta, intensity, hkl, d_hkl, recip.

    Order matters and follows the physics: shift positions, redistribute intensity by texture,
    convolve with the peak profile, add background, then apply counting noise last (noise is a
    detector effect and therefore acts on the final signal, background included).
    """
    two_theta = np.asarray(sticks["two_theta"], dtype=np.float32)
    inten = np.asarray(sticks["intensity"], dtype=np.float32).copy()
    hkl = np.asarray(sticks["hkl"])
    d_hkl = np.asarray(sticks["d_hkl"], dtype=np.float32)
    recip = np.asarray(sticks["recip"], dtype=np.float32)

    if any(e != 0.0 for e in p.axis_strain):
        d_hkl = _axis_strained_d(hkl, recip, p.axis_strain)
        two_theta = np.degrees(2.0 * np.arcsin(np.clip(WAVELENGTH / (2.0 * d_hkl), -1.0, 1.0)))
    tt = _shift_two_theta(two_theta, d_hkl, p)
    inten = inten * _march_dollase(hkl, recip, p)
    fwhm = _fwhm(tt, p)

    y = _profile(grid, tt, inten, fwhm, p.eta)

    m = y.max()
    if m <= 0:
        return np.zeros(grid.size, dtype=np.float32)
    y = y / m

    y = y + _background(grid, p)

    if p.counts > 0:
        if rng is None:
            rng = np.random.default_rng()
        y = rng.poisson(np.clip(y, 0, None) * p.counts).astype(np.float32) / p.counts

    m = y.max()
    return (y / m).astype(np.float32) if m > 0 else y.astype(np.float32)


def mixture(sticks_a: dict, sticks_b: dict, weight: float, p: Params = CLEAN,
            rng: "np.random.Generator | None" = None, grid: np.ndarray = GRID) -> np.ndarray:
    """Two-phase query: w*A + (1-w)*B, each rendered then normalised before mixing (SPEC §3.3)."""
    ya = render_sticks(sticks_a, p, rng, grid)
    yb = render_sticks(sticks_b, p, rng, grid)
    y = weight * ya + (1.0 - weight) * yb
    m = y.max()
    return (y / m).astype(np.float32) if m > 0 else y


def standardize(y: np.ndarray) -> np.ndarray:
    """Zero-mean/unit-variance, the form the encoder consumes."""
    mu, sd = y.mean(), y.std()
    return ((y - mu) / sd).astype(np.float32) if sd > 1e-8 else (y - mu).astype(np.float32)


@dataclass
class Realism:
    """Measured-pattern effects for the revision retrain (tasks/todo.md Step 3, R4c), on top of `Ranges`.

    Chosen from the R4a ladder (scripts/30_model_gap.py): K-alpha2 and the queries' background removal were the
    two effects that cost the model most on its own entries; the follow-up rungs test intensity mismatch.
    Ranges cover the 188 measured queries (120 lab Cu; 46 do not span 5-50 deg; B from library CIFs 4.45 A^2).
    """
    p_ka2: float = 0.6                 # Cu K-alpha2 doublet, lambda2/lambda1 = 1.0024889, I2/I1 = 0.5
    ka2_rho: float = 1.5444274 / 1.5405929
    ka2_ratio: float = 0.5
    B: tuple = (0.0, 8.0)              # overall Debye-Waller B (A^2), uniform
    jitter: tuple = (0.0, 0.5)         # log-normal sigma of per-reflection intensity factors, uniform
    p_crop: float = 0.25               # measured range shorter than 5-50 deg
    crop_lo: tuple = (5.0, 12.0)
    crop_hi: tuple = (30.0, 50.0)


def realism_sticks(sticks: dict, rng: np.random.Generator, R: Realism) -> dict:
    """Intensity effects (Debye-Waller, per-reflection jitter) and a K-alpha2 copy of every reflection."""
    tt = np.asarray(sticks["two_theta"], np.float64)
    inten = np.asarray(sticks["intensity"], np.float64)
    d = np.asarray(sticks["d_hkl"], np.float64)
    hkl = np.asarray(sticks["hkl"])
    B = rng.uniform(*R.B)
    if B > 0:
        inten = inten * np.exp(-B / (2 * d ** 2))                 # exp(-2 B s^2), s = 1/(2d)
    sig = rng.uniform(*R.jitter)
    if sig > 0:
        inten = inten * rng.lognormal(0.0, sig, inten.shape)
    if rng.random() < R.p_ka2:
        s2 = np.sin(np.radians(tt / 2)) * R.ka2_rho
        ok = s2 < 1
        tt = np.r_[tt, np.degrees(2 * np.arcsin(s2[ok]))]
        inten = np.r_[inten, R.ka2_ratio * inten[ok]]
        d = np.r_[d, d[ok]]
        hkl = np.r_[hkl, hkl[ok]]
    return {**sticks, "two_theta": tt, "intensity": inten, "d_hkl": d, "hkl": hkl}


def realism_mask(rng: np.random.Generator, R: Realism, grid: np.ndarray = GRID) -> np.ndarray:
    """Measured 2theta coverage: the full grid, or a shorter range with probability p_crop."""
    if rng.random() >= R.p_crop:
        return np.ones(grid.size, bool)
    return (grid >= rng.uniform(*R.crop_lo)) & (grid <= rng.uniform(*R.crop_hi))


# Convenience for the robustness sweep: clean everywhere, one dial turned.
def sweep_params(**kwargs) -> Params:
    """`sweep_params(zero_offset=0.2)` -> otherwise-clean pattern with a 0.2 deg shift."""
    return replace(CLEAN, **kwargs)
