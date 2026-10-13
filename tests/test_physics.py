"""Physics correctness tests for the simulation + augmentation pipeline.

The load-bearing test is `test_peak_positions_match_bragg`: it checks that a rendered peak lands
where Bragg's law analytically says it must, for a cell whose answer we can compute by hand.
Every downstream number in both papers rests on that being true, and a silent unit error here
(degrees vs radians, d vs 2d) would produce a pipeline that trains happily and means nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrdclip.data import augment as A


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------

def cubic_sticks(a: float = 10.0, reflections=((1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0))) -> dict:
    """Analytic stick pattern for a simple cubic cell of parameter `a`.

    For cubic:  d(hkl) = a / sqrt(h^2+k^2+l^2),  and  lambda = 2 d sin(theta).
    """
    hkl = np.array(reflections, dtype=np.int16)
    n = np.sqrt((hkl.astype(np.float64) ** 2).sum(axis=1))
    d = a / n
    sin_theta = A.WAVELENGTH / (2.0 * d)
    ok = sin_theta <= 1.0
    hkl, d, sin_theta = hkl[ok], d[ok], sin_theta[ok]
    two_theta = np.degrees(2.0 * np.arcsin(sin_theta))
    recip = np.eye(3, dtype=np.float32) / a  # cubic reciprocal lattice
    return {
        "two_theta": two_theta.astype(np.float32),
        "intensity": np.full(len(d), 100.0, dtype=np.float32),
        "hkl": hkl,
        "d_hkl": d.astype(np.float32),
        "recip": recip,
    }


def peak_positions(y: np.ndarray, grid: np.ndarray = A.GRID, thresh: float = 0.2) -> np.ndarray:
    """Grid positions of local maxima above `thresh`."""
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] >= thresh and y[i] > y[i - 1] and y[i] >= y[i + 1]:
            peaks.append(grid[i])
    return np.array(peaks)


# ------------------------------------------------------------------------------------------
# the test that matters
# ------------------------------------------------------------------------------------------

def test_peak_positions_match_bragg():
    """A rendered clean pattern must put peaks at the analytic Bragg angles."""
    a = 10.0
    sticks = cubic_sticks(a)
    y = A.render_sticks(sticks, A.CLEAN)

    found = peak_positions(y)
    expected = sticks["two_theta"][
        (sticks["two_theta"] >= A.TWO_THETA_MIN) & (sticks["two_theta"] <= A.TWO_THETA_MAX)
    ]

    assert len(found) == len(expected), f"expected {len(expected)} peaks, found {len(found)}"
    for e in expected:
        assert np.min(np.abs(found - e)) < 0.05, f"no peak within 0.05 deg of Bragg angle {e:.3f}"


def test_bragg_law_self_consistency():
    """d-spacings in the stick dict must satisfy lambda = 2 d sin(theta) against their own 2theta."""
    s = cubic_sticks(10.0)
    theta = np.radians(s["two_theta"] / 2.0)
    lam = 2.0 * s["d_hkl"] * np.sin(theta)
    assert np.allclose(lam, A.WAVELENGTH, atol=1e-4)


# ------------------------------------------------------------------------------------------
# each degradation does what it claims
# ------------------------------------------------------------------------------------------

def test_zero_offset_shifts_all_peaks_equally():
    s = cubic_sticks()
    y0 = A.render_sticks(s, A.CLEAN)
    off = 0.2
    y1 = A.render_sticks(s, A.sweep_params(zero_offset=off))

    p0, p1 = peak_positions(y0), peak_positions(y1)
    assert len(p0) == len(p1)
    shifts = p1 - p0
    assert np.allclose(shifts, off, atol=0.03), f"expected uniform {off} deg shift, got {shifts}"


def test_strain_shift_is_theta_dependent_not_uniform():
    """Strain must shift high-angle peaks MORE than low-angle ones.

    This is the physical signature that separates strain from a zero offset, and it is precisely
    what a correlation baseline cannot absorb. If this test ever passes with a uniform shift, the
    strain path has degenerated into a constant offset and the headline claim is dead.
    """
    s = cubic_sticks()
    p0 = peak_positions(A.render_sticks(s, A.CLEAN))
    p1 = peak_positions(A.render_sticks(s, A.sweep_params(strain=0.01)))

    assert len(p0) == len(p1) >= 2
    shifts = np.abs(p1 - p0)
    # positive strain = bigger cell = smaller 2theta
    assert np.all(p1 < p0), "positive strain must move peaks to LOWER 2theta"
    assert shifts[-1] > shifts[0] * 1.2, (
        f"strain shift must grow with angle; got {shifts[0]:.4f} (low) vs {shifts[-1]:.4f} (high)"
    )


def test_scherrer_broadening_widens_peaks():
    s = cubic_sticks()
    sharp = A.render_sticks(s, A.sweep_params(domain_nm=1e4))
    broad = A.render_sticks(s, A.sweep_params(domain_nm=15.0))

    # Small crystallites -> broader peaks -> lower, wider maxima -> fewer resolved local maxima
    # and a larger fraction of the pattern above half-height.
    frac_sharp = (sharp > 0.5).mean()
    frac_broad = (broad > 0.5).mean()
    assert frac_broad > frac_sharp * 1.5, (
        f"15 nm domains must broaden peaks vs bulk; above-half-height fraction "
        f"{frac_broad:.4f} vs {frac_sharp:.4f}"
    )


def test_march_dollase_redistributes_intensity():
    """Texture must change RELATIVE peak intensities, not just rescale the pattern."""
    s = cubic_sticks()
    y0 = A.render_sticks(s, A.CLEAN)
    y1 = A.render_sticks(s, A.sweep_params(march=0.6, po_axis=(0, 0, 1)))

    # Both are max-normalised, so a pure rescale would leave them identical.
    assert not np.allclose(y0, y1, atol=1e-3), "March-Dollase r=0.6 must alter the pattern"

    # r = 1 must be an exact no-op.
    y_none = A.render_sticks(s, A.sweep_params(march=1.0))
    assert np.allclose(y0, y_none, atol=1e-6), "march=1.0 must be identity"


def test_march_dollase_is_unity_at_r_one():
    s = cubic_sticks()
    corr = A._march_dollase(s["hkl"], s["recip"], A.sweep_params(march=1.0))
    assert np.allclose(corr, 1.0)


def test_background_is_additive_and_positive():
    s = cubic_sticks()
    y = A.render_sticks(s, A.sweep_params(bkg_scale=0.1, bkg_coef=(1, 0.5, -0.3, 0, 0, 0), hump_scale=0.05))
    assert y.min() >= 0.0
    assert y.max() <= 1.0 + 1e-5


def test_noise_degrades_but_preserves_peaks():
    s = cubic_sticks()
    rng = np.random.default_rng(0)
    clean = A.render_sticks(s, A.CLEAN)
    noisy = A.render_sticks(s, A.sweep_params(counts=5e3), rng=rng)

    assert not np.allclose(clean, noisy)
    # Peaks must survive: the noisy pattern should still correlate strongly with the clean one.
    c = np.corrcoef(clean, noisy)[0, 1]
    assert c > 0.8, f"Poisson noise at 5e3 counts destroyed the pattern (corr={c:.3f})"


# ------------------------------------------------------------------------------------------
# renderer contract
# ------------------------------------------------------------------------------------------

def test_output_shape_and_normalisation():
    s = cubic_sticks()
    y = A.render_sticks(s, A.CLEAN)
    assert y.shape == (A.N_POINTS,) == (2251,)
    assert np.isclose(y.max(), 1.0, atol=1e-5)
    assert y.dtype == np.float32


def test_sampled_params_are_finite_and_render():
    """Every draw from the augmentation distribution must produce a usable pattern."""
    s = cubic_sticks()
    rng = np.random.default_rng(42)
    for _ in range(200):
        p = A.sample_params(rng)
        y = A.render_sticks(s, p, rng=rng)
        assert np.all(np.isfinite(y)), f"non-finite pattern from {p}"
        assert y.shape == (A.N_POINTS,)
        assert y.max() > 0, "augmentation annihilated the pattern"


def test_augmentation_is_stochastic():
    """Two draws must differ — otherwise the dataloader is silently serving a fixed pattern."""
    s = cubic_sticks()
    rng = np.random.default_rng(7)
    a = A.render_sticks(s, A.sample_params(rng), rng=rng)
    b = A.render_sticks(s, A.sample_params(rng), rng=rng)
    assert not np.allclose(a, b)


def test_mixture_contains_both_phases():
    sa = cubic_sticks(a=10.0)
    sb = cubic_sticks(a=8.0)
    y = A.mixture(sa, sb, 0.5)

    pa = peak_positions(A.render_sticks(sa, A.CLEAN), thresh=0.9)
    pb = peak_positions(A.render_sticks(sb, A.CLEAN), thresh=0.9)
    pm = peak_positions(y, thresh=0.2)

    for p in list(pa[:1]) + list(pb[:1]):
        assert np.min(np.abs(pm - p)) < 0.1, f"mixture lost the phase with a peak at {p:.2f}"


def test_march_dollase_requires_3index_hkl():
    """Guard against the Miller-Bravais regression: hkl must be 3-index for hkl @ B (3x3).

    pymatgen returns 4-index (h k i l) reflections for hexagonal/trigonal cells. If a 4-index
    array ever reaches the renderer, this is where it explodes -- so pin the contract with a test.
    """
    s = cubic_sticks()
    assert s["hkl"].shape[1] == 3
    # a 4-index array must be rejected loudly, not silently broadcast
    bad = {**s, "hkl": np.hstack([s["hkl"], np.zeros((len(s["hkl"]), 1), dtype=np.int16)])}
    with pytest.raises(Exception):
        A.render_sticks(bad, A.sweep_params(march=0.6))


def test_standardize():
    y = A.render_sticks(cubic_sticks(), A.CLEAN)
    z = A.standardize(y)
    assert abs(z.mean()) < 1e-5
    assert abs(z.std() - 1.0) < 1e-4


def test_realism_adds_ka2_and_keeps_shape():
    import numpy as np
    from xrdclip.data import augment as A
    st = {"two_theta": np.array([20.0, 40.0], np.float32), "intensity": np.array([100.0, 50.0], np.float32),
          "hkl": np.array([[1, 0, 0], [2, 0, 0]]), "d_hkl": np.array([4.4361, 2.2520], np.float32),
          "recip": np.eye(3, dtype=np.float32)}
    R = A.Realism(p_ka2=1.0, B=(0.0, 0.0), jitter=(0.0, 0.0))
    out = A.realism_sticks(st, np.random.default_rng(0), R)
    assert len(out["two_theta"]) == 4 and np.allclose(out["intensity"][2:], [50.0, 25.0])
    # K-alpha2 sits at higher angle, by ~0.05 deg at 20 deg and ~0.1 deg at 40 deg
    d = out["two_theta"][2:] - out["two_theta"][:2]
    assert 0.04 < d[0] < 0.07 and 0.09 < d[1] < 0.13
    m = A.realism_mask(np.random.default_rng(1), A.Realism(p_crop=1.0))
    assert m.dtype == bool and 0 < m.sum() < A.N_POINTS
