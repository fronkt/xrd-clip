"""Correctness tests for the v2 strong baselines and the anisotropic-strain dial.

Each baseline claims an invariance; each test checks that invariance on a cell where the answer is
known, and checks that the score still separates the true cell from a decoy (an invariance that
also makes every candidate look identical is worthless).
"""
from __future__ import annotations

import itertools

import numpy as np

from xrdclip.data import augment as A
from xrdclip.eval import strong_baselines as SB


def ortho_sticks(a: float, b: float, c: float, seed: int = 0, hmax: int = 6) -> dict:
    """Orthorhombic stick pattern over 5-50 deg with fixed pseudo-random intensities."""
    rng = np.random.default_rng(seed)
    hkl = np.array([h for h in itertools.product(range(hmax + 1), repeat=3) if any(h)], dtype=np.int16)
    recip = np.diag([1 / a, 1 / b, 1 / c]).astype(np.float32)
    d = 1.0 / np.linalg.norm(hkl @ recip, axis=1)
    s = A.WAVELENGTH / (2 * d)
    tt = np.degrees(2 * np.arcsin(np.clip(s, 0, 1)))
    ok = (s < 1) & (tt > A.TWO_THETA_MIN) & (tt < A.TWO_THETA_MAX)
    inten = rng.uniform(5, 100, ok.sum()).astype(np.float32)
    inten = inten / inten.max() * 100
    return {"two_theta": tt[ok].astype(np.float32), "intensity": inten, "hkl": hkl[ok],
            "d_hkl": d[ok].astype(np.float32), "recip": recip,
            "cell": np.array([a, b, c, 90, 90, 90], dtype=np.float32)}


TRUE = ortho_sticks(11.0, 7.3, 15.2, seed=1)
DECOY = ortho_sticks(11.0, 7.9, 15.2, seed=2)   # "polymorph": same a, c; different b + intensities


def test_axis_strain_moves_h00_by_bragg():
    s = ortho_sticks(11.0, 7.3, 15.2)
    p = A.sweep_params(axis_strain=(0.02, 0.0, 0.0))
    d = A._axis_strained_d(s["hkl"], s["recip"], p.axis_strain)
    i = int(np.where((s["hkl"] == (1, 0, 0)).all(axis=1))[0][0])
    j = int(np.where((s["hkl"] == (0, 1, 0)).all(axis=1))[0][0])
    assert np.isclose(d[i], 11.0 * 1.02, rtol=1e-5)
    assert np.isclose(d[j], 7.3, rtol=1e-5)          # b untouched


def test_wcc_absorbs_zero_offset_and_separates_decoy():
    ref = np.stack([A.render_sticks(TRUE), A.render_sticks(DECOY)])
    q = A.render_sticks(TRUE, A.sweep_params(zero_offset=0.3))[None]
    s = SB.wcc_scores(q, ref, l_deg=1.0, shift_deg=0.5)[0]
    assert s[0] > 0.97 and s[0] > s[1] + 0.05
    pear = SB.wcc_scores(q, ref, l_deg=0.0, shift_deg=0.5)[0]   # Pearson + zero scan
    assert pear[0] > 0.97 and pear[0] > pear[1]


def test_lnq_absorbs_isotropic_strain():
    ref = np.stack([A.render_sticks(TRUE), A.render_sticks(DECOY)])
    q = A.render_sticks(TRUE, A.sweep_params(strain=0.02))[None]
    s = SB.lnq_wcc_scores(q, ref, max_strain=0.03)[0]
    assert s[0] > 0.95 and s[0] > s[1] + 0.05


def test_lnq_offsets_v1_reproduced_v2_contains_zero():
    v1 = SB.lnq_offsets(1)
    assert np.array_equal(v1, np.arange(-0.3, 0.3 + 1e-9, 0.04)) and len(v1) == 16
    assert np.abs(v1).min() > 0.01                       # the draft v1 flaw: 0 is not scanned
    v2 = SB.lnq_offsets(2)
    assert len(v2) == 17 and np.isclose(v2, -v2[::-1]).all() and np.abs(v2).min() == 0.0
    assert np.isclose(v2[-1], 0.32)


def test_lnq_v1_resampling_unchanged():
    y = A.render_sticks(TRUE)
    old = np.interp(SB._TT_OF_LNQ + 0.1, A.GRID, y).astype(np.float32)
    assert np.array_equal(SB._to_lnq(y, 0.1), old) and np.array_equal(SB._to_lnq(y, 0.1, 1), old)


def test_lnq_v2_resampling_keeps_peak_area():
    # one sharp peak at high angle, where the ln q axis step (~0.05 deg) is wider than the data step
    y = A.render_sticks({"two_theta": np.array([47.31], np.float32), "intensity": np.array([100.0], np.float32),
                         "hkl": np.array([[1, 0, 0]]), "d_hkl": np.array([1.92], np.float32),
                         "recip": np.eye(3, dtype=np.float32)})
    z = SB._to_lnq(y, 0.0, 2)
    area_tt = np.trapezoid(y, dx=SB.STEP)
    area_lnq = float((z * np.diff(SB._LNQ_EDGES)).sum())
    assert abs(area_lnq - area_tt) < 1e-3 * area_tt


def test_lnq_v2_self_score_is_one_and_strain_found():
    ref = np.stack([A.render_sticks(TRUE), A.render_sticks(DECOY)])
    s0 = SB.lnq_wcc_scores(ref[:1], ref, version=2)[0]
    assert abs(s0[0] - 1.0) < 1e-5 and s0[0] > s0[1] + 0.05
    q = A.render_sticks(TRUE, A.sweep_params(strain=0.01))[None]
    s = SB.lnq_wcc_scores(q, ref, version=2)[0]
    assert s[0] > 0.97 and s[0] > s[1] + 0.05


def test_lnq_torch_matches_numpy_v2():
    from xrdclip.eval import wcc_torch as WT
    ref = np.stack([A.render_sticks(TRUE), A.render_sticks(DECOY)])
    q = np.stack([A.render_sticks(TRUE, A.sweep_params(zero_offset=0.13, strain=0.007)), ref[1]])
    sc = WT.TorchWCC(lnq=True, version=2)
    got = sc.scores(sc.prep_rows(q), sc.prep_refs(ref)).numpy()
    want = SB.WCCLibrary(ref, lnq=True, version=2).scores(q)
    assert np.abs(got - want).max() < 1e-5


def test_vc_gpwdf_recovers_anisotropic_strain_and_offset():
    # The ceiling is < 1: the query keeps its 40 strongest peaks, the candidate keeps every
    # reflection. So compare against the unstrained self-score, not against 1.
    ceiling = SB.vc_gpwdf_scores([SB.query_peaks(A.render_sticks(TRUE))], [TRUE], [(0, 0)])[0]
    p = A.sweep_params(axis_strain=(0.03, -0.01, 0.02), zero_offset=0.1)
    qp = [SB.query_peaks(A.render_sticks(TRUE, p))]
    scores, prm = SB.vc_gpwdf_scores(qp, [TRUE, DECOY], [(0, 0), (0, 1)], return_params=True)
    assert scores[0] > 0.85 * ceiling and scores[0] > scores[1] + 0.1
    assert np.allclose(prm[0, :3], (0.03, -0.01, 0.02), atol=0.005)
    assert abs(prm[0, 6] - 0.1) < 0.03


def test_wcc_library_matches_pairwise():
    ref = np.stack([A.render_sticks(TRUE), A.render_sticks(DECOY)])
    q = A.render_sticks(TRUE, A.sweep_params(zero_offset=0.3))[None]
    pair = SB.wcc_scores(q, ref, l_deg=1.0, shift_deg=0.5)
    lib = SB.WCCLibrary(ref, l_deg=1.0, shift_deg=0.5, chunk=1).scores(q)
    assert np.allclose(pair, lib, atol=1e-3)
