"""Resubmission D1 fixes (tasks/todo.md Section D): K-alpha2 under strain, March-Dollase over equivalent members,
CIF symmetry operators, and the symmetry-aware structure encoder."""
import itertools
import os

import numpy as np
import pytest
import torch

from xrdclip.data import augment as A
from xrdclip.data.symops import expand, parse_op, parse_symops


def _one_peak(tt=30.0):
    d = A.WAVELENGTH / (2 * np.sin(np.radians(tt / 2)))
    return {"two_theta": np.array([tt], np.float32), "intensity": np.array([100.0], np.float32),
            "hkl": np.array([[1, 0, 0]], np.int16), "d_hkl": np.array([d], np.float32),
            "recip": np.eye(3, dtype=np.float32) / d}


def _peaks(y, thr=0.05):
    return [i for i in range(1, len(y) - 1) if y[i] > thr and y[i] >= y[i - 1] and y[i] > y[i + 1]]


def test_ka2_doublet_survives_strain_through_the_training_path():
    R = A.Realism(p_ka2=1.0, B=(0.0, 0.0), jitter=(0.0, 0.0))
    s = A.realism_sticks(_one_peak(), np.random.default_rng(0), R)
    assert np.allclose(s["lam"], [1.0, R.ka2_rho])
    p = A.sweep_params(strain=0.003)
    tt = A._shift_two_theta(np.asarray(s["two_theta"]), np.asarray(s["d_hkl"]), p, np.asarray(s["lam"]))
    split = tt[1] - tt[0]
    assert 0.07 < split < 0.09, split            # Cu K-alpha1/2 split at ~30 deg is ~0.076 deg
    # and the rendered doublet differs from the single line (before the fix it matched it to ~1e-7)
    sharp = A.sweep_params(strain=0.003, u=0.0, v=0.0, w=0.0003)
    y2 = A.render_sticks(s, sharp)
    y1 = A.render_sticks(_one_peak(), sharp)
    assert np.abs(y2 - y1).max() > 0.1
    assert len(_peaks(y2)) == 2


def test_ka2_without_strain_unchanged():
    R = A.Realism(p_ka2=1.0, B=(0.0, 0.0), jitter=(0.0, 0.0))
    s = A.realism_sticks(_one_peak(), np.random.default_rng(0), R)
    tt = A._shift_two_theta(np.asarray(s["two_theta"]), np.asarray(s["d_hkl"]), A.CLEAN, np.asarray(s["lam"]))
    assert np.allclose(tt, s["two_theta"])


def _cubic_rotations():
    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3), np.float32)
            for i, (j, sg) in enumerate(zip(perm, signs)):
                m[i, j] = sg
            rots.append(m)
    return np.stack(rots)


def test_march_dollase_averages_over_equivalent_members():
    p = A.sweep_params(march=0.5, po_axis=(0, 0, 1))
    hkl = np.array([[1, 0, 0]])
    recip = np.eye(3, dtype=np.float32)
    old = A._march_dollase(hkl, recip, p)                       # representative only: 90 deg to the axis
    new = A._march_dollase(hkl, recip, p, _cubic_rotations())
    assert np.isclose(old[0], 2 ** -1.5, rtol=1e-5)
    assert np.isclose(new[0], (4 * 2 ** -1.5 + 2 * 8.0) / 6, rtol=1e-5)   # 2.902


def test_march_dollase_identity_group_is_the_old_behaviour():
    rng = np.random.default_rng(1)
    hkl = rng.integers(-3, 4, size=(20, 3))
    hkl[hkl.sum(1) == 0] = 1
    recip = np.array([[0.1, 0, 0], [0.02, 0.12, 0], [0.01, 0.03, 0.09]], np.float32)
    p = A.sweep_params(march=0.7, po_axis=(0, 1, 1))
    assert np.allclose(A._march_dollase(hkl, recip, p), A._march_dollase(hkl, recip, p, np.eye(3)[None]))


def test_march_dollase_r_above_one_and_unity():
    hkl = np.array([[0, 0, 1], [1, 0, 0]])
    recip = np.eye(3, dtype=np.float32)
    f = A._march_dollase(hkl, recip, A.sweep_params(march=1.5, po_axis=(0, 0, 1)))
    assert f[0] < 1 < f[1]                                      # needles: along-axis reflections weakened
    assert np.allclose(A._march_dollase(hkl, recip, A.CLEAN), 1.0)


@pytest.mark.parametrize("op,R,t", [
    ("x,y,z", np.eye(3), [0, 0, 0]),
    ("-x,1/2+y,1/2-z", np.diag([-1, 1, -1]), [0, 0.5, 0.5]),
    ("'x-y, x, z+1/6'", [[1, -1, 0], [1, 0, 0], [0, 0, 1]], [0, 0, 1 / 6]),
    ("-x+0.25,-y,z-1/2", np.diag([-1, -1, 1]), [0.25, 0, 0.5]),
])
def test_parse_op(op, R, t):
    m = parse_op(op)
    assert np.allclose(m[:, :3], R) and np.allclose(m[:, 3], t)


def test_parse_symops_loop_and_expand():
    cif = """data_x
_symmetry_space_group_name_H-M 'P 21/c'
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 x,y,z
2 -x,1/2+y,1/2-z
3 -x,-y,-z
4 x,1/2-y,1/2+z
_cell_length_a 5
"""
    ops = parse_symops(cif)
    assert ops.shape == (4, 3, 4)
    assert len(expand(np.array([[0.1, 0.2, 0.3]], np.float32), ops)) == 4
    assert len(expand(np.array([[0.0, 0.0, 0.0]], np.float32), ops)) == 2    # special position
    assert parse_symops("data_y\n_cell_length_a 5\n").shape == (1, 3, 4)       # no loop -> identity


def test_parse_symops_from_symbol_without_loop():
    """Older COD CIFs give only the symbols (e.g. COD 2001164): the operators come from the Hall symbol."""
    hall_only = "data_z\n_symmetry_space_group_name_Hall  'C -2yc'\n_symmetry_space_group_name_H-M   'C 1 c 1'\n"
    ops = parse_symops(hall_only)
    want = {"x,y,z", "x,-y,z+1/2", "x+1/2,y+1/2,z", "x+1/2,-y+1/2,z+1/2"}
    assert ops.shape == (4, 3, 4)
    assert {tuple(np.round(o, 4).ravel()) for o in ops} == {tuple(np.round(parse_op(w), 4).ravel()) for w in want}
    assert len(expand(np.array([[0.1, 0.2, 0.3]], np.float32), ops)) == 4
    hm_only = "data_w\n_space_group_name_H-M_alt 'P 1 21/c 1'\n"
    assert parse_symops(hm_only).shape == (4, 3, 4)


def test_parse_symops_skips_loops_quoted_in_text_fields():
    """COD 2004836 quotes an old, incomplete operator loop inside its depositor comments; that loop must be ignored and
    the operators taken from the Hall symbol (Pbca, 8 operators)."""
    cif = ("data_q\n_symmetry_space_group_name_Hall  '-P 2ac 2ab'\n_symmetry_space_group_name_H-M   'P b c a'\n"
           "_cod_depositor_comments\n;\n Commenting out the incomplete symmetry operator list:\n\n loop_\n"
           " _symmetry_equiv_pos_as_xyz\n '1/2 + x, y, 1/2 - z'\n\n The spacegroup symbols seem to be correct.\n;\n"
           "loop_\n_atom_site_label\n_atom_site_fract_x\nC1 0.1\n")
    assert parse_symops(cif).shape == (8, 3, 4)
    real = cif + "loop_\n_symmetry_equiv_pos_as_xyz\nx,y,z\n-x,-y,-z\n"   # a real loop after the text field still wins
    assert parse_symops(real).shape == (2, 3, 4)


def test_symops_encoder_forward_and_operator_order_invariance():
    from xrdclip.data.dataset import structure_tensors
    from xrdclip.models.clip import XRDClip
    torch.manual_seed(0)
    m = XRDClip(embed_dim=32, symops=True).eval()
    ops = parse_symops("loop_\n_symmetry_equiv_pos_as_xyz\nx,y,z\n-x,1/2+y,1/2-z\n-x,-y,-z\nx,1/2-y,1/2+z\n")
    rec = {"atom_z": np.array([6, 7, 8]), "atom_frac": np.random.default_rng(0).random((3, 3)).astype(np.float32),
           "cell": np.array([5, 6, 7, 90, 100, 90], np.float32), "n_atoms": 3, "ops": ops}
    rec2 = {**rec, "ops": ops[::-1].copy()}
    t1, t2 = structure_tensors([rec]), structure_tensors([rec2])
    with torch.no_grad():
        z1 = m.encode_structure(**t1)
        z2 = m.encode_structure(**t2)
        z3 = m.encode_structure(**{**t1, "ops": t1["ops"][:, :2], "ops_mask": t1["ops_mask"][:, :2]})
    assert z1.shape == (1, 32)
    assert torch.allclose(z1, z2, atol=1e-5)          # a set of operators, not a sequence
    assert not torch.allclose(z1, z3, atol=1e-3)      # the operators matter


def test_plain_encoder_ignores_operators():
    from xrdclip.data.dataset import structure_tensors
    from xrdclip.models.clip import XRDClip
    m = XRDClip(embed_dim=32).eval()
    rec = {"atom_z": np.array([6, 7]), "atom_frac": np.zeros((2, 3), np.float32),
           "cell": np.array([5, 6, 7, 90, 90, 90], np.float32), "n_atoms": 2}
    t = structure_tensors([rec])
    with torch.no_grad():
        a = m.encode_structure(**t)
        b = m.encode_structure(t["atom_z"], t["atom_frac"], t["lattice"], t["mask"])
    assert torch.allclose(a, b)


@pytest.mark.skipif(not os.path.exists("runs/xrdclip_poly/best.pt"), reason="original checkpoint not present")
def test_original_checkpoint_still_loads():
    from xrdclip.models.clip import load_checkpoint
    m = load_checkpoint("runs/xrdclip_poly/best.pt")
    assert not m.symops


def test_compound_ids_merges_families_that_share_a_same_form_entry():
    """Two benchmark refcode families linked by a shared same-form entry are one compound (GOFWUV / QQQAUJ04);
    an unlinked family stays its own compound, and a link to a non-benchmark family changes nothing."""
    import importlib, sys
    sys.path.insert(0, "scripts")
    bench = importlib.import_module("19_real_bench")
    truth = [{"true_stems": ["QQQAUJ"], "same_form": [{"refcode": "QQQAUJ04"}, {"refcode": "GOFWUV01"}]},
             {"true_stems": ["GOFWUV"], "same_form": [{"refcode": "GOFWUV"}, {"refcode": "QQQAUJ04"}]},
             {"true_stems": ["NAPHTA"], "same_form": [{"refcode": "NAPHTA10"}, {"refcode": "ZZZZZZ01"}]},
             {"true_stems": ["QQQAUJ"], "same_form": [{"refcode": "QQQAUJ04"}]}]
    assert bench.compound_ids(truth) == ["GOFWUV", "GOFWUV", "NAPHTA", "GOFWUV"]
