"""Space-group symmetry operators read straight from a CIF, in the CIF's own setting.

Why read them from the CIF instead of asking a symmetry finder: the stick shards store the asymmetric unit in the
frame pymatgen parsed from that CIF, so the CIF's own operator loop is exactly the set of operators that regenerates the
full cell from those atoms. A symbol lookup (e.g. 'P21/c') can return another setting or origin.

`parse_symops(cif)` returns an (n, 3, 4) float32 array: rotation (3 x 3) and translation (3,) of x' = R x + t, with t in
[0, 1). Centring translations appear as separate operators, as CIFs list them.
"""
from __future__ import annotations

import re
import shlex

import numpy as np

_XYZ_TAGS = ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz")
_TERM = re.compile(r"([+-]?)([^+-]+)")


def _parse_component(expr: str) -> "tuple[np.ndarray, float]":
    """'-x+1/2' -> ([-1, 0, 0], 0.5). Handles '1/2+x', 'x-y', '0.25-z', '2x' (rare)."""
    expr = expr.replace(" ", "").lower()
    row, t = np.zeros(3, np.float32), 0.0
    for sign, body in _TERM.findall(expr):
        s = -1.0 if sign == "-" else 1.0
        axis = next((i for i, c in enumerate("xyz") if c in body), None)
        if axis is None:
            num, den = (body.split("/") + ["1"])[:2]
            t += s * float(num) / float(den)
        else:
            coef = body.replace("xyz"[axis], "").replace("*", "")
            row[axis] += s * (float(coef) if coef else 1.0)
    return row, t


def parse_op(op: str) -> np.ndarray:
    parts = op.strip().strip("'\"").split(",")
    if len(parts) != 3:
        raise ValueError(f"bad operator {op!r}")
    m = np.zeros((3, 4), np.float32)
    for i, p in enumerate(parts):
        m[i, :3], t = _parse_component(p)
        m[i, 3] = t % 1.0
    return m


def parse_symops(cif: str) -> np.ndarray:
    """All operators of the first symop loop in `cif`; without a loop, the operators of its space-group symbol
    (ops_from_symbol); identity only if the CIF has neither."""
    lines = cif.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        heads, j = [], i + 1
        while j < len(lines) and lines[j].strip().startswith("_"):
            heads.append(lines[j].strip().split()[0].lower())
            j += 1
        col = next((k for k, h in enumerate(heads) if h in _XYZ_TAGS), None)
        if col is None:
            continue
        ops = []
        while j < len(lines):
            s = lines[j].strip()
            if not s or s.startswith("_") or s.startswith("loop_") or s.startswith("#"):
                break
            toks = shlex.split(s)
            # an operator written without quotes but with spaces ('-x, y, -z') splits into several tokens
            if len(toks) > len(heads):
                toks = toks[:col] + ["".join(toks[col:])]
            ops.append(parse_op(toks[col]))
            j += 1
        if ops:
            return np.stack(ops)
    sym = ops_from_symbol(cif)
    if sym is not None:
        return sym
    return np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), np.float32)], axis=1)[None]


_HALL_TAGS = ("_symmetry_space_group_name_hall", "_space_group_name_hall")
_HM_TAGS = ("_symmetry_space_group_name_h-m", "_space_group_name_h-m_alt")


def _tag_value(lines: "list[str]", tags: tuple) -> "str | None":
    for ln in lines:
        parts = ln.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() in tags:
            v = parts[1].strip().strip("'\"").strip()
            if v and v not in ("?", "."):
                return v
    return None


def ops_from_symbol(cif: str) -> "np.ndarray | None":
    """Operators generated from the space-group symbol, for a CIF without an operator loop (some older COD entries
    give only the Hall and H-M symbols; every CSD CIF written by the CSD API has the loop). The Hall symbol is
    preferred because it fixes setting and origin; an H-M symbol takes gemmi's default setting. pymatgen built those
    entries' cells from the same symbols, and scripts/40 --check compares the expanded sites with pymatgen's."""
    lines = cif.splitlines()
    try:
        import gemmi
        hall = _tag_value(lines, _HALL_TAGS)
        if hall:
            g = gemmi.symops_from_hall(hall)
        else:
            hm = _tag_value(lines, _HM_TAGS)
            sg = gemmi.find_spacegroup_by_name(hm) if hm else None
            if sg is None:
                return None
            g = sg.operations()
        return np.stack([parse_op(op.triplet()) for op in g])
    except (ImportError, RuntimeError, ValueError):
        return None


def expand(frac: np.ndarray, ops: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Apply every operator to fractional coordinates; return the unique sites in [0, 1)."""
    pts = (np.einsum("nij,aj->nai", ops[:, :, :3], frac) + ops[:, None, :, 3]) % 1.0
    pts = pts.reshape(-1, 3)
    keep: list = []
    for p in pts:
        d = np.abs(p[None] - np.asarray(keep)) if keep else None
        if d is not None:
            d = np.minimum(d, 1 - d).max(axis=1)
            if (d < tol).any():
                continue
        keep.append(p)
    return np.asarray(keep, np.float32)
