"""PyMOL venv: outline-mode renders of ROY polymorphs (look copied from sts-report pymol_atoms.py).

    python render_roy.py roy.json outdir [mol|cell] [form ...] [--axis a|b|c]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pymol2
from chempy import Atom, Bond, models
from pymol.cgo import CYLINDER

JMOL = {"H": (1, 1, 1), "C": (0x90 / 255,) * 3, "N": (0x30 / 255, 0x50 / 255, 0xF8 / 255),
        "O": (1, 0x0D / 255, 0x0D / 255), "S": (1, 1, 0x30 / 255)}
BALL_R = {"H": 0.20, "C": 0.34, "N": 0.34, "O": 0.34, "S": 0.44}
STICK_R = 0.12

data = json.load(open(sys.argv[1]))
outdir = Path(sys.argv[2]); outdir.mkdir(exist_ok=True)
mode = sys.argv[3]
args = sys.argv[4:]
axis = "a"
if "--axis" in args:
    axis = args[args.index("--axis") + 1]; args = args[:args.index("--axis")]
forms = args or list(data)


def kabsch(P, Q):
    """Rotation R, translation t with R @ P_i + t ~ Q_i."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def ring_frame(mol):
    """Canonical anchor atoms: S, ring C bonded to amine N, the rest of the thiophene, then the amine N."""
    els = [a["el"] for a in mol["atoms"]]
    nb = {i: set() for i in range(len(els))}
    for i, j in mol["bonds"]:
        nb[i].add(j); nb[j].add(i)
    s = els.index("S")
    c2 = next(c for c in nb[s] if any(els[n] == "N" for n in nb[c]))
    n_am = next(n for n in nb[c2] if els[n] == "N")
    c3 = next(c for c in nb[c2] if c not in (s, n_am) and els[c] == "C")
    c5 = next(c for c in nb[s] if c != c2)
    c4 = next(c for c in nb[c3] if c in nb[c5])
    return [s, c2, c3, c4, c5, n_am]


def torsion(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1n) * b1n; w = b2 - (b2 @ b1n) * b1n
    return math.degrees(math.atan2(np.cross(b1n, v) @ w, v @ w))


def render(atoms, bonds, out_png, width_px, extra_cgo=None, pad=1.0, fog_start=0.35):
    P = np.array([a["p"] for a in atoms])
    p = pymol2.PyMOL(); p.start(); cmd = p.cmd
    m = models.Indexed()
    for k, a in enumerate(atoms):
        at = Atom(); at.symbol = a["el"]; at.name = f"A{k}"; at.resn = "ROY"; at.resi = "1"
        at.coord = [float(x) for x in a["p"]]; at.vdw = BALL_R[a["el"]]
        m.add_atom(at)
    for i, j in bonds:
        bd = Bond(); bd.index = [i, j]; bd.order = 1; m.add_bond(bd)
    cmd.load_model(m, "st")
    for k, a in enumerate(atoms):
        cmd.set_color(f"c{k}", list(a.get("rgb", JMOL[a["el"]])))
        cmd.color(f"c{k}", f"st and name A{k}")
    cmd.hide("everything"); cmd.show("spheres", "st"); cmd.show("sticks", "st")
    cmd.set("sphere_scale", 1.0); cmd.set("stick_radius", STICK_R)
    cmd.set("valence", 0); cmd.set("stick_ball", 0)
    pts = [P]
    if extra_cgo:
        cmd.load_cgo(extra_cgo[0], "cell"); pts.append(extra_cgo[1])
    allp = np.vstack(pts)
    lo, hi = allp.min(0) - pad, allp.max(0) + pad
    w, h = hi[0] - lo[0], hi[1] - lo[1]
    ctr = (lo + hi) / 2
    fov = 20.0
    dist = h / (2 * math.tan(math.radians(fov) / 2))
    depth = np.abs(allp[:, 2] - ctr[2]).max() + 5
    cmd.set("orthoscopic", 1)
    cmd.set_view((1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -dist, *ctr, dist - depth, dist + depth, fov))
    cmd.set("orthoscopic", 1)
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("ray_shadows", 1); cmd.set("ray_shadow_decay_factor", 0.1)
    cmd.set("ambient", 0.35); cmd.set("direct", 0.55); cmd.set("reflect", 0.35)
    cmd.set("specular", 0.55); cmd.set("shininess", 40); cmd.set("light_count", 2)
    cmd.set("depth_cue", 1); cmd.set("ray_trace_fog", 1); cmd.set("fog_start", fog_start)
    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 1); cmd.set("ray_trace_gain", 0.12); cmd.set("ray_trace_color", "black")
    hpx = int(round(width_px * h / w))
    cmd.ray(width_px, hpx)
    cmd.png(str(out_png), dpi=600)
    p.stop()
    print("RENDERED", out_png, width_px, hpx)


TILT = -35.0
YROT = 35.0
TSIGN = 1
if mode == "mol":
    ref = data["Y"]["mol"]; ref_idx = ring_frame(ref)
    Q = np.array([ref["atoms"][i]["p"] for i in ref_idx])
    # camera frame from the reference: thiophene plane = screen plane, S->C2 along +x
    q0 = Q - Q[:5].mean(0)
    x = q0[1] - q0[0]; x /= np.linalg.norm(x)
    nrm = np.cross(q0[1] - q0[0], q0[2] - q0[0]); nrm /= np.linalg.norm(nrm)
    y = np.cross(nrm, x)
    Rcam = np.vstack([x, y, nrm])
    for f in forms:
        mol = data[f]["mol"]; idx = ring_frame(mol)
        Pm = np.array([a["p"] for a in mol["atoms"]])
        s_, c2_, _, _, _, n_ = idx
        nb0 = {i: set() for i in range(len(mol["atoms"]))}
        for i, j in mol["bonds"]:
            nb0[i].add(j); nb0[j].add(i)
        ip_ = next(c for c in nb0[n_] if c != c2_ and mol["atoms"][c]["el"] == "C")
        if torsion(*(Pm[k] for k in (s_, c2_, n_, ip_))) * TSIGN < 0:
            Pm = -Pm      # centrosymmetric crystal: use the inversion-related copy so all forms twist the same way
        R, t = kabsch(Pm[idx], Q)
        if np.linalg.det(R) < 0:
            raise SystemExit("reflection")
        X = (R @ Pm.T).T + t
        X = (Rcam @ (X - Q[:5].mean(0)).T).T
        tl = math.radians(TILT)
        X = X @ np.array([[1, 0, 0], [0, math.cos(tl), -math.sin(tl)], [0, math.sin(tl), math.cos(tl)]]).T
        yr = math.radians(YROT)
        X = X @ np.array([[math.cos(yr), 0, math.sin(yr)], [0, 1, 0], [-math.sin(yr), 0, math.cos(yr)]]).T
        # torsion C3-C2-N-C(phenyl ipso) is ROY's conformational coordinate
        els = [a["el"] for a in mol["atoms"]]
        nb = {i: set() for i in range(len(els))}
        for i, j in mol["bonds"]:
            nb[i].add(j); nb[j].add(i)
        s, c2, c3, c4, c5, n = idx
        ipso = next(c for c in nb[n] if c != c2 and els[c] == "C")
        print(f, "|theta(S-C2-N-Cipso)| =", round(abs(torsion(*(Pm[k] for k in (s, c2, n, ipso)))), 1))
        atoms = [dict(a, p=X[k].tolist()) for k, a in enumerate(mol["atoms"])]
        render(atoms, mol["bonds"], outdir / f"roy_mol_{f}.png", 1400, pad=0.7, fog_start=0.75)
else:
    for f in forms:
        d = data[f]; V = np.array(d["vec"])
        P = np.array([a["p"] for a in d["packed"]["atoms"]])
        ia = "abc".index(axis)
        view = V[ia] / np.linalg.norm(V[ia])
        o1, o2 = [V[k] for k in range(3) if k != ia]
        x = o1 - (o1 @ view) * view; x /= np.linalg.norm(x)
        y = np.cross(view, x)
        Rcam = np.vstack([x, y, view])
        corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) @ V
        cen = corners.mean(0)
        X = (Rcam @ (P - cen).T).T
        Cc = (Rcam @ (corners - cen).T).T
        cgo = []
        rgb = [0.12, 0.47, 0.71]
        for i in range(8):
            for j in range(i + 1, 8):
                di = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
                if np.abs(di[i] - di[j]).sum() == 1:
                    cgo += [CYLINDER, *Cc[i], *Cc[j], 0.06, *rgb, *rgb]
        atoms = [dict(a, p=X[k].tolist()) for k, a in enumerate(d["packed"]["atoms"])]
        render(atoms, d["packed"]["bonds"], outdir / f"roy_cell_{f}_{axis}.png", 1400,
               extra_cgo=(cgo, Cc), pad=0.8)
