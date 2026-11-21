"""Camera, light, solids and drawing helpers shared by the poster's schematics.

One camera and one light for every icon, so they read as one set. Layout is done in page terms: P(a, b, z) is a
point `a` to the right, `b` toward the viewer and `z` up. 1 world unit is 1 inch on the poster.

Each object is painted on its own layer (the caller knows what stands in front of what); within an object the
faces sort by depth, then its outline goes on at layer + 0.4 and its inner edges at layer + 0.5.
"""
from __future__ import annotations

import numpy as np

import palette as pal
from iso3d import Camera, face_normal, icosphere, rotation_to

CAM = Camera(elev=24, azim=-58)
RH = CAM.right                                  # runs right on the page
TH = CAM.horizontal_basis()[0]                  # runs toward the viewer (down the page)
UP = np.array([0.0, 0.0, 1.0])
_l = -0.50 * RH + 0.42 * TH + 0.76 * UP         # light from the upper left, in front
LIGHT = _l / np.linalg.norm(_l)
_h = LIGHT + CAM.forward
HALF = _h / np.linalg.norm(_h)                  # half vector, for highlights on metal and glass
rgb = pal.hex2rgb
OUTLINE_LW = 0.7


def P(a, b, z=0.0):
    return a * RH + b * TH + z * UP


def screen(pt):
    return CAM.project(np.asarray(pt, float)[None, :])[0][0]


# ---------------------------------------------------------------- solids
def ring(center, r, n=96):
    """Horizontal circle, counter-clockwise seen from above."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.asarray(center, float) + np.c_[r * np.cos(th), r * np.sin(th), np.zeros(n)]


def frustum(base, r0, r1, h, n=96, bottom=True, top=True):
    """Upright cone frustum: radius r0 at the base, r1 at height h. Convex; the top cap is the last face."""
    b, t = ring(base, r0, n), ring(np.asarray(base, float) + h * UP, r1, n)
    faces = [np.array([b[i], b[(i + 1) % n], t[(i + 1) % n], t[i]]) for i in range(n)]
    if bottom:
        faces.append(b[::-1])
    if top:
        faces.append(t)
    return faces


def lying(start, direction, r, length, n=64):
    """Cylinder lying along a horizontal direction from `start`; its far-end cap is the last face."""
    R = rotation_to(direction)
    return [f @ R.T + np.asarray(start, float) for f in frustum((0, 0, 0), r, r, length, n)]


def obox(center, size):
    """Box squared up to the page: size = (width to the right, depth away from the viewer, height)."""
    c = np.asarray(center, float)
    sx, sy, sz = np.asarray(size, float) / 2
    away = -TH                                   # (RH, away, UP) is right-handed
    v = np.array([c + x * RH + y * away + z * UP for x, y, z in
                  [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
                   (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)]])
    return [v[i] for i in ([0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3])]


def silhouette(center, r, z0, z1, n=64):
    """Page outline of an upright open cylinder: front half of the base, back half of the top."""
    c = np.asarray(center, float)
    front = [c + r * np.cos(p) * RH + r * np.sin(p) * TH + z0 * UP for p in np.linspace(np.pi, 0, n)]
    back = [c + r * np.cos(p) * RH + r * np.sin(p) * TH + z1 * UP for p in np.linspace(0, -np.pi, n)]
    return np.array(front + back)


def arc(center, r, z, p0, p1, n=48):
    """Part of a horizontal circle; 90 deg is the point nearest the viewer, 0 deg the rightmost."""
    c = np.asarray(center, float)
    return np.array([c + r * np.cos(p) * RH + r * np.sin(p) * TH + z * UP for p in np.radians(np.linspace(p0, p1, n))])


# ---------------------------------------------------------------- light
def lit(base, normal, spec=0.0, gloss=18, ambient=0.60, diffuse=0.44):
    lam = float(normal @ LIGHT)
    c = np.asarray(base, float) * (ambient + diffuse * max(lam, 0.0) + 0.07 * max(-lam, 0.0))
    if spec:
        c = c + spec * max(float(normal @ HALF), 0.0) ** gloss
    return tuple(np.clip(c, 0, 1))


def dark(colour, f=0.55):
    return tuple(np.asarray(rgb(colour) if isinstance(colour, str) else colour, float) * f)


def _hull(xy):
    """Indices of the convex hull of 2-D points (monotone chain)."""
    order = sorted(range(len(xy)), key=lambda i: (xy[i, 0], xy[i, 1]))

    def turn(o, a, b):
        return (xy[a, 0] - xy[o, 0]) * (xy[b, 1] - xy[o, 1]) - (xy[a, 1] - xy[o, 1]) * (xy[b, 0] - xy[o, 0])

    lower, upper = [], []
    for i in order:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    for i in reversed(order):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


# ---------------------------------------------------------------- painting
def paint(sc, faces, colour, layer, spec=0.0, gloss=18, alpha=1.0, outline=True, edge=None, ambient=0.60,
          diffuse=0.44):
    """Shade and add the camera-facing faces of a solid; outline their joint silhouette. Returns those faces."""
    base = rgb(colour) if isinstance(colour, str) else np.asarray(colour, float)
    seen = []
    for f in faces:
        n = face_normal(f)
        if float(n @ CAM.forward) <= 1e-9:
            continue
        col = lit(base, n, spec, gloss, ambient, diffuse)
        sc.add_polygon(f, col, layer=layer, alpha=alpha, ec=col if alpha == 1.0 else None, lw=0.3 if alpha == 1.0 else 0.0)
        seen.append(np.asarray(f, float))
    if outline and seen:
        outline_of(sc, seen, edge or dark(base), layer + 0.4)
    return seen


def outline_of(sc, faces, colour, layer, lw=OUTLINE_LW):
    """Silhouette of a convex set of faces, as an unfilled polygon."""
    pts = np.vstack(faces)
    sc.add_polygon(pts[_hull(CAM.project(pts)[0])], "none", layer=layer, ec=colour, lw=lw)


def edge(sc, pts, colour, layer, lw=0.6, closed=True):
    pts = np.asarray(pts, float)
    sc.add_line(np.vstack([pts, pts[:1]]) if closed else pts, colour, layer=layer, lw=lw)


def cylinder(sc, base, r0, r1, h, colour, layer, spec=0.0, gloss=18, n=96, top_edge=True, **light):
    """Painted frustum with its outline and the edge of its top face."""
    faces = frustum(base, r0, r1, h, n)
    paint(sc, faces, colour, layer, spec, gloss, **light)
    if top_edge:
        edge(sc, faces[-1], dark(colour), layer + 0.5)
    return faces


def ball(sc, center, r, colour, layer):
    paint(sc, icosphere(center, r, 3), colour, layer, spec=0.3, gloss=6, ambient=0.66, diffuse=0.4)


SHADOW_STEPS = 24                               # many faint steps, so the falloff shows no rings


def shadow(sc, center, r, strength=1.0, layer=0.0):
    """Soft contact shadow: stacked faint ellipses on the ground, nudged away from the light."""
    c = np.asarray(center, float) * np.array([1.0, 1.0, 0.0]) + 0.05 * r * RH - 0.04 * r * TH
    for k in range(SHADOW_STEPS):
        rr = r * (0.80 + 0.42 * k / (SHADOW_STEPS - 1))
        sc.add_polygon(ring(c, rr, 64), pal.SHADOW, layer=layer, alpha=0.011 * strength)


def shadow_strip(sc, start, end, r, strength=1.0, layer=0.0):
    """The same, under something long and lying down."""
    a, b = (np.asarray(p, float) * np.array([1.0, 1.0, 0.0]) + 0.05 * RH - 0.04 * TH for p in (start, end))
    d = (b - a) / np.linalg.norm(b - a)
    s = np.cross(UP, d)
    for k in range(SHADOW_STEPS):
        rr = r * (0.8 + 1.3 * k / (SHADOW_STEPS - 1))
        ph = np.linspace(-np.pi / 2, np.pi / 2, 16)
        cap_b = [b + rr * (np.cos(p) * d + np.sin(p) * s) for p in ph]
        cap_a = [a + rr * (-np.cos(p) * d - np.sin(p) * s) for p in ph]
        sc.add_polygon(np.array(cap_b + cap_a), pal.SHADOW, layer=layer, alpha=0.011 * strength)


def box(sc, center, size, colour, layer, spec=0.0, gloss=18, **light):
    """Painted page-squared box with its outline and the edge of its top face."""
    faces = obox(center, size)
    paint(sc, faces, colour, layer, spec, gloss, **light)
    edge(sc, faces[1], dark(colour), layer + 0.5)
    return faces
