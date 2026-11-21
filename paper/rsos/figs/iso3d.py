"""Minimal vector 3-D renderer for publication schematics.

Why not mpl_toolkits.mplot3d: it cannot clip solids, its depth sorting is per-artist,
and it will not give exact face-contact geometry between a sphere and a cube. This
module does the small amount of geometry we actually need, exactly:

  * orthographic camera
  * convex-solid clipping against a half-space (Sutherland-Hodgman + cap generation)
    -- this is what makes both quarter cut-aways and face contact between two solids
       ("this cube seats against a flat facet of that sphere") geometrically true
       rather than hand-faked
  * painter's algorithm with explicit layer overrides
  * Lambertian shading

Everything is emitted as matplotlib Polygon patches, so PDF/SVG output stays fully
vector and remains editable.

A convex solid is represented as a list of faces; each face is an (N, 3) float array
of vertices wound counter-clockwise when viewed from outside the solid.
"""
import numpy as np
from matplotlib.patches import Polygon

EPS = 1e-9


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------
class Camera:
    """Orthographic camera.

    `forward` points from the scene toward the viewer, so a larger projected depth
    means closer to the camera.
    """

    def __init__(self, elev=18.0, azim=35.0):
        self.elev = float(elev)
        self.azim = float(azim)
        e, a = np.radians(elev), np.radians(azim)
        forward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)
        self.forward = forward
        self.right = right
        self.up = up
        self.R = np.stack([right, up, forward])  # world -> (u, v, depth)

    def project(self, pts):
        """(..., 3) world -> ((..., 2) screen, (...,) depth)."""
        pts = np.asarray(pts, dtype=float)
        c = pts @ self.R.T
        return c[..., :2], c[..., 2]

    def horizontal_basis(self):
        """Horizontal unit vectors (toward-camera, camera-left).

        Used to place cut planes so the exposed section faces the viewer almost
        head-on, which keeps any glyphs drawn on that section crisp.
        """
        n_to_cam = np.array([self.forward[0], self.forward[1], 0.0])
        n_to_cam /= np.linalg.norm(n_to_cam)
        n_left = np.array([-n_to_cam[1], n_to_cam[0], 0.0])
        return n_to_cam, n_left


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def rotation_to(n):
    """Rotation matrix taking +z onto unit vector n."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3) if n[2] > 0 else np.diag([1.0, -1.0, -1.0])
    c = float(np.dot(z, n))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def face_normal(poly):
    """Outward normal of a wound polygon (Newell's method)."""
    p = np.asarray(poly, float)
    n = np.zeros(3)
    m = len(p)
    for i in range(m):
        a, b = p[i], p[(i + 1) % m]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])


def cube_faces(center, edge, rot=None):
    """Axis-aligned cube (optionally rotated), faces wound CCW from outside."""
    c = np.asarray(center, float)
    h = edge / 2.0
    v = np.array([[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
                  [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]])
    if rot is not None:
        v = v @ np.asarray(rot, float).T
    v = v + c
    idx = [[0, 3, 2, 1],   # -z
           [4, 5, 6, 7],   # +z
           [0, 1, 5, 4],   # -y
           [2, 3, 7, 6],   # +y
           [1, 2, 6, 5],   # +x
           [0, 4, 7, 3]]   # -x
    return [v[i] for i in idx]


def box_faces(center, size):
    """Rectangular box; `size` is (sx, sy, sz)."""
    c = np.asarray(center, float)
    sx, sy, sz = (np.asarray(size, float) / 2.0)
    v = np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                  [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]]) + c
    idx = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
           [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
    return [v[i] for i in idx]


def icosphere(center=(0, 0, 0), radius=1.0, subdiv=2):
    """Subdivided icosahedron. Convex, so it clips correctly as a solid."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    tris = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    tris = [np.array([verts[i] for i in tri]) for tri in tris]
    for _ in range(int(subdiv)):
        out = []
        for tri in tris:
            a, b, c = tri
            ab = (a + b) / 2; bc = (b + c) / 2; ca = (c + a) / 2
            ab /= np.linalg.norm(ab); bc /= np.linalg.norm(bc); ca /= np.linalg.norm(ca)
            out += [np.array([a, ab, ca]), np.array([b, bc, ab]),
                    np.array([c, ca, bc]), np.array([ab, bc, ca])]
        tris = out
    c = np.asarray(center, float)
    return [t_ * radius + c for t_ in tris]


# ---------------------------------------------------------------------------
# clipping
# ---------------------------------------------------------------------------
def clip_polygon(poly, n, d):
    """Sutherland-Hodgman: keep the part of `poly` with n.x <= d."""
    p = np.asarray(poly, float)
    m = len(p)
    out = []
    for i in range(m):
        cur, nxt = p[i], p[(i + 1) % m]
        dc = float(n @ cur - d)
        dn = float(n @ nxt - d)
        if dc <= EPS:
            out.append(cur)
        if (dc < -EPS and dn > EPS) or (dc > EPS and dn < -EPS):
            out.append(cur + (dc / (dc - dn)) * (nxt - cur))
    if len(out) < 3:
        return None
    # drop duplicates introduced at the cut
    ded = [out[0]]
    for q in out[1:]:
        if np.linalg.norm(q - ded[-1]) > 1e-9:
            ded.append(q)
    if len(ded) > 1 and np.linalg.norm(ded[0] - ded[-1]) < 1e-9:
        ded.pop()
    return np.array(ded) if len(ded) >= 3 else None


def _cap_polygon(on_plane, n):
    """Order points lying in a plane into the convex cap polygon, CCW about +n."""
    uniq = []
    for q in on_plane:
        if not any(np.linalg.norm(q - u) < 1e-7 for u in uniq):
            uniq.append(q)
    if len(uniq) < 3:
        return None
    uniq = np.array(uniq)
    u_ax = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(u_ax) < 1e-8:
        u_ax = np.cross(n, [1.0, 0.0, 0.0])
    u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(n, u_ax)
    rel = uniq - uniq.mean(axis=0)
    return uniq[np.argsort(np.arctan2(rel @ v_ax, rel @ u_ax))]


def clip_solid(faces, n, d, want_cap=True):
    """Clip a convex solid to the half-space n.x <= d.

    Returns (kept_faces, cap) where `cap` is the newly exposed cross-section
    polygon lying in the plane (wound CCW about +n), or None. The cap is NOT
    included in kept_faces -- append it yourself, or the solid is open and renders
    hollow.
    """
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    kept, on_plane = [], []
    for f in faces:
        cf = clip_polygon(f, n, d)
        if cf is None:
            continue
        kept.append(cf)
        for q in cf:
            if abs(float(n @ q - d)) < 1e-7:
                on_plane.append(q)
    cap = _cap_polygon(on_plane, n) if want_cap else None
    return kept, cap


def clip_solid_multi(faces, planes):
    """Clip successively by several (n, d) half-spaces. Returns (faces, caps).

    `caps` are (normal, polygon) pairs for the exposed cross-sections, kept apart
    from `faces` only so the caller can style a cut differently from a surface.

    A cap is a face of the resulting solid and is treated as one here: it is carried
    into every later clip, and later caps are built from it as well. Clipping only
    the ORIGINAL faces looks right for two or three planes and then fails silently --
    seat a sphere against enough neighbours and every original face is eventually
    trimmed away, so the solid reports itself empty and vanishes from the figure even
    though its true intersection is large.
    """
    # (source plane normal or None for an original face, polygon)
    cur = [(None, np.asarray(f, float)) for f in faces]
    for n, d in planes:
        n = np.asarray(n, float)
        n = n / np.linalg.norm(n)
        d = float(d)
        kept, on_plane = [], []
        for src, f in cur:
            cf = clip_polygon(f, n, d)
            if cf is None:
                continue
            kept.append((src, cf))
            for q in cf:
                if abs(float(n @ q - d)) < 1e-7:
                    on_plane.append(q)
        if not kept:
            return [], []
        cap = _cap_polygon(on_plane, n)
        if cap is not None:
            kept.append((n, cap))
        cur = kept
    return ([f for src, f in cur if src is None],
            [(src, f) for src, f in cur if src is not None])


# ---------------------------------------------------------------------------
# shading
# ---------------------------------------------------------------------------
def shade(rgb, normal, light, ambient=0.70, diffuse=0.34, back=0.10):
    """Lambertian shading with a soft back-light.

    Deliberately high ambient: these are schematics printed small, so unlit faces
    must stay clearly the same colour as lit ones rather than going muddy.
    """
    rgb = np.asarray(rgb, float)
    lam = float(np.asarray(normal) @ np.asarray(light))
    f = ambient + diffuse * max(lam, 0.0) + back * max(-lam, 0.0)
    return tuple(np.clip(rgb * f, 0, 1))


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------
class Scene:
    """Collects primitives and paints them back-to-front.

    `layer` overrides depth: everything in layer 0 is painted before layer 1, and so
    on. Needed because the schematics deliberately mix a flat backdrop section plane
    with true 3-D geometry standing in front of it.
    """

    DEFAULT_LIGHT = np.array([-0.42, -0.34, 0.84])

    def __init__(self, camera, light=None):
        self.cam = camera
        light = self.DEFAULT_LIGHT if light is None else np.asarray(light, float)
        self.light = light / np.linalg.norm(light)
        self.items = []

    # -- primitives ---------------------------------------------------------
    def add_polygon(self, poly, color, layer=1, alpha=1.0, ec=None, lw=0.0,
                    shade_it=False, base_rgb=None, zbias=0.0, joinstyle="round"):
        poly = np.asarray(poly, float)
        if len(poly) < 3:
            return
        xy, depth = self.cam.project(poly)
        fc = color
        if shade_it and base_rgb is not None:
            fc = shade(base_rgb, face_normal(poly), self.light)
        self.items.append((layer, float(depth.mean()) + zbias,
                           dict(kind="poly", xy=xy, fc=fc, ec=ec, lw=lw,
                                alpha=alpha, joinstyle=joinstyle)))

    def add_solid(self, faces, base_rgb, layer=1, alpha=1.0, ec=None, lw=0.0,
                  cull=True, face_colors=None, zbias=0.0):
        """Add a convex solid with per-face shading and back-face culling."""
        for i, f in enumerate(faces):
            nrm = face_normal(f)
            if cull and float(nrm @ self.cam.forward) <= 0.0:
                continue
            rgb = base_rgb if face_colors is None else face_colors[i]
            col = shade(rgb, nrm, self.light)
            edge = col if ec == "match" else ec
            self.add_polygon(f, col, layer=layer, alpha=alpha, ec=edge, lw=lw,
                             zbias=zbias)

    def add_line(self, pts, color, layer=2, lw=1.0, alpha=1.0, ls="-",
                 zbias=0.0, capstyle="round", zorder_pt=None):
        pts = np.asarray(pts, float)
        xy, depth = self.cam.project(pts)
        key = float(depth.mean()) if zorder_pt is None else float(zorder_pt)
        self.items.append((layer, key + zbias,
                           dict(kind="line", xy=xy, color=color, lw=lw,
                                alpha=alpha, ls=ls, capstyle=capstyle)))

    def add_marker(self, pt, color, layer=3, size=8, marker="o", alpha=1.0,
                   ec="none", lw=0.0, zbias=0.0):
        xy, depth = self.cam.project(np.asarray(pt, float)[None, :])
        self.items.append((layer, float(depth[0]) + zbias,
                           dict(kind="marker", xy=xy[0], color=color, size=size,
                                marker=marker, alpha=alpha, ec=ec, lw=lw)))

    def add_text(self, pt, s, layer=4, zbias=0.0, **kw):
        xy, depth = self.cam.project(np.asarray(pt, float)[None, :])
        self.items.append((layer, float(depth[0]) + zbias,
                           dict(kind="text", xy=xy[0], s=s, kw=kw)))

    # -- output -------------------------------------------------------------
    def draw(self, ax):
        self.items.sort(key=lambda it: (it[0], it[1]))
        for z, (_layer, _depth, it) in enumerate(self.items):
            k = it["kind"]
            if k == "poly":
                ax.add_patch(Polygon(
                    it["xy"], closed=True, facecolor=it["fc"],
                    edgecolor=("none" if it["ec"] is None else it["ec"]),
                    linewidth=it["lw"], alpha=it["alpha"], zorder=z,
                    joinstyle=it["joinstyle"], snap=False))
            elif k == "line":
                ax.plot(it["xy"][:, 0], it["xy"][:, 1], color=it["color"],
                        lw=it["lw"], alpha=it["alpha"], ls=it["ls"],
                        solid_capstyle=it["capstyle"], zorder=z,
                        clip_on=True)
            elif k == "marker":
                ax.plot([it["xy"][0]], [it["xy"][1]], marker=it["marker"],
                        color=it["color"], markersize=it["size"],
                        markeredgecolor=it["ec"], markeredgewidth=it["lw"],
                        alpha=it["alpha"], zorder=z, linestyle="none")
            elif k == "text":
                ax.text(it["xy"][0], it["xy"][1], it["s"], zorder=z, **it["kw"])

    def bounds(self):
        """Screen-space bounding box of everything added so far."""
        xs, ys = [], []
        for _l, _d, it in self.items:
            xy = np.atleast_2d(it["xy"])
            xs += list(xy[:, 0]); ys += list(xy[:, 1])
        if not xs:
            return (-1, 1, -1, 1)
        return min(xs), max(xs), min(ys), max(ys)


# ---------------------------------------------------------------------------
# 2-D helpers used on exposed section planes
# ---------------------------------------------------------------------------
def plane_basis(n):
    """Orthonormal (u, v) spanning the plane with normal n, with u x v = n."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(n @ ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def to3d(pts2d, origin, u, v):
    p = np.atleast_2d(np.asarray(pts2d, float))
    return origin + p[:, 0:1] * u + p[:, 1:2] * v


def disc_polygon(radius, n_seg=72):
    a = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)


def clip_poly2d(poly, clipper):
    """Clip a 2-D polygon by a convex 2-D polygon (Sutherland-Hodgman)."""
    out = np.asarray(poly, float)
    m = len(clipper)
    for i in range(m):
        a, b = clipper[i], clipper[(i + 1) % m]
        e = b - a
        nrm = np.array([-e[1], e[0]])          # inward for CCW clipper
        nrm /= (np.linalg.norm(nrm) + 1e-15)
        d = float(nrm @ a)
        new = []
        k = len(out)
        if k == 0:
            return None
        for j in range(k):
            cur, nxt = out[j], out[(j + 1) % k]
            dc = float(nrm @ cur) - d
            dn = float(nrm @ nxt) - d
            if dc >= -EPS:
                new.append(cur)
            if (dc < -EPS and dn > EPS) or (dc > EPS and dn < -EPS):
                new.append(cur + (dc / (dc - dn)) * (nxt - cur))
        if len(new) < 3:
            return None
        out = np.array(new)
    return out


def rect_polygon(u0, u1, v0, v1):
    return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], float)


def voronoi_cells(points, clipper, ghost_scale=8.0):
    """Voronoi cells of `points`, each clipped to the convex polygon `clipper`.

    Used for grain structures in a sectioned matrix. Returns (cell_polygon, seed_index)
    pairs; cells that fall entirely outside the clipper are dropped.
    """
    from scipy.spatial import Voronoi
    pts = np.asarray(points, float)
    c = np.asarray(clipper, float)
    ctr = c.mean(axis=0)
    rad = float(np.max(np.linalg.norm(c - ctr, axis=1))) * ghost_scale
    a = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ghost = ctr + np.stack([rad * np.cos(a), rad * np.sin(a)], axis=1)
    vor = Voronoi(np.vstack([pts, ghost]))
    out = []
    for i in range(len(pts)):
        reg = vor.regions[vor.point_region[i]]
        if not reg or -1 in reg:
            continue
        cell = clip_poly2d(vor.vertices[reg], c)
        if cell is not None and len(cell) >= 3:
            out.append((cell, i))
    return out


def voronoi_grains(radius, n_grains, seed=0, n_seg=64, margin=1.35):
    """Voronoi cells clipped to a disc - a polycrystalline cross-section.

    Returns a list of 2-D polygons covering the disc.
    """
    from scipy.spatial import Voronoi
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0.0, 0.85, n_grains))
    th = rng.uniform(0, 2 * np.pi, n_grains)
    pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    # mirror points far outside so every interior cell is bounded
    ring_a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ghost = np.stack([radius * margin * 4 * np.cos(ring_a),
                      radius * margin * 4 * np.sin(ring_a)], axis=1)
    vor = Voronoi(np.vstack([pts, ghost]))
    clipper = disc_polygon(radius, n_seg)
    cells = []
    for i in range(n_grains):
        reg = vor.regions[vor.point_region[i]]
        if not reg or -1 in reg:
            continue
        poly = vor.vertices[reg]
        c = clip_poly2d(poly, clipper)
        if c is not None and len(c) >= 3:
            cells.append(c)
    return cells
