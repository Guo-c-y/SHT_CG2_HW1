# code/viz_suite.py — polygons & hulls viewer (self-contained)
from __future__ import annotations
from typing import List, Tuple, Sequence, Deque
from collections import deque
import random, math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

Point = Tuple[int, int]
PolyLike = Sequence[Point]

# ---------------- basic geometry utils ----------------
def as_points(P: PolyLike | Deque[Point]) -> List[Point]:
    return list(P)

def poly_area_signed(points: Sequence[Point]) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s

def _orient(a: Point, b: Point, c: Point) -> int:
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 1 if v > 0 else (-1 if v < 0 else 0)

def _on_segment(a: Point, b: Point, p: Point) -> bool:
    if _orient(a, b, p) != 0:
        return False
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])

def _seg_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = _orient(a, b, c); o2 = _orient(a, b, d)
    o3 = _orient(c, d, a); o4 = _orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(a, b, c): return True
    if o2 == 0 and _on_segment(a, b, d): return True
    if o3 == 0 and _on_segment(c, d, a): return True
    if o4 == 0 and _on_segment(c, d, b): return True
    return False

def is_simple(points: Sequence[Point]) -> bool:
    P = as_points(points); n = len(P)
    if n < 3: return False
    for i in range(n):
        if P[i] == P[(i + 1) % n]:
            return False
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        for j in range(i + 1, n):
            c, d = P[j], P[(j + 1) % n]
            if j == i or (j == (i + 1) % n) or (i == 0 and j == n - 1):
                continue
            if _seg_intersect(a, b, c, d):
                return False
    return True

# ---------------- style ----------------
def _apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.titlesize": 13,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 160,
    })

# ---------------- helpers ----------------
def _summary(points: List[Point]):
    area = poly_area_signed(points)
    return {"n": len(points), "direction": "CCW" if area > 0 else "CW"}

def _label_vertices(ax, points: List[Point], base_px: int = 4, step: int = 1):
    n = len(points)
    step = max(1, int(step))
    idxs = list(range(0, n, step))
    if (n - 1) not in idxs:
        idxs.append(n - 1)
    peff = [pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()]
    for i in idxs:
        x, y = points[i]
        ax.annotate(
            str(i), xy=(x, y), xytext=(base_px, base_px),
            textcoords="offset points", fontsize=6, color="black",
            ha="left", va="bottom", path_effects=peff, clip_on=True, zorder=9
        )

def _subtitle(ax, P: List[Point], title: str):
    meta = _summary(P)
    ax.set_title(title, fontsize=13, pad=15)
    ax.text(0.5, 1.0, f"n = {meta['n']}   orientation = {meta['direction']}",
            transform=ax.transAxes, fontsize=9, color="black",
            ha="center", va="bottom")

def _finish_axes(ax, coord_range: int):
    ax.set_xlim(0, coord_range)
    ax.set_ylim(0, coord_range)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

# ---------------- random simple polygon generator ----------------
def _rand_unique_points(rng: random.Random, k: int, image_size: int) -> List[Point]:
    seen, pts = set(), []
    while len(pts) < k:
        p = (rng.randrange(0, image_size), rng.randrange(0, image_size))
        if p not in seen:
            seen.add(p); pts.append(p)
    return pts

def _order_by_angle(pts: List[Point]) -> List[Point]:
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

def _make_simple_polygon(rng: random.Random, n: int, image_size: int, retries: int) -> Deque[Point]:
    for _ in range(max(1, retries)):
        pts = _rand_unique_points(rng, n, image_size)
        cand = _order_by_angle(pts)
        if is_simple(cand): return deque(cand)
        jitter = [(x + rng.randint(-2, 2), y + rng.randint(-2, 2)) for x, y in pts]
        cand = _order_by_angle(jitter)
        if is_simple(cand): return deque(cand)
    return deque(_order_by_angle(_rand_unique_points(rng, n, image_size)))

def build_complexity_test_polys_random_simple(
    sizes: Sequence[int],
    image_size: int = 1000,
    retries: int = 500,
    seed: int = 1024
) -> List[Deque[Point]]:
    rng = random.Random(seed)
    return [ _make_simple_polygon(rng, max(3, int(n)), image_size, retries) for n in sizes ]

# ---------------- polygon renderer ----------------
def render_ploys(
    ploys: Sequence[PolyLike],
    mark_every: int = 1,
    coord_range: int = 1000,
    cols_per_row: int = 3,
):
    if not ploys:
        raise ValueError("ploys is empty")
    _apply_paper_style()

    P_list: List[List[Point]] = [as_points(P) for P in ploys]
    N = len(P_list)
    ncols = max(1, int(cols_per_row))
    nrows = (N + ncols - 1) // ncols

    fig_w = 3.6 * ncols + 0.2 * (ncols - 1)
    fig_h = 3.6 * nrows + 0.2 * (nrows - 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True, constrained_layout=True)

    if N == 1:
        axes = [[axes]]  # type: ignore
    elif nrows == 1:
        axes = [list(axes)]  # type: ignore
    else:
        axes = [list(row) for row in axes]  # type: ignore

    step_cfg = max(1, int(mark_every))

    k = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            if k < N:
                P = P_list[k]
                xs, ys = zip(*P)
                ax.plot(list(xs) + [P[0][0]], list(ys) + [P[0][1]],
                        linewidth=1.2, color="black", zorder=3)
                ax.scatter(xs, ys, s=36, color="black", zorder=4)
                _subtitle(ax, P, f"polygon {k+1}")
                _label_vertices(ax, P, base_px=4, step=step_cfg)
                _finish_axes(ax, coord_range)
            else:
                ax.axis("off")
            k += 1
    plt.show()

# ---------------- hulls renderer ----------------
def render_hulls(
    hulls: Sequence[PolyLike],
    polys: Sequence[PolyLike],
    mark_every_hull: int = 1,
    mark_every_poly: int = 0,
    coord_range: int = 1000,
    cols_per_row: int = 3,
):
    """
    Visualize convex hulls over their polygons.
    Style matches render_ploys. Hulls are emphasized.

    - hulls, polys: same length and order.
    - mark_every_hull: label stride on hull vertices (>=1).
    - mark_every_poly: label stride on polygon vertices; 0 disables labels.
    - coord_range: axes set to [0, coord_range]^2 for every subplot.
    - cols_per_row: number of columns per row.
    """
    if not hulls or not polys:
        raise ValueError("hulls and polys must be non-empty")
    if len(hulls) != len(polys):
        raise ValueError("length mismatch: hulls vs polys")

    _apply_paper_style()

    H_list: List[List[Point]] = [as_points(H) for H in hulls]
    P_list: List[List[Point]] = [as_points(P) for P in polys]

    N = len(H_list)
    ncols = max(1, int(cols_per_row))
    nrows = (N + ncols - 1) // ncols

    fig_w = 3.6 * ncols + 0.2 * (ncols - 1)
    fig_h = 3.6 * nrows + 0.2 * (nrows - 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True, constrained_layout=True)

    if N == 1:
        axes = [[axes]]  # type: ignore
    elif nrows == 1:
        axes = [list(axes)]  # type: ignore
    else:
        axes = [list(row) for row in axes]  # type: ignore

    step_h = max(1, int(mark_every_hull))
    step_p = max(0, int(mark_every_poly))

    k = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            if k < N:
                P = P_list[k]
                H = H_list[k]

                # 1) draw polygon lightly (context)
                px, py = zip(*P)
                ax.plot(list(px) + [P[0][0]], list(py) + [P[0][1]],
                        linewidth=0.9, color="#7f7f7f", zorder=2)
                ax.scatter(px, py, s=18, color="#7f7f7f", zorder=3)
                if step_p >= 1:
                    _label_vertices(ax, P, base_px=3, step=step_p)

                # 2) draw hull with emphasis: light fill + bold edge + prominent vertices
                hx, hy = zip(*H)
                ax.fill(list(hx) + [H[0][0]], list(hy) + [H[0][1]],
                        facecolor="#d9d9d9", alpha=0.35, edgecolor="none", zorder=4)
                ax.plot(list(hx) + [H[0][0]], list(hy) + [H[0][1]],
                        linewidth=1.8, color="black", zorder=5)
                ax.scatter(hx, hy, s=46, facecolor="white", edgecolor="black", linewidth=1.2, zorder=6)

                _subtitle(ax, H, f"convex hull {k+1}")
                _label_vertices(ax, H, base_px=4, step=step_h)
                _finish_axes(ax, coord_range)
            else:
                ax.axis("off")
            k += 1
    plt.show()