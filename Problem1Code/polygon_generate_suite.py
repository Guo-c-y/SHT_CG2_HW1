from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import json
import math
import random

Point = Tuple[int, int]


# ---------------- Geometry utils ----------------
def _cross(ax: int, ay: int, bx: int, by: int) -> int:
    return ax * by - ay * bx


def _orient(a: Point, b: Point, c: Point) -> int:
    return _cross(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1])


def _on_seg(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    )


def _proper_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    if o1 == 0 and _on_seg(a, b, c):
        return True
    if o2 == 0 and _on_seg(a, b, d):
        return True
    if o3 == 0 and _on_seg(c, d, a):
        return True
    if o4 == 0 and _on_seg(c, d, b):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _segment_intersection_point(a: Point, b: Point, c: Point, d: Point) -> tuple[float, float] | None:
    """Return intersection point of segments ab and cd for diagnostics.
    Parallel or coincident lines return None.
    """
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return float(px), float(py)


def _is_simple(poly: List[Point]) -> bool:
    """Return True if polygon has no self-intersections. Print diagnostics if not."""
    n = len(poly)
    if n < 3:
        print(f"[diagnose] polygon has <3 points (n={n}).")
        return False

    problems = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if a == b:
            problems.append(
                {
                    "type": "zero-length-edge",
                    "edge": (i, (i + 1) % n),
                    "pts": (a, b),
                }
            )
        for j in range(i + 1, n):
            # skip adjacent edges or those sharing endpoints
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            c, d = poly[j], poly[(j + 1) % n]

            o1 = _orient(a, b, c)
            o2 = _orient(a, b, d)
            o3 = _orient(c, d, a)
            o4 = _orient(c, d, b)

            touched = None
            if o1 == 0 and _on_seg(a, b, c):
                touched = ("touch", "c_on_ab")
            elif o2 == 0 and _on_seg(a, b, d):
                touched = ("touch", "d_on_ab")
            elif o3 == 0 and _on_seg(c, d, a):
                touched = ("touch", "a_on_cd")
            elif o4 == 0 and _on_seg(c, d, b):
                touched = ("touch", "b_on_cd")

            if touched:
                problems.append(
                    {
                        "type": touched[0],
                        "subtype": touched[1],
                        "edges": ((i, (i + 1) % n), (j, (j + 1) % n)),
                        "points": (a, b, c, d),
                        "at": None,
                    }
                )
                continue

            if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
                ip = _segment_intersection_point(a, b, c, d)
                problems.append(
                    {
                        "type": "proper-cross",
                        "edges": ((i, (i + 1) % n), (j, (j + 1) % n)),
                        "points": (a, b, c, d),
                        "at": ip,
                    }
                )

    if problems:
        print(f"[diagnose] NOT simple. Found {len(problems)} issue(s). Showing up to first 10:")
        for k, pr in enumerate(problems[:10], 1):
            if pr["type"] == "zero-length-edge":
                i0, i1 = pr["edge"]
                a, b = pr["pts"]
                print(f"  #{k}: zero-length edge on ({i0}->{i1}) with points {a} == {b}")
            elif pr["type"] == "touch":
                (e1, e2) = pr["edges"]
                a, b, c, d = pr["points"]
                print(f"  #{k}: touching edges {e1} {a}->{b} and {e2} {c}->{d}  [{pr['subtype']}]")
            else:
                (e1, e2) = pr["edges"]
                a, b, c, d = pr["points"]
                ip = pr["at"]
                print(f"  #{k}: proper cross between {e1} {a}->{b} and {e2} {c}->{d} at {ip}")
        if len(problems) > 10:
            print(f"  ... {len(problems) - 10} more")
        return False

    return True


# ---------------- Load polygons from params.json ----------------
def load_polys_cases(expect_closed: bool = False) -> List[deque[Point]]:
    """Load caseA/B/C polygons from params.json into deques.

    Args:
        expect_closed: If True, require first point == last point and check simplicity.

    Returns:
        List of three deques of points in the order A, B, C.
    """
    p = Path("poly_cases.json")
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    keys = ["caseA_points", "caseB_points", "caseC_points"]
    for k in keys:
        if k not in cfg:
            raise KeyError(f"Missing key in params.json: {k}")

    result: List[deque[Point]] = []
    for k in keys:
        pts = list(map(tuple, cfg[k]))
        if len(pts) < 3:
            raise ValueError(f"{k} has fewer than 3 points: {len(pts)}")
        if expect_closed:
            if pts[0] != pts[-1]:
                raise ValueError(
                    f"{k} is expected to be closed but first and last points differ: {pts[0]} vs {pts[-1]}"
                )
            if not _is_simple(pts):
                raise ValueError(f"{k} is not a simple polygon")
        else:
            if not _is_simple(pts):
                print(f"Warning: {k} polyline is self-intersecting; algorithms may fail")

        result.append(deque(pts))
    return result


# ---------------- Random polygon generator ----------------
def generate_ploys(
    sizes: Iterable[int],
    image_size: int,
    retries: int = 2000,
    seed: Optional[int] = None,
) -> List[deque[Point]]:
    """Generate simple polygons with integer coordinates on an image_size×image_size grid.

    Each n in `sizes` yields one simple polygon with exactly n distinct vertices.
    Orientation is not enforced. Retries up to `retries` times per n.

    A feasibility check ensures the grid is large enough:
        image_size >= ceil(sqrt(6 * max_n)) + 6
    """
    if image_size < 8:
        raise ValueError("image_size too small")

    sizes = list(sizes)
    if not sizes:
        return []

    max_n = max(sizes)
    min_need = int(math.ceil(math.sqrt(6 * max_n))) + 6
    if image_size < min_need:
        raise ValueError(f"image_size={image_size} too small for n={max_n}. Use >= {min_need}.")

    rng = random.Random(seed)
    cx = cy = (image_size - 1) / 2.0
    margin = 3.0
    R_out = (image_size - 1) / 2.0 - margin
    R_in = max(R_out * 0.55, 6.0)

    out_polys: List[deque[Point]] = []

    for n in sizes:
        if n < 3:
            raise ValueError(f"n must be >= 3, got {n}")

        ok = False
        for _ in range(retries):
            # angle-sorted points with random radius within an annulus
            angles = sorted(rng.random() * 2.0 * math.pi for _ in range(n))
            pts_f = []
            for th in angles:
                r = R_in + (R_out - R_in) * (0.35 + 0.65 * rng.random())
                x = cx + r * math.cos(th)
                y = cy + r * math.sin(th)
                pts_f.append((x, y))

            # quantize and clamp to grid
            pts = [(int(round(x)), int(round(y))) for (x, y) in pts_f]
            pts = [
                (min(max(px, 0), image_size - 1), min(max(py, 0), image_size - 1))
                for (px, py) in pts
            ]

            # deduplicate; must keep exactly n points
            seen = set()
            dedup: List[Point] = []
            for p in pts:
                if p not in seen:
                    seen.add(p)
                    dedup.append(p)
            if len(dedup) != n:
                continue

            # simple and non-zero area
            if not _is_simple(dedup):
                continue
            area2 = 0
            for i in range(n):
                x1, y1 = dedup[i]
                x2, y2 = dedup[(i + 1) % n]
                area2 += x1 * y2 - x2 * y1
            if area2 == 0:
                continue

            out_polys.append(deque(dedup))
            ok = True
            break

        if not ok:
            raise RuntimeError(
                f"Failed to generate a simple polygon with n={n} in {retries} retries "
                f"under image_size={image_size}."
            )

    return out_polys