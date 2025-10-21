from collections import deque
from typing import List, Tuple, Iterable, Optional
from importlib import resources
import json
import math
import random
from collections import deque
from typing import List, Tuple
from pathlib import Path
import json

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
    if o1 == 0 and _on_seg(a, b, c): return True
    if o2 == 0 and _on_seg(a, b, d): return True
    if o3 == 0 and _on_seg(c, d, a): return True
    if o4 == 0 and _on_seg(c, d, b): return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

# ---- 新增：计算两线段的交点（仅用于诊断输出；平行或重合返回 None）----
def _segment_intersection_point(a: Point, b: Point, c: Point, d: Point):
    x1, y1 = a; x2, y2 = b; x3, y3 = c; x4, y4 = d
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if den == 0:
        return None
    # 使用浮点以便诊断
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
    return (float(px), float(py))

# ---- 修改：保持签名与返回值不变；发现问题时打印详细诊断 ----
def _is_simple(poly: List[Point]) -> bool:
    n = len(poly)
    if n < 3:
        print(f"[diagnose] polygon has <3 points (n={n}).")
        return False

    problems = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        # 零长度边诊断
        if a == b:
            problems.append({
                "type": "zero-length-edge",
                "edge": (i, (i + 1) % n),
                "pts": (a, b)
            })
        for j in range(i + 1, n):
            # 跳过相邻边与共享端点
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            c, d = poly[j], poly[(j + 1) % n]

            o1 = _orient(a, b, c)
            o2 = _orient(a, b, d)
            o3 = _orient(c, d, a)
            o4 = _orient(c, d, b)

            touched = None
            if o1 == 0 and _on_seg(a, b, c): touched = ("touch", "c_on_ab")
            elif o2 == 0 and _on_seg(a, b, d): touched = ("touch", "d_on_ab")
            elif o3 == 0 and _on_seg(c, d, a): touched = ("touch", "a_on_cd")
            elif o4 == 0 and _on_seg(c, d, b): touched = ("touch", "b_on_cd")

            if touched:
                problems.append({
                    "type": touched[0],
                    "subtype": touched[1],
                    "edges": ((i, (i + 1) % n), (j, (j + 1) % n)),
                    "points": (a, b, c, d),
                    "at": None  # 端点或在线上，不唯一
                })
                continue

            if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
                ip = _segment_intersection_point(a, b, c, d)
                problems.append({
                    "type": "proper-cross",
                    "edges": ((i, (i + 1) % n), (j, (j + 1) % n)),
                    "points": (a, b, c, d),
                    "at": ip
                })

    if problems:
        print(f"[diagnose] NOT simple. Found {len(problems)} issue(s). Show up to first 10:")
        for k, pr in enumerate(problems[:10], 1):
            if pr["type"] == "zero-length-edge":
                i0, i1 = pr["edge"]
                a, b = pr["pts"]
                print(f"  #{k}: zero-length-edge on edge ({i0}->{i1}) with points {a} == {b}")
            elif pr["type"] == "touch":
                (e1, e2) = pr["edges"]
                a, b, c, d = pr["points"]
                print(f"  #{k}: touching edges {e1} {a}->{b}  and  {e2} {c}->{d}  [{pr['subtype']}]")
            else:  # proper-cross
                (e1, e2) = pr["edges"]
                a, b, c, d = pr["points"]
                ip = pr["at"]
                print(f"  #{k}: proper-cross between edges {e1} {a}->{b}  and  {e2} {c}->{d}  at {ip}")
        if len(problems) > 10:
            print(f"  ... {len(problems)-10} more")
        return False

    return True

# ---------------- Generator ----------------
def load_polys_cases( expect_closed: bool = False ) -> List[deque[Point]]:
    p = Path("params.json")
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    keys = ["caseA_points", "caseB_points", "caseC_points"]
    for k in keys:
        if k not in cfg:
            raise KeyError(f"Missing key in params.json: {k}")

    result = []
    for k in keys:
        pts = list(map(tuple, cfg[k]))
        if len(pts) < 3:
            raise ValueError(f"{k} has fewer than 3 points: {len(pts)}")
        if expect_closed:
            if pts[0] != pts[-1]:
                raise ValueError(f"{k} is expected to be a closed polygon but first and last points differ: {pts[0]} vs {pts[-1]}")
        # 检查自交，只在闭合多边形情形下 meaningful
        if expect_closed:
            if not _is_simple(pts):
                raise ValueError(f"{k} is not a simple polygon (self-intersecting)")
        # 即便是折线模式，也可检查是否自交（更严格）
        else:
            if not _is_simple(pts):
                print(f"Warning: {k} polyline itself is self-intersecting; algorithm may fail")

        result.append(deque(pts))
    return result


def generate_ploys(
    sizes: Iterable[int],
    image_size: int,
    retries: int = 2000,
    seed: Optional[int] = None,
) -> List[deque[Point]]:
    """
    在 image_size×image_size 网格上为每个 n 生成恰好 n 点的 simple polygon（整数坐标）。
    方向不统一，可为 CW 或 CCW。失败会重试，超过 retries 抛错。

    可行性检查：image_size >= ceil(sqrt(6*max_n)) + 6，否则直接报错。
    """
    if image_size < 8:
        raise ValueError("image_size too small")

    sizes = list(sizes)
    if not sizes:
        return []

    max_n = max(sizes)
    min_need = int(math.ceil(math.sqrt(6 * max_n))) + 6
    if image_size < min_need:
        raise ValueError(
            f"image_size={image_size} too small for n={max_n}. Use >= {min_need}."
        )

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
            # 角度排序 + 随机半径（环带内），减少自交概率
            angles = sorted(rng.random() * 2.0 * math.pi for _ in range(n))
            pts_f = []
            for th in angles:
                r = R_in + (R_out - R_in) * (0.35 + 0.65 * rng.random())
                x = cx + r * math.cos(th)
                y = cy + r * math.sin(th)
                pts_f.append((x, y))

            # 量化与裁剪
            pts = [(int(round(x)), int(round(y))) for (x, y) in pts_f]
            pts = [
                (
                    min(max(px, 0), image_size - 1),
                    min(max(py, 0), image_size - 1),
                )
                for (px, py) in pts
            ]

            # 去重，严格保证点数
            seen = set()
            dedup: List[Point] = []
            for p in pts:
                if p not in seen:
                    seen.add(p)
                    dedup.append(p)
            if len(dedup) != n:
                continue

            # simple + 非零面积
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
                f"Fail to generate simple polygon with n={n} in {retries} retries "
                f"under image_size={image_size}."
            )

    return out_polys