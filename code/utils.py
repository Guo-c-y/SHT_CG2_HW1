# code/utils.py
# 几何与生成工具函数（无类）
from __future__ import annotations
from typing import List, Tuple, Dict, Iterable
import math

Point = Tuple[int, int]

# --------- shared helpers moved from viz_suite ---------
def as_points(seq: Iterable[Tuple[int, int]]) -> List[Point]:
    """Normalize any iterable of (x,y) to List[Point] with ints."""
    return [(int(x), int(y)) for x, y in list(seq)]

def unit(vx: float, vy: float) -> Tuple[float, float]:
    """Return normalized vector; (0,0) if input is zero."""
    n = math.hypot(vx, vy)
    if n == 0:
        return (0.0, 0.0)
    return (vx / n, vy / n)

def left_normal(vx: float, vy: float) -> Tuple[float, float]:
    """Left-hand normal (rotate 90 degrees CCW)."""
    return (-vy, vx)

# ------------------------- 基础几何 -------------------------
def orient(a: Point, b: Point, c: Point) -> int:
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def on_segment(a: Point, b: Point, p: Point) -> bool:
    if orient(a,b,p) != 0:
        return False
    x1,x2 = sorted((a[0], b[0]))
    y1,y2 = sorted((a[1], b[1]))
    return x1 <= p[0] <= x2 and y1 <= p[1] <= y2

def segments_properly_intersect(a: Point,b: Point,c: Point,d: Point) -> bool:
    # 严格相交（排除端点接触/共线重叠）
    o1,o2,o3,o4 = orient(a,b,c), orient(a,b,d), orient(c,d,a), orient(c,d,b)
    if o1 == 0 and on_segment(a,b,c): return False
    if o2 == 0 and on_segment(a,b,d): return False
    if o3 == 0 and on_segment(c,d,a): return False
    if o4 == 0 and on_segment(c,d,b): return False
    return (o1>0) != (o2>0) and (o3>0) != (o4>0)

def dist_pp(a: Point,b: Point) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def dist_ps(p: Point, a: Point, b: Point) -> float:
    vx, vy = b[0]-a[0], b[1]-a[1]
    wx, wy = p[0]-a[0], p[1]-a[1]
    vv = vx*vx + vy*vy
    if vv == 0:
        return dist_pp(p,a)
    t = max(0.0, min(1.0, (wx*vx + wy*vy)/vv))
    proj = (a[0]+t*vx, a[1]+t*vy)
    return math.hypot(p[0]-proj[0], p[1]-proj[1])

def poly_area_signed(P: List[Point]) -> float:
    s = 0
    for i in range(len(P)):
        x1,y1 = P[i]; x2,y2 = P[(i+1)%len(P)]
        s += x1*y2 - x2*y1
    return 0.5*s

def is_simple(P: List[Point]) -> bool:
    n = len(P)
    if n < 3:
        return False
    for i in range(n):
        a1,a2 = P[i], P[(i+1)%n]
        for j in range(i+1, n):
            if j == i or (j+1)%n == i or (i+1)%n == j:
                continue
            b1,b2 = P[j], P[(j+1)%n]
            if segments_properly_intersect(a1,a2,b1,b2):
                return False
    return True

def min_vertex_dist(P: List[Point]) -> float:
    m = float("inf")
    for i in range(len(P)):
        for j in range(i+1, len(P)):
            d = dist_pp(P[i], P[j])
            if d < m:
                m = d
    return m

def min_nonadjacent_point_edge_dist(P: List[Point]) -> float:
    n = len(P); m = float("inf")
    for i in range(n):
        p = P[i]
        for j in range(n):
            if j in {i, (i-1)%n, (i+1)%n}:
                continue
            a,b = P[j], P[(j+1)%n]
            d = dist_ps(p,a,b)
            if d < m:
                m = d
    return m

# ------------------------- 生成与校验原子操作 -------------------------
def try_insert(P: List[Point], idx: int, p: Point,
               eps_len: int, eps_point: int, eps_edge: int) -> bool:
    # 将点 p 插入到边 (idx, idx+1) 之间，若满足简单性与阈值则提交
    if any(p == q for q in P):
        return False
    if dist_pp(p, P[idx]) < eps_len:
        return False
    if dist_pp(p, P[(idx+1)%len(P)]) < eps_len:
        return False
    Q = P[:idx+1] + [p] + P[idx+1:]
    if not is_simple(Q):
        return False
    if min_vertex_dist(Q) < eps_point:
        return False
    if min_nonadjacent_point_edge_dist(Q) < eps_edge:
        return False
    P[:] = Q
    return True

def final_check(P: List[Point], cfg: Dict) -> Dict:
    area = poly_area_signed(P)
    if abs(area) < cfg["epsilon_area_min"]:
        raise RuntimeError("area too small")
    if not is_simple(P):
        raise RuntimeError("not simple")
    if min_vertex_dist(P) < cfg["epsilon_point"]:
        raise RuntimeError("vertex too close")
    if min_nonadjacent_point_edge_dist(P) < cfg["epsilon_edge"]:
        raise RuntimeError("point-edge too close")
    return {"area_signed": area, "direction": "CCW" if area>0 else "CW"}