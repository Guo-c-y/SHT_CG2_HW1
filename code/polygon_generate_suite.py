# polygon_generate_suite.py — 生成三类 simple polygon，并返回双端队列（deque）
from __future__ import annotations
from typing import List, Tuple, Dict, Set
import json, math, random
from collections import deque

from utils import (
    is_simple, poly_area_signed, try_insert
)

Point = Tuple[int, int]

# ---------------- 参数与工具 ----------------
def load_params(path: str = "params.json") -> Dict:
    """
    已精简：无 grid_min / grid_max。
    必需键：
      seed, epsilon_point, epsilon_len, epsilon_edge, epsilon_area_min, max_retries
    """
    with open(path, "r", encoding="utf-8") as f:
        g = json.load(f)
    need = ["seed", "epsilon_point", "epsilon_len", "epsilon_edge", "epsilon_area_min", "max_retries"]
    for k in need:
        if k not in g:
            raise KeyError(f"missing key in params.json: {k}")
    return g


def _randi(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def _scaffold_simple_polygon(rng: random.Random, cfg: Dict, k_lo: int, k_hi: int) -> List[Point]:
    """
    随机散点按极角排序，直到 simple 且面积达阈值。
    失败提供诊断信息。
    """
    last = {"k": None, "simple": None, "area": None}
    for _ in range(cfg["max_retries"]):
        k = _randi(rng, k_lo, k_hi)
        cx, cy = _randi(rng, 30, 70), _randi(rng, 30, 70)
        pts = set()
        while len(pts) < k:
            pts.add((_randi(rng, 20, 80), _randi(rng, 20, 80)))
        P = sorted(list(pts), key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        ok_simple = is_simple(P)
        area = abs(poly_area_signed(P))
        last.update({"k": k, "simple": ok_simple, "area": area})
        if ok_simple and area >= cfg["epsilon_area_min"]:
            return P
    raise RuntimeError(
        "scaffold fail: "
        f"retries={cfg['max_retries']}, k_range=[{k_lo},{k_hi}], "
        f"area_min={cfg['epsilon_area_min']}. "
        f"last_try: k={last['k']}, simple={last['simple']}, area={last['area']:.3f}"
    )


# --------- 生成过程标记：区分脚手架点 vs 后续插入点（可视化空心/实心） ---------
class GenContext:
    def __init__(self) -> None:
        self.scaffold_points: Set[Point] = set()
        self.inserted_points: Set[Point] = set()

    def mark_scaffold(self, pts: List[Point]) -> None:
        self.scaffold_points = set(pts)

    def mark_inserted(self, q: Point) -> None:
        if q not in self.scaffold_points:
            self.inserted_points.add(q)

    def is_inserted(self, q: Point) -> bool:
        return q in self.inserted_points


# ---------------- 顶层小工具 ----------------
def unit_int(vx: int, vy: int) -> Tuple[int, int]:
    """整向量归一为互质整步。"""
    if vx == 0 and vy == 0:
        return 0, 0
    g = max(1, math.gcd(abs(vx), abs(vy)))
    return vx // g, vy // g


def centroid_int(P: List[Point]) -> Tuple[int, int]:
    """整型质心近似。"""
    return sum(x for x, _ in P) // len(P), sum(y for _, y in P) // len(P)


def try_insert_and_mark_on_edge(P: List[Point], edge_idx: int, q: Point, cfg: Dict, ctx: GenContext) -> bool:
    ok = try_insert(P, edge_idx, q, cfg["epsilon_len"], cfg["epsilon_point"], cfg["epsilon_edge"])
    if ok:
        ctx.mark_inserted(q)
    return ok


def try_insert_anywhere_and_mark(P: List[Point], q: Point, cfg: Dict, ctx: GenContext) -> bool:
    for i in range(len(P)):
        if try_insert(P, i, q, cfg["epsilon_len"], cfg["epsilon_point"], cfg["epsilon_edge"]):
            ctx.mark_inserted(q)
            return True
    return False


def add_pocket(P: List[Point], rng: random.Random, cfg: Dict, ctx: GenContext) -> bool:
    """按边中点朝质心方向插 3 点（向心/离心/凹入），三点都成功才算一次口袋。"""
    tries = 0
    while tries < cfg["max_retries"]:
        tries += 1
        idx = _randi(rng, 0, len(P) - 1)
        a, b = P[idx], P[(idx + 1) % len(P)]
        gx, gy = centroid_int(P)
        mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        vx, vy = unit_int(gx - mx, gy - my)
        if vx == 0 and vy == 0:
            continue
        h1 = (mx + vx, my + vy)           # 向心
        h2 = (mx - vx, my - vy)           # 离心
        rdepth = _randi(rng, 2, 5)
        r = (mx + rdepth * vx, my + rdepth * vy)   # 凹入
        ok1 = try_insert_and_mark_on_edge(P, idx, h1, cfg, ctx)
        ok2 = try_insert_anywhere_and_mark(P, h2, cfg, ctx)
        ok3 = try_insert_anywhere_and_mark(P, r, cfg, ctx)
        if ok1 and ok2 and ok3:
            return True
    return False


def _synthesize_collinear_edge(P: List[Point], rng: random.Random, cnum: int, cfg: Dict, ctx: GenContext) -> bool:
    """
    人工构造一条水平或垂直的长边，使其内部整点 >= cnum：
      1) 选原始边 (j,j+1)，在其位置先插入 p1，再在其后插入 p2，使 p1-p2 成为相邻顶点。
      2) 在 (p1,p2) 之间的格点补入 cnum 个点。
    全过程若任一步失败则重试。
    """
    for _ in range(cfg["max_retries"]):
        P_try = P[:]
        j = _randi(rng, 0, len(P_try) - 1)
        a, b = P_try[j], P_try[(j + 1) % len(P_try)]
        mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2

        horiz = rng.random() < 0.5
        span = cnum + 2  # g = span, 内部 = span-1 >= cnum
        if horiz:
            y = max(20, min(80, my))
            x1 = max(20, min(80 - span, mx - span // 2))
            x2 = x1 + span
            p1, p2 = (x1, y), (x2, y)
            targets = [(x1 + t, y) for t in range(1, span)]
        else:
            x = max(20, min(80, mx))
            y1 = max(20, min(80 - span, my - span // 2))
            y2 = y1 + span
            p1, p2 = (x, y1), (x, y2)
            targets = [(x, y1 + t) for t in range(1, span)]

        # 插入 p1、p2，形成新边 (p1,p2)
        if not try_insert(P_try, j, p1, cfg["epsilon_len"], cfg["epsilon_point"], cfg["epsilon_edge"]):
            continue
        if not try_insert(P_try, j + 1, p2, cfg["epsilon_len"], cfg["epsilon_point"], cfg["epsilon_edge"]):
            continue

        # 在新边之间补 cnum 个格点（从 p1 后开始）
        placed = 0
        insert_idx = j + 1
        added_points: List[Point] = [p1, p2]
        for q in targets:
            if placed >= cnum:
                break
            if try_insert(P_try, insert_idx, q, cfg["epsilon_len"], cfg["epsilon_point"], cfg["epsilon_edge"]):
                added_points.append(q)
                insert_idx += 1
                placed += 1

        if placed >= cnum and is_simple(P_try):
            # 提交
            P[:] = P_try
            for q in added_points:
                if q not in ctx.scaffold_points:
                    ctx.mark_inserted(q)
            return True
    return False


# ---------------- A: 人工长边保证“多重共线”；其余保持 ----------------
def generate_A(cfg: Dict, ctx: GenContext) -> deque:
    rng = random.Random(cfg["seed"] + 101)

    P = _scaffold_simple_polygon(rng, cfg, 5, 8)
    ctx.mark_scaffold(P)

    # 仅保留：人工合成一条水平/垂直长边，再补 cnum 个共线点
    cnum = 2 if len(P) >= 6 else 3
    ok = _synthesize_collinear_edge(P, rng, cnum, cfg, ctx)
    if not ok:
        raise RuntimeError("A: collinear synthesis fail")

    # 近接触：边中点 10×10 方框随机点，任意边插入一次即止
    for _ in range(10):
        j = _randi(rng, 0, len(P) - 1)
        a, b = P[j], P[(j + 1) % len(P)]
        mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        px, py = _randi(rng, mx - 10, mx + 10), _randi(rng, my - 10, my + 10)
        if try_insert_anywhere_and_mark(P, (px, py), cfg, ctx):
            break

    # 近共线：沿外法线偏 1 格
    j = _randi(rng, 0, len(P) - 1)
    a, b = P[j], P[(j + 1) % len(P)]
    dx, dy = b[0] - a[0], b[1] - a[1]
    nx, ny = unit_int(-dy, dx)
    if nx or ny:
        tnum, tden = _randi(rng, 1, 3), _randi(rng, 4, 6)
        qx = a[0] + round(dx * tnum / tden) + nx
        qy = a[1] + round(dy * tnum / tden) + ny
        try_insert_anywhere_and_mark(P, (int(qx), int(qy)), cfg, ctx)

    # 近退化：边中点附近 ±1 扰动插 1–2 点
    added, tries = 0, 0
    while added < 2 and tries < cfg["max_retries"]:
        tries += 1
        i = _randi(rng, 0, len(P) - 1)
        a, b = P[i], P[(i + 1) % len(P)]
        mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        px, py = mx + _randi(rng, -1, 1), my + _randi(rng, -1, 1)
        if try_insert_and_mark_on_edge(P, i, (px, py), cfg, ctx):
            added += 1

    # 强制 A 为 CW
    if poly_area_signed(P) > 0:
        P.reverse()

    return deque(P)


# ---------------- B: 两个口袋 + 随机补点 ----------------
def generate_B(cfg: Dict, ctx: GenContext) -> deque:
    rng = random.Random(cfg["seed"] + 202)

    P = _scaffold_simple_polygon(rng, cfg, 6, 9)
    ctx.mark_scaffold(P)

    if not add_pocket(P, rng, cfg, ctx):
        raise RuntimeError("B: pocket1 fail")
    if not add_pocket(P, rng, cfg, ctx):
        raise RuntimeError("B: pocket2 fail")

    target = _randi(rng, 2, 6)
    added, tries = 0, 0
    while added < target and tries < cfg["max_retries"]:
        tries += 1
        p = (_randi(rng, 20, 80), _randi(rng, 20, 80))
        if try_insert_anywhere_and_mark(P, p, cfg, ctx):
            added += 1

    return deque(P)


# ---------------- C: 内部走廊 + 稀疏外扩 ----------------
def generate_C(cfg: Dict, ctx: GenContext) -> deque:
    rng = random.Random(cfg["seed"] + 303)

    P = _scaffold_simple_polygon(rng, cfg, 5, 7)
    ctx.mark_scaffold(P)

    # 内部走廊：沿切向微移后按质心方向内偏
    for _ in range(_randi(rng, 2, 3)):
        idx = _randi(rng, 0, len(P) - 1)
        a, b = P[idx], P[(idx + 1) % len(P)]
        gx, gy = sum(x for x, _ in P) / len(P), sum(y for _, y in P) / len(P)
        mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        vx, vy = unit_int(int(gx - mx), int(gy - my))
        if vx == 0 and vy == 0:
            continue
        inset = _randi(rng, 2, 6)
        kcorr = _randi(rng, 2, 5)
        tries = 0
        added = 0
        while added < kcorr and tries < cfg["max_retries"]:
            tries += 1
            t = _randi(rng, -3, 3)
            step = max(1, abs(b[0] - a[0]) + abs(b[1] - a[1]))
            px = mx + t * (b[0] - a[0]) // step
            py = my + t * (b[1] - a[1]) // step
            q = (int(px + inset * vx), int(py + inset * vy))
            if try_insert_anywhere_and_mark(P, q, cfg, ctx):
                added += 1

    # 稀疏外扩：沿法线外推 6–14
    for _ in range(_randi(rng, 2, 3)):
        i = _randi(rng, 0, len(P) - 1)
        a, b = P[i], P[(i + 1) % len(P)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        fx, fy = unit_int(dy, -dx) if rng.random() < 0.5 else unit_int(-dy, dx)
        if fx == 0 and fy == 0:
            continue
        r = _randi(rng, 6, 14)
        q = (a[0] + fx * r, a[1] + fy * r)
        try_insert_anywhere_and_mark(P, q, cfg, ctx)

    return deque(P)


# ---------------- 统一入口 ----------------
def generate_with_marks(which: str, params_path: str = "params.json") -> Tuple[deque, Set[Point]]:
    """
    扩展：返回 (points_deque, inserted_points_set)。
    inserted_points_set 为空心黑点，其余为实心黑点。
    """
    cfg = load_params(params_path)
    ctx = GenContext()
    w = which.upper()
    if w == "A":
        pts = generate_A(cfg, ctx)
    elif w == "B":
        pts = generate_B(cfg, ctx)
    elif w == "C":
        pts = generate_C(cfg, ctx)
    else:
        raise ValueError("which must be 'A'|'B'|'C'")
    return pts, ctx.inserted_points

# ---------------- 仅 simple 约束的生成器（复杂度测试） ----------------
def _scaffold_simple_only(rng: random.Random, n: int, max_retries: int) -> List[Point]:
    """
    仅保证 simple：随机散点 → 以质心极角排序 → 检查 simple，失败重试。
    不设面积阈值，不依赖 params.json 的 epsilon_area_min。
    """
    for _ in range(max_retries):
        # 取整数格点，避免重复
        cx, cy = rng.randint(30, 70), rng.randint(30, 70)
        pts = set()
        # 在 20..80 的 61×61 网格中采样，足以容纳 n=1000
        while len(pts) < n:
            pts.add((rng.randint(20, 80), rng.randint(20, 80)))
        P = list(pts)

        # 以质心为参考做极角排序
        gx = sum(x for x, _ in P) / n
        gy = sum(y for _, y in P) / n
        P.sort(key=lambda p: math.atan2(p[1] - gy, p[0] - gx))

        if is_simple(P):
            return P
    raise RuntimeError(f"simple-only scaffold fail: retries={max_retries}, n={n}")


def generate_simple_only(n: int, params_path: str = "params.json") -> deque:
    """
    复杂度测试用：生成 n 点 simple polygon（仅 simple 约束）。
    返回 deque(points)，不区分脚手架/插入点。
    """
    cfg = load_params(params_path)
    # 与现有 A/B/C 的种子相互独立，避免干扰
    seed_offset = {10: 10001, 100: 10002, 1000: 10003}.get(n, 19997)
    rng = random.Random(cfg["seed"] + seed_offset)
    P = _scaffold_simple_only(rng, n, max_retries=cfg["max_retries"])
    # 方向不强制，若需要 CCW 可按需翻转：
    # if poly_area_signed(P) < 0: P.reverse()
    return deque(P)


def generate_complexity_cases(params_path: str = "params.json"):
    """
    一键得到三组规模：10、100、1000。
    返回 dict: {10: deque, 100: deque, 1000: deque}
    """
    return {
        10: generate_simple_only(10, params_path),
        100: generate_simple_only(100, params_path),
        1000: generate_simple_only(1000, params_path),
    }