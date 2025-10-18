# code/viz_suite.py
# 输入：
#  - render_triptych_from_polys(polys): 三份 simple polygon，可为 points 或 (points, inserted_set) —— 保持不变
#  - render_triptych_hulls(pairs): 三个 (polygon, hull) 对，专注凸包可视化（黑白灰论文风）
# 输出：论文风格可视化（每隔 label_every 点标号；白描边；三图等大；副标题在标题下方显示 n 和 orientation）
# 规则：
#  - simple polygon：细灰线 + 小灰点，不标号
#  - convex hull：浅灰填充 + 粗黑边 + 白心黑边顶点，给 hull 顶点编号
#  - 三图标题分别为：convex hull A / B / C

from __future__ import annotations
from typing import List, Tuple, Sequence, Iterable, Set, Union
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
import json
from utils import poly_area_signed, as_points

Point = Tuple[int, int]
PolyLike = Sequence[Point]
Item = Union[PolyLike, Tuple[PolyLike, Iterable[Point]]]

# ---------- 样式 ----------
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

def _read_label_every(default: int = 1) -> int:
    """Read label frequency from params.json; fallback to default."""
    try:
        cfg_path = Path(__file__).with_name("params.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = int(data.get("label_every", default))
        return max(1, v)
    except Exception:
        return default

# ---------- 工具 ----------
def _summary(points: List[Point]):
    area = poly_area_signed(points)
    return {"n": len(points), "direction": "CCW" if area > 0 else "CW", "ccw": area > 0}

def _label_vertices(ax, points: List[Point], base_px: int = 4, step: int = 1):
    n = len(points)
    idxs = list(range(0, n, max(1, step)))
    if (n-1) not in idxs:
        idxs.append(n-1)
    peff = [pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()]
    for i in idxs:
        x, y = points[i]
        ax.annotate(
            str(i), xy=(x, y), xytext=(base_px, base_px),
            textcoords="offset points",
            fontsize=6, alpha=0.95, color="black",
            ha="left", va="bottom", path_effects=peff, clip_on=True, zorder=9
        )

def _draw_polygon_bg(ax, points_seq: Sequence[Point]):
    """以次要灰阶风格画被包裹的 simple polygon（不编号）"""
    P = as_points(points_seq)
    xs, ys = zip(*P)
    # 细灰线闭合边
    ax.plot(list(xs) + [P[0][0]], list(ys) + [P[0][1]],
            linewidth=0.9, color="#777777", zorder=2)
    # 小灰点
    ax.scatter(xs, ys, s=20, color="#777777", zorder=3)
    return P

def _draw_hull_main(ax, hull_seq: Sequence[Point]):
    """以主体黑白风格画凸包，并返回点列"""
    H = as_points(hull_seq)
    if len(H) == 0:
        return H
    hx, hy = zip(*H)
    # 浅灰填充（闭合）
    ax.fill(list(hx) + [H[0][0]], list(hy) + [H[0][1]],
            alpha=0.12, linewidth=0, color="#777777", zorder=1)
    # 粗黑边（闭合）
    ax.plot(list(hx) + [H[0][0]], list(hy) + [H[0][1]],
            linewidth=2.0, color="black", zorder=8)
    # 顶点：白心黑边
    ax.scatter(hx, hy, s=56, facecolors='white',
               edgecolors='black', linewidths=1.6, zorder=9)
    return H

def _bbox_union(list_of_pointlists: List[List[Point]], pad: float = 5.0):
    all_x = [x for P in list_of_pointlists for (x, _) in P] or [0.0]
    all_y = [y for P in list_of_pointlists for (_, y) in P] or [0.0]
    xmin, xmax = min(all_x)-pad, max(all_x)+pad
    ymin, ymax = min(all_y)-pad, max(all_y)+pad
    return xmin, xmax, ymin, ymax

def _subtitle(ax, P: List[Point], title: str):
    meta = _summary(P)
    ax.set_title(title, fontsize=13, pad=15)
    ax.text(0.5, 1.0, f"n = {meta['n']}   orientation = {meta['direction']}",
            transform=ax.transAxes, fontsize=9, color="black",
            ha="center", va="bottom")

def _finish_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

# ---------- 保持不变：三幅 simple polygon ----------
def render_triptych_from_polys(polys: Sequence[Item],
                               titles: Sequence[str] = ("simple polygon A","simple polygon B","simple polygon C"),
                               save_path: str | None = None):
    """
    polys: 长度为 3 的序列。
      - 兼容旧用法：每项为 list/tuple/deque of (x,y)
      - 新用法：每项为 (points, inserted_set) 二元组，inserted_set 中的点将绘制为黑边白心点
    """
    if len(polys) != 3:
        raise ValueError("polys must contain exactly 3 items")
    _apply_paper_style()

    # 拆解 points 与 inserted
    pts_list: List[List[Point]] = []
    ins_list: List[Set[Point]] = []
    for item in polys:
        if isinstance(item, tuple) and len(item) == 2:
            points, inserted_iter = item
            P = as_points(points)
            pts_list.append(P)
            ins_list.append(set(tuple(p) for p in inserted_iter))
        else:
            P = as_points(item)  # type: ignore[arg-type]
            pts_list.append(P)
            ins_list.append(set())

    # 统一坐标范围
    xmin, xmax, ymin, ymax = _bbox_union(pts_list, pad=5)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True, constrained_layout=True)
    step_cfg = _read_label_every(default=1)
    for ax, P, ins, title in zip(axes, pts_list, ins_list, titles):
        # 线条
        xs, ys = zip(*P)
        ax.plot(list(xs) + [P[0][0]], list(ys) + [P[0][1]],
                linewidth=1.2, color="black", zorder=3)
        # 点
        if ins:
            Pset = set(P)
            ins_set = set(tuple(p) for p in ins) & Pset
            base_pts = [p for p in P if p not in ins_set]
            if base_pts:
                bx, by = zip(*base_pts)
                ax.scatter(bx, by, s=36, color="black", zorder=4)
            if ins_set:
                sx, sy = zip(*ins_set)
                ax.scatter(sx, sy, s=64, facecolors='white',
                           edgecolors='black', linewidths=1.3, zorder=5)
        else:
            ax.scatter(xs, ys, s=36, color="black", zorder=4)

        _subtitle(ax, P, title)
        # 顶点编号（simple polygon 模式仍按原逻辑）
        _label_vertices(ax, P, base_px=4, step=step_cfg)

        _finish_axes(ax, (xmin, xmax), (ymin, ymax))
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()

# ---------- 新增：三图 simple polygon + convex hull（仅凸包编号，黑白灰） ----------
def render_triptych_hulls(pairs: Sequence[Tuple[PolyLike, Sequence[Point]]],
                          titles: Sequence[str] = ("convex hull A","convex hull B","convex hull C"),
                          save_path: str | None = None):
    """
    pairs: 长度为 3 的序列，每项为 (polygon, hull)
      - polygon: 原始 simple polygon（仅作为背景辅助，不编号）
      - hull: 按顺序给出的凸包顶点（将编号）
    """
    if len(pairs) != 3:
        raise ValueError("pairs must contain exactly 3 (polygon, hull) items")
    _apply_paper_style()

    P_list: List[List[Point]] = []
    H_list: List[List[Point]] = []
    for poly, hull in pairs:
        P_list.append(as_points(poly))
        H_list.append(as_points(hull))

    xmin, xmax, ymin, ymax = _bbox_union(P_list + H_list, pad=5)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True, constrained_layout=True)

    step_cfg = _read_label_every(default=1)

    for ax, P, H, title in zip(axes, P_list, H_list, titles):
        _draw_polygon_bg(ax, P)           # 背景 simple polygon（细灰，不编号）
        H_drawn = _draw_hull_main(ax, H)  # 主体凸包（黑白风格）

        # 标题与副标题改为基于 hull 的统计；若 hull 为空则退回 polygon
        base_for_meta = H_drawn if H_drawn else P
        _subtitle(ax, base_for_meta, title)

        # 仅凸包顶点编号
        if H_drawn:
            _label_vertices(ax, H_drawn, base_px=4, step=step_cfg)

        _finish_axes(ax, (xmin, xmax), (ymin, ymax))

    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()