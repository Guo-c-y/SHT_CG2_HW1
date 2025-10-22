# viz_suite.py
from __future__ import annotations
from typing import List, Tuple, Sequence, Optional
import math, itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

Point  = Tuple[float, float]
WPoint = Tuple[int, int, int]
HalfPlane = Tuple[float, float, float]

_MIN_EXPAND = 1.2
_MAX_EXPAND = 4.0
_TITLE_Y = 1.08
_INFO_Y  = 1.035

# =========================
# ---- Geometry helpers ----
# =========================
def _area2(poly: Sequence[Point]) -> float:
    s = 0.0
    for i in range(len(poly)):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%len(poly)]
        s += x1*y2 - y1*x2
    return s

def _ensure_ccw(poly: List[Point]) -> List[Point]:
    return list(reversed(poly)) if len(poly)>=3 and _area2(poly)<0 else poly

def _clip_polygon_with_halfplane(poly: List[Point], a: float, b: float, c: float, eps: float=1e-9) -> List[Point]:
    if not poly: return []
    def inside(P): return a*P[0]+b*P[1] <= c+eps
    def intersect(P,Q):
        u = a*P[0]+b*P[1]-c; v = a*Q[0]+b*Q[1]-c
        den = u - v
        if abs(den) < 1e-15: return Q
        t = u/den
        return (P[0]+t*(Q[0]-P[0]), P[1]+t*(Q[1]-P[1]))
    out: List[Point] = []
    n = len(poly)
    for i in range(n):
        S = poly[i]; E = poly[(i+1)%n]
        Si, Ei = inside(S), inside(E)
        if   Si and Ei: out.append(E)
        elif Si and not Ei: out.append(intersect(S,E))
        elif (not Si) and Ei: out.append(intersect(S,E)); out.append(E)
    cleaned: List[Point] = []
    for p in out:
        if not cleaned or (abs(cleaned[-1][0]-p[0])>1e-12 or abs(cleaned[-1][1]-p[1])>1e-12):
            cleaned.append(p)
    if len(cleaned)>=2 and abs(cleaned[0][0]-cleaned[-1][0])<1e-12 and abs(cleaned[0][1]-cleaned[-1][1])<1e-12:
        cleaned.pop()
    return cleaned

def _power_vertex_from_triplet(si: WPoint, sj: WPoint, sk: WPoint) -> Optional[Point]:
    xi,yi,wi = si; xj,yj,wj = sj; xk,yk,wk = sk
    a1 = 2.0*(xj - xi); b1 = 2.0*(yj - yi); c1 = (xj*xj+yj*yj)-(xi*xi+yi*yi)+(wi-wj)
    a2 = 2.0*(xk - xi); b2 = 2.0*(yk - yi); c2 = (xk*xk+yk*yk)-(xi*xi+yi*yi)+(wi-wk)
    det = a1*b2 - a2*b1
    if abs(det)<1e-12: return None
    x = (c1*b2 - c2*b1)/det; y = (a1*c2 - a2*c1)/det
    return (x,y)

def _power_value(p: Point, s: WPoint) -> float:
    x,y = p; xi,yi,wi = s
    return (x-xi)*(x-xi) + (y-yi)*(y-yi) - wi

def _finite_power_vertices(sites: Sequence[WPoint], eps: float=1e-9) -> List[Point]:
    verts: List[Point] = []
    n = len(sites)
    for i,j,k in itertools.combinations(range(n),3):
        p = _power_vertex_from_triplet(sites[i],sites[j],sites[k])
        if p is None or not math.isfinite(p[0]) or not math.isfinite(p[1]): continue
        pv = _power_value(p, sites[i])
        if abs(pv-_power_value(p,sites[j]))>1e-6 or abs(pv-_power_value(p,sites[k]))>1e-6: continue
        ok = True
        for t in range(n):
            if t in (i,j,k): continue
            if pv - _power_value(p, sites[t]) > eps: ok=False; break
        if ok: verts.append(p)
    return verts

def _bbox_points(pts: Sequence[Point]):
    if not pts: return None
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return (min(xs),min(ys),max(xs),max(ys))

def _expand_to_square(box, margin_ratio: float):
    xmin,ymin,xmax,ymax = box
    cx=0.5*(xmin+xmax); cy=0.5*(ymin+ymax)
    span=max(xmax-xmin, ymax-ymin) or 1.0
    span *= (1.0+margin_ratio)
    return (cx-span/2, cy-span/2, cx+span/2, cy+span/2)

def _sites_square(sites: Sequence[WPoint], margin_ratio: float=0.20):
    xs=[float(x) for x,_,_ in sites]; ys=[float(y) for _,y,_ in sites]
    return _expand_to_square((min(xs),min(ys),max(xs),max(ys)), margin_ratio)

def _weiszfeld(points: List[Point], iters: int=50, eps: float=1e-8) -> Point:
    x = sum(px for px,_ in points)/len(points)
    y = sum(py for _,py in points)/len(points)
    for _ in range(iters):
        numx=numy=den=0.0; moved=False
        for px,py in points:
            dx=x-px; dy=y-py; d=math.hypot(dx,dy)
            if d<eps: x,y=px,py; moved=True; break
            w=1.0/d; numx+=w*px; numy+=w*py; den+=w
        if moved: break
        nx,ny=numx/den,numy/den
        if math.hypot(nx-x,ny-y)<eps: x,y=nx,ny; break
        x,y=nx,ny
    return (x,y)

def _robust_center(sites: Sequence[WPoint]) -> Point:
    pts=[(float(x),float(y)) for (x,y,_) in sites]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    centroid=(sum(xs)/len(xs), sum(ys)/len(ys))
    sxs=sorted(xs); sys=sorted(ys); m=len(xs)//2
    median=(sxs[m] if len(xs)%2 else 0.5*(sxs[m-1]+sxs[m]),
            sys[m] if len(ys)%2 else 0.5*(sys[m-1]+sys[m]))
    gmed=_weiszfeld(pts)
    w0=10.0; ws=[max(abs(w),w0) for *_,w in sites]; wsum=sum(ws)
    wcent=(sum(ws[i]*xs[i] for i in range(len(xs)))/wsum,
           sum(ws[i]*ys[i] for i in range(len(ys)))/wsum)
    return (0.25*(centroid[0]+median[0]+gmed[0]+wcent[0]),
            0.25*(centroid[1]+median[1]+gmed[1]+wcent[1]))

def _auto_view_from_cells_clamped(sites: Sequence[WPoint], max_expand: float=_MAX_EXPAND):
    bxmin,bymin,bxmax,bymax = _sites_square(sites, 0.20)
    bspan = max(bxmax-bxmin, bymax-bymin)
    verts = _finite_power_vertices(sites)
    pts = verts + [(float(x),float(y)) for (x,y,_) in sites]
    box = _bbox_points(pts) or (bxmin,bymin,bxmax,bymax)
    vxmin,vymin,vxmax,vymax = _expand_to_square(box, 0.20)
    vspan = max(vxmax-vxmin, vymax-vymin)
    span = min(max(vspan, _MIN_EXPAND*bspan), max_expand*bspan)
    cx,cy = _robust_center(sites)
    xmin, ymin = cx - span/2, cy - span/2
    xmax, ymax = cx + span/2, cy + span/2
    return (xmin,ymin,xmax,ymax)

def _view_polygon(view):
    xmin,ymin,xmax,ymax = view
    return [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]

# =========================
# ---- Drawing helpers ----
# =========================
def _apply_axes_with_ticks(ax, xmin, ymin, xmax, ymax):
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    for side in ("top","right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=8)

def _apply_axes_no_ticks_no_border(ax, xmin, ymin, xmax, ymax):
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top","right","left","bottom"):
        ax.spines[s].set_visible(False)
    ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                           fill=False, edgecolor="white", linewidth=8,
                           zorder=10, clip_on=False))

def _draw_weight_circles(ax, sites: Sequence[WPoint], span: float):
    D = span; r_base = 0.05*D; w0 = 10.0
    pos_style = (0,(1,2))
    neg_style = (0,(6,2,1,2))
    for (x,y,w) in sites:
        mag = max(abs(w), w0); r = r_base*math.sqrt(mag/w0)
        face = (0.7,0.7,0.7,0.15) if w>=0 else (1,1,1,0.15)
        ls   = pos_style if w>=0 else neg_style
        ax.add_patch(Circle((x,y), r, facecolor=face, edgecolor="black",
                            linewidth=0.5, linestyle=ls, zorder=1))

def _clip_all_cells(power_regions: Sequence[Sequence[HalfPlane]],
                    view_poly: List[Point]) -> List[List[Point]]:
    cells_polys: List[List[Point]] = []
    for hi in power_regions:
        poly = list(view_poly)
        for (a,b,c) in hi:
            poly = _clip_polygon_with_halfplane(poly,a,b,c,eps=1e-9)
            if not poly: break
        if poly: poly = _ensure_ccw(poly)
        cells_polys.append(poly)
    return cells_polys

# ==========================================
# ---- Public: grid renderers ----
# ==========================================
def render_sites(sites_list: Sequence[Sequence[WPoint]],
                 coord_range: int = 1000,
                 cols_per_row: int = 3) -> plt.Figure:
    m = len(sites_list)
    if m == 0: raise ValueError("empty sites_list")
    rows = (m + cols_per_row - 1)//cols_per_row
    fig_w = 4 * cols_per_row
    fig_h = 4 * rows
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(fig_w, fig_h), dpi=160)
    if rows == 1 and cols_per_row == 1: axes = [[axes]]
    elif rows == 1: axes = [axes]
    xmin=ymin=0.0; xmax=ymax=float(coord_range)
    span = xmax - xmin

    for idx in range(rows*cols_per_row):
        r, c = divmod(idx, cols_per_row)
        ax = axes[r][c]
        if idx >= m: ax.axis("off"); continue
        sites = sites_list[idx]
        _apply_axes_with_ticks(ax, xmin, ymin, xmax, ymax)
        _draw_weight_circles(ax, sites, span)
        ax.scatter([x for x,_,_ in sites], [y for _,y,_ in sites], s=10, c="black", zorder=3)

        wvals = [w for *_, w in sites]
        info = f"n = {len(sites)}    w ∈ [{min(wvals)}, {max(wvals)}]"
        ax.text(0.0, _TITLE_Y, f"sites case {idx+1}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=13)
        ax.text(0.5, _INFO_Y, info, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    return fig

def render_power_diagrams(sites_list: Sequence[Sequence[WPoint]],
                          power_regions_list: Sequence[Sequence[Sequence[HalfPlane]]],
                          cols_per_row: int = 3,
                          max_expand: float = 3.0) -> plt.Figure:
    if len(sites_list) != len(power_regions_list):
        raise ValueError("sites_list and power_regions_list length mismatch")
    m = len(sites_list)
    if m == 0: raise ValueError("empty inputs")
    rows = (m + cols_per_row - 1)//cols_per_row
    fig_w = 4 * cols_per_row
    fig_h = 4 * rows
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(fig_w, fig_h), dpi=160)
    if rows == 1 and cols_per_row == 1: axes = [[axes]]
    elif rows == 1: axes = [axes]

    for idx in range(rows*cols_per_row):
        r, c = divmod(idx, cols_per_row)
        ax = axes[r][c]
        if idx >= m: ax.axis("off"); continue

        sites = sites_list[idx]
        regions = power_regions_list[idx]
        xmin,ymin,xmax,ymax = _auto_view_from_cells_clamped(sites, max_expand=max_expand)
        Vpoly = _view_polygon((xmin,ymin,xmax,ymax))
        span = max(xmax-xmin, ymax-ymin)

        _apply_axes_no_ticks_no_border(ax, xmin, ymin, xmax, ymax)
        _draw_weight_circles(ax, sites, span)
        cells_polys = _clip_all_cells(regions, Vpoly)
        n_cells = sum(1 for P in cells_polys if len(P)>=3)

        for poly in cells_polys:
            if len(poly)>=2:
                xs=[p[0] for p in poly]+[poly[0][0]]
                ys=[p[1] for p in poly]+[poly[0][1]]
                ax.plot(xs,ys,linestyle='-',color='black',linewidth=1.5,zorder=3)
        ax.scatter([x for x,_,_ in sites],[y for _,y,_ in sites], s=10, c="black", zorder=4)

        ax.text(0.0, _TITLE_Y, f"power diagram {idx+1}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=13)
        info = f"cells = {n_cells}    origin=({int(xmin)},{int(ymin)})    edge={int(round(span))}"
        ax.text(0.5, _INFO_Y, info, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    return fig