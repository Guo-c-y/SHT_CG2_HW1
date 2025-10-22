from typing import List, Tuple, Sequence, Optional
import math, itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

Point  = Tuple[float, float]
WPoint = Tuple[int, int, int]
HalfPlane = Tuple[float, float, float]

# ---- parameters for auto view / styling ----
_MIN_EXPAND = 1.2
_MAX_EXPAND = 4.0
_GLOBAL_MIN_CORNER = (-1000.0, -1000.0)

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

def _coord_median(vals: List[float]) -> float:
    s = sorted(vals); m = len(s)//2
    return s[m] if len(s)%2==1 else 0.5*(s[m-1]+s[m])

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
    median=(_coord_median(xs), _coord_median(ys))
    gmed=_weiszfeld(pts)
    w0=10.0; ws=[max(abs(w),w0) for *_,w in sites]; wsum=sum(ws)
    wcent=(sum(ws[i]*xs[i] for i in range(len(xs)))/wsum,
           sum(ws[i]*ys[i] for i in range(len(ys)))/wsum)
    return (0.25*(centroid[0]+median[0]+gmed[0]+wcent[0]),
            0.25*(centroid[1]+median[1]+gmed[1]+wcent[1]))

def _auto_view_from_cells_clamped(sites: Sequence[WPoint]):
    bxmin,bymin,bxmax,bymax = _sites_square(sites, 0.20)
    bspan = max(bxmax-bxmin, bymax-bymin)
    verts = _finite_power_vertices(sites)
    pts = verts + [(float(x),float(y)) for (x,y,_) in sites]
    box = _bbox_points(pts) or (bxmin,bymin,bxmax,bymax)
    vxmin,vymin,vxmax,vymax = _expand_to_square(box, 0.20)
    vspan = max(vxmax-vxmin, vymax-vymin)
    span = min(max(vspan, _MIN_EXPAND*bspan), _MAX_EXPAND*bspan)
    cx,cy = _robust_center(sites)
    xmin, ymin = cx - span/2, cy - span/2
    xmax, ymax = cx + span/2, cy + span/2
    gminx,gminy = _GLOBAL_MIN_CORNER
    if xmin < gminx: shift=gminx-xmin; xmin+=shift; xmax+=shift
    if ymin < gminy: shift=gminy-ymin; ymin+=shift; ymax+=shift
    return (xmin,ymin,xmax,ymax)

def _view_polygon(view):
    xmin,ymin,xmax,ymax = view
    return [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]

def render_power_diagram(sites_in: Sequence[WPoint],
                         power_regions: Sequence[Sequence[HalfPlane]]) -> plt.Figure:
    if len(sites_in) != len(power_regions):
        raise ValueError("sites and power_regions length mismatch")

    view = _auto_view_from_cells_clamped(sites_in)
    xmin,ymin,xmax,ymax = view
    span = max(xmax-xmin, ymax-ymin)
    Vpoly0 = _view_polygon(view)

    # clip per-site polygon
    cells_polys: List[List[Point]] = []
    for hi in power_regions:
        poly = Vpoly0[:]
        for (a,b,c) in hi:
            poly = _clip_polygon_with_halfplane(poly,a,b,c,eps=1e-9)
            if not poly: break
        if poly: poly = _ensure_ccw(poly)
        cells_polys.append(poly)

    # draw
    fig = plt.figure(figsize=(5,5), dpi=160, facecolor="white")
    ax  = fig.add_subplot(111)
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top","right","left","bottom"): ax.spines[s].set_visible(False)

    # weight circles (below edges)
    D = span; r_base = 0.05*D; w0 = 10.0
    pos_style = (0,(1,2))       # positive: dotted
    neg_style = (0,(6,2,1,2))   # negative: dash-dot-dot
    for (x,y,w) in sites_in:
        mag = max(abs(w), w0); r = r_base*math.sqrt(mag/w0)
        face = (0.7,0.7,0.7,0.15) if w>=0 else (1,1,1,0.15)
        ls   = pos_style if w>=0 else neg_style
        ax.add_patch(Circle((x,y), r, facecolor=face, edgecolor="black",
                            linewidth=0.5, linestyle=ls, zorder=1))

    # cell boundaries
    for poly in cells_polys:
        if len(poly)>=2:
            xs=[p[0] for p in poly]+[poly[0][0]]
            ys=[p[1] for p in poly]+[poly[0][1]]
            ax.plot(xs,ys,linestyle='-',color='black',linewidth=1.5,zorder=3)

    # sites on top-most
    ax.scatter([x for x,_,_ in sites_in],[y for _,y,_ in sites_in], s=10, c="black", zorder=4)

    # white overlay to hide any residual axis border
    ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                           fill=False, edgecolor="white", linewidth=8,
                           zorder=10, clip_on=False))

    # compact label under the square
    label = f"({int(xmin)},{int(ymin)}) -> ({int(xmax)},{int(ymax)})  edge={int(round(span))}"
    ax.text(xmin, ymin - 0.015*span, label, ha="left", va="top",
            fontsize=8, color="black", clip_on=False)
    return fig