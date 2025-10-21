import time, numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# --------- timing ----------
def _median_time(fn, arg, runs=5):
    ts=[]
    for _ in range(runs):
        t0=time.perf_counter(); fn(arg); ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[len(ts)//2] if ts else float("nan")

# --------- style ----------
def _apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.titlesize": 13,
        "axes.titlelocation": "left",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": "#aaaaaa",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": 160,
    })

# --------- helpers ----------
def _get(d, k, default=0):
    v = d.get(k, default)
    try: return int(v)
    except Exception:
        try: return float(v)
        except Exception: return default

def _normalize_stats(s):
    s = dict(s)
    o = {}
    for k in ["n","hull_len","total_ops","comparisons","inside_tests",
              "cross","orient","is_left","is_right_or_on",
              "while_top_iters","while_bot_iters"]:
        o[k] = _get(s,k)
    ar=_get(s,"deq_append") or _get(s,"append")
    al=_get(s,"deq_appendleft") or _get(s,"appendleft")
    pr=_get(s,"deq_pop") or _get(s,"pop")
    pl=_get(s,"deq_popleft") or _get(s,"popleft")
    o.update({
        "append_right":ar,"append_left":al,
        "pop_right":pr,"pop_left":pl,
        "append_total":ar+al,"pop_total":pr+pl
    })
    return o

def _call_hull_for_time(fn,P):
    for args in [(False,False),(False,None),(None,False),(None,None)]:
        try:
            return fn(P,stats=args[0],verbose=args[1])
        except TypeError: pass
    return fn(P)

def _call_hull_for_stats(fn,P):
    for args in [(True,False),(True,None)]:
        try:
            return fn(P,stats=args[0],verbose=args[1])
        except TypeError: pass
    hull=_call_hull_for_time(fn,P)
    return {"n":len(P),"hull_len":len(hull) if hasattr(hull,"__len__") else 0,"total_ops":0}

def _fmt_float(x,prec=2):
    if not np.isfinite(x): return "NA"
    if abs(x)>=100: return f"{x:.0f}"
    if abs(x)>=10:  return f"{x:.1f}"
    return f"{x:.{prec}f}"

def _fmt_signed(x):
    if not np.isfinite(x): return " NA"
    sign="+" if x>=0 else "-"
    return f" {sign}{_fmt_float(abs(x))}"

# --------- main ----------
def analyze_melkman_complexity(TestPolys,hull_fn,runs=5,show=True,save_path=None,style="paper"):
    if style=="paper": _apply_paper_style()
    ns=[len(P) for P in TestPolys]
    times=[_median_time(lambda X:_call_hull_for_time(hull_fn,X),P,runs) for P in TestPolys]
    stats=[_normalize_stats(_call_hull_for_stats(hull_fn,P) or {}) for P in TestPolys]
    ops=[int(s["total_ops"]) for s in stats]
    ns_a,t_a,o_a=np.array(ns,float),np.array(times,float),np.array(ops,float)

    mask=(ns_a>0)&(t_a>0)
    if mask.sum()>=2:
        x,y=np.log(ns_a[mask]),np.log(t_a[mask])
        k,A=np.polyfit(x,y,1); C=np.exp(A)
        fit_t=C*ns_a**k
        ss_res=((t_a[mask]-C*ns_a[mask]**k)**2).sum()
        ss_tot=((t_a[mask]-t_a[mask].mean())**2).sum()
        r2_t=1-ss_res/ss_tot if ss_tot>0 else 1
    else: k,C,r2_t=np.nan,np.nan,np.nan; fit_t=np.full_like(ns_a,np.nan)

    if len(ns_a)>=2:
        a,b=np.polyfit(ns_a,o_a,1); fit_o=a*ns_a+b
        ss_res=((o_a-fit_o)**2).sum(); ss_tot=((o_a-o_a.mean())**2).sum()
        r2_o=1-ss_res/ss_tot if ss_tot>0 else 1
    else: a,b,r2_o=np.nan,np.nan,np.nan; fit_o=np.full_like(ns_a,np.nan)

    fig=None
    if show:
        fig,axs=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)

        # (a)
        ax=axs[0]
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.scatter(ns,times,color="black",label="median")
        if np.isfinite(fit_t).any():
            ax.plot(ns,fit_t,color="black",
                label=f"fit: { _fmt_float(C) }·n^{ _fmt_float(k) }  (R²={ _fmt_float(r2_t) })")
        ax.set_xlabel("n"); ax.set_ylabel("time (s)")
        ax.set_title("(a) Time vs n")
        ax.grid(True,which="both",alpha=0.6)
        ax.xaxis.set_major_locator(FixedLocator(ns))
        ax.set_xticklabels([str(n) for n in ns])
        ax.legend(loc="lower right",fontsize=10)

        # (b)
        ax=axs[1]
        ax.plot(ns,ops,"s",color="black",label="ops")
        if np.isfinite(fit_o).any():
            ax.plot(ns,fit_o,"-",color="black",
                label=f"fit: { _fmt_float(a) }·n{ _fmt_signed(b) }  (R²={ _fmt_float(r2_o) })")
        ax.set_xlabel("n"); ax.set_ylabel("total ops")
        ax.set_title("(b) Operation Count")
        ax.grid(True,alpha=0.6)
        ax.legend(loc="upper left",fontsize=10)

        # (c)
        ax=axs[2]
        idx=int(np.argmax(ns_a)); last=stats[idx]
        parts=[
            ("orient",["orient"]),("cross",["cross"]),
            ("is_left",["is_left"]),("is_right_or_on",["is_right_or_on"]),
            ("comparisons",["comparisons"]),("inside_tests",["inside_tests"]),
            ("append_total",["append_total"]),("pop_total",["pop_total"]),
            ("append_right",["append_right"]),("append_left",["append_left"]),
            ("pop_right",["pop_right"]),("pop_left",["pop_left"]),
            ("while_top",["while_top_iters"]),("while_bot",["while_bot_iters"])
        ]
        labels=[p[0] for p in parts]
        vals=[sum(last.get(k,0) for k in p[1]) for p in parts]
        gray=["#000000","#222222","#444444","#666666","#888888","#aaaaaa"]*3
        ax.bar(labels,vals,color=gray[:len(labels)],edgecolor="black",lw=0.6)
        ax.set_ylabel("count"); ax.set_title(f"(c) Breakdown @ n={last.get('n','NA')}")
        for t in ax.get_xticklabels():
            t.set_rotation(45); t.set_ha("right")
        ax.grid(axis="y",alpha=0.6)
        if save_path: fig.savefig(save_path,bbox_inches="tight")

    return {"ns":ns,"times":times,"ops":ops,
            "k_time":float(k),"C_time":float(C),"a_ops":float(a),"b_ops":float(b),
            "r2_time":float(r2_t),"r2_ops":float(r2_o),
            "breakdown_last":stats[int(np.argmax(ns_a))],"figure":fig}