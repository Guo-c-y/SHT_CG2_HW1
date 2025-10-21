import time
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


# --------- timing ----------
def _median_time(fn: Callable[[Any], Any], arg: Any, runs: int = 5) -> float:
    """Return median runtime over `runs` executions."""
    ts: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(arg)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] if ts else float("nan")


# --------- style ----------
def _apply_paper_style() -> None:
    """Apply a clean plotting style."""
    plt.rcParams.update(
        {
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
        }
    )


# --------- helpers ----------
def _get(d: Dict[str, Any], k: str, default: float = 0) -> float:
    """Fetch numeric value with fallback."""
    v = d.get(k, default)
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception:
        try:
            return float(v)  # type: ignore[arg-type]
        except Exception:
            return default


def _normalize_stats(s: Dict[str, Any]) -> Dict[str, float]:
    """Normalize stats keys from different hull implementations."""
    s = dict(s)
    o: Dict[str, float] = {}
    for k in [
        "n",
        "hull_len",
        "total_ops",
        "comparisons",
        "inside_tests",
        "cross",
        "orient",
        "is_left",
        "is_right_or_on",
        "while_top_iters",
        "while_bot_iters",
    ]:
        o[k] = _get(s, k)

    ar = _get(s, "deq_append") or _get(s, "append")
    al = _get(s, "deq_appendleft") or _get(s, "appendleft")
    pr = _get(s, "deq_pop") or _get(s, "pop")
    pl = _get(s, "deq_popleft") or _get(s, "popleft")

    o.update(
        {
            "append_right": ar,
            "append_left": al,
            "pop_right": pr,
            "pop_left": pl,
            "append_total": ar + al,
            "pop_total": pr + pl,
        }
    )
    return o


def _call_hull_for_time(fn: Callable[..., Any], P: Any) -> Any:
    """Call hull function in a way compatible with different signatures."""
    for args in [(False, False), (False, None), (None, False), (None, None)]:
        try:
            return fn(P, stats=args[0], verbose=args[1])
        except TypeError:
            pass
    return fn(P)


def _call_hull_for_stats(fn: Callable[..., Any], P: Any) -> Dict[str, Any]:
    """Call hull function expecting a stats dict; fallback to minimal stats."""
    for args in [(True, False), (True, None)]:
        try:
            return fn(P, stats=args[0], verbose=args[1])
        except TypeError:
            pass
    hull = _call_hull_for_time(fn, P)
    n = len(P) if hasattr(P, "__len__") else 0
    h = len(hull) if hasattr(hull, "__len__") else 0
    return {"n": n, "hull_len": h, "total_ops": 0}


def _fmt_float(x: float, prec: int = 2) -> str:
    if not np.isfinite(x):
        return "NA"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.{prec}f}"


def _fmt_signed(x: float) -> str:
    if not np.isfinite(x):
        return " NA"
    sign = "+" if x >= 0 else "-"
    return f" {sign}{_fmt_float(abs(x))}"


# --------- main ----------
def analyze_melkman_complexity(
    TestPolys: Sequence[Sequence[tuple]],
    hull_fn: Callable[..., Any],
    runs: int = 5,
    show: bool = True,
    save_path: str | None = None,
    style: str = "paper",
) -> Dict[str, Any]:
    """Profile Melkman implementations and visualize scaling.

    Args:
        TestPolys: Sequence of polygons (each a sequence of points).
        hull_fn: Function implementing the hull. Should accept
                 (polygon, stats=..., verbose=...) or just (polygon).
        runs: Repetitions for median timing.
        show: If True, create the figure.
        save_path: If provided, save the figure.
        style: "paper" applies a clean style.

    Returns:
        A dict with n, times, ops, fitted parameters, R², last-case breakdown, and the figure.
    """
    if style == "paper":
        _apply_paper_style()

    ns = [len(P) for P in TestPolys]
    times = [_median_time(lambda X: _call_hull_for_time(hull_fn, X), P, runs) for P in TestPolys]
    stats = [_normalize_stats(_call_hull_for_stats(hull_fn, P) or {}) for P in TestPolys]
    ops = [int(s["total_ops"]) for s in stats]

    ns_a = np.array(ns, float)
    t_a = np.array(times, float)
    o_a = np.array(ops, float)

    # time ~ C * n^k (log-log fit)
    mask = (ns_a > 0) & (t_a > 0)
    if mask.sum() >= 2:
        x, y = np.log(ns_a[mask]), np.log(t_a[mask])
        k, A = np.polyfit(x, y, 1)
        C = np.exp(A)
        fit_t = C * ns_a ** k
        ss_res = ((t_a[mask] - C * ns_a[mask] ** k) ** 2).sum()
        ss_tot = ((t_a[mask] - t_a[mask].mean()) ** 2).sum()
        r2_t = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    else:
        k, C, r2_t = np.nan, np.nan, np.nan
        fit_t = np.full_like(ns_a, np.nan)

    # ops ~ a * n + b (linear fit)
    if len(ns_a) >= 2:
        a, b = np.polyfit(ns_a, o_a, 1)
        fit_o = a * ns_a + b
        ss_res = ((o_a - fit_o) ** 2).sum()
        ss_tot = ((o_a - o_a.mean()) ** 2).sum()
        r2_o = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    else:
        a, b, r2_o = np.nan, np.nan, np.nan
        fit_o = np.full_like(ns_a, np.nan)

    fig = None
    if show:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

        # (a) time vs n (log-log)
        ax = axs[0]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(ns, times, color="black", label="median")
        if np.isfinite(fit_t).any():
            ax.plot(
                ns,
                fit_t,
                color="black",
                label=f"fit: { _fmt_float(C) }·n^{ _fmt_float(k) }  (R²={ _fmt_float(r2_t) })",
            )
        ax.set_xlabel("n")
        ax.set_ylabel("time (s)")
        ax.set_title("(a) Time vs n")
        ax.grid(True, which="both", alpha=0.6)
        ax.xaxis.set_major_locator(FixedLocator(ns))
        ax.set_xticklabels([str(n) for n in ns])
        ax.legend(loc="lower right", fontsize=10)

        # (b) total ops vs n (linear)
        ax = axs[1]
        ax.plot(ns, ops, "s", color="black", label="ops")
        if np.isfinite(fit_o).any():
            ax.plot(
                ns,
                fit_o,
                "-",
                color="black",
                label=f"fit: { _fmt_float(a) }·n{ _fmt_signed(b) }  (R²={ _fmt_float(r2_o) })",
            )
        ax.set_xlabel("n")
        ax.set_ylabel("total ops")
        ax.set_title("(b) Operation Count")
        ax.grid(True, alpha=0.6)
        ax.legend(loc="upper left", fontsize=10)

        # (c) breakdown for the largest n
        ax = axs[2]
        idx = int(np.argmax(ns_a))
        last = stats[idx]
        parts = [
            ("orient", ["orient"]),
            ("cross", ["cross"]),
            ("is_left", ["is_left"]),
            ("is_right_or_on", ["is_right_or_on"]),
            ("comparisons", ["comparisons"]),
            ("inside_tests", ["inside_tests"]),
            ("append_total", ["append_total"]),
            ("pop_total", ["pop_total"]),
            ("append_right", ["append_right"]),
            ("append_left", ["append_left"]),
            ("pop_right", ["pop_right"]),
            ("pop_left", ["pop_left"]),
            ("while_top", ["while_top_iters"]),
            ("while_bot", ["while_bot_iters"]),
        ]
        labels = [p[0] for p in parts]
        vals = [sum(last.get(k, 0) for k in p[1]) for p in parts]
        gray = ["#000000", "#222222", "#444444", "#666666", "#888888", "#aaaaaa"] * 3
        ax.bar(labels, vals, color=gray[: len(labels)], edgecolor="black", lw=0.6)
        ax.set_ylabel("count")
        ax.set_title(f"(c) Breakdown @ n={last.get('n', 'NA')}")
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha("right")
        ax.grid(axis="y", alpha=0.6)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

    return {
        "ns": ns,
        "times": times,
        "ops": ops,
        "k_time": float(k),
        "C_time": float(C),
        "a_ops": float(a),
        "b_ops": float(b),
        "r2_time": float(r2_t),
        "r2_ops": float(r2_o),
        "breakdown_last": stats[int(np.argmax(ns_a))] if len(ns_a) else {},
        "figure": fig,
    }