from __future__ import annotations
from typing import Sequence, List, Tuple, Dict, Any, Callable
import time
import math
import numpy as np
import matplotlib.pyplot as plt

WPoint = Tuple[int, int, int]
HalfPlane = Tuple[float, float, float]
BuildFn = Callable[..., Any]


def _apply_paper_style() -> None:
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


def _median_time(fn: Callable[[], None], runs: int = 5) -> float:
    ts: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] if ts else float("nan")


def _safe_stats_call(build_fn: BuildFn, sites: Sequence[WPoint]) -> Dict[str, Any]:
    s = build_fn(sites, stats=True)
    if not isinstance(s, dict) or "ops" not in s:
        raise TypeError("build_fn(sites, stats=True) must return a dict with key 'ops'")
    return s


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    m = (x > 0) & (y > 0)
    X = np.log(x[m])
    Y = np.log(y[m])
    if len(X) < 2:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([X, np.ones_like(X)]).T
    p, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    Yhat = A @ np.array([p, b])
    ss_res = float(np.sum((Y - Yhat) ** 2))
    ss_tot = float(np.sum((Y - Y) .mean() ** 2))  # placeholder to keep structure; overwritten next line
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    R2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(p), float(math.exp(b)), R2


def _fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = A @ np.array([a, b])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    R2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(a), float(b), R2


def analyze_power_cells_complexity(
    sites_list: Sequence[Sequence[WPoint]],
    build_fn: BuildFn,
    runs: int = 5,
    show: bool = True,
) -> Dict[str, Any]:
    ns: List[int] = []
    t_med: List[float] = []
    ops_total: List[float] = []
    hp_counts: List[float] = []
    approx_mb: List[float] = []

    for sites in sites_list:
        n = len(sites)
        ns.append(n)
        t_med.append(_median_time(lambda: build_fn(sites), runs=runs))
        s = _safe_stats_call(build_fn, sites)
        ops_total.append(float(sum(int(v) for v in s.get("ops", {}).values())))
        hp = int(s.get("halfplanes_stored", n * (n - 1)))
        hp_counts.append(float(hp))
        approx_mb.append(hp * 24.0 / (1024.0**2))  # float64 triplet lower bound

    ns_arr = np.array(ns, dtype=float)
    t_arr = np.array(t_med, dtype=float)
    ops_arr = np.array(ops_total, dtype=float)
    hp_arr = np.array(hp_counts, dtype=float)
    mem_arr = np.array(approx_mb, dtype=float)

    p, c, r2_time = _fit_power_law(ns_arr, t_arr)
    a_ops, b_ops, r2_ops = _fit_linear(ns_arr * (ns_arr - 1.0), ops_arr)
    a_sp, b_sp, r2_sp = _fit_linear(ns_arr * (ns_arr - 1.0), hp_arr)

    out: Dict[str, Any] = {
        "n": ns,
        "time_median": t_med,
        "ops_total": ops_total,
        "halfplanes": hp_counts,
        "approx_mem_MB_lowerbound": approx_mb,
        "fits": {
            "time_powerlaw": {"p": p, "c": c, "R2": r2_time},
            "ops_linear_in_n2": {"a": a_ops, "b": b_ops, "R2": r2_ops},
            "space_linear_in_n2": {"a": a_sp, "b": b_sp, "R2": r2_sp},
        },
    }

    if show:
        _apply_paper_style()
        fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=False)

        xticks = list(map(int, sorted(ns)))
        n_line = np.linspace(ns_arr.min(), ns_arr.max(), 400)

        ax0 = ax[0]
        ax0.set_title("(a) Time vs n")
        ax0.set_xlabel("n")
        ax0.set_ylabel("time (s)")
        ax0.set_yscale("log")
        ax0.grid(True, which="both")
        ax0.scatter(ns_arr, t_arr, facecolors="white", edgecolors="black", marker="o", label="median")
        if np.isfinite(p) and np.isfinite(c):
            ax0.plot(n_line, c * n_line**p, color="black", label=f"fit: {c:.1g}·n^{p:.2f}  (R²={r2_time:.2f})")
        ax0.set_xticks(xticks)
        ax0.legend(loc="lower left", bbox_to_anchor=(0.28, 0.02))

        ax1 = ax[1]
        ax1.set_title("(b) Operation Count")
        ax1.set_xlabel("n")
        ax1.set_ylabel("total ops")
        ax1.grid(True, which="both")
        ax1.scatter(ns_arr, ops_arr, facecolors="black", edgecolors="black", marker="s", label="ops")
        ax1.plot(n_line, a_ops * (n_line * (n_line - 1.0)) + b_ops, color="black",
                 label=f"fit: {a_ops:.2f}·n(n-1)+{b_ops:.1f}  (R²={r2_ops:.2f})")
        ax1.set_xticks(xticks)
        ax1.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

        ax2 = ax[2]
        ax2.set_title("(c) Space vs n")
        ax2.set_xlabel("n")
        ax2.set_ylabel("half-planes (count)")
        ax2.grid(True, which="both")
        ax2.plot(ns_arr, hp_arr, linestyle="None", marker="o", color="black", label="count")
        ax2.plot(n_line, a_sp * (n_line * (n_line - 1.0)) + b_sp, color="black",
                 label=f"fit: {a_sp:.2f}·n(n-1)+{b_sp:.1f}  (R²={r2_sp:.2f})")
        ax2.set_xticks(xticks)
        ax2.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

        ax2b = ax2.twinx()
        ax2b.set_ylabel("lower-bound memory (MB)")
        ax2b.plot(ns_arr, mem_arr, linestyle=":", color="gray")
        ax2b.tick_params(axis="y", colors="gray")
        ax2b.spines["right"].set_visible(False)
        ax2b.spines["top"].set_visible(False)

        for a in (ax0, ax1, ax2):
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.subplots_adjust(top=0.90, bottom=0.12, left=0.07, right=0.93, wspace=0.35)

    return out