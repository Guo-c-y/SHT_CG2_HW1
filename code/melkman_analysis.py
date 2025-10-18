# melkman_analysis.py
# 论文风格 · 三图合一 · 黑白灰
import time, numpy as np, matplotlib.pyplot as plt

# ---------- 计时 ----------
def _median_time(fn, arg, runs=5):
    ts=[]
    for _ in range(runs):
        t0=time.perf_counter(); fn(arg); ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[len(ts)//2]

# ---------- 风格 ----------
def _apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": "#aaaaaa",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "figure.dpi": 150,
    })

# ---------- 主分析 ----------
def analyze_melkman_complexity(complex_cases, complex_stats, hull_fn, runs=5, show=True,
                               save_path=None, style="paper"):
    """
    complex_cases: [P10,P100,P1000]
    complex_stats: [stats10,stats100,stats1000]（或留空将现场统计）
    hull_fn: melkman_convex_hull
    返回: report(dict)；若 show=True 生成一张三联图；save_path 可保存 png/pdf/svg
    """
    if style == "paper":
        _apply_paper_style()

    ns = [len(P) for P in complex_cases]

    # 时间（正常模式）
    times = [_median_time(lambda X: hull_fn(X, test=False), P, runs=runs)
             for P in complex_cases]

    # 操作数（测试模式）
    ops, fresh_stats = [], []
    for P, st in zip(complex_cases, complex_stats or [None]*len(complex_cases)):
        if not isinstance(st, dict) or ("total_ops" not in st):
            st = hull_fn(P, test=True)
        ops.append(int(st.get("total_ops", 0))); fresh_stats.append(st)

    ns_arr = np.array(ns, float)
    t_arr  = np.array(times, float)
    o_arr  = np.array(ops, float)

    # 拟合：time ~ n^k
    x_log = np.log(ns_arr); y_log = np.log(t_arr)
    k, A = np.polyfit(x_log, y_log, 1)
    t_fit = np.exp(A + k * x_log)
    r2_time = 1 - ((t_arr - t_fit)**2).sum() / ((t_arr - t_arr.mean())**2).sum()

    # 拟合：ops ~ a n + b
    a, b = np.polyfit(ns_arr, o_arr, 1)
    o_fit = a*ns_arr + b
    r2_ops = 1 - ((o_arr - o_fit)**2).sum() / ((o_arr - o_arr.mean())**2).sum()

    # 派生
    time_doubling = [round(times[i+1]/times[i], 3) for i in range(len(times)-1)]
    ops_per_n = [round(ops[i]/ns[i], 3) for i in range(len(ns))]

    # —— 可视化（单图三联）——
    fig = None
    if show:
        fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2), constrained_layout=True)

        # S1: 时间 vs n（log-log）
        ax = axs[0]
        ax.loglog(ns, times, marker="o", linestyle="none", color="black", label="median")
        ax.loglog(ns, t_fit, linestyle="-", color="black", label=f"fit  n^{k:.2f}")
        ax.set_xlabel("n"); ax.set_ylabel("time (s)")
        ax.set_title("(a) Time vs n")
        ax.grid(True, which="both", alpha=0.6)
        ax.legend(loc="lower right")

        # S2: 操作数 vs n（线性）
        ax = axs[1]
        ax.plot(ns, ops, marker="s", linestyle="none", color="black", label="ops")
        ax.plot(ns, o_fit, linestyle="-", color="black", label=f"fit  {a:.1f}·n{b:+.0f}")
        ax.set_xlabel("n"); ax.set_ylabel("total ops")
        ax.set_title("(b) Operation Count")
        ax.grid(True, alpha=0.6)
        ax.legend(loc="upper left")

        # S3: 分项操作（最大 n）
        ax = axs[2]
        last = fresh_stats[-1]
        parts = ["is_left","orient","cross","append","appendleft","pop","popleft"]
        vals  = [last.get(k,0) for k in parts]
        # 柱形用黑灰序列
        gray_seq = ["#000000","#333333","#555555","#777777","#999999","#bbbbbb","#dddddd"]
        ax.bar(parts, vals, color=gray_seq[:len(parts)], edgecolor="black", linewidth=0.6)
        ax.set_ylabel("count"); ax.set_title(f"(c) Breakdown @ n={ns[-1]}")
        for tick in ax.get_xticklabels(): tick.set_rotation(30)
        ax.grid(axis="y", alpha=0.6)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

    # 汇总
    report = {
        "ns": ns,
        "times": times,
        "ops": ops,
        "time_doubling": time_doubling,
        "ops_per_n": ops_per_n,
        "k_time": float(k),
        "a_ops": float(a),
        "r2_time": float(r2_time),
        "r2_ops": float(r2_ops),
        "breakdown_last": {k:int(fresh_stats[-1].get(k,0)) for k in fresh_stats[-1]},
        "figure": fig
    }
    return report