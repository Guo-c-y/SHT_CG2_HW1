# voronoi_site_suite.py
from __future__ import annotations
from typing import List, Tuple, Sequence, Dict, Any, Iterable
from pathlib import Path
import json, random

# -------- types --------
WPoint = Tuple[int, int, int]   # (x, y, w) —— w 为整数
WSites = List[WPoint]
CaseBundleW = List[WSites]

__all__ = [
    "WPoint", "WSites", "CaseBundleW",
    "load_sites_cases", "generate_sites"
]

# -------- utils --------
def _coerce_triplet(item: Iterable[Any], key: str, idx: int) -> WPoint:
    """
    支持 [x,y] 或 [x,y,w]；缺省 w→0。x,y,w 最终转为 int。
    """
    seq = list(item)
    if len(seq) not in (2, 3):
        raise ValueError(f"{key}[{idx}] must be [x,y] or [x,y,w]")
    x, y = seq[0], seq[1]
    w = 0 if len(seq) == 2 else seq[2]
    try:
        xi = int(x); yi = int(y); wi = int(round(float(w)))
    except Exception:
        raise ValueError(f"{key}[{idx}] coords/weight must be numeric")
    return (xi, yi, wi)

# -------- I/O: load weighted cases from json --------
def load_sites_cases(path: str | Path | None = None) -> CaseBundleW:
    """
    读取 voronoi_cases.json，返回：
        [
          [(x,y,w), ...],  # caseA
          [(x,y,w), ...],  # caseB
          [(x,y,w), ...],  # caseC
        ]
    JSON 期望键：caseA_points / caseB_points / caseC_points
    每个元素可为 [x,y,w] 或 [x,y]（缺省 w=0）。
    """
    path = Path(__file__).with_name("voronoi_cases.json") if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"voronoi_cases.json not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    required = ["caseA_points", "caseB_points", "caseC_points"]
    for k in required:
        if k not in data:
            raise KeyError(f"missing key in json: {k}")

    bundle: CaseBundleW = []
    for key in required:
        arr = data[key]
        if not isinstance(arr, list):
            raise TypeError(f"{key} must be a list")
        sites: WSites = []
        for i, item in enumerate(arr):
            sites.append(_coerce_triplet(item, key, i))
        bundle.append(sites)
    return bundle

# -------- generation: fixed integer-weight scheme --------
def generate_sites(
    sizes: Sequence[int],
    image_size: int = 1000,
    seed: int = 1024,
) -> CaseBundleW:
    """
    按 sizes 生成随机整点与“固定模式”的整数权重，返回 [[(x,y,w)], ...]。
    固定模式（经验分布，用于覆盖轻度/中等/极端及正负权）：
      - 75% 近零小权：w ∈ [-5, 5]
      - 20% 中权：随机选正/负区间，w ∈ [20, 100] 或 [-100, -20]
      - 5% 极端权：随机选正/负区间，w ∈ [200, 800] 或 [-800, -200]
    说明：x,y ∈ [0, image_size] 且为整数；w 为整数。
    """
    if any((not isinstance(n, int) or n < 0) for n in sizes):
        raise ValueError("sizes must be a sequence of non-negative integers")
    if image_size < 0:
        raise ValueError("image_size must be >= 0")

    rng = random.Random(seed)

    def sample_w() -> int:
        r = rng.random()
        if r < 0.75:
            return rng.randint(-5, 5)
        if r < 0.95:
            if rng.random() < 0.5:
                return rng.randint(20, 100)
            else:
                return -rng.randint(20, 100)
        # top 5%
        if rng.random() < 0.5:
            return rng.randint(200, 800)
        else:
            return -rng.randint(200, 800)

    bundle: CaseBundleW = []
    for n in sizes:
        sites: WSites = []
        for _ in range(n):
            x = rng.randint(0, image_size)
            y = rng.randint(0, image_size)
            w = sample_w()
            sites.append((x, y, w))
        bundle.append(sites)
    return bundle