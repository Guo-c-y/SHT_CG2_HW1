# voronoi_site_suite.py
from __future__ import annotations
from typing import List, Tuple, Sequence, Dict, Any, Iterable
from pathlib import Path
import json, random, math

# -------- types --------
WPoint = Tuple[int, int, int]   # (x, y, w) —— w 为整数
WSites = List[WPoint]
CaseBundleW = List[WSites]

__all__ = [
    "WPoint", "WSites", "CaseBundleW",
    "load_sites_cases", "generate_sites"
]

# -------- helpers: validity & dedup --------
def _finite_int(v: Any, name: str) -> int:
    """确保 v 可转为有限整数：先转 float 做 isfinite，再转 int。"""
    fv = float(v)
    if not math.isfinite(fv):
        raise ValueError(f"{name} must be finite, got {v}")
    return int(round(fv))

def _coerce_triplet(item: Iterable[Any], key: str, idx: int) -> WPoint:
    """支持 [x,y] 或 [x,y,w]；缺省 w→0；统一为 int 并检查有限性。"""
    seq = list(item)
    if len(seq) not in (2, 3):
        raise ValueError(f"{key}[{idx}] must be [x,y] or [x,y,w]")
    x = _finite_int(seq[0], f"{key}[{idx}].x")
    y = _finite_int(seq[1], f"{key}[{idx}].y")
    w = _finite_int(seq[2], f"{key}[{idx}].w") if len(seq) == 3 else 0
    return (x, y, w)

def _dedup_preserve_order(sites: WSites) -> WSites:
    """按 (x,y,w) 完全相同去重，保留首次出现顺序。"""
    seen = set()
    out: WSites = []
    for p in sites:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _validate_sites(sites: WSites, key: str) -> WSites:
    """再次确保三元组与有限性、整型，并去重。"""
    cleaned: WSites = []
    for i, (x, y, w) in enumerate(sites):
        xi = _finite_int(x, f"{key}[{i}].x")
        yi = _finite_int(y, f"{key}[{i}].y")
        wi = _finite_int(w, f"{key}[{i}].w")
        cleaned.append((xi, yi, wi))
    return _dedup_preserve_order(cleaned)

# -------- I/O: load weighted cases from json --------
def load_sites_cases(path: str | Path | None = None) -> CaseBundleW:
    """
    读取 voronoi_cases.json，返回：
        [
          [(x,y,w), ...],  # caseA
          [(x,y,w), ...],  # caseB
          [(x,y,w), ...],  # caseC
        ]
    JSON 键：caseA_points / caseB_points / caseC_points
    元素可为 [x,y,w] 或 [x,y]（缺省 w=0）。含显式有限性检查与去重。
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
        sites = [_coerce_triplet(item, key, i) for i, item in enumerate(arr)]
        bundle.append(_validate_sites(sites, key))
    return bundle

# -------- generation: fixed integer-weight scheme + uniqueness --------
def generate_sites(
    sizes: Sequence[int],
    image_size: int = 1000,
    seed: int = 1024,
) -> CaseBundleW:
    """
    按 sizes 生成随机整点与“固定模式”的整数权重，返回 [[(x,y,w)], ...]。
    固定模式分布：
      - 75% 近零小权：w ∈ [-5, 5]
      - 20% 中权：±[20, 100]
      - 5%  极端权：±[200, 800]
    生成阶段：
      - x,y 用 randint(0, image_size)，整数与有限性天然满足
      - w 按上面区间生成整数
      - 对每一组强制唯一 (x,y,w)，使用拒绝采样直至凑满 n（设上限避免死循环）
      - 末尾调用 _validate_sites 再次校验
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
            return rng.randint(20, 100) * (1 if rng.random() < 0.5 else -1)
        return rng.randint(200, 800) * (1 if rng.random() < 0.5 else -1)

    bundle: CaseBundleW = []
    for n in sizes:
        sset = set()
        attempts = 0
        max_attempts = max(1000, 20 * max(1, n))  # 宽松上限
        while len(sset) < n and attempts < max_attempts:
            attempts += 1
            x = rng.randint(0, image_size)
            y = rng.randint(0, image_size)
            w = sample_w()
            sset.add((x, y, w))
        if len(sset) < n:
            raise RuntimeError(f"could not generate {n} unique sites after {attempts} attempts")
        sites = list(sset)
        bundle.append(_validate_sites(sites, "generated"))
    return bundle