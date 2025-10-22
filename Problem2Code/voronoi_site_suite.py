from __future__ import annotations
from typing import List, Tuple, Sequence, Dict, Any, Iterable, Optional, Union
from pathlib import Path
import json
import random
import math

# -------- types --------
WPoint = Tuple[int, int, int]  # (x, y, w)
WSites = List[WPoint]
CaseBundleW = List[WSites]

__all__ = [
    "WPoint",
    "WSites",
    "CaseBundleW",
    "load_sites_cases",
    "generate_sites",
]


# -------- helpers --------
def _finite_int(v: Any, name: str) -> int:
    """Cast to finite int with validation."""
    fv = float(v)
    if not math.isfinite(fv):
        raise ValueError(f"{name} must be finite, got {v}")
    return int(round(fv))


def _coerce_triplet(item: Iterable[Any], key: str, idx: int) -> WPoint:
    """Accept [x,y] or [x,y,w]; default w=0."""
    seq = list(item)
    if len(seq) not in (2, 3):
        raise ValueError(f"{key}[{idx}] must be [x,y] or [x,y,w]")
    x = _finite_int(seq[0], f"{key}[{idx}].x")
    y = _finite_int(seq[1], f"{key}[{idx}].y")
    w = _finite_int(seq[2], f"{key}[{idx}].w") if len(seq) == 3 else 0
    return (x, y, w)


def _dedup_preserve_order(sites: WSites) -> WSites:
    """Remove exact duplicates while preserving order."""
    seen = set()
    out: WSites = []
    for p in sites:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _validate_sites(sites: WSites, key: str) -> WSites:
    """Re-validate triples as finite ints and deduplicate."""
    cleaned: WSites = []
    for i, (x, y, w) in enumerate(sites):
        xi = _finite_int(x, f"{key}[{i}].x")
        yi = _finite_int(y, f"{key}[{i}].y")
        wi = _finite_int(w, f"{key}[{i}].w")
        cleaned.append((xi, yi, wi))
    return _dedup_preserve_order(cleaned)


# -------- generation --------
def load_sites_cases(path: Optional[Union[str, Path]] = None) -> CaseBundleW:
    """
    Load weighted site cases from JSON with keys:
    caseA_points, caseB_points, caseC_points.
    Elements may be [x,y] or [x,y,w].
    """
    jpath = Path(__file__).with_name("voronoi_cases.json") if path is None else Path(path)
    if not jpath.exists():
        raise FileNotFoundError(f"voronoi_cases.json not found: {jpath}")

    with jpath.open("r", encoding="utf-8") as f:
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


def generate_sites(
    sizes: Sequence[int],
    image_size: int = 1000,
    seed: int = 1024,
) -> CaseBundleW:
    """
    Generate random integer sites with integer weights.
    Weight scheme:
      75% in [-5, 5], 20% in ±[20, 100], 5% in ±[200, 800].
    Ensure uniqueness per group on (x,y,w).
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
        max_attempts = max(1000, 20 * max(1, n))
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