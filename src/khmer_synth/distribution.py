"""Image-level real-vs-synthetic distribution statistics (no trained model needed).

Used by ``scripts/compare_distributions.py`` and the Streamlit playground's
"real-gap check" tab.  See the script docstring for the metric list.
"""

from __future__ import annotations

import random
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from .text.graphemes import segment_graphemes

METRICS = [
    ("paper", "paper brightness", "higher = whiter"),
    ("contrast", "ink contrast", "paper − darkest ink"),
    ("blue_frac", "ink blueness", "blue − red channel of ink"),
    ("sw_over_h", "stroke width / text h", ""),
    ("sw_cv", "stroke-width CV", "pen dynamics; lower = monoline"),
    ("dark_cv", "ink darkness CV", ""),
    ("rough", "contour roughness", "1 = smooth; jitter raises it"),
    ("slant", "slant (deg)", "negative = leans right here"),
    ("slant_std", "slant consistency (deg std)", "adult hand = low"),
    ("h_cv", "letter height CV", ""),
    ("band_h", "x-height band / text h", "real ≈ 0.35"),
    ("base_res", "baseline residual / text h", ""),
    ("touch", "touching-letter rate", ""),
    ("cc_per_g", "components / grapheme", "low = merged letters"),
    ("ng", "graphemes per crop", ""),
    ("aspect", "aspect ratio (w/h)", ""),
    ("ruled", "ruled/grid background", "fraction"),
]


def read_list(path: Path) -> list[tuple[Path, str]]:
    rows = []
    for line in path.read_text("utf-8").splitlines():
        if "\t" not in line:
            continue
        p, t = line.rstrip("\n").split("\t", 1)
        rows.append((path.parent / p, unicodedata.normalize("NFC", t.strip())))
    return rows


def slant_of(mask: np.ndarray, step: int = 3) -> int:
    h, w = mask.shape
    best, best_a = -1.0, 0
    for a in range(-45, 46, step):
        t = np.tan(np.radians(a))
        M = np.float32([[1, -t, t * h / 2], [0, 1, 0]])
        sh = cv2.warpAffine(mask.astype(np.uint8), M, (w + int(abs(t) * h) + 2, h))
        v = sh.sum(axis=0).astype(np.float64).var()
        if v > best:
            best, best_a = v, a
    return best_a


def measure(path: Path | None, text: str, rgb_u8: np.ndarray | None = None) -> dict | None:
    """Measure one crop given a file path or an RGB uint8 array."""
    if rgb_u8 is None:
        im = Image.open(path)
        if im.mode == "RGBA":
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.alpha_composite(im)
            im = bg
        rgb = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
    else:
        rgb = np.asarray(rgb_u8).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    grey = rgb.mean(axis=2)
    paper = float(np.median(grey))
    if paper < 0.45:
        return None
    g8 = (grey * 255).astype(np.uint8)
    thr, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (grey <= thr / 255.0) & (grey < paper - 0.12)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(4, 0.00015 * h * w)
    ink = keep[lab]
    if ink.sum() < 30:
        return None
    rows = ink.sum(axis=1)
    cum = np.cumsum(rows) / rows.sum()
    ty0, ty1 = int(np.searchsorted(cum, 0.03)), int(np.searchsorted(cum, 0.97))
    th = max(ty1 - ty0, 1)
    # x-height band = densest rows holding 55 % of ink
    pk = int(np.argmax(np.convolve(rows, np.ones(max(3, th // 8)) / max(3, th // 8), mode="same")))
    lo = hi = pk
    tot = rows.sum()
    while rows[lo : hi + 1].sum() < 0.55 * tot and (lo > 0 or hi < h - 1):
        if lo > 0 and (hi >= h - 1 or rows[lo - 1] >= rows[hi + 1]):
            lo -= 1
        else:
            hi += 1
    band_h = (hi - lo + 1) / th
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    sk = skeletonize(ink)
    d = dt[sk]
    sw = float(2 * np.median(d)) if len(d) else 0.0
    sw_cv = float(np.std(d) / max(np.mean(d), 1e-6)) if len(d) else 0.0
    dark = paper - grey[sk]
    dark_cv = float(np.std(dark) / max(np.mean(dark), 1e-6)) if len(dark) else 0.0
    ink_rgb = np.median(rgb[ink], axis=0)
    contrast = float(paper - np.percentile(grey[ink], 20))
    # roughness at a normalised scale (text height 48)
    s = 48.0 / th
    m = cv2.resize(ink.astype(np.uint8) * 255, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) > 127
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    L = sum(cv2.arcLength(c, True) for c in cs if len(c) > 20)
    sm = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5) > 0.5
    cs2, _ = cv2.findContours(sm.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    L2 = sum(cv2.arcLength(c, True) for c in cs2 if len(c) > 20)
    rough = L / max(L2, 1e-6)
    # components
    ids = [i for i in range(1, n) if keep[i]]
    big = [i for i in ids if st[i, 3] > 0.35 * th and st[i, 2] > 0.15 * th]
    slants, hs = [], []
    for i in big[:25]:
        x, y, ww, hh = st[i, :4]
        slants.append(slant_of(lab[y : y + hh, x : x + ww] == i, step=4))
        hs.append(hh)
    slant_std = float(np.std(slants)) if len(slants) > 2 else np.nan
    h_cv = float(np.std(hs) / np.mean(hs)) if len(hs) > 2 else np.nan
    if len(big) > 3:
        xs = np.array([st[i, 0] + st[i, 2] / 2 for i in big])
        ys = np.array([st[i, 1] + st[i, 3] for i in big])
        A = np.vstack([xs, np.ones_like(xs)]).T
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        base_res = float(np.std(ys - A @ coef) / th)
        bs = sorted((st[i, 0], st[i, 2]) for i in big)
        gaps = np.array([b[0] - (a[0] + a[1]) for a, b in zip(bs[:-1], bs[1:])])
        touch = float((gaps <= 0).mean())
    else:
        base_res = touch = np.nan
    ng = max(len([c for c in segment_graphemes(text) if c.text.strip()]), 1)
    faint = (grey < paper - 0.04) & (grey > paper - 0.5) & ~ink
    ruled = float((faint.mean(axis=1) > 0.5).any())
    return dict(
        paper=paper, contrast=contrast, blue_frac=float(ink_rgb[2] - ink_rgb[0]), sw_over_h=sw / th,
        sw_cv=sw_cv, dark_cv=dark_cv, rough=rough, slant=float(slant_of(ink)), slant_std=slant_std,
        h_cv=h_cv, band_h=band_h, base_res=base_res, touch=touch, cc_per_g=len(ids) / ng, ng=float(ng),
        aspect=w / h, ruled=ruled,
    )




def summarise_results(res: list[dict]) -> dict[str, tuple[float, float, float]]:
    out: dict = {}
    for key, *_ in METRICS:
        v = np.array([r[key] for r in res], float)
        v = v[~np.isnan(v)]
        out[key] = tuple(np.percentile(v, [10, 50, 90])) if len(v) else (np.nan,) * 3
    out["_n"] = len(res)
    return out


def summarise(rows: list[tuple[Path, str]], n: int, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    rng = random.Random(seed)
    rows = rng.sample(rows, min(n, len(rows)))
    res = [r for r in (measure(p, t) for p, t in rows if p.exists()) if r]
    return summarise_results(res)


def verdicts(real: dict, synth: dict) -> dict[str, bool]:
    """Metric -> True when the synthetic median lies inside the real p10-p90 band."""
    return {key: bool(real[key][0] <= synth[key][1] <= real[key][2]) for key, *_ in METRICS}
