"""Binary mask cleanup + centreline (skeleton) extraction.

Font outlines describe *filled shapes*.  A handwriting trajectory is a
*centreline*.  There is no exact inverse of "stroke -> outline" (the outline
lost the pen width and the stroke order), so we approximate the centreline by
thinning the filled mask.  Several thinning algorithms are offered because they
fail differently:

``zhang``       Zhang-Suen (via scikit-image) - fast, slightly noisier ends.
``lee``         Lee's 3-D-capable algorithm (scikit-image) - smoother.
``guo_hall``    Guo-Hall parallel thinning - implemented here in numpy, tends to
                keep diagonal strokes straighter than Zhang-Suen.
``medial_axis`` exact medial axis - most faithful but very spur-prone.
``opencv``      cv2.ximgproc.thinning if the contrib module is installed.

None of these produce an *ordered* stroke sequence; that is the job of
``skeleton/graph.py`` + ``strokes/planner.py``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, medial_axis, skeletonize

METHODS = ("zhang", "lee", "guo_hall", "medial_axis", "opencv")


def binarize(coverage: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.asarray(coverage) >= threshold


def clean_mask(
    mask: np.ndarray,
    closing_radius: int = 2,
    opening_radius: int = 0,
    min_component_area_ratio: float = 0.0008,
    max_hole_fill_ratio: float = 0.0,
) -> np.ndarray:
    """Morphological cleanup before thinning.

    * closing bridges hairline gaps where two contours *almost* touch (very
      common between a base consonant and its coeng subscript), preventing the
      skeleton from fragmenting into disconnected pieces;
    * opening removes 1-px specks that would become spurious strokes;
    * tiny components are dropped, tiny holes optionally filled.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    if closing_radius > 0:
        m = ndi.binary_closing(m, structure=disk(closing_radius), border_value=0)
    if opening_radius > 0:
        m = ndi.binary_opening(m, structure=disk(opening_radius), border_value=0)
    area = m.shape[0] * m.shape[1]
    if min_component_area_ratio > 0:
        # done by hand rather than via skimage.remove_small_objects, whose
        # min_size/max_size keyword changed meaning across versions
        min_size = max(int(min_component_area_ratio * area), 2)
        lab, n = ndi.label(m, structure=np.ones((3, 3), dtype=int))
        if n:
            sizes = np.bincount(lab.ravel())
            keep = np.zeros(sizes.shape, dtype=bool)
            keep[1:] = sizes[1:] >= min_size
            m = keep[lab]
    if max_hole_fill_ratio > 0:
        filled = ndi.binary_fill_holes(m)
        holes = filled & ~m
        lab, n = ndi.label(holes)
        if n:
            sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
            small = np.isin(lab, np.nonzero(sizes <= max_hole_fill_ratio * area)[0] + 1)
            m = m | small
    return m


# --------------------------------------------------------------------------- #
# Guo-Hall
# --------------------------------------------------------------------------- #

_SHIFTS = {
    "p2": (-1, 0),
    "p3": (-1, 1),
    "p4": (0, 1),
    "p5": (1, 1),
    "p6": (1, 0),
    "p7": (1, -1),
    "p8": (0, -1),
    "p9": (-1, -1),
}


def _neighbors(m: np.ndarray) -> dict:
    out = {}
    for name, (dy, dx) in _SHIFTS.items():
        out[name] = np.roll(np.roll(m, -dy, axis=0), -dx, axis=1)
        # zero the wrapped border
        if dy == -1:
            out[name][-1, :] = False
        elif dy == 1:
            out[name][0, :] = False
        if dx == -1:
            out[name][:, -1] = False
        elif dx == 1:
            out[name][:, 0] = False
    return out


def guo_hall_thin(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Vectorised Guo-Hall parallel thinning (Guo & Hall, CACM 1989)."""
    m = np.asarray(mask, dtype=bool).copy()
    for _ in range(max_iter):
        changed = False
        for parity in (0, 1):
            p = _neighbors(m)
            p2, p3, p4, p5 = p["p2"], p["p3"], p["p4"], p["p5"]
            p6, p7, p8, p9 = p["p6"], p["p7"], p["p8"], p["p9"]
            C = (
                (~p2 & (p3 | p4)).astype(np.int8)
                + (~p4 & (p5 | p6)).astype(np.int8)
                + (~p6 & (p7 | p8)).astype(np.int8)
                + (~p8 & (p9 | p2)).astype(np.int8)
            )
            N1 = (
                (p9 | p2).astype(np.int8)
                + (p3 | p4).astype(np.int8)
                + (p5 | p6).astype(np.int8)
                + (p7 | p8).astype(np.int8)
            )
            N2 = (
                (p2 | p3).astype(np.int8)
                + (p4 | p5).astype(np.int8)
                + (p6 | p7).astype(np.int8)
                + (p8 | p9).astype(np.int8)
            )
            N = np.minimum(N1, N2)
            if parity == 0:
                cond = ((p6 | p7 | ~p9) & p8)
            else:
                cond = ((p2 | p3 | ~p5) & p4)
            delete = m & (C == 1) & (N >= 2) & (N <= 3) & (~cond)
            if delete.any():
                m &= ~delete
                changed = True
        if not changed:
            break
    return m


def skeletonize_mask(mask: np.ndarray, method: str = "zhang") -> np.ndarray:
    """Thin ``mask`` to a ~1-pixel-wide skeleton."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    if method == "zhang":
        return skeletonize(m, method="zhang").astype(bool)
    if method == "lee":
        return skeletonize(m, method="lee").astype(bool)
    if method == "guo_hall":
        return guo_hall_thin(m)
    if method == "medial_axis":
        return medial_axis(m).astype(bool)
    if method == "opencv":  # pragma: no cover - optional contrib module
        try:
            import cv2

            thin = cv2.ximgproc.thinning((m * 255).astype(np.uint8))
            return thin > 0
        except Exception:
            return skeletonize(m, method="zhang").astype(bool)
    raise ValueError(f"unknown thinning method {method!r} (expected one of {METHODS})")


def stroke_width_map(mask: np.ndarray) -> np.ndarray:
    """Distance transform of the mask - an estimate of local half stroke width.

    Used to (a) sanity-check skeleton quality and (b) give the renderer a
    font-derived width prior, so a synthetic writer can inherit the *relative*
    thick/thin modulation of the source typeface.
    """
    return ndi.distance_transform_edt(np.asarray(mask, dtype=bool)).astype(np.float32)
