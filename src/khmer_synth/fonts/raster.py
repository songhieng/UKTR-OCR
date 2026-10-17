"""High-resolution nonzero-winding rasterisation of glyph contours.

Why not just use PIL polygons?
------------------------------
``ImageDraw.polygon`` fills one contour at a time; compositing several contours
with XOR gives *even-odd* fill.  Khmer glyphs - especially composite ones and
stacked coeng clusters - routinely contain contours that overlap in the same
winding direction, and even-odd fill punches spurious holes exactly where two
strokes cross.  Those holes survive skeletonisation as fake loops, which is one
of the classic silent failures this pipeline has to avoid.

So we implement the correct rule: a scanline **nonzero-winding** fill, fully
vectorised in numpy.  For every sample row we accumulate +/-1 at each edge
crossing and take a cumulative sum along x; a pixel is inside where the winding
number is non-zero.  Anti-aliasing comes from supersampling the sample grid and
box-averaging back down.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RasterFrame:
    """Mapping between font units (y-UP) and raster pixels (y-DOWN)."""

    x_min: float  # font units at pixel column 0 (left edge)
    y_max: float  # font units at pixel row 0 (top edge)
    scale: float  # pixels per font unit
    width: int
    height: int
    upem: int

    def px_to_em(self, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Pixel centres -> em space (y-DOWN, origin on the baseline)."""
        x_fu = self.x_min + (np.asarray(cols, dtype=np.float64) + 0.5) / self.scale
        y_fu = self.y_max - (np.asarray(rows, dtype=np.float64) + 0.5) / self.scale
        return np.stack([x_fu / self.upem, -y_fu / self.upem], axis=-1)

    def em_to_px(self, pts_em: np.ndarray) -> np.ndarray:
        pts_em = np.asarray(pts_em, dtype=np.float64)
        x_fu = pts_em[..., 0] * self.upem
        y_fu = -pts_em[..., 1] * self.upem
        cols = (x_fu - self.x_min) * self.scale - 0.5
        rows = (self.y_max - y_fu) * self.scale - 0.5
        return np.stack([cols, rows], axis=-1)


def contours_to_edges(contours: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate closed contours into an ``(E, 4)`` array of (x0,y0,x1,y1)."""
    segs: list[np.ndarray] = []
    for c in contours:
        c = np.asarray(c, dtype=np.float64)
        if len(c) < 2:
            continue
        if not np.allclose(c[0], c[-1]):
            c = np.vstack([c, c[:1]])
        segs.append(np.concatenate([c[:-1], c[1:]], axis=1))
    if not segs:
        return np.zeros((0, 4), dtype=np.float64)
    return np.vstack(segs)


def fill_nonzero(edges: np.ndarray, width: int, height: int) -> np.ndarray:
    """Binary nonzero-winding fill sampled at pixel centres.

    ``edges`` are already in device (pixel) coordinates, y-DOWN.
    Returns a ``(height, width)`` bool array.
    """
    out = np.zeros((height, width), dtype=bool)
    if len(edges) == 0 or width <= 0 or height <= 0:
        return out

    x0, y0, x1, y1 = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]
    horizontal = np.isclose(y0, y1)
    x0, y0, x1, y1 = x0[~horizontal], y0[~horizontal], x1[~horizontal], y1[~horizontal]
    if len(x0) == 0:
        return out

    direction = np.where(y1 > y0, 1, -1).astype(np.int32)
    y_lo = np.minimum(y0, y1)
    y_hi = np.maximum(y0, y1)

    # rows whose pixel centre (r + 0.5) lies in [y_lo, y_hi)
    r0 = np.clip(np.ceil(y_lo - 0.5), 0, height).astype(np.int64)
    r1 = np.clip(np.ceil(y_hi - 0.5), 0, height).astype(np.int64)
    counts = np.maximum(r1 - r0, 0)
    total = int(counts.sum())
    if total == 0:
        return out

    edge_idx = np.repeat(np.arange(len(x0), dtype=np.int64), counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    offsets = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    rows = r0[edge_idx] + offsets

    y = rows.astype(np.float64) + 0.5
    inv_slope = (x1 - x0) / (y1 - y0)
    x = x0[edge_idx] + (y - y0[edge_idx]) * inv_slope[edge_idx]

    # crossing at x affects every pixel whose centre is >= x
    cols = np.clip(np.ceil(x - 0.5), 0, width).astype(np.int64)

    acc = np.zeros((height, width + 1), dtype=np.int32)
    np.add.at(acc, (rows, cols), direction[edge_idx])
    winding = np.cumsum(acc[:, :width], axis=1)
    return winding != 0


def rasterize_contours(
    contours: Sequence[np.ndarray],
    upem: int,
    px_per_em: float,
    supersample: int = 3,
    margin_em: float = 0.02,
    bounds: tuple[float, float, float, float] | None = None,
    max_side: int = 2048,
) -> tuple[np.ndarray, RasterFrame]:
    """Rasterise font-unit contours into an anti-aliased coverage map.

    Returns ``(coverage[H, W] float32 in [0, 1], frame)``.
    """
    from .glyphs import outline_bbox

    bb = bounds if bounds is not None else outline_bbox(contours)
    if bb is None:
        frame = RasterFrame(0.0, 0.0, px_per_em / upem, 1, 1, upem)
        return np.zeros((1, 1), dtype=np.float32), frame

    margin_fu = margin_em * upem
    x_min = bb[0] - margin_fu
    y_min = bb[1] - margin_fu
    x_max = bb[2] + margin_fu
    y_max = bb[3] + margin_fu

    scale = px_per_em / float(upem)
    width = int(np.ceil((x_max - x_min) * scale))
    height = int(np.ceil((y_max - y_min) * scale))
    width = int(np.clip(width, 1, max_side))
    height = int(np.clip(height, 1, max_side))

    ss = max(1, int(supersample))
    edges = contours_to_edges(contours)
    if len(edges):
        dev = np.empty_like(edges)
        dev[:, 0] = (edges[:, 0] - x_min) * scale * ss
        dev[:, 2] = (edges[:, 2] - x_min) * scale * ss
        dev[:, 1] = (y_max - edges[:, 1]) * scale * ss
        dev[:, 3] = (y_max - edges[:, 3]) * scale * ss
    else:
        dev = edges

    mask = fill_nonzero(dev, width * ss, height * ss)
    if ss > 1:
        cov = mask.reshape(height, ss, width, ss).mean(axis=(1, 3)).astype(np.float32)
    else:
        cov = mask.astype(np.float32)

    frame = RasterFrame(
        x_min=x_min, y_max=y_max, scale=scale, width=width, height=height, upem=upem
    )
    return cov, frame
