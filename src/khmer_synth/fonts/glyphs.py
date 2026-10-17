"""Glyph outline extraction and flattening.

fontTools gives us quadratic (TrueType) or cubic (CFF/OTF) Bezier contours plus
composite-glyph references.  :class:`ContourPen` decomposes components and
flattens every curve into a polyline with a curvature-aware number of segments,
producing the intermediate vector representation used by the rest of the
pipeline:

    GlyphOutline.contours : list of (N, 2) float arrays, font units, y-UP,
                            closed (first point repeated at the end),
                            winding direction preserved (needed for
                            nonzero-winding fill).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from fontTools.pens.basePen import BasePen

if TYPE_CHECKING:  # pragma: no cover
    from .loader import LoadedFont


def _n_segments(points: Sequence[tuple[float, float]], tol: float, degree: int) -> int:
    """Number of line segments needed to flatten a Bezier within ``tol``.

    Uses the standard bound  err <= L / (8 n^2)  (quadratic) and
    err <= L / (6 n^2) (cubic-ish), where L is the control-polygon length.
    """
    L = 0.0
    for a, b in zip(points[:-1], points[1:]):
        L += float(np.hypot(b[0] - a[0], b[1] - a[1]))
    if L <= 0:
        return 2
    k = 8.0 if degree == 2 else 6.0
    n = int(np.ceil(np.sqrt(L / max(k * tol, 1e-9))))
    return int(np.clip(n, 2, 96))


class ContourPen(BasePen):
    """Collect flattened, closed contours in font units."""

    def __init__(self, glyph_set, tolerance: float = 2.0):
        super().__init__(glyph_set)
        self.tolerance = float(tolerance)
        self.contours: list[list[tuple[float, float]]] = []
        self._current: list[tuple[float, float]] = []

    # --- BasePen protocol --- #
    def _moveTo(self, pt):
        self._flush()
        self._current = [(float(pt[0]), float(pt[1]))]

    def _lineTo(self, pt):
        self._current.append((float(pt[0]), float(pt[1])))

    def _curveToOne(self, p1, p2, p3):
        p0 = self._current[-1]
        n = _n_segments([p0, p1, p2, p3], self.tolerance, 3)
        t = np.linspace(0.0, 1.0, n + 1)[1:]
        a = np.array(p0), np.array(p1), np.array(p2), np.array(p3)
        tt = t[:, None]
        pts = (
            (1 - tt) ** 3 * a[0]
            + 3 * (1 - tt) ** 2 * tt * a[1]
            + 3 * (1 - tt) * tt**2 * a[2]
            + tt**3 * a[3]
        )
        self._current.extend([(float(x), float(y)) for x, y in pts])

    def _qCurveToOne(self, p1, p2):
        p0 = self._current[-1]
        n = _n_segments([p0, p1, p2], self.tolerance, 2)
        t = np.linspace(0.0, 1.0, n + 1)[1:]
        a = np.array(p0), np.array(p1), np.array(p2)
        tt = t[:, None]
        pts = (1 - tt) ** 2 * a[0] + 2 * (1 - tt) * tt * a[1] + tt**2 * a[2]
        self._current.extend([(float(x), float(y)) for x, y in pts])

    def _closePath(self):
        self._flush()

    def _endPath(self):
        self._flush()

    def _flush(self):
        if len(self._current) >= 3:
            c = self._current
            if c[0] != c[-1]:
                c = c + [c[0]]
            self.contours.append(c)
        self._current = []

    def finish(self) -> list[np.ndarray]:
        self._flush()
        return [np.asarray(c, dtype=np.float64) for c in self.contours]


@dataclass
class GlyphOutline:
    """Flattened outline of one glyph in font units (y-UP, origin = baseline)."""

    glyph_id: int
    glyph_name: str
    upem: int
    contours: list[np.ndarray]
    advance: float

    @property
    def is_empty(self) -> bool:
        return not self.contours

    def bbox(self) -> tuple[float, float, float, float] | None:
        return outline_bbox(self.contours)

    def transformed(self, dx: float, dy: float) -> list[np.ndarray]:
        return [c + np.array([dx, dy]) for c in self.contours]


def outline_bbox(contours: Sequence[np.ndarray]) -> tuple[float, float, float, float] | None:
    """(x_min, y_min, x_max, y_max) over all contours, or None if empty."""
    pts = [c for c in contours if len(c)]
    if not pts:
        return None
    allp = np.vstack(pts)
    return (
        float(allp[:, 0].min()),
        float(allp[:, 1].min()),
        float(allp[:, 0].max()),
        float(allp[:, 1].max()),
    )


def extract_outline(font: LoadedFont, glyph_id: int, tolerance_em: float = 0.0015) -> GlyphOutline:
    """Extract and flatten the outline of ``glyph_id``.

    ``tolerance_em`` is the maximum flattening error in em units; the default
    (1.5/1000 em) is far below the analysis raster resolution, so flattening is
    never the limiting factor for skeleton quality.
    """
    name = font.glyph_name(glyph_id)
    gs = font.glyph_set
    tol = max(tolerance_em * font.upem, 0.05)
    contours: list[np.ndarray] = []
    advance = 0.0
    try:
        glyph = gs[name]
        pen = ContourPen(gs, tolerance=tol)
        glyph.draw(pen)
        contours = pen.finish()
        advance = float(getattr(glyph, "width", 0.0))
    except KeyError:
        pass
    return GlyphOutline(
        glyph_id=glyph_id,
        glyph_name=name,
        upem=font.upem,
        contours=contours,
        advance=advance,
    )


_OUTLINE_CACHE: dict[tuple[str, int], GlyphOutline] = {}


def extract_outline_cached(font: LoadedFont, glyph_id: int, tolerance_em: float = 0.0015) -> GlyphOutline:
    key = (f"{font.key}:{tuple(sorted(font.variations.items()))}", glyph_id)
    hit = _OUTLINE_CACHE.get(key)
    if hit is None:
        hit = extract_outline(font, glyph_id, tolerance_em)
        _OUTLINE_CACHE[key] = hit
    return hit
