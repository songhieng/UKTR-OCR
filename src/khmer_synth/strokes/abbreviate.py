"""Reduction of the printed letterform toward a practised hand.

Everything upstream of this module produces the *textbook* shape of a letter,
because the geometry is the medial axis of a font outline.  Wobbling that shape
- however cleverly - does not produce adult handwriting; it produces a carefully
copied textbook shape with wobble, which is what a child writing from the board
actually does.  An adult hand writes a genuinely **different, reduced** shape:

* loops that a typeface closes are pinched shut into cusps,
* subscripts (coeng) shrink and pull up toward the base consonant,
* fine detail that costs pen time and buys no legibility disappears.

This module applies those three reductions in line space (pixels, baseline at
y = 0, y down), scaled by ``WriterStyle.abbreviation``, before stroke connection
runs - a reduced letterform has shorter gaps, so it also connects more readily.

Like :mod:`khmer_synth.writer.deformation`, the amount is drawn from a
*per-allograph* stream keyed on ``(writer, cluster text)``: how far this writer
abbreviates a given letter is part of the hand, not per-occurrence noise.
"""

from __future__ import annotations

import numpy as np

from ..config import AbbreviationConfig
from ..utils.geometry import cumulative_length, rdp
from ..utils.rng import make_rng
from ..writer.style import WriterStyle
from .planner import Stroke

__all__ = ["abbreviate_strokes", "collapse_loops", "compress_below_baseline", "soften_detail"]


# --------------------------------------------------------------------------- #
# individual reductions
# --------------------------------------------------------------------------- #


def collapse_loops(
    pts: np.ndarray, amount: float, cfg: AbbreviationConfig, em_px: float
) -> np.ndarray:
    """Pinch small closed loops shut, leaving a cusp.

    A loop is a sub-path whose two ends come back within ``loop_close_em`` of
    each other after between ``loop_min_em`` and ``loop_max_em`` of travel, *and*
    whose arc length exceeds ``loop_min_ratio`` times that closing distance.
    The tortuosity test is what distinguishes a loop from a straight run - a
    point 0.05 em along a straight line is also 0.05 em away from where it
    started.  The upper bound matters too: past it the "loop" is the body of the
    letter, and a writer who collapsed that would be writing a different letter.

    The pinch is tapered smoothly to zero at both ends of the sub-path, so the
    stroke stays a single connected polyline with no kink where the loop begins -
    the loop closes, the path does not break.
    """
    if amount <= 0.0 or len(pts) < 8:
        return pts
    s = cumulative_length(pts)
    if s[-1] <= 0:
        return pts

    lo = cfg.loop_min_em * em_px
    hi = cfg.loop_max_em * em_px
    close = cfg.loop_close_em * em_px
    strength = cfg.loop_max_collapse * amount

    out = pts.astype(np.float64, copy=True)
    n = len(pts)
    i = 0
    while i < n - 3:
        j0 = int(np.searchsorted(s, s[i] + lo))
        if j0 >= n:
            break
        j1 = min(int(np.searchsorted(s, s[i] + hi)), n - 1)
        if j1 <= j0:
            i += 1
            continue
        d = np.linalg.norm(pts[j0 : j1 + 1] - pts[i], axis=1)
        k = int(np.argmin(d))
        j = j0 + k
        arc = s[j] - s[i]
        if d[k] > close or arc < cfg.loop_min_ratio * d[k]:
            i += 1
            continue

        seg = out[i : j + 1]
        centre = seg.mean(axis=0)
        u = (s[i : j + 1] - s[i]) / max(s[j] - s[i], 1e-9)
        # sin^2 rather than sin: it is zero *and* flat at both ends, so the
        # pinch blends into the untouched path instead of putting a visible
        # kink at the point where the loop begins.
        taper = (np.sin(np.pi * u) ** 2)[:, None]
        out[i : j + 1] = seg + (centre - seg) * strength * taper
        i = j + 1
    return out


def compress_below_baseline(pts: np.ndarray, factor: float, depth: float) -> np.ndarray:
    """Pull everything below the baseline up toward it.

    Applied to *points*, not whole strokes.  Per-stroke detection does not work:
    the planner routinely traces a base consonant and its coeng in one stroke, so
    almost nothing is ever "a subscript stroke" (measured: 2 of 29 on a line of
    stacked clusters).  A continuous piecewise map has no such problem, is exact
    at the boundary - the displacement is zero at ``depth``, so strokes crossing
    the baseline stay connected - and it compresses below-base vowel signs along
    with the coeng, which is what a practised hand does to the whole descender
    zone.
    """
    out = pts.astype(np.float64, copy=True)
    below = out[:, 1] > depth
    if np.any(below):
        out[below, 1] = depth + (out[below, 1] - depth) * factor
    return out


def soften_detail(
    pts: np.ndarray, amount: float, cfg: AbbreviationConfig, em_px: float
) -> np.ndarray:
    """Drop fine detail with Douglas-Peucker, in proportion to ``amount``."""
    eps = cfg.detail_epsilon_em * amount * em_px
    if eps <= 1e-9 or len(pts) <= cfg.min_points:
        return pts
    out = rdp(np.asarray(pts, dtype=np.float64), eps)
    if len(out) < cfg.min_points:
        step = max(1, len(pts) // cfg.min_points)
        out = pts[::step]
    return out


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #


def abbreviate_strokes(
    strokes: list[Stroke],
    style: WriterStyle,
    cfg: AbbreviationConfig,
    em_px: float,
    cluster_keys: dict[int, str] | None = None,
    allograph_seed: int = 0,
) -> list[Stroke]:
    """Reduce every cluster's letterform according to the writer's hand."""
    if not cfg.enabled or style.abbreviation <= 0.0 or not strokes:
        return strokes

    by_cluster: dict[int, list[int]] = {}
    for i, s in enumerate(strokes):
        by_cluster.setdefault(s.cluster, []).append(i)

    out = [
        Stroke(
            np.asarray(s.points, np.float64),
            list(s.edges),
            s.component,
            s.cluster,
            list(s.ligature_spans),
        )
        for s in strokes
    ]

    depth = cfg.subscript_min_depth_em * em_px
    for cluster, idx in by_cluster.items():
        key = (cluster_keys or {}).get(cluster)
        if key:
            rng = make_rng("abbrev", allograph_seed, style.writer_id, key)
            amount = float(
                np.clip(style.abbreviation * (1.0 + rng.normal(0.0, cfg.amount_jitter)), 0.0, 1.0)
            )
        else:
            amount = float(np.clip(style.abbreviation, 0.0, 1.0))
        if amount <= 0.0:
            continue

        sub_factor = 1.0 - cfg.subscript_max_shrink * amount
        for i in idx:
            p = out[i].points
            if len(p) < 2:
                continue
            p = collapse_loops(p, amount, cfg, em_px)
            p = compress_below_baseline(p, sub_factor, depth)
            out[i].points = soften_detail(p, amount, cfg, em_px)

    return [s for s in out if len(s.points) >= 2]
