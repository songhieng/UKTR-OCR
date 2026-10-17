"""Stroke connection - the main visual cue of *fast* handwriting.

A careful writer lifts the pen between every stroke.  A fast writer does not:
the pen stays on the paper (or barely leaves it) and drags a light connecting
ligature into the next stroke.  That single behaviour is most of what separates
"neatly printed" from "written at speed", and it is also why fast handwriting has
far fewer, much longer strokes than a font-derived skeleton suggests.

Given the planned stroke sequence, this module walks consecutive pairs and, when
the pen-up gap is short enough, splices them into one stroke joined by a cubic
Bezier whose tangents continue the outgoing and incoming stroke directions.  The
spliced-in region is recorded in ``Stroke.ligature_spans`` (as fractions of arc
length) so that:

* the pressure model can draw it lighter than real glyph ink, and
* the renderer therefore draws it thinner - exactly like a real carry-over.

Connections are made in *layout* order, so the ligature can cross from one
grapheme cluster into the next, which is what actually happens on paper.
"""

from __future__ import annotations

import numpy as np

from ..utils.geometry import cubic_bezier
from .planner import Stroke


def _exit_dir(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.array([1.0, 0.0])
    d = pts[-1] - pts[max(0, len(pts) - 5)]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def _entry_dir(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.array([1.0, 0.0])
    d = pts[min(4, len(pts) - 1)] - pts[0]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def _ligature(a_end: np.ndarray, a_dir: np.ndarray, b_start: np.ndarray, b_dir: np.ndarray, spacing: float) -> np.ndarray:
    """Cubic Bezier carry-over from the end of one stroke into the next.

    The control-point extension is scaled by how much each stroke's tangent
    actually points *towards* the other end.  With a fixed extension, two strokes
    whose tangents oppose each other produce a Bezier that doubles back on
    itself, drawing a conspicuous loop through the glyph.
    """
    gap = float(np.linalg.norm(b_start - a_end))
    if gap < 1e-9:
        return np.zeros((0, 2))
    u = (b_start - a_end) / gap
    fa = float(np.clip(np.dot(a_dir, u), 0.0, 1.0))
    fb = float(np.clip(np.dot(b_dir, u), 0.0, 1.0))
    c1 = a_end + a_dir * (gap * (0.12 + 0.28 * fa))
    c2 = b_start - b_dir * (gap * (0.12 + 0.28 * fb))
    n = int(np.clip(gap / max(spacing, 0.5), 3, 64))
    pts = cubic_bezier(a_end, c1, c2, b_start, n)
    return pts[1:-1] if len(pts) > 2 else np.zeros((0, 2))


def connect_strokes(
    strokes: list[Stroke],
    rng: np.random.Generator,
    em_px: float,
    probability: float,
    max_gap_em: float,
    spacing_px: float = 2.0,
    backward_penalty: float = 0.55,
    max_chain: int = 5,
    vertical_ratio: float = 0.95,
    max_rise_em: float = 0.42,
) -> list[Stroke]:
    """Splice consecutive strokes separated by a short gap into single strokes.

    ``probability``   chance of carrying the pen through an eligible gap
    ``max_gap_em``    longest gap that may be bridged
    ``backward_penalty`` multiplies the probability when the next stroke starts
                      to the *left* of the current pen position (writers do carry
                      backwards, but far less often than forwards)
    ``max_chain``     safety cap on how many strokes may be fused into one
    ``vertical_ratio`` reject hops steeper than this |dy|/|dx| ratio
    ``max_rise_em``   reject hops that climb/drop more than this many em
    """
    if not strokes or probability <= 0.0 or max_gap_em <= 0.0:
        return strokes

    max_gap = max_gap_em * em_px
    out: list[Stroke] = []
    cur: Stroke | None = None
    # arc lengths of the pieces making up ``cur``, and which of them are ligature
    seg_lengths: list[float] = []
    seg_is_lig: list[bool] = []
    chain = 0

    def flush() -> None:
        nonlocal cur, seg_lengths, seg_is_lig, chain
        if cur is None:
            return
        total = float(sum(seg_lengths))
        spans: list[tuple[float, float]] = []
        if total > 1e-9:
            acc = 0.0
            for L, is_lig in zip(seg_lengths, seg_is_lig):
                if is_lig and L > 0:
                    spans.append((acc / total, (acc + L) / total))
                acc += L
        cur.ligature_spans = spans
        out.append(cur)
        cur = None
        seg_lengths = []
        seg_is_lig = []
        chain = 0

    def arc(p: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0

    for s in strokes:
        pts = np.asarray(s.points, dtype=np.float64)
        if len(pts) == 0:
            continue
        if cur is None:
            cur = Stroke(pts.copy(), list(s.edges), s.component, s.cluster)
            seg_lengths = [arc(pts)]
            seg_is_lig = [False]
            chain = 1
            continue

        a_end = cur.points[-1]
        b_start = pts[0]
        delta = b_start - a_end
        gap = float(np.linalg.norm(delta))
        dx, dy = float(delta[0]), float(delta[1])

        # Short gaps are carried over almost always, long ones rarely: the
        # probability decays with gap length rather than being a hard cutoff, so
        # a writer connects neighbouring strokes but still lifts to cross the
        # page.  (Median pen-up gap in a Khmer line is ~0.55 em, so the ceiling
        # has to be around a character width, not a fraction of one.)
        p_connect = probability * (1.0 - 0.55 * min(gap / max_gap, 1.0))
        if dx < 0:
            p_connect *= backward_penalty

        # A real carry-over runs roughly along the writing direction, from the
        # end of one letter into the start of the next.  Connecting a base
        # consonant to a diacritic sitting above or below it instead drags a long
        # swooping line straight through the glyph and destroys legibility -
        # writers lift for those.  So: reject predominantly *vertical* hops, and
        # reject any hop that climbs or drops more than `max_rise`.
        too_steep = abs(dy) > vertical_ratio * max(abs(dx), 1e-6)
        too_tall = abs(dy) > max_rise_em * em_px
        if (
            gap > max_gap
            or too_steep
            or too_tall
            or chain >= max_chain
            or len(pts) < 2
            or len(cur.points) < 2
        ):
            p_connect = 0.0

        if p_connect > 0.0 and rng.random() < p_connect:
            lig = _ligature(a_end, _exit_dir(cur.points), b_start, _entry_dir(pts), spacing_px)
            joined = np.vstack([cur.points, lig, pts]) if len(lig) else np.vstack([cur.points, pts])
            seg_lengths.append(arc(np.vstack([a_end[None, :], lig, b_start[None, :]])))
            seg_is_lig.append(True)
            seg_lengths.append(arc(pts))
            seg_is_lig.append(False)
            cur.points = joined
            cur.edges = list(cur.edges) + list(s.edges)
            chain += 1
        else:
            flush()
            cur = Stroke(pts.copy(), list(s.edges), s.component, s.cluster)
            seg_lengths = [arc(pts)]
            seg_is_lig = [False]
            chain = 1

    flush()
    return out


def ligature_mask(points: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    """Per-point boolean mask of the ligature regions of a stroke.

    Recomputed from *current* arc length, so it stays correct after the stroke
    has been smoothed, resampled and deformed.
    """
    n = len(points)
    mask = np.zeros(n, dtype=bool)
    if not spans or n < 2:
        return mask
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total <= 1e-9:
        return mask
    u = s / total
    for a, b in spans:
        mask |= (u >= a) & (u <= b)
    return mask
