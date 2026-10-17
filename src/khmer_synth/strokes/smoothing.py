"""Skeleton polyline -> smooth, uniformly sampled pen trajectory.

A raw skeleton path is staircased and vertex-dense; drawn directly it looks
mechanical.  We smooth it, but with two safeguards that matter for Khmer:

1. **Corner preservation.**  Vertices whose turn angle exceeds
   ``corner_angle_deg`` are treated as hard corners and the spline is fitted
   piecewise between them.  Without this, the sharp junctions and the small
   angular hooks that distinguish several Khmer consonants get rounded away.
2. **Centripetal Catmull-Rom** (alpha = 0.5) rather than the uniform variant,
   which provably cannot form cusps or self-intersections on tight turns.

The output is resampled to an approximately constant arc-length spacing
(``point_spacing_px``), which is what the timing model assumes.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ..config import SmoothingConfig
from ..utils.geometry import (
    catmull_rom,
    dedupe_consecutive,
    polyline_length,
    resample_uniform,
    turn_angles,
)
from .planner import Stroke, StrokePlan


def _split_at_corners(pts: np.ndarray, angle_deg: float) -> list[np.ndarray]:
    if len(pts) < 3 or angle_deg <= 0:
        return [pts]
    ang = np.abs(turn_angles(pts))
    corners = np.nonzero(ang > np.radians(angle_deg))[0] + 1
    if len(corners) == 0:
        return [pts]
    pieces: list[np.ndarray] = []
    prev = 0
    for c in corners:
        if c - prev >= 1:
            pieces.append(pts[prev : c + 1])
            prev = c
    pieces.append(pts[prev:])
    return [p for p in pieces if len(p) >= 2]


def _bspline(pts: np.ndarray, samples_per_segment: int = 12) -> np.ndarray:
    """Uniform cubic B-spline (approximating, not interpolating).

    Smoother than Catmull-Rom but pulls away from the control polygon, so it is
    only offered as an option for deliberately "loose" writers.
    """
    if len(pts) < 4:
        return pts
    p = np.vstack([pts[:1], pts[:1], pts, pts[-1:], pts[-1:]])
    out = []
    t = np.linspace(0, 1, samples_per_segment, endpoint=False)[:, None]
    b0 = ((1 - t) ** 3) / 6.0
    b1 = (3 * t**3 - 6 * t**2 + 4) / 6.0
    b2 = (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6.0
    b3 = (t**3) / 6.0
    for i in range(len(p) - 3):
        out.append(b0 * p[i] + b1 * p[i + 1] + b2 * p[i + 2] + b3 * p[i + 3])
    out.append(pts[-1][None, :])
    return np.vstack(out)


def smooth_stroke(
    points: np.ndarray,
    cfg: SmoothingConfig,
    spacing: float | None = None,
    smoothness: float = 1.0,
    corner_angle_deg: float | None = None,
) -> np.ndarray:
    """Smooth + resample one stroke.

    ``smoothness`` in [0, 1] scales the effect.  ``corner_angle_deg`` overrides
    the config threshold above which a vertex is preserved as a hard corner -
    raising it makes the hand round off corners, which is what writing at speed
    does.
    """
    pts = dedupe_consecutive(np.asarray(points, dtype=np.float64))
    spacing = float(spacing if spacing is not None else cfg.point_spacing_px)
    if len(pts) < 2:
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    if polyline_length(pts) < 1e-9:
        return pts[:1]

    method = cfg.method if smoothness > 0.02 else "none"
    if method == "none":
        out = pts
    else:
        angle = cfg.corner_angle_deg if corner_angle_deg is None else float(corner_angle_deg)
        pieces = _split_at_corners(pts, angle)
        smoothed: list[np.ndarray] = []
        for piece in pieces:
            if len(piece) < 3:
                smoothed.append(piece)
                continue
            if method == "catmull_rom":
                s = catmull_rom(piece, samples_per_segment=10, alpha=cfg.alpha)
            elif method == "bspline":
                s = _bspline(piece)
            else:
                s = piece
            smoothed.append(s)
        out = smoothed[0]
        for s in smoothed[1:]:
            out = np.vstack([out, s[1:]])
        if smoothness < 1.0:
            # blend between the raw polyline and the smoothed curve
            raw = resample_uniform(pts, spacing)
            sm = resample_uniform(out, spacing)
            n = min(len(raw), len(sm))
            if n >= 2:
                out = (1.0 - smoothness) * raw[:n] + smoothness * sm[:n]

    out = resample_uniform(out, spacing, min_points=cfg.min_points)

    if cfg.savgol_window >= 5 and len(out) > cfg.savgol_window and cfg.savgol_mix > 0:
        w = int(cfg.savgol_window) | 1
        try:
            f = savgol_filter(out, window_length=w, polyorder=cfg.savgol_poly, axis=0, mode="nearest")
            mix = float(np.clip(cfg.savgol_mix * smoothness, 0.0, 1.0))
            f[0] = out[0]
            f[-1] = out[-1]
            out = (1 - mix) * out + mix * f
        except ValueError:  # pragma: no cover - window/poly mismatch
            pass
    return out


def smooth_plan(
    plan: StrokePlan, cfg: SmoothingConfig, spacing: float | None = None, smoothness: float = 1.0
) -> StrokePlan:
    strokes = []
    for s in plan.strokes:
        pts = smooth_stroke(s.points, cfg, spacing=spacing, smoothness=smoothness)
        strokes.append(Stroke(points=pts.astype(np.float32), edges=list(s.edges), component=s.component))
    return StrokePlan(strokes=strokes, cost=plan.cost, breakdown=dict(plan.breakdown))
