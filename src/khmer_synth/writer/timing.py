"""Synthetic pen timing.

Constant-speed timestamps are the single most obvious tell of synthetic online
handwriting, and they teach a downstream model a velocity distribution that no
real pen ever produces.  We instead derive velocity from the geometry, following
the well-established relation between curvature and speed in human drawing
(the "two-thirds power law": angular velocity is roughly constant, so tangential
speed falls as curvature rises):

    v(s) = v_base / (1 + lambda * kappa_em(s))          [px / ms]

on top of which we apply

* an acceleration / deceleration ramp at both ends of every stroke,
* a smooth (correlated) multiplicative speed noise per stroke,
* a faster pen-up transit plus a constant reaction time between strokes.

Timestamps are integer-friendly milliseconds and strictly increasing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import TimingConfig
from ..strokes.planner import Stroke
from ..utils.geometry import curvature
from ..utils.noise import smooth_noise_along
from .style import WriterStyle


def synth_timing(
    strokes: Sequence[Stroke],
    style: WriterStyle,
    cfg: TimingConfig,
    rng: np.random.Generator,
    em_px: float,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Return ``(times_ms per stroke, speeds px/ms per stroke, total ms)``."""
    v_base = cfg.base_speed_em_per_s * max(style.writing_speed, 1e-3) * em_px / 1000.0
    v_base = max(v_base, 1e-4)

    times: list[np.ndarray] = []
    speeds: list[np.ndarray] = []
    t_now = 0.0
    prev_end = None

    for s in strokes:
        p = np.asarray(s.points, dtype=np.float64)
        n = len(p)
        if n == 0:
            times.append(np.zeros(0))
            speeds.append(np.zeros(0))
            continue

        if prev_end is not None:
            d = float(np.linalg.norm(p[0] - prev_end))
            t_now += cfg.pen_up_reaction_ms + d / max(v_base * cfg.pen_up_speed_factor, 1e-6)

        if n == 1:  # a dot: a short dwell
            times.append(np.array([t_now]))
            speeds.append(np.array([0.0]))
            t_now += 30.0
            prev_end = p[0]
            continue

        ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(ds)])
        total = float(s_cum[-1])

        # curvature in 1/px -> dimensionless by multiplying with the em size
        kappa = curvature(p) * em_px
        v = v_base / np.power(
            1.0 + cfg.curvature_lambda * np.clip(kappa, 0.0, 60.0), cfg.curvature_exponent
        )

        # start / end ramps
        if total > 1e-9 and cfg.ramp_fraction > 0:
            u = s_cum / total
            r = np.minimum(u / cfg.ramp_fraction, (1.0 - u) / cfg.ramp_fraction)
            r = np.clip(r, 0.0, 1.0)
            # a fluent hand is still moving when the pen lands and lifts
            end = cfg.end_speed_factor + (1.0 - cfg.end_speed_factor) * 0.6 * style.fluency
            ramp = end + (1.0 - end) * np.sqrt(r)
            v = v * ramp

        # correlated speed noise (per stroke)
        sigma = cfg.speed_noise_sigma * (0.5 + style.speed_variation)
        if sigma > 0:
            g = smooth_noise_along(rng, s_cum, scale=max(cfg.speed_noise_scale_px, 1.0), amplitude=sigma)
            v = v * np.exp(np.clip(g, -1.5, 1.5))

        v = np.maximum(v, v_base * 0.05)

        # dt from the mean speed over each segment
        v_seg = 0.5 * (v[:-1] + v[1:])
        dt = np.clip(ds / np.maximum(v_seg, 1e-9), cfg.min_dt_ms, cfg.max_dt_ms)
        t = t_now + np.concatenate([[0.0], np.cumsum(dt)])

        times.append(t)
        speeds.append(v)
        t_now = float(t[-1])
        prev_end = p[-1]

    return times, speeds, t_now


def check_monotonic(times: Sequence[np.ndarray]) -> bool:
    prev = -np.inf
    for t in times:
        if len(t) == 0:
            continue
        if not np.all(np.diff(t) > 0) or t[0] <= prev:
            if len(t) > 1 and not np.all(np.diff(t) > 0):
                return False
            if t[0] <= prev:
                return False
        prev = float(t[-1])
    return True
