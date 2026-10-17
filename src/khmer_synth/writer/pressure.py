"""Synthetic pen pressure.

Digitiser pressure traces have three robust properties that we reproduce:

1. **A stroke envelope** - pressure ramps up as the nib is planted, plateaus,
   then falls as the pen is lifted.  A raw ``sin(pi t)`` envelope is a poor model
   (it peaks only at the exact midpoint); we use a plateau with tapered ends.
2. **Low-frequency drift** - the writer's grip force wanders slowly; pressure is
   temporally smooth, never i.i.d. noise.
3. **Speed coupling** - the pen presses harder where it slows down (corners,
   stroke ends), which is why pressure and curvature correlate in real data.

Output is clipped to [0, 1] as required by the trajectory schema.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..config import PressureConfig
from ..strokes.connect import ligature_mask
from ..strokes.planner import Stroke
from ..utils.noise import smooth_noise_along
from .style import WriterStyle


def _envelope(
    s_cum: np.ndarray, ramp: float, exponent: float, min_ramp_points: float = 6.0, ramp_end: float | None = None
) -> np.ndarray:
    """Plateau envelope with tapered ends, in normalised arc length.

    The rise/fall is additionally forced to span at least ``min_ramp_points``
    samples.  A pen's pressure rise time is roughly constant (tens of ms), not
    proportional to how long the stroke happens to be, so a purely fractional
    ramp makes short ticks jump from 0.35 to 1.0 between two consecutive
    samples - a step no digitiser can produce.  ``ramp_end`` (default ``ramp``)
    sets the lift-off taper separately from the landing.
    """
    n = len(s_cum)
    total = float(s_cum[-1]) if n else 0.0
    if total <= 1e-9 or ramp <= 0:
        return np.ones_like(s_cum)
    lo = min_ramp_points / max(n, 1)
    ramp_end = float(np.clip(max(ramp if ramp_end is None else ramp_end, lo), 1e-6, 0.6))
    ramp = float(np.clip(max(ramp, lo), 1e-6, 0.5))
    u = s_cum / total
    r = np.clip(np.minimum(u / ramp, (1.0 - u) / ramp_end), 0.0, 1.0)
    # smoothstep: zero derivative at both ends, so the pressure eases in and out
    # instead of jumping on the first sample (a plain power < 1 rises almost
    # vertically at r = 0)
    return (r * r * (3.0 - 2.0 * r)) ** exponent


def synth_pressure(
    strokes: Sequence[Stroke],
    style: WriterStyle,
    cfg: PressureConfig,
    rng: np.random.Generator,
    speeds: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Per-stroke pressure arrays in [0, 1]."""
    out: list[np.ndarray] = []
    for i, s in enumerate(strokes):
        p = np.asarray(s.points, dtype=np.float64)
        n = len(p)
        if n == 0:
            out.append(np.zeros(0))
            continue
        if n == 1:
            out.append(np.array([float(np.clip(style.pressure_mean, cfg.min_pressure, cfg.max_pressure))]))
            continue

        ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(ds)])

        if n < 8:
            # A dot or a 2-3 point tick is a single tap: spreading the full
            # rise/plateau/fall envelope over 2 samples would produce a physically
            # impossible pressure step of ~0.5 between consecutive points.
            out.append(
                np.full(n, float(np.clip(style.pressure_mean * 0.85, cfg.min_pressure, cfg.max_pressure)))
            )
            continue

        # A fast writer plants the pen quickly and lifts it while still moving,
        # leaving a long thin exit flick; a slow writer stops before lifting, so
        # the stroke ends blunt.  The envelope floor (35 % of the mean pressure
        # for the slowest hand) and the exit ramp length both follow fluency.
        f = float(style.fluency)
        env = _envelope(
            s_cum,
            cfg.ramp_fraction * (1.0 - 0.4 * f),
            cfg.envelope_exponent,
            ramp_end=cfg.ramp_fraction * (1.0 + 2.5 * f),
        )
        floor = 0.35 - 0.30 * f
        env = floor + (1.0 - floor) * env

        base = float(style.pressure_mean)

        # Grip force and speed modulate pressure *multiplicatively*.  An additive
        # formulation drives the value negative whenever the noise and the speed
        # term both go low, which then sits pinned at the clipping floor for
        # hundreds of milliseconds - a shape no real digitiser produces.
        noise_rel = smooth_noise_along(
            rng, s_cum, scale=max(cfg.noise_scale_px, 1.0), amplitude=style.pressure_variation
        )
        noise_rel = np.clip(noise_rel, -0.45, 0.45)

        coupling_rel = 0.0
        if cfg.speed_coupling > 0 and speeds is not None and i < len(speeds) and len(speeds[i]) == n:
            v = np.asarray(speeds[i], dtype=np.float64)
            vm = float(np.median(v[v > 0])) if np.any(v > 0) else 1.0
            coupling_rel = cfg.speed_coupling * np.clip(1.0 - v / max(vm, 1e-9), -1.0, 1.0)
            # grip force has mass: it cannot track instantaneous speed changes,
            # so the coupling term is low-pass filtered before it is applied
            if n > 5:
                coupling_rel = gaussian_filter1d(
                    coupling_rel, sigma=max(cfg.coupling_smooth_points, 0.5), mode="nearest"
                )

        pres = base * env * (1.0 + noise_rel) * (1.0 + coupling_rel)

        # A connecting ligature is drawn with the pen barely touching the paper,
        # so it must be markedly lighter (and therefore, since width follows
        # pressure, thinner) than the glyph ink it joins.
        if s.ligature_spans and cfg.ligature_pressure < 1.0:
            m = ligature_mask(p, s.ligature_spans)
            if m.any():
                factor = np.where(m, cfg.ligature_pressure, 1.0)
                factor = gaussian_filter1d(factor, sigma=2.0, mode="nearest")
                pres = pres * factor
        # Final band-limit.  Pen pressure is a mechanical quantity with limited
        # bandwidth; this bounds the per-sample derivative regardless of what the
        # envelope, the noise and the speed term individually did.
        if cfg.smooth_points > 0 and n > 4:
            pres = gaussian_filter1d(pres, sigma=cfg.smooth_points, mode="nearest")
        out.append(np.clip(pres, cfg.min_pressure, cfg.max_pressure))
    return out
