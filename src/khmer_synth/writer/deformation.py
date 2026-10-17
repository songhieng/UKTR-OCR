"""Multi-scale handwriting deformation.

Deformation is applied at five nested scales, because that is how handwriting
variability actually decomposes:

============  ===============================================================
global        slant, x/y scale, small rotation - constant for a writer
baseline      drift + sinusoid + random walk along the line
cluster       per-grapheme translation / rotation / scale / slant wobble
stroke        overshoot, shortening, low-frequency bending (elasticity)
point         correlated jitter (a smooth 2-D field) + high-frequency tremor
============  ===============================================================

The key modelling choice is that point-level noise is **spatially correlated**,
not i.i.d.:  a shared smooth 2-D displacement field is sampled at every point of
every stroke, so neighbouring strokes distort *together*.  Independent per-point
Gaussian noise produces the unmistakable "wobbly font" look and, worse, teaches a
downstream model that stroke geometry is high-frequency noise.

Coordinates here are "line space": pixels, x increasing left to right, y = 0 on
the baseline (y down).  Fitting into the final canvas happens afterwards, so
deformation can never push ink outside the image.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import DeformationConfig
from ..strokes.planner import Stroke
from ..utils.geometry import make_affine, tangents, transform_affine
from ..utils.noise import smooth_noise_1d, smooth_noise_2d, smooth_noise_along
from .style import WriterStyle

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class _VectorField:
    """A smooth 2-D displacement field sampled with bilinear interpolation."""

    def __init__(self, rng: np.random.Generator, bbox, scale_px: float, amplitude: float, step: float = 4.0):
        x0, y0, x1, y1 = bbox
        self.x0, self.y0 = x0 - scale_px, y0 - scale_px
        self.step = max(step, 1.0)
        w = int(max(4, np.ceil((x1 - x0 + 2 * scale_px) / self.step)))
        h = int(max(4, np.ceil((y1 - y0 + 2 * scale_px) / self.step)))
        s = max(scale_px / self.step, 0.8)
        self.fx = smooth_noise_2d(rng, (h, w), scale=s, amplitude=amplitude, octaves=2)
        self.fy = smooth_noise_2d(rng, (h, w), scale=s, amplitude=amplitude, octaves=2)

    def sample(self, pts: np.ndarray) -> np.ndarray:
        gx = (pts[:, 0] - self.x0) / self.step
        gy = (pts[:, 1] - self.y0) / self.step
        h, w = self.fx.shape
        gx = np.clip(gx, 0, w - 1.001)
        gy = np.clip(gy, 0, h - 1.001)
        x0 = np.floor(gx).astype(int)
        y0 = np.floor(gy).astype(int)
        tx = (gx - x0)[:, None]
        ty = (gy - y0)[:, None]
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        f = np.stack([self.fx, self.fy], axis=-1)
        v = (
            f[y0, x0] * (1 - ty) * (1 - tx)
            + f[y0, x1] * (1 - ty) * tx
            + f[y1, x0] * ty * (1 - tx)
            + f[y1, x1] * ty * tx
        )
        return v


def _bbox_of(strokes: Sequence[Stroke]) -> tuple[float, float, float, float]:
    pts = np.vstack([s.points for s in strokes if len(s.points)])
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def _extend_endpoints(pts: np.ndarray, start: float, end: float) -> np.ndarray:
    """Overshoot (>0) or shorten (<0) a stroke along its end tangents."""
    if len(pts) < 3 or (abs(start) < 1e-6 and abs(end) < 1e-6):
        return pts
    out = pts.copy()
    t0 = pts[min(2, len(pts) - 1)] - pts[0]
    n0 = np.linalg.norm(t0)
    t1 = pts[-1] - pts[max(0, len(pts) - 3)]
    n1 = np.linalg.norm(t1)
    if n0 > 1e-9 and start != 0:
        out = np.vstack([pts[0] - t0 / n0 * start, out])
    if n1 > 1e-9 and end != 0:
        out = np.vstack([out, pts[-1] + t1 / n1 * end])
    return out


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #


def deform_strokes(
    strokes: list[Stroke],
    style: WriterStyle,
    cfg: DeformationConfig,
    rng: np.random.Generator,
    em_px: float,
) -> list[Stroke]:
    """Apply the full deformation stack in line space (baseline at y = 0)."""
    strokes = [
        Stroke(
            np.asarray(s.points, np.float64), list(s.edges), s.component, s.cluster, list(s.ligature_spans)
        )
        for s in strokes
    ]
    strokes = [s for s in strokes if len(s.points)]
    if not strokes:
        return strokes

    max_disp = cfg.max_displacement_em * em_px

    # ---------------- cluster level ---------------- #
    if cfg.cluster_level:
        by_cluster: dict[int, list[int]] = {}
        for i, s in enumerate(strokes):
            by_cluster.setdefault(s.cluster, []).append(i)
        for idx in by_cluster.values():
            pts = np.vstack([strokes[i].points for i in idx])
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            rot = float(rng.normal(0.0, style.glyph_rotation))
            sc = 1.0 + float(rng.normal(0.0, style.glyph_scale_var))
            scy = 1.0 + float(rng.normal(0.0, style.glyph_scale_var))
            tx = float(rng.normal(0.0, style.glyph_translation * em_px))
            ty = float(rng.normal(0.0, style.glyph_translation * em_px))
            shear = float(rng.normal(0.0, style.slant_variation))
            m = make_affine(
                scale=(sc, scy), rotation=rot, shear_x=shear, translate=(tx, ty), center=(cx, cy)
            )
            for i in idx:
                strokes[i].points = transform_affine(strokes[i].points, m)

    # ---------------- stroke level ---------------- #
    if cfg.stroke_level:
        for s in strokes:
            p = s.points
            if len(p) < 3:
                continue
            if style.overshoot > 0:
                a = float(rng.normal(0.0, style.overshoot * em_px))
                b = float(rng.normal(0.0, style.overshoot * em_px))
                p = _extend_endpoints(p, np.clip(a, -max_disp, max_disp), np.clip(b, -max_disp, max_disp))
            if style.local_elasticity > 0 and len(p) > 4:
                seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
                amp = style.local_elasticity * em_px
                bend = smooth_noise_along(rng, seg, scale=max(seg[-1] * 0.6, 1.0), amplitude=amp)
                # taper to zero at the endpoints so strokes stay attached
                taper = np.sin(np.pi * np.clip(seg / max(seg[-1], 1e-9), 0, 1))
                nrm = np.stack([-tangents(p)[:, 1], tangents(p)[:, 0]], axis=1)
                p = p + nrm * (bend * taper)[:, None]
            s.points = p

    # ---------------- point level ---------------- #
    if cfg.point_level:
        bb = _bbox_of(strokes)
        if style.local_jitter > 0:
            field = _VectorField(
                rng, bb, scale_px=cfg.noise_scale_px, amplitude=style.local_jitter * em_px
            )
            for s in strokes:
                d = field.sample(s.points)
                s.points = s.points + np.clip(d, -max_disp, max_disp)
        if style.tremor > 0:
            for s in strokes:
                n = len(s.points)
                if n < 3:
                    continue
                amp = style.tremor * em_px
                dx = smooth_noise_1d(rng, n, scale=max(cfg.tremor_scale_px, 1.0), amplitude=amp)
                dy = smooth_noise_1d(rng, n, scale=max(cfg.tremor_scale_px, 1.0), amplitude=amp)
                s.points = s.points + np.clip(np.stack([dx, dy], axis=1), -max_disp, max_disp)

    # ---------------- baseline ---------------- #
    if cfg.baseline:
        bb = _bbox_of(strokes)
        x0, x1 = bb[0], bb[2]
        span = max(x1 - x0, 1e-6)
        walk_n = 128
        walk = (
            smooth_noise_1d(rng, walk_n, scale=18.0, amplitude=style.baseline_walk * em_px, octaves=2)
            if style.baseline_walk > 0
            else np.zeros(walk_n)
        )
        for s in strokes:
            x = s.points[:, 0]
            u = (x - x0) / em_px
            dy = style.baseline_amplitude * em_px * np.sin(
                2 * np.pi * style.baseline_frequency * u + style.baseline_phase
            )
            dy = dy + style.baseline_drift * (x - x0)
            if style.baseline_walk > 0:
                g = np.clip((x - x0) / span * (walk_n - 1), 0, walk_n - 1)
                dy = dy + np.interp(g, np.arange(walk_n), walk)
            s.points = s.points + np.stack([np.zeros_like(dy), dy], axis=1)

    # ---------------- global ---------------- #
    if cfg.global_transform:
        bb = _bbox_of(strokes)
        cx = 0.5 * (bb[0] + bb[2])
        m = make_affine(
            scale=(style.horizontal_scale, style.vertical_scale),
            rotation=style.rotation,
            shear_x=-style.slant,  # x' = x - slant * y ; y is negative above the baseline
            translate=(0.0, 0.0),
            center=(cx, 0.0),
        )
        for s in strokes:
            s.points = transform_affine(s.points, m)

    for s in strokes:
        s.points = np.asarray(s.points, dtype=np.float64)
    return strokes
