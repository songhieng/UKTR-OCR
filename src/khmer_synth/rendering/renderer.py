"""Trajectory -> anti-aliased ink image.

Naively drawing constant-width OpenCV polylines produces the flat, obviously
synthetic look this project is trying to avoid.  Instead each stroke is rendered
as a swept disk of *varying radius and opacity*:

1. the stroke centreline is drawn 1-px thin into a local ROI, together with a
   per-centreline-pixel radius map and opacity map;
2. ``cv2.distanceTransformWithLabels`` gives, for every pixel of the ROI, the
   Euclidean distance to the nearest centreline pixel **and which one it is**;
3. coverage is then the analytic disk footprint
   ``clip(radius(nearest) + 0.5 - distance, 0, 1)`` multiplied by
   ``opacity(nearest)``.

That yields exact round caps and joins, continuous width modulation, per-point
darkness, and costs a single OpenCV pass per stroke instead of one draw call per
sample point.  Supersampling (``RenderConfig.supersample``) is applied on top.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import RenderConfig
from ..dataset.schema import Trajectory, TrajectoryStroke
from ..utils.noise import smooth_noise_2d
from ..utils.rng import uniform
from ..writer.style import WriterStyle
from .ink import PenModel, get_ink_color, get_pen


def _stroke_radius_alpha(
    stroke: TrajectoryStroke, base_radius: float, pen: PenModel, cfg: RenderConfig
) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(stroke.pressure, 0.0, 1.0)
    wp = cfg.width_pressure_bias + cfg.width_pressure_gain * p
    wp = 1.0 + pen.width_pressure * (wp - 1.0)
    radius = base_radius * pen.width_scale * wp

    if pen.speed_thinning > 0 and len(stroke.points) > 2:
        dt = np.diff(stroke.time_ms, prepend=stroke.time_ms[0])
        ds = np.concatenate([[0.0], np.linalg.norm(np.diff(stroke.points, axis=0), axis=1)])
        v = ds / np.maximum(dt, 1e-6)
        vm = float(np.median(v[v > 0])) if np.any(v > 0) else 1.0
        radius = radius * (1.0 - pen.speed_thinning * np.clip(v / max(vm, 1e-9) - 1.0, -0.8, 0.8))

    if pen.nib_ratio < 1.0 and len(stroke.points) > 1:
        d = np.gradient(stroke.points, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0]) - np.radians(cfg.nib_angle_deg)
        radius = radius * (pen.nib_ratio + (1.0 - pen.nib_ratio) * np.abs(np.sin(ang)))

    ap = cfg.alpha_pressure_bias + cfg.alpha_pressure_gain * p
    ap = 1.0 + pen.alpha_pressure * (ap - 1.0)
    alpha = np.clip(pen.alpha * ap, 0.0, 1.0)
    return np.maximum(radius, 0.35), alpha


def _render_stroke(
    canvas: np.ndarray,
    stroke: TrajectoryStroke,
    scale: float,
    base_radius: float,
    pen: PenModel,
    cfg: RenderConfig,
) -> None:
    """Composite one stroke into ``canvas`` (float32 alpha, supersampled)."""
    pts = stroke.points * scale
    if len(pts) == 0:
        return
    radius, alpha = _stroke_radius_alpha(stroke, base_radius, pen, cfg)
    radius = radius * scale

    H, W = canvas.shape
    pad = int(np.ceil(float(radius.max()))) + 3
    x0 = int(np.floor(pts[:, 0].min())) - pad
    y0 = int(np.floor(pts[:, 1].min())) - pad
    x1 = int(np.ceil(pts[:, 0].max())) + pad
    y1 = int(np.ceil(pts[:, 1].max())) + pad
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, W), min(y1, H)
    if x1c <= x0c or y1c <= y0c:
        return

    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0 or w * h > 64_000_000:
        return

    mask = np.zeros((h, w), dtype=np.uint8)
    rad_img = np.zeros((h, w), dtype=np.float32)
    alp_img = np.zeros((h, w), dtype=np.float32)

    local = pts - np.array([x0, y0], dtype=np.float64)

    # Densify so that consecutive stamp centres are closer than the pen radius:
    # the ink is the union of disks around the sample points, so a spacing below
    # ~half the radius makes the swept shape indistinguishable from a continuous
    # sweep while keeping the point count (and therefore the cost) low.
    if len(local) > 1:
        ds = np.linalg.norm(np.diff(local, axis=0), axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(ds)])
        total = float(s_cum[-1])
        step = max(float(np.min(radius)) * 0.5, 0.5)
        n = int(np.clip(total / step + 1, 2, 20000))
        u = np.linspace(0.0, total, n)
        lx = np.interp(u, s_cum, local[:, 0])
        ly = np.interp(u, s_cum, local[:, 1])
        rr = np.interp(u, s_cum, radius)
        aa = np.interp(u, s_cum, alpha)
    else:
        lx, ly, rr, aa = local[:, 0], local[:, 1], radius, alpha

    ix = np.clip(np.rint(lx).astype(np.int32), 0, w - 1)
    iy = np.clip(np.rint(ly).astype(np.int32), 0, h - 1)
    mask[iy, ix] = 255
    rad_img[iy, ix] = rr
    alp_img[iy, ix] = aa

    dist, labels = cv2.distanceTransformWithLabels(
        255 - mask, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    on = mask > 0
    lab_on = labels[on]
    n_lab = int(labels.max()) + 1
    rad_lut = np.zeros(n_lab, dtype=np.float32)
    alp_lut = np.zeros(n_lab, dtype=np.float32)
    rad_lut[lab_on] = rad_img[on]
    alp_lut[lab_on] = alp_img[on]

    r_map = rad_lut[labels]
    a_map = alp_lut[labels]
    soft = 0.5 + pen.edge_softness * max(base_radius * scale, 1.0) * 0.5
    cov = np.clip((r_map + soft - dist) / max(2 * soft, 1e-3), 0.0, 1.0)
    ink = cov * a_map

    sub = ink[y0c - y0 : y1c - y0, x0c - x0 : x1c - x0]
    dst = canvas[y0c:y1c, x0c:x1c]
    if pen.overlap_darkening:
        np.copyto(dst, 1.0 - (1.0 - dst) * (1.0 - sub))
    else:
        np.maximum(dst, sub, out=dst)


def render_ink_alpha(
    traj: Trajectory,
    style: WriterStyle,
    cfg: RenderConfig,
    rng: np.random.Generator,
    em_px: float,
    pen: PenModel | None = None,
) -> np.ndarray:
    """Render the ink coverage (alpha) map at final resolution, in [0, 1]."""
    pen = pen or get_pen(style.pen_model)
    ss = max(1, int(cfg.supersample))
    H, W = traj.height * ss, traj.width * ss
    canvas = np.zeros((H, W), dtype=np.float32)

    base_radius = 0.5 * style.stroke_width * em_px
    for s in traj.strokes:
        if len(s) == 0:
            continue
        _render_stroke(canvas, s, ss, base_radius, pen, cfg)

    if cfg.draw_pen_up and len(traj.strokes) > 1:
        for a, b in zip(traj.strokes[:-1], traj.strokes[1:]):
            p0 = tuple(np.rint(a.points[-1] * ss).astype(int))
            p1 = tuple(np.rint(b.points[0] * ss).astype(int))
            cv2.line(canvas, p0, p1, 0.18, 1, cv2.LINE_AA)

    # ink texture (grain) - generated at half resolution and upsampled, which is
    # visually equivalent for a band-limited field and much cheaper
    tex = pen.texture * uniform(rng, cfg.texture_strength) * 4.0
    if tex > 0.01:
        noise = smooth_noise_2d(rng, (max(H // 2, 4), max(W // 2, 4)), scale=1.2, amplitude=1.0, octaves=2)
        noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_LINEAR)
        canvas *= np.clip(1.0 - tex * (0.5 + 0.5 * noise), 0.0, 1.0)

    # ink bleed into the paper
    if pen.bleed > 0 and rng.random() < cfg.bleed_probability:
        sigma = uniform(rng, cfg.bleed_sigma_px) * pen.bleed * ss
        if sigma > 0.15:
            blurred = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sigma, sigmaY=sigma)
            canvas = np.maximum(canvas, blurred * 0.75)

    if ss > 1:
        canvas = canvas.reshape(traj.height, ss, traj.width, ss).mean(axis=(1, 3))
    return np.clip(canvas, 0.0, 1.0)


def render_trajectory(
    traj: Trajectory,
    style: WriterStyle,
    cfg: RenderConfig,
    rng: np.random.Generator,
    em_px: float,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Render to an RGB uint8 image."""
    pen = get_pen(style.pen_model)
    alpha = render_ink_alpha(traj, style, cfg, rng, em_px, pen)
    if background is None:
        bg = np.ones((traj.height, traj.width, 3), dtype=np.float32)
    else:
        bg = np.asarray(background, dtype=np.float32)
        if bg.ndim == 2:
            bg = np.repeat(bg[:, :, None], 3, axis=2)
    colour = np.asarray(get_ink_color(style.ink_color, pen), dtype=np.float32)
    a = alpha[:, :, None]
    out = bg * (1.0 - a) + colour[None, None, :] * a
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
