"""Document / camera degradation.

Applied *on top of* the clean render, in an order that mimics the physical chain

    ink fading -> paper stains -> optical blur -> geometry (page/camera pose)
    -> illumination -> sensor noise -> resampling -> JPEG

Every operation is independently switchable and probabilistic.  The clean image
is always kept, so a dataset can be used both for OCR robustness training and as
a paired clean/degraded restoration set.

Geometric ops (rotation, perspective) invalidate the pixel-alignment between the
degraded image and the trajectory, so the composed 3x3 homography is returned in
``info["homography"]`` - apply it to the trajectory if you need aligned targets.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..config import DegradationConfig
from ..utils.noise import smooth_noise_2d
from ..utils.rng import uniform


def _to_float(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def degrade(
    image: np.ndarray, cfg: DegradationConfig, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(degraded uint8 RGB, info)``."""
    info: dict[str, Any] = {"ops": []}
    if not cfg.enabled:
        return image, info

    img = _to_float(image)
    h, w = img.shape[:2]
    H = np.eye(3, dtype=np.float64)

    # --- ink fading ------------------------------------------------------- #
    if rng.random() < cfg.p_ink_fade:
        f = uniform(rng, cfg.ink_fade)
        img = 1.0 - (1.0 - img) * (1.0 - f)
        info["ops"].append(("ink_fade", round(f, 3)))

    # --- stains / blotches ------------------------------------------------ #
    if rng.random() < cfg.p_stains:
        n = int(rng.integers(int(cfg.stain_count[0]), int(cfg.stain_count[1]) + 1))
        for _ in range(n):
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            r = rng.uniform(0.05, 0.3) * min(h, w)
            yy = np.arange(h, dtype=np.float32)[:, None]
            xx = np.arange(w, dtype=np.float32)[None, :]
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1e-6)
            m = np.clip(1.0 - d, 0.0, 1.0) ** 2 * rng.uniform(0.05, 0.22)
            tint = np.array([0.75, 0.68, 0.55], dtype=np.float32)
            img = img * (1 - m[:, :, None]) + tint * m[:, :, None]
        info["ops"].append(("stains", n))

    # --- optical blur ----------------------------------------------------- #
    if rng.random() < cfg.p_blur:
        s = uniform(rng, cfg.blur_sigma)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=s)
        info["ops"].append(("blur", round(s, 3)))

    if rng.random() < cfg.p_motion_blur:
        k = int(rng.integers(int(cfg.motion_kernel[0]), int(cfg.motion_kernel[1]) + 1)) | 1
        ang = rng.uniform(0, np.pi)
        kern = np.zeros((k, k), dtype=np.float32)
        cv2.line(
            kern,
            (int(k / 2 - np.cos(ang) * k / 2), int(k / 2 - np.sin(ang) * k / 2)),
            (int(k / 2 + np.cos(ang) * k / 2), int(k / 2 + np.sin(ang) * k / 2)),
            1.0,
            1,
        )
        s = kern.sum()
        if s > 0:
            img = cv2.filter2D(img, -1, kern / s, borderType=cv2.BORDER_REPLICATE)
            info["ops"].append(("motion_blur", k))

    # --- geometry --------------------------------------------------------- #
    # the border colour is only needed if a warp actually fires; computing the
    # median of a 3-channel megapixel image is not free
    _border_cache = []

    def border_colour():
        if not _border_cache:
            sub = img[:: max(h // 64, 1), :: max(w // 64, 1)].reshape(-1, 3)
            _border_cache.append(tuple(float(v) for v in np.median(sub, axis=0)))
        return _border_cache[0]

    if rng.random() < cfg.p_rotate:
        deg = uniform(rng, cfg.rotate_deg)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_colour()
        )
        H = np.vstack([M, [0, 0, 1]]) @ H
        info["ops"].append(("rotate", round(deg, 3)))

    if rng.random() < cfg.p_perspective:
        s = uniform(rng, cfg.perspective_strength)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        d = np.float32(rng.uniform(-s, s, size=(4, 2))) * np.float32([w, h])
        M = cv2.getPerspectiveTransform(src, src + d)
        img = cv2.warpPerspective(
            img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_colour()
        )
        H = M @ H
        info["ops"].append(("perspective", round(s, 4)))

    # --- illumination ----------------------------------------------------- #
    if rng.random() < cfg.p_uneven_light:
        s = uniform(rng, cfg.uneven_light_strength)
        field = smooth_noise_2d(rng, (h, w), scale=max(h, w) / 3.0, amplitude=s, octaves=2)
        img *= (1.0 + field[:, :, None])
        info["ops"].append(("uneven_light", round(s, 3)))

    if rng.random() < cfg.p_shadow:
        s = uniform(rng, cfg.shadow_strength)
        yy = np.arange(h, dtype=np.float32)[:, None]
        xx = np.arange(w, dtype=np.float32)[None, :]
        ang = rng.uniform(0, 2 * np.pi)
        proj = np.cos(ang) * xx / max(w, 1) + np.sin(ang) * yy / max(h, 1)
        edge = rng.uniform(0.15, 0.85)
        soft = rng.uniform(0.05, 0.35)
        m = np.clip((proj - edge) / soft, 0, 1) * s
        img *= (1.0 - m[:, :, None])
        info["ops"].append(("shadow", round(s, 3)))

    if rng.random() < cfg.p_brightness:
        b = uniform(rng, cfg.brightness)
        img *= b
        info["ops"].append(("brightness", round(b, 3)))

    if rng.random() < cfg.p_contrast:
        c = uniform(rng, cfg.contrast)
        m = float(img.mean())
        img = (img - m) * c + m
        info["ops"].append(("contrast", round(c, 3)))

    img = np.clip(img, 0.0, 1.0)

    # --- resolution ------------------------------------------------------- #
    if rng.random() < cfg.p_downscale:
        f = uniform(rng, cfg.downscale_factor)
        nw, nh = max(int(w * f), 8), max(int(h * f), 8)
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        info["ops"].append(("downscale", round(f, 3)))

    if rng.random() < cfg.p_sharpen:
        blur = cv2.GaussianBlur(img, (0, 0), 1.0)
        img = np.clip(img + 0.6 * (img - blur), 0, 1)
        info["ops"].append(("sharpen", 1))

    # --- sensor noise ----------------------------------------------------- #
    if rng.random() < cfg.p_noise:
        s = uniform(rng, cfg.noise_sigma)
        img += rng.normal(0.0, s, size=img.shape).astype(np.float32)
        info["ops"].append(("noise", round(s, 4)))

    out = _to_uint8(np.clip(img, 0, 1))

    # --- compression ------------------------------------------------------ #
    if rng.random() < cfg.p_jpeg:
        q = int(rng.integers(int(cfg.jpeg_quality[0]), int(cfg.jpeg_quality[1]) + 1))
        ok, buf = cv2.imencode(".jpg", out[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
            info["ops"].append(("jpeg", q))

    info["homography"] = H.tolist()
    return np.ascontiguousarray(out), info
