"""Paper simulation (the layer the ink is composited onto)."""

from __future__ import annotations

import numpy as np

from ..config import BackgroundConfig
from ..utils.noise import smooth_noise_2d
from ..utils.rng import choice_weighted, uniform


def make_background(
    width: int,
    height: int,
    cfg: BackgroundConfig,
    rng: np.random.Generator,
    kind: str | None = None,
    baseline_y: float | None = None,
    em_px: float | None = None,
) -> tuple[np.ndarray, str]:
    """Return ``(rgb float32 [H, W, 3] in [0, 1], kind)``."""
    kind = kind or choice_weighted(rng, cfg.kinds)
    base = uniform(rng, cfg.base_brightness)

    # slight paper tint
    tint = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if kind in ("off_white", "textured", "ruled", "grid"):
        tint = np.array(
            [1.0, 1.0 - rng.uniform(0.0, 0.035), 1.0 - rng.uniform(0.01, 0.075)], dtype=np.float32
        )

    img = np.ones((height, width, 3), dtype=np.float32) * base * tint

    strength = uniform(rng, cfg.texture_strength)
    if kind in ("textured", "ruled", "grid") or strength > 0.02:
        fibre = smooth_noise_2d(rng, (height, width), scale=2.2, amplitude=strength, octaves=3)
        img += fibre[:, :, None]
        grain = rng.normal(0.0, strength * 0.35, size=(height, width, 1)).astype(np.float32)
        img += grain

    if kind in ("ruled", "grid"):
        alpha = uniform(rng, cfg.ruled_line_alpha)
        spacing = max(uniform(rng, cfg.ruled_spacing_frac) * height, 12.0)
        colour = np.array([0.45, 0.55, 0.75], dtype=np.float32)
        offset = (
            baseline_y + rng.uniform(-2, 2) if baseline_y is not None else rng.uniform(0, spacing)
        )
        y = offset
        while y > 0:
            y -= spacing
        while y < height:
            yi = int(round(y))
            if 0 <= yi < height:
                thick = 1 if rng.random() < 0.8 else 2
                img[yi : yi + thick] = img[yi : yi + thick] * (1 - alpha) + colour * alpha
            y += spacing
        if kind == "grid":
            x = rng.uniform(0, spacing)
            while x < width:
                xi = int(round(x))
                if 0 <= xi < width:
                    img[:, xi : xi + 1] = img[:, xi : xi + 1] * (1 - alpha * 0.8) + colour * alpha * 0.8
                x += spacing

    return np.clip(img, 0.0, 1.0), kind
