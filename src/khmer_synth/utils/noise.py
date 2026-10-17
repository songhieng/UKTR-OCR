"""Smooth / correlated noise.

Independent per-point Gaussian noise makes a trajectory look like a *noisy font*,
not like handwriting.  Real pen motion error is strongly auto-correlated, so all
noise in this package goes through one of the helpers below, which produce
band-limited (low-frequency) signals with a controllable correlation length.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_noise_1d(
    rng: np.random.Generator,
    n: int,
    scale: float,
    amplitude: float = 1.0,
    octaves: int = 1,
    persistence: float = 0.5,
) -> np.ndarray:
    """Band-limited 1-D noise of length ``n``.

    ``scale`` is the correlation length expressed in samples.  Multi-octave
    summation gives a cheap Perlin-like spectrum.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    amp = 1.0
    total = 0.0
    for o in range(max(1, octaves)):
        # Clamp sigma to the signal length.  Smoothing a 3-sample sequence with
        # sigma=6 leaves an almost constant signal whose std is ~1e-16; dividing
        # by it (to normalise) amplifies pure float noise by ~1e5 and silently
        # explodes the trajectory.  This bit us once - keep the guard.
        s = float(np.clip(scale / (2**o), 0.75, max(n / 3.0, 0.75)))
        w = rng.standard_normal(n)
        f = gaussian_filter1d(w, sigma=s, mode="reflect")
        std = float(f.std())
        f = f / std if std > 1e-6 else np.zeros_like(f)
        out += amp * f
        total += amp
        amp *= persistence
    out = amplitude * out / max(total, 1e-9)
    return np.clip(out, -4.0 * abs(amplitude), 4.0 * abs(amplitude))


def smooth_noise_2d(
    rng: np.random.Generator,
    shape: tuple[int, int],
    scale: float,
    amplitude: float = 1.0,
    octaves: int = 1,
    persistence: float = 0.5,
) -> np.ndarray:
    """Band-limited 2-D field (used for lighting, paper texture, ink texture)."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for o in range(max(1, octaves)):
        s = max(scale / (2**o), 0.75)
        # generate at reduced resolution for speed, then upsample smoothly
        sub = max(1, int(s // 3))
        small = rng.standard_normal((max(h // sub, 4), max(w // sub, 4))).astype(np.float32)
        # cv2 is ~5x faster than scipy here and this runs for every sample
        sigma = float(np.clip(s / sub, 0.5, max(min(small.shape) / 3.0, 0.5)))
        f = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
        std = float(f.std())
        f = f / std if std > 1e-6 else np.zeros_like(f)
        if f.shape != (h, w):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        out += amp * f
        total += amp
        amp *= persistence
    out = amplitude * out / max(total, 1e-9)
    return np.clip(out, -4.0 * abs(amplitude), 4.0 * abs(amplitude)).astype(np.float32)


def smooth_noise_along(
    rng: np.random.Generator,
    positions: np.ndarray,
    scale: float,
    amplitude: float = 1.0,
    octaves: int = 1,
) -> np.ndarray:
    """Noise sampled at arbitrary 1-D ``positions`` (e.g. arc length in px).

    Unlike :func:`smooth_noise_1d` the correlation length is expressed in the
    same units as ``positions``, so re-sampling the trajectory does not change
    the perceived frequency of the noise.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.size == 0:
        return np.zeros(0, dtype=np.float64)
    span = float(positions.max() - positions.min())
    n = int(np.clip(span / max(scale, 1e-6) * 8.0 + 8, 8, 4096))
    grid = np.linspace(positions.min(), positions.max() + 1e-9, n)
    base = smooth_noise_1d(rng, n, scale=8.0, amplitude=amplitude, octaves=octaves)
    return np.interp(positions, grid, base)
