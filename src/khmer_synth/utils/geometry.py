"""Small geometry helpers shared by the stroke / writer / render stages.

All functions operate on ``(N, 2)`` float arrays in a y-DOWN coordinate frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def as_points(p: Sequence) -> np.ndarray:
    a = np.asarray(p, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"expected (N, 2) points, got {a.shape}")
    return a


def segment_lengths(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(np.diff(pts, axis=0), axis=1)


def polyline_length(pts: np.ndarray) -> float:
    return float(segment_lengths(pts).sum())


def cumulative_length(pts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(segment_lengths(pts))])


def dedupe_consecutive(pts: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Drop points that coincide with their predecessor (breaks splines)."""
    if len(pts) < 2:
        return pts
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], d > eps])
    out = pts[keep]
    return out if len(out) >= 1 else pts[:1]


def resample_uniform(pts: np.ndarray, spacing: float, min_points: int = 2) -> np.ndarray:
    """Resample a polyline to ~uniform arc-length spacing (endpoints preserved)."""
    pts = dedupe_consecutive(as_points(pts))
    if len(pts) < 2:
        return pts
    s = cumulative_length(pts)
    total = float(s[-1])
    if total <= 1e-9:
        return pts[:1]
    n = max(int(round(total / max(spacing, 1e-6))) + 1, min_points)
    target = np.linspace(0.0, total, n)
    x = np.interp(target, s, pts[:, 0])
    y = np.interp(target, s, pts[:, 1])
    return np.stack([x, y], axis=1)


def resample_count(pts: np.ndarray, n: int) -> np.ndarray:
    pts = dedupe_consecutive(as_points(pts))
    if len(pts) < 2:
        return np.repeat(pts[:1], max(n, 1), axis=0)
    s = cumulative_length(pts)
    target = np.linspace(0.0, s[-1], max(n, 2))
    return np.stack([np.interp(target, s, pts[:, 0]), np.interp(target, s, pts[:, 1])], axis=1)


def tangents(pts: np.ndarray) -> np.ndarray:
    """Unit tangent per point (central differences, endpoints one-sided)."""
    if len(pts) < 2:
        return np.zeros_like(pts)
    d = np.gradient(pts, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return d / n


def normals(pts: np.ndarray) -> np.ndarray:
    t = tangents(pts)
    return np.stack([-t[:, 1], t[:, 0]], axis=1)


def curvature(pts: np.ndarray) -> np.ndarray:
    """Discrete curvature |x' y'' - y' x''| / |r'|^3 (per point, >= 0)."""
    if len(pts) < 3:
        return np.zeros(len(pts), dtype=np.float64)
    d1 = np.gradient(pts, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = np.power(np.linalg.norm(d1, axis=1), 3) + 1e-12
    k = num / den
    return np.nan_to_num(k, nan=0.0, posinf=0.0)


def turn_angles(pts: np.ndarray) -> np.ndarray:
    """Signed turn angle (radians) at every interior vertex."""
    if len(pts) < 3:
        return np.zeros(max(len(pts) - 2, 0), dtype=np.float64)
    v = np.diff(pts, axis=0)
    a = np.arctan2(v[:, 1], v[:, 0])
    d = np.diff(a)
    return (d + np.pi) % (2 * np.pi) - np.pi


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Unsigned angle in radians between two 2-vectors (0 = same direction)."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def rdp(pts: np.ndarray, epsilon: float) -> np.ndarray:
    """Iterative Douglas-Peucker simplification (no recursion limit issues)."""
    pts = as_points(pts)
    n = len(pts)
    if n < 3 or epsilon <= 0:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack: list[tuple[int, int]] = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        seg = pts[i + 1 : j]
        L = float(np.linalg.norm(ab))
        if L < 1e-12:
            d = np.linalg.norm(seg - a, axis=1)
        else:
            # 2-D cross product written out: np.cross on 2-vectors is deprecated
            v = seg - a
            d = np.abs(ab[0] * v[:, 1] - ab[1] * v[:, 0]) / L
        k = int(np.argmax(d))
        if d[k] > epsilon:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return pts[keep]


def catmull_rom(pts: np.ndarray, samples_per_segment: int = 12, alpha: float = 0.5) -> np.ndarray:
    """Centripetal Catmull-Rom interpolation through ``pts``.

    ``alpha=0.5`` (centripetal) is used by default because it provably avoids
    cusps and self-intersections that the uniform (alpha=0) variant produces on
    the tight corners common in Khmer glyph skeletons.
    """
    pts = dedupe_consecutive(as_points(pts))
    n = len(pts)
    if n < 3:
        return pts
    # phantom endpoints by reflection
    p = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    m = n - 1  # number of spans
    p0 = p[0:m][:, None, :]
    p1 = p[1 : m + 1][:, None, :]
    p2 = p[2 : m + 2][:, None, :]
    p3 = p[3 : m + 3][:, None, :]

    d01 = np.maximum(np.linalg.norm(p[1 : m + 1] - p[0:m], axis=1) ** alpha, 1e-6)
    d12 = np.maximum(np.linalg.norm(p[2 : m + 2] - p[1 : m + 1], axis=1) ** alpha, 1e-6)
    d23 = np.maximum(np.linalg.norm(p[3 : m + 3] - p[2 : m + 2], axis=1) ** alpha, 1e-6)
    t0 = np.zeros(m)
    t1 = d01
    t2 = t1 + d12
    t3 = t2 + d23
    u = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[None, :]
    t = (t1[:, None] + (t2 - t1)[:, None] * u)[:, :, None]

    def _c(a, b):  # column vector helper
        return (a - b)[:, None, None]

    a1 = (t1[:, None, None] - t) / _c(t1, t0) * p0 + (t - t0[:, None, None]) / _c(t1, t0) * p1
    a2 = (t2[:, None, None] - t) / _c(t2, t1) * p1 + (t - t1[:, None, None]) / _c(t2, t1) * p2
    a3 = (t3[:, None, None] - t) / _c(t3, t2) * p2 + (t - t2[:, None, None]) / _c(t3, t2) * p3
    b1 = (t2[:, None, None] - t) / _c(t2, t0) * a1 + (t - t0[:, None, None]) / _c(t2, t0) * a2
    b2 = (t3[:, None, None] - t) / _c(t3, t1) * a2 + (t - t1[:, None, None]) / _c(t3, t1) * a3
    c = (t2[:, None, None] - t) / _c(t2, t1) * b1 + (t - t1[:, None, None]) / _c(t2, t1) * b2
    return np.vstack([c.reshape(-1, 2), pts[-1][None, :]])


def quad_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def bbox(pts: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.min(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 0])),
        float(np.max(pts[:, 1])),
    )


def transform_affine(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to ``(N, 2)`` points."""
    return pts @ m[:, :2].T + m[:, 2]


def make_affine(
    scale: tuple[float, float] = (1.0, 1.0),
    rotation: float = 0.0,
    shear_x: float = 0.0,
    translate: tuple[float, float] = (0.0, 0.0),
    center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Compose translate(center) . T . R . Shear . S . translate(-center)."""
    cx, cy = center
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    S = np.array([[scale[0], 0.0], [0.0, scale[1]]], dtype=np.float64)
    H = np.array([[1.0, shear_x], [0.0, 1.0]], dtype=np.float64)
    A = R @ H @ S
    t = np.array([cx + translate[0], cy + translate[1]], dtype=np.float64) - A @ np.array([cx, cy])
    return np.concatenate([A, t[:, None]], axis=1)
