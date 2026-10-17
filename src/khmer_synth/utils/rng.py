"""Deterministic RNG helpers.

Every stochastic decision in the pipeline draws from a :class:`numpy.random.Generator`
that is derived from a single integer seed, so a sample is exactly reproducible
from ``(seed, text, writer_id, font)`` alone.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np


def stable_hash(*parts: object) -> int:
    """A process-independent 64-bit hash (``hash()`` is randomised per process)."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "little", signed=False)


def make_rng(*parts: object) -> np.random.Generator:
    """Build a generator whose stream depends on all ``parts``."""
    return np.random.default_rng(stable_hash(*parts))


def spawn(rng: np.random.Generator, tag: object) -> np.random.Generator:
    """Derive an independent sub-stream, tagged for readability/stability."""
    seed = int(rng.integers(0, 2**63 - 1))
    return make_rng(seed, tag)


def uniform(rng: np.random.Generator, rng_range: Sequence[float]) -> float:
    lo, hi = float(rng_range[0]), float(rng_range[1])
    if hi <= lo:
        return lo
    return float(rng.uniform(lo, hi))


def triangular(rng: np.random.Generator, rng_range: Sequence[float], mode: float | None = None) -> float:
    """Sample with a central tendency - avoids writers that are all extremes."""
    lo, hi = float(rng_range[0]), float(rng_range[1])
    if hi <= lo:
        return lo
    m = 0.5 * (lo + hi) if mode is None else float(mode)
    return float(rng.triangular(lo, m, hi))


def choice_weighted(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    if not keys:
        raise ValueError("empty weight table")
    w = np.asarray([max(0.0, float(weights[k])) for k in keys], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return str(rng.choice(keys, p=w))


def softmax_sample(rng: np.random.Generator, costs: Iterable[float], temperature: float) -> int:
    """Sample an index with probability ~ exp(-cost / T).  ``T <= 0`` -> argmin."""
    c = np.asarray(list(costs), dtype=np.float64)
    if c.size == 0:
        raise ValueError("no candidates")
    if temperature <= 1e-9:
        return int(np.argmin(c))
    z = -(c - c.min()) / max(temperature, 1e-9)
    z = np.clip(z, -60.0, 60.0)
    p = np.exp(z)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return int(np.argmin(c))
    return int(rng.choice(len(c), p=p / s))


def split_seed(seed: int, n: int) -> tuple[int, ...]:
    ss = np.random.SeedSequence(seed)
    return tuple(int(s.generate_state(1, dtype=np.uint64)[0] >> 1) for s in ss.spawn(n))
