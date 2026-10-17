"""Trajectory + metadata schema (the on-disk contract).

Two equivalent representations are emitted, because downstream consumers want
different things:

**Stroke-separated** (human readable, used by ``trajectories/*.json``)::

    {"strokes": [{"stroke_id": 0, "points": [{"x":..,"y":..,"pressure":..,"timestamp":..}, ...]}]}

**Compact sequence** (ready for an LSTM/Transformer/diffusion model)::

    (N, 5) float32 array of  [dx, dy, pen_state, pressure, dt_ms]

Conventions - fixed, and mirrored in the README:

* ``x``/``y`` are **normalised to [0, 1]** by the canvas width/height, y down.
* ``dx``/``dy`` are normalised deltas *from the previous point of the whole
  sequence* (so they encode the pen-up transit too).  The very first point's
  delta is measured from the origin (0, 0), i.e. it is its absolute position.
* ``pen_state``  ``0 = pen down``, ``1 = pen up / end of stroke``,
  ``2 = end of sequence``.  The last point of every stroke carries ``1``; the
  final point of the last stroke carries ``2``.
* ``timestamp`` is milliseconds from the start of the sample, strictly
  increasing across the whole sequence; ``dt`` is the difference to the previous
  point (0 for the first).
* ``pressure`` is in [0, 1].
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PEN_DOWN = 0
PEN_UP = 1
PEN_END = 2


@dataclass
class TrajectoryStroke:
    points: np.ndarray  # (N, 2) canvas pixels, y down
    pressure: np.ndarray  # (N,)
    time_ms: np.ndarray  # (N,)
    cluster: int = -1
    component: int = 0

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)
        self.pressure = np.asarray(self.pressure, dtype=np.float64).reshape(-1)
        self.time_ms = np.asarray(self.time_ms, dtype=np.float64).reshape(-1)

    def __len__(self) -> int:
        return len(self.points)

    @property
    def length_px(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


@dataclass
class Trajectory:
    strokes: list[TrajectoryStroke]
    width: int
    height: int
    sample_id: str = ""
    writer_id: str = ""
    text: str = ""
    duration_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def num_points(self) -> int:
        return int(sum(len(s) for s in self.strokes))

    @property
    def num_strokes(self) -> int:
        return len(self.strokes)

    def bbox(self) -> tuple[float, float, float, float] | None:
        if not self.strokes:
            return None
        p = np.vstack([s.points for s in self.strokes if len(s)])
        if len(p) == 0:
            return None
        return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))

    def total_length_px(self) -> float:
        return float(sum(s.length_px for s in self.strokes))

    # ------------------------------------------------------------------ #
    def to_dict(self, normalized: bool = True, precision: int = 5) -> dict[str, Any]:
        sx = 1.0 / max(self.width, 1) if normalized else 1.0
        sy = 1.0 / max(self.height, 1) if normalized else 1.0
        strokes = []
        for i, s in enumerate(self.strokes):
            pts = [
                {
                    "x": round(float(p[0]) * sx, precision),
                    "y": round(float(p[1]) * sy, precision),
                    "pressure": round(float(pr), 4),
                    "timestamp": round(float(t), 2),
                }
                for p, pr, t in zip(s.points, s.pressure, s.time_ms)
            ]
            strokes.append(
                {
                    "stroke_id": i,
                    "cluster_index": int(s.cluster),
                    "points": pts,
                }
            )
        return {
            "sample_id": self.sample_id,
            "writer_id": self.writer_id,
            "text": self.text,
            "canvas_width": int(self.width),
            "canvas_height": int(self.height),
            "coordinate_space": "normalized" if normalized else "pixels",
            "pen_state_convention": {"0": "pen_down", "1": "pen_up_end_of_stroke", "2": "end_of_sequence"},
            "duration_ms": round(float(self.duration_ms), 2),
            "num_strokes": self.num_strokes,
            "num_points": self.num_points,
            "strokes": strokes,
            "meta": self.meta,
        }

    def to_compact(self, normalized: bool = True) -> np.ndarray:
        """``(N, 5)`` array of ``[dx, dy, pen_state, pressure, dt_ms]``."""
        if not self.strokes:
            return np.zeros((0, 5), dtype=np.float32)
        sx = 1.0 / max(self.width, 1) if normalized else 1.0
        sy = 1.0 / max(self.height, 1) if normalized else 1.0
        xs, ys, pen, pres, ts = [], [], [], [], []
        for s in self.strokes:
            n = len(s)
            if n == 0:
                continue
            xs.append(s.points[:, 0] * sx)
            ys.append(s.points[:, 1] * sy)
            state = np.zeros(n, dtype=np.float32)
            state[-1] = PEN_UP
            pen.append(state)
            pres.append(s.pressure)
            ts.append(s.time_ms)
        if not xs:
            return np.zeros((0, 5), dtype=np.float32)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        pen_arr = np.concatenate(pen)
        pres_arr = np.concatenate(pres)
        t = np.concatenate(ts)
        dx = np.diff(x, prepend=0.0)
        dy = np.diff(y, prepend=0.0)
        dt = np.diff(t, prepend=t[0])
        pen_arr[-1] = PEN_END
        return np.stack([dx, dy, pen_arr, pres_arr, dt], axis=1).astype(np.float32)

    @staticmethod
    def from_compact(
        compact: np.ndarray, width: int, height: int, normalized: bool = True, **kw
    ) -> Trajectory:
        """Inverse of :meth:`to_compact` (used by tests and by real-data import)."""
        c = np.asarray(compact, dtype=np.float64)
        if c.size == 0:
            return Trajectory([], width, height, **kw)
        x = np.cumsum(c[:, 0])
        y = np.cumsum(c[:, 1])
        if normalized:
            x = x * width
            y = y * height
        t = np.cumsum(c[:, 4])
        strokes: list[TrajectoryStroke] = []
        start = 0
        for i in range(len(c)):
            if c[i, 2] != PEN_DOWN or i == len(c) - 1:
                sl = slice(start, i + 1)
                strokes.append(
                    TrajectoryStroke(np.stack([x[sl], y[sl]], axis=1), c[sl, 3], t[sl])
                )
                start = i + 1
        return Trajectory(strokes, width, height, duration_ms=float(t[-1]), **kw)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Trajectory:
        w = int(data.get("canvas_width", 1))
        h = int(data.get("canvas_height", 1))
        scale = data.get("coordinate_space", "normalized") == "normalized"
        strokes = []
        for s in data.get("strokes", []):
            pts = s.get("points", [])
            if not pts:
                continue
            arr = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float64)
            if scale:
                arr[:, 0] *= w
                arr[:, 1] *= h
            strokes.append(
                TrajectoryStroke(
                    arr,
                    np.array([p.get("pressure", 0.5) for p in pts]),
                    np.array([p.get("timestamp", 0.0) for p in pts]),
                    cluster=int(s.get("cluster_index", -1)),
                )
            )
        return Trajectory(
            strokes,
            w,
            h,
            sample_id=data.get("sample_id", ""),
            writer_id=data.get("writer_id", ""),
            text=data.get("text", ""),
            duration_ms=float(data.get("duration_ms", 0.0)),
            meta=data.get("meta", {}),
        )

    def save(self, path: str | Path, normalized: bool = True, compact: bool = False) -> None:
        d = self.to_dict(normalized=normalized)
        if compact:
            d["compact"] = {
                "format": ["dx", "dy", "pen_state", "pressure", "dt_ms"],
                "data": [[round(float(v), 6) for v in row] for row in self.to_compact(normalized)],
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), "utf-8")


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #


@dataclass
class SampleMetadata:
    sample_id: str
    text: str
    normalized_text: str
    graphemes: list[str]
    writer_id: str
    font: str
    font_size: float
    width: int
    height: int
    num_strokes: int
    num_points: int
    duration_ms: float
    clean_image: str = ""
    degraded_image: str = ""
    trajectory_file: str = ""
    seed: int = 0
    pen_model: str = ""
    ink_color: str = ""
    foreground_ratio: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def validate_trajectory(
    traj: Trajectory, tolerance: float = 0.06, min_points: int = 4
) -> list[str]:
    """Return a list of problems (empty == valid).  See §21 of the spec."""
    problems: list[str] = []
    if traj.num_strokes == 0:
        problems.append("no strokes")
        return problems
    if traj.num_points < min_points:
        problems.append(f"too few points ({traj.num_points})")

    prev_t = -np.inf
    for i, s in enumerate(traj.strokes):
        if len(s) == 0:
            problems.append(f"stroke {i} is empty")
            continue
        if not np.all(np.isfinite(s.points)):
            problems.append(f"stroke {i} has non-finite coordinates")
        if not np.all(np.isfinite(s.pressure)) or np.any(s.pressure < 0) or np.any(s.pressure > 1):
            problems.append(f"stroke {i} pressure out of [0, 1]")
        if len(s) > 1 and not np.all(np.diff(s.time_ms) > 0):
            problems.append(f"stroke {i} timestamps not strictly increasing")
        if s.time_ms[0] <= prev_t:
            problems.append(f"stroke {i} starts before the previous stroke ended")
        prev_t = float(s.time_ms[-1])

    bb = traj.bbox()
    if bb is None:
        problems.append("empty bounding box")
    else:
        w, h = max(traj.width, 1), max(traj.height, 1)
        if bb[0] < -tolerance * w or bb[1] < -tolerance * h:
            problems.append(f"ink outside the canvas (min corner {bb[0]:.1f}, {bb[1]:.1f})")
        if bb[2] > (1 + tolerance) * w or bb[3] > (1 + tolerance) * h:
            problems.append(f"ink outside the canvas (max corner {bb[2]:.1f}, {bb[3]:.1f})")
        if (bb[2] - bb[0]) < 0.01 * w and (bb[3] - bb[1]) < 0.01 * h:
            problems.append("degenerate (collapsed) layout")
    return problems
