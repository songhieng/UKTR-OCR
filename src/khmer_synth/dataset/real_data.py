"""Interfaces for (future) real online-handwriting data.

The synthetic pipeline is meant to be *bootstrapped*, then corrected with a small
amount of real pen data.  This module defines the seam:

1. :func:`load_real_trajectory` reads the minimal real-data contract

       text, writer_id, ordered (x, y), timestamps, pressure, pen up/down

   from JSON / CSV, or from an already-parsed array, and returns the exact same
   :class:`~khmer_synth.dataset.schema.Trajectory` object the synthesiser emits.
   Downstream code (PyTorch datasets, models, metrics) therefore does not need
   to know whether a sample is real or synthetic.

2. :func:`estimate_calibration` measures the statistics that the procedural
   model gets wrong most easily - pen speed, pressure distribution, stroke count
   per grapheme, pen-lift rate, tremor spectrum - and returns config overrides
   that pull the synthetic distribution towards the real one.  This is a moment
   matching step, not learning: it is deliberately simple, auditable, and works
   with a few dozen real samples.

3. The trajectory format is already the one a learned generator consumes
   (``schema.Trajectory.to_compact``), so "synthetic pretraining -> real
   fine-tuning" is a data-mixing decision, not a redesign.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..text.graphemes import segment_graphemes
from ..text.normalization import normalize_khmer
from .schema import Trajectory, TrajectoryStroke


def trajectory_from_arrays(
    x: Sequence[float],
    y: Sequence[float],
    pen_up: Sequence[bool],
    timestamps: Sequence[float] | None = None,
    pressure: Sequence[float] | None = None,
    text: str = "",
    writer_id: str = "real_0000",
    width: int | None = None,
    height: int | None = None,
    sample_id: str = "real",
) -> Trajectory:
    """Build a :class:`Trajectory` from flat real-data arrays.

    ``pen_up[i]`` is True on the LAST point of each stroke (matching the
    ``pen_state == 1`` convention of the synthetic format).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n == 0:
        raise ValueError("empty trajectory")
    up = np.asarray(pen_up, dtype=bool)
    t = np.asarray(timestamps, dtype=np.float64) if timestamps is not None else np.arange(n, dtype=np.float64) * 8.0
    p = np.asarray(pressure, dtype=np.float64) if pressure is not None else np.full(n, 0.6)
    p = np.clip(p, 0.0, 1.0)

    if width is None:
        width = int(max(np.ceil(x.max() - min(x.min(), 0)) + 1, 1))
    if height is None:
        height = int(max(np.ceil(y.max() - min(y.min(), 0)) + 1, 1))

    strokes: list[TrajectoryStroke] = []
    start = 0
    for i in range(n):
        if up[i] or i == n - 1:
            sl = slice(start, i + 1)
            if i + 1 - start >= 1:
                strokes.append(
                    TrajectoryStroke(np.stack([x[sl], y[sl]], axis=1), p[sl], t[sl])
                )
            start = i + 1
    return Trajectory(
        strokes=strokes,
        width=int(width),
        height=int(height),
        sample_id=sample_id,
        writer_id=writer_id,
        text=normalize_khmer(text),
        duration_ms=float(t[-1] - t[0]),
        meta={"source": "real"},
    )


def load_real_trajectory(path: str | Path) -> Trajectory:
    """Load one real sample.

    Accepted layouts
    ----------------
    * a JSON file already in the synthetic schema (``strokes`` -> ``points``);
    * a JSON file with flat arrays ``{"x": [], "y": [], "pen_up": [], ...}``;
    * a CSV with columns ``x,y[,t][,pressure][,pen_up|stroke_id]``.
    """
    p = Path(path)
    if p.suffix.lower() == ".csv":
        rows = list(csv.DictReader(p.open("r", encoding="utf-8", newline="")))
        if not rows:
            raise ValueError(f"{p} is empty")
        cols = {k.lower(): k for k in rows[0].keys()}

        def col(name: str, default: float | None = None):
            key = cols.get(name)
            if key is None:
                return None if default is None else [default] * len(rows)
            return [float(r[key]) for r in rows]

        x, y = col("x"), col("y")
        t = col("t") or col("time") or col("timestamp")
        pr = col("pressure")
        if "pen_up" in cols:
            up = [bool(float(r[cols["pen_up"]])) for r in rows]
        elif "stroke_id" in cols:
            sid = [float(r[cols["stroke_id"]]) for r in rows]
            up = [i == len(sid) - 1 or sid[i] != sid[i + 1] for i in range(len(sid))]
        else:
            up = [i == len(rows) - 1 for i in range(len(rows))]
        return trajectory_from_arrays(x, y, up, t, pr, text=p.stem)

    data = json.loads(p.read_text("utf-8"))
    if "strokes" in data and data["strokes"] and isinstance(data["strokes"][0], dict) and "points" in data["strokes"][0]:
        return Trajectory.from_dict(data)
    return trajectory_from_arrays(
        data["x"],
        data["y"],
        data.get("pen_up", [i == len(data["x"]) - 1 for i in range(len(data["x"]))]),
        data.get("t") or data.get("timestamps"),
        data.get("pressure"),
        text=data.get("text", ""),
        writer_id=str(data.get("writer_id", "real_0000")),
        width=data.get("canvas_width"),
        height=data.get("canvas_height"),
    )


def normalize_real_trajectory(
    traj: Trajectory, target_em_px: float = 96.0, baseline_quantile: float = 0.82
) -> tuple[Trajectory, float]:
    """Rescale a real sample to the synthetic em convention.

    Real digitiser coordinates are in device units; to compare or mix with
    synthetic data we normalise so that the ink height corresponds to a known em
    size and the baseline sits at y = 0 (line space).  Returns
    ``(trajectory, estimated_em_px)``.
    """
    pts = np.vstack([s.points for s in traj.strokes if len(s)])
    if len(pts) == 0:
        return traj, target_em_px
    y = pts[:, 1]
    ink_height = float(np.percentile(y, 99) - np.percentile(y, 1))
    # a Khmer line's ink height is empirically ~1.15 em (ascenders + descenders)
    em = max(ink_height / 1.15, 1e-6)
    scale = target_em_px / em
    baseline = float(np.percentile(y, baseline_quantile * 100))
    out = Trajectory(
        strokes=[
            TrajectoryStroke(
                (s.points - np.array([pts[:, 0].min(), baseline])) * scale,
                s.pressure,
                s.time_ms,
                s.cluster,
                s.component,
            )
            for s in traj.strokes
        ],
        width=traj.width,
        height=traj.height,
        sample_id=traj.sample_id,
        writer_id=traj.writer_id,
        text=traj.text,
        duration_ms=traj.duration_ms,
        meta=dict(traj.meta),
    )
    return out, target_em_px


def measure_trajectory(traj: Trajectory, em_px: float | None = None) -> dict[str, float]:
    """Per-sample statistics used for calibration and for synthetic/real gap reports."""
    if em_px is None:
        em_px = float(traj.meta.get("em_px", 0.0)) or 1.0
    n_clusters = max(len([c for c in segment_graphemes(traj.text) if c.kind != "space"]), 1)

    speeds: list[np.ndarray] = []
    lengths = 0.0
    press: list[np.ndarray] = []
    for s in traj.strokes:
        if len(s) < 2:
            continue
        ds = np.linalg.norm(np.diff(s.points, axis=0), axis=1)
        dt = np.diff(s.time_ms)
        ok = dt > 0
        if ok.any():
            speeds.append(ds[ok] / dt[ok])
        lengths += float(ds.sum())
        press.append(s.pressure)

    v = np.concatenate(speeds) if speeds else np.zeros(1)
    p = np.concatenate(press) if press else np.zeros(1)
    duration_s = max(traj.duration_ms, 1e-6) / 1000.0
    return {
        "strokes_per_cluster": traj.num_strokes / n_clusters,
        "points_per_stroke": traj.num_points / max(traj.num_strokes, 1),
        "path_em_per_cluster": lengths / em_px / n_clusters,
        "speed_em_per_s_mean": float(np.mean(v)) * 1000.0 / em_px,
        "speed_em_per_s_p90": float(np.percentile(v, 90)) * 1000.0 / em_px,
        "speed_cv": float(np.std(v) / max(np.mean(v), 1e-9)),
        "pressure_mean": float(np.mean(p)),
        "pressure_std": float(np.std(p)),
        "duration_s_per_cluster": duration_s / n_clusters,
    }


def estimate_calibration(
    real: Sequence[Trajectory], synthetic: Sequence[Trajectory] | None = None
) -> dict[str, Any]:
    """Moment-match the procedural model against a small real corpus.

    Returns a dict of suggested config overrides (in the dotted form accepted by
    ``--set``), plus the measured statistics of both populations so the gap is
    auditable rather than hidden inside a fitted model.
    """
    if not real:
        raise ValueError("no real trajectories supplied")

    def aggregate(trajs: Sequence[Trajectory]) -> dict[str, float]:
        rows = [measure_trajectory(t) for t in trajs]
        keys = rows[0].keys()
        return {k: float(np.median([r[k] for r in rows])) for k in keys}

    real_stats = aggregate(real)
    out: dict[str, Any] = {"real": real_stats, "overrides": {}}

    if synthetic:
        syn_stats = aggregate(synthetic)
        out["synthetic"] = syn_stats
        speed_ratio = real_stats["speed_em_per_s_mean"] / max(syn_stats["speed_em_per_s_mean"], 1e-9)
        out["overrides"]["timing.base_speed_em_per_s"] = round(7.0 * speed_ratio, 3)
        out["overrides"]["writer.pressure_mean"] = [
            round(max(0.05, real_stats["pressure_mean"] - real_stats["pressure_std"]), 3),
            round(min(1.0, real_stats["pressure_mean"] + real_stats["pressure_std"]), 3),
        ]
        out["overrides"]["timing.speed_noise_sigma"] = round(
            float(np.clip(real_stats["speed_cv"] * 0.5, 0.02, 0.6)), 3
        )
        lift_ratio = real_stats["strokes_per_cluster"] / max(syn_stats["strokes_per_cluster"], 1e-9)
        if lift_ratio > 1.15:
            out["overrides"]["planner.pen_lift_bias"] = round(
                min(0.5, 0.05 * lift_ratio * 2.0), 3
            )
        elif lift_ratio < 0.85:
            out["overrides"]["planner.w_pen_up_count"] = round(1.0 / max(lift_ratio, 0.2), 3)
        out["gap"] = {
            k: round(real_stats[k] - syn_stats.get(k, 0.0), 4) for k in real_stats
        }
    else:
        out["overrides"]["timing.base_speed_em_per_s"] = round(
            real_stats["speed_em_per_s_mean"], 3
        )
        out["overrides"]["writer.pressure_mean"] = [
            round(max(0.05, real_stats["pressure_mean"] - real_stats["pressure_std"]), 3),
            round(min(1.0, real_stats["pressure_mean"] + real_stats["pressure_std"]), 3),
        ]
    return out
