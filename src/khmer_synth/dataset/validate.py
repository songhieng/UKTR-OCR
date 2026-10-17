"""Dataset quality validation, statistics and contact sheets.

Silent corruption is the main risk in a synthetic-data pipeline: a skeleton that
fragments, a deformation that explodes, a layout that collapses to a speck, a
font that renders .notdef boxes.  None of those raise an exception - they just
produce a plausible-looking file.  This module is the automated guard.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..text.normalization import looks_corrupted
from .schema import Trajectory, validate_trajectory


def check_record(
    record: dict[str, Any],
    root: Path | None = None,
    load_trajectory: bool = True,
    load_image: bool = True,
    min_fg: float = 0.002,
    max_fg: float = 0.45,
) -> list[str]:
    """Validate one metadata record.  Returns a list of problems."""
    problems: list[str] = []
    text = record.get("normalized_text") or record.get("text") or ""
    if not text.strip():
        problems.append("empty text")
    elif looks_corrupted(text):
        problems.append("corrupted unicode")
    if record.get("num_strokes", 0) <= 0:
        problems.append("no strokes")
    if record.get("num_points", 0) < 4:
        problems.append("too few points")
    fg = float(record.get("foreground_ratio", 0.0))
    if load_image and fg and not (min_fg <= fg <= max_fg):
        problems.append(f"implausible foreground ratio {fg:.4f}")
    if record.get("extra", {}).get("has_notdef"):
        problems.append("font produced .notdef glyphs")

    if root is not None and load_trajectory and record.get("trajectory_file"):
        p = root / record["trajectory_file"]
        if not p.exists():
            problems.append("missing trajectory file")
        else:
            try:
                traj = Trajectory.from_dict(json.loads(p.read_text("utf-8")))
                problems.extend(validate_trajectory(traj))
            except Exception as exc:
                problems.append(f"unreadable trajectory ({type(exc).__name__})")

    if root is not None and load_image:
        for key in ("clean_image", "degraded_image"):
            rel = record.get(key)
            if rel and not (root / rel).exists():
                problems.append(f"missing {key}")
    return problems


def dataset_statistics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate statistics for a dataset (§21 of the spec)."""
    if not records:
        return {"samples": 0}

    def arr(key: str, default: float = 0.0) -> np.ndarray:
        return np.array([float(r.get(key, default) or 0.0) for r in records], dtype=np.float64)

    strokes = arr("num_strokes")
    points = arr("num_points")
    fg = arr("foreground_ratio")
    widths = arr("width")
    heights = arr("height")
    dur = arr("duration_ms")
    lengths = np.array(
        [float((r.get("extra") or {}).get("trajectory_length_px", 0.0)) for r in records]
    )

    writers = Counter(r.get("writer_id", "?") for r in records)
    fonts = Counter(r.get("font", "?") for r in records)
    pens = Counter(r.get("pen_model", "?") for r in records)
    inks = Counter(r.get("ink_color", "?") for r in records)

    clusters: Counter = Counter()
    for r in records:
        for g in r.get("graphemes", []) or []:
            if g.strip():
                clusters[g] += 1

    def describe(a: np.ndarray) -> dict[str, float]:
        if a.size == 0:
            return {}
        return {
            "mean": round(float(a.mean()), 4),
            "std": round(float(a.std()), 4),
            "min": round(float(a.min()), 4),
            "p50": round(float(np.percentile(a, 50)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "max": round(float(a.max()), 4),
        }

    return {
        "samples": len(records),
        "strokes_per_sample": describe(strokes),
        "points_per_sample": describe(points),
        "trajectory_length_px": describe(lengths),
        "duration_ms": describe(dur),
        "foreground_ratio": describe(fg),
        "image_width": describe(widths),
        "image_height": describe(heights),
        "unique_writers": len(writers),
        "writer_distribution_top": writers.most_common(10),
        "font_distribution": fonts.most_common(),
        "pen_distribution": pens.most_common(),
        "ink_distribution": inks.most_common(),
        "grapheme_coverage": {
            "unique_clusters": len(clusters),
            "total_clusters": int(sum(clusters.values())),
            "top": clusters.most_common(25),
            "singletons": int(sum(1 for _, c in clusters.items() if c == 1)),
        },
    }


def validate_dataset(
    root: str | Path,
    limit: int = 0,
    load_trajectory: bool = True,
    load_image: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    meta = root / "metadata.jsonl"
    if not meta.exists():
        raise FileNotFoundError(meta)
    records: list[dict[str, Any]] = []
    with meta.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if limit and len(records) >= limit:
                break

    failures: list[dict[str, Any]] = []
    problem_counter: Counter = Counter()
    for rec in records:
        problems = check_record(
            rec, root=root, load_trajectory=load_trajectory, load_image=load_image
        )
        if problems:
            failures.append({"sample_id": rec.get("sample_id"), "problems": problems})
            for p in problems:
                problem_counter[p.split("(")[0].strip()] += 1

    report = {
        "root": str(root),
        "checked": len(records),
        "failed": len(failures),
        "pass_rate": round(1.0 - len(failures) / max(len(records), 1), 5),
        "problem_histogram": problem_counter.most_common(),
        "failures_sample": failures[:25],
        "statistics": dataset_statistics(records),
    }
    return report


# --------------------------------------------------------------------------- #
# contact sheet
# --------------------------------------------------------------------------- #


def contact_sheet(
    images: Sequence[np.ndarray],
    captions: Sequence[str] | None = None,
    cols: int = 4,
    cell_width: int = 512,
    cell_height: int = 96,
    pad: int = 6,
    background: int = 245,
) -> np.ndarray:
    """Tile sample images into one inspection image."""
    from PIL import Image

    n = len(images)
    if n == 0:
        return np.full((cell_height, cell_width, 3), background, dtype=np.uint8)
    rows = int(np.ceil(n / cols))
    cap_h = 0 if captions is None else 16
    sheet = np.full(
        (rows * (cell_height + cap_h + pad) + pad, cols * (cell_width + pad) + pad, 3),
        background,
        dtype=np.uint8,
    )
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        pil = Image.fromarray(np.asarray(img, dtype=np.uint8))
        pil.thumbnail((cell_width, cell_height))
        a = np.asarray(pil)
        if a.ndim == 2:
            a = np.repeat(a[:, :, None], 3, axis=2)
        y = pad + r * (cell_height + cap_h + pad)
        x = pad + c * (cell_width + pad)
        sheet[y : y + a.shape[0], x : x + a.shape[1]] = a[:, :, :3]

    if captions is not None:
        try:
            from PIL import ImageDraw

            pil = Image.fromarray(sheet)
            draw = ImageDraw.Draw(pil)
            font = _khmer_pil_font(13)
            for i, cap in enumerate(captions[:n]):
                r, c = divmod(i, cols)
                y = pad + r * (cell_height + cap_h + pad) + cell_height
                x = pad + c * (cell_width + pad)
                draw.text((x, y), cap[:80], fill=(40, 40, 40), font=font)
            sheet = np.asarray(pil)
        except Exception:  # pragma: no cover - captions are cosmetic
            pass
    return sheet


def _khmer_pil_font(size: int):
    """A PIL font that can show Khmer captions if one is available."""
    from PIL import ImageFont

    for candidate in (
        "fonts/Battambang-Regular.ttf",
        "fonts/Hanuman-Regular.ttf",
        "C:/Windows/Fonts/KhmerUI.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()
