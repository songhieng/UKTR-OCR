"""Stage-by-stage debug visualisation.

Skeleton and graph algorithms fail *silently*: they still return a plausible
object after producing a wrong path.  The only reliable defence is to look at
every intermediate stage, so this module renders all nine of them:

1. shaped font outlines        6. smoothed trajectory
2. glyph / cluster mask        7. writer deformation
3. skeleton                    8. clean render
4. skeleton graph              9. degraded OCR image
5. planned stroke order

Used by ``scripts/debug_pipeline.py`` and by ``generate.py --debug``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import PipelineConfig
from .dataset.generator import SampleResult
from .fonts.glyphs import extract_outline_cached
from .fonts.raster import rasterize_contours
from .skeleton.thinning import binarize, clean_mask, skeletonize_mask

STROKE_COLORS = [
    (214, 39, 40),
    (31, 119, 180),
    (44, 160, 44),
    (255, 127, 14),
    (148, 103, 189),
    (23, 190, 207),
    (140, 86, 75),
    (227, 119, 194),
    (188, 189, 34),
    (127, 127, 127),
]
NODE_COLORS = {
    "junction": (0, 90, 255),
    "end": (0, 150, 0),
    "loop": (255, 140, 0),
    "dot": (255, 0, 255),
}


def _blank(w: int, h: int, colour=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (w, h), colour)


def _fit_transform(fit: dict[str, float]):
    s, tx, ty = fit["scale"], fit["tx"], fit["ty"]

    def f(pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64)
        return np.stack([p[:, 0] * s + tx, p[:, 1] * s + ty], axis=1)

    return f


def _draw_strokes(
    img: Image.Image,
    strokes: Sequence,
    transform=None,
    width: int = 2,
    numbers: bool = True,
    pen_up: bool = True,
    start_dots: bool = True,
    index_offset: int = 0,
    prev_end=None,
) -> Image.Image:
    d = ImageDraw.Draw(img)
    for j, s in enumerate(strokes):
        i = j + index_offset
        pts = np.asarray(getattr(s, "points", s), dtype=np.float64)
        if transform is not None:
            pts = transform(pts)
        col = STROKE_COLORS[i % len(STROKE_COLORS)]
        seq = [(float(p[0]), float(p[1])) for p in pts]
        if len(seq) > 1:
            d.line(seq, fill=col, width=width, joint="curve")
        else:
            x, y = seq[0]
            d.ellipse([x - width, y - width, x + width, y + width], fill=col)
        if pen_up and prev_end is not None:
            _dashed(d, prev_end, seq[0], (170, 170, 170))
        if start_dots:
            x, y = seq[0]
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col, outline=(0, 0, 0))
        if numbers:
            d.text((seq[0][0] + 4, seq[0][1] - 12), str(i), fill=(20, 20, 20))
        prev_end = seq[-1]
    return prev_end


def _dashed(d: ImageDraw.ImageDraw, a, b, colour, dash: float = 5.0) -> None:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-6:
        return
    n = max(int(L / dash), 1)
    for k in range(0, n, 2):
        p0 = a + (b - a) * (k / n)
        p1 = a + (b - a) * (min(k + 1, n) / n)
        d.line([tuple(p0), tuple(p1)], fill=colour, width=1)


def render_debug_stages(
    result: SampleResult, cfg: PipelineConfig, px_per_em_preview: float = 128.0
) -> dict[str, Image.Image]:
    """Produce every intermediate visualisation for one generated sample."""
    if not result.debug:
        raise ValueError("result was generated without want_debug=True")
    dbg = result.debug
    shaped = dbg["shaped"]
    font = dbg["font"]
    fit = dbg["fit"]
    W, H = result.trajectory.width, result.trajectory.height
    out: dict[str, Image.Image] = {}

    # ---- 1. shaped font outlines ---- #
    contours: list[np.ndarray] = []
    for g in shaped.glyphs:
        o = extract_outline_cached(font, g.glyph_id)
        if not o.is_empty:
            contours.extend(o.transformed(g.x, g.y))
    if contours:
        cov, frame = rasterize_contours(
            contours, upem=font.upem, px_per_em=px_per_em_preview, supersample=2, margin_em=0.08
        )
        out["01_shaped_font"] = Image.fromarray(((1 - cov) * 255).astype(np.uint8)).convert("RGB")

        mask = clean_mask(
            binarize(cov, cfg.raster.threshold),
            closing_radius=cfg.raster.closing_radius,
            opening_radius=cfg.raster.opening_radius,
            min_component_area_ratio=cfg.raster.min_component_area_ratio,
        )
        out["02_mask"] = Image.fromarray(((~mask) * 255).astype(np.uint8)).convert("RGB")

        skel = skeletonize_mask(mask, cfg.skeleton.method)
        vis = np.stack([(1 - mask * 0.78) * 255] * 3, axis=-1).astype(np.uint8)
        vis[skel] = (220, 0, 0)
        out["03_skeleton"] = Image.fromarray(vis)

    # ---- 4. graph + 5. plan (per cluster, composed on the line) ---- #
    em = dbg["em_px"] / max(fit["scale"], 1e-6)  # em size before the canvas fit
    graph_img = _blank(W, H)
    plan_img = _blank(W, H)
    gd = ImageDraw.Draw(graph_img)
    tf = _fit_transform(fit)

    stroke_counter = 0
    plan_prev_end = None
    for entry in dbg.get("clusters", []):
        skel_obj = entry["skeleton"]
        origin = entry["origin_em"]

        def to_canvas(p: np.ndarray, origin: float = origin) -> np.ndarray:
            p = np.asarray(p, dtype=np.float64).reshape(-1, 2).copy()
            p[:, 0] += origin
            return tf(p * em)

        for e in skel_obj.graph.edges:
            pts = to_canvas(e.points)
            gd.line([(float(a), float(b)) for a, b in pts], fill=(90, 90, 90), width=2)
        for node, kind in zip(skel_obj.graph.nodes, skel_obj.graph.node_kind):
            x, y = to_canvas(node[None, :])[0]
            c = NODE_COLORS.get(kind, (0, 0, 0))
            gd.ellipse([x - 3, y - 3, x + 3, y + 3], fill=c)

        plan = entry["plan"]
        shifted = [type("S", (), {"points": to_canvas(s.points)})() for s in plan.strokes]
        plan_prev_end = _draw_strokes(
            plan_img,
            shifted,
            transform=None,
            width=2,
            numbers=True,
            index_offset=stroke_counter,
            prev_end=plan_prev_end,
        )
        stroke_counter += len(plan.strokes)

    out["04_graph"] = graph_img
    out["05_stroke_plan"] = plan_img

    # ---- 6/7. smoothed + deformed (line space -> canvas) ---- #
    if "smoothed_strokes" in dbg:
        img = _blank(W, H)
        _draw_strokes(img, dbg["smoothed_strokes"], transform=tf, width=2)
        out["06_smoothed"] = img
    if "deformed_strokes" in dbg:
        img = _blank(W, H)
        _draw_strokes(img, dbg["deformed_strokes"], transform=tf, width=2)
        out["07_deformed"] = img

    # ---- 7b. final trajectory with pressure ---- #
    img = _blank(W, H)
    d = ImageDraw.Draw(img)
    for i, s in enumerate(result.trajectory.strokes):
        col = STROKE_COLORS[i % len(STROKE_COLORS)]
        for k in range(len(s.points) - 1):
            w = max(1, int(1 + 5 * float(s.pressure[k])))
            d.line(
                [tuple(s.points[k]), tuple(s.points[k + 1])],
                fill=col,
                width=w,
            )
    out["08_trajectory_pressure"] = img

    if result.clean is not None:
        out["09_clean"] = Image.fromarray(result.clean)
    if result.degraded is not None:
        out["10_degraded"] = Image.fromarray(result.degraded)
    return out


def save_debug_stages(
    result: SampleResult, cfg: PipelineConfig, out_dir: str | Path, make_sheet: bool = True
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = render_debug_stages(result, cfg)
    paths = []
    for name, img in stages.items():
        p = out_dir / f"{name}.png"
        img.save(p)
        paths.append(p)
    if make_sheet:
        sheet = stack_stages(stages)
        p = out_dir / "00_all_stages.png"
        sheet.save(p)
        paths.append(p)
    return paths


def stack_stages(stages: dict[str, Image.Image], width: int = 1100, pad: int = 8) -> Image.Image:
    """Stack all stage images vertically with labels."""
    scaled: list[tuple[str, Image.Image]] = []
    for name, img in stages.items():
        w, h = img.size
        nh = max(int(h * (width / max(w, 1))), 1)
        scaled.append((name, img.resize((width, nh), Image.LANCZOS)))
    total_h = sum(im.size[1] + 20 + pad for _, im in scaled) + pad
    sheet = Image.new("RGB", (width + 2 * pad, total_h), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    y = pad
    for name, im in scaled:
        d.text((pad, y), name.replace("_", " "), fill=(30, 30, 30))
        y += 18
        sheet.paste(im, (pad, y))
        y += im.size[1] + pad
    return sheet
