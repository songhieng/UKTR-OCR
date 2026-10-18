"""End-to-end sample generation: text -> trajectory + images + metadata.

This is the module that wires every stage together.  It is written as a pure
function of ``(text, writer_id, font, seed, config)`` so a sample can be
reproduced bit-for-bit from its metadata, and so it can be called either from the
offline CLI generator or from the on-the-fly PyTorch dataset.

Layout pipeline (all in "line space": px, x from 0, baseline at y = 0):

    shape -> per-cluster centreline graph (cached) -> stroke plan (cached pool)
          -> place at the cluster's pen position -> smooth/resample
          -> writer deformation -> fit into the canvas -> timing -> pressure
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..augmentation.degradation import degrade
from ..augmentation.paper import apply_paper
from ..config import PipelineConfig
from ..fonts.loader import FontLibrary, LoadedFont
from ..rendering.backgrounds import make_background
from ..rendering.renderer import render_trajectory
from ..skeleton.pipeline import cluster_skeleton
from ..strokes.abbreviate import abbreviate_strokes
from ..strokes.connect import connect_strokes
from ..strokes.planner import Stroke, StrokePlan, plan_variants
from ..strokes.smoothing import smooth_stroke
from ..text.graphemes import segment_graphemes
from ..text.normalization import normalize_khmer
from ..text.shaping import ShapedText, Shaper
from ..utils.caching import LRUCache, TwoLevelCache
from ..utils.geometry import make_affine, transform_affine
from ..utils.noise import smooth_noise_1d
from ..utils.rng import make_rng, stable_hash, uniform
from ..writer.deformation import deform_strokes
from ..writer.pressure import synth_pressure
from ..writer.style import WriterStyle, sample_writer
from ..writer.timing import synth_timing
from .schema import SampleMetadata, Trajectory, TrajectoryStroke, validate_trajectory


@dataclass
class SampleResult:
    trajectory: Trajectory
    clean: np.ndarray | None
    degraded: np.ndarray | None
    metadata: SampleMetadata
    style: WriterStyle
    problems: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


class SampleGenerator:
    """Reusable generator (holds the font library and the skeleton caches)."""

    def __init__(self, cfg: PipelineConfig, cache: TwoLevelCache | None = None):
        self.cfg = cfg
        self.fonts = FontLibrary(cfg.fonts)
        self.shaper = Shaper(cfg.shaping)
        disk_root = cfg.dataset.cache_dir if cfg.dataset.use_disk_cache else None
        self.cache = cache or TwoLevelCache(capacity=16384, disk_root=disk_root)
        self.plan_cache = LRUCache(capacity=16384)

    # ------------------------------------------------------------------ #
    # font / writer selection
    # ------------------------------------------------------------------ #

    def pick_font(self, style: WriterStyle, rng: np.random.Generator, text: str) -> LoadedFont:
        if style.preferred_fonts and rng.random() < style.font_loyalty:
            for key in rng.permutation(np.array(style.preferred_fonts, dtype=object)):
                try:
                    font = self.fonts.get(str(key))
                except KeyError:
                    continue
                if font.covers(text):
                    return font
        return self.fonts.sample(rng, text)

    # ------------------------------------------------------------------ #
    # stroke assembly
    # ------------------------------------------------------------------ #

    def _cluster_plans(self, font: LoadedFont, placements: Sequence[tuple[int, float, float]]):
        """Cached ``(skeleton, list-of-plans)`` for one grapheme cluster.

        Planning a cluster costs ~200 ms, so this goes through the *two-level*
        cache: an in-process LRU plus the shared on-disk tier.  The disk tier
        matters enormously under multiprocessing - without it every worker
        re-plans every cluster from scratch and throughput barely improves with
        more workers.
        """
        sig = tuple(sorted((int(g), round(float(dx), 1), round(float(dy), 1)) for g, dx, dy in placements))
        key_parts = (
            # v2: font identity includes the file's content hash - see the
            # cluster_skeleton_v2 comment in skeleton/pipeline.py for the proven
            # stem collision this fixes.
            "cluster_plans_v2",
            font.uid,
            tuple(sorted(font.variations.items())),
            sig,
            self.cfg.raster,
            self.cfg.skeleton,
            self.cfg.planner,
        )
        key = f"{stable_hash(*key_parts):016x}"
        hit = self.plan_cache.get(key)
        if hit is not None:
            return hit

        def _compute():
            skel = cluster_skeleton(font, placements, self.cfg.raster, self.cfg.skeleton, self.cache)
            n = max(1, self.cfg.planner.plan_variants)
            plans = plan_variants(skel.graph, self.cfg.planner, n=n, seed=stable_hash(key) % (2**31))
            return skel, plans

        value = self.cache.get_or_compute(key_parts, _compute)
        self.plan_cache.put(key, value)
        return value

    def _build_line(
        self,
        font: LoadedFont,
        shaped: ShapedText,
        em_px: float,
        style: WriterStyle,
        rng: np.random.Generator,
        debug: dict[str, Any] | None = None,
    ) -> list[Stroke]:
        """Place every cluster's planned strokes into line space (px)."""
        strokes: list[Stroke] = []
        upem = float(font.upem)
        order = sorted(
            shaped.cluster_glyphs.keys(),
            key=lambda ci: min(shaped.glyphs[i].x for i in shaped.cluster_glyphs[ci]),
        )
        for ci in order:
            idx = shaped.cluster_glyphs[ci]
            if ci < len(shaped.clusters) and shaped.clusters[ci].kind == "space":
                continue
            origin = min(shaped.glyphs[i].x - shaped.glyphs[i].x_offset for i in idx)
            placements = [
                (shaped.glyphs[i].glyph_id, shaped.glyphs[i].x - origin, shaped.glyphs[i].y)
                for i in idx
            ]
            skel, plans = self._cluster_plans(font, placements)
            if skel.is_empty or not plans:
                continue
            # Which stroke order this writer uses for this letter is part of the
            # hand, not per-occurrence noise: a writer who draws ក with one
            # particular pen path draws it that way every time.  Keying the
            # choice on (writer, cluster text) is what stops the same letter
            # from being re-invented at every occurrence in the line.
            key = shaped.clusters[ci].text if ci < len(shaped.clusters) else ""
            if key:
                prng = make_rng("plan", self.cfg.dataset.seed, style.writer_id, key, font.key)
                plan: StrokePlan = plans[int(prng.integers(0, len(plans)))]
            else:
                plan = plans[int(rng.integers(0, len(plans)))]
            dx = origin / upem
            for s in plan.strokes:
                pts = np.asarray(s.points, dtype=np.float64).copy()
                pts[:, 0] += dx
                pts *= em_px
                strokes.append(Stroke(pts, list(s.edges), s.component, cluster=int(ci)))
            if debug is not None:
                debug.setdefault("clusters", []).append(
                    {
                        "cluster_index": int(ci),
                        "text": shaped.clusters[ci].text if ci < len(shaped.clusters) else "",
                        "skeleton": skel,
                        "plan": plan,
                        "origin_em": dx,
                    }
                )
        return strokes

    # ------------------------------------------------------------------ #
    def generate(
        self,
        text: str,
        writer_id: int = 0,
        seed: int = 0,
        font_key: str | None = None,
        want_debug: bool = False,
        render: bool = True,
        degrade_image: bool | None = None,
        sample_id: str = "sample",
    ) -> SampleResult:
        cfg = self.cfg
        t_start = time.perf_counter()
        debug: dict[str, Any] = {} if want_debug else None  # type: ignore[assignment]

        rng = make_rng("sample", seed, writer_id, text)
        norm = normalize_khmer(text, cfg.text)
        if not norm.strip():
            raise ValueError("text is empty after normalisation")

        style = sample_writer(
            writer_id,
            cfg.writer,
            seed=cfg.dataset.seed,
            font_keys=self.fonts.keys,
            font_weights=cfg.fonts.weights,
            font_default_weight=getattr(cfg.fonts, "default_weight", 1.0),
        ).jittered(rng, amount=0.07)

        font = self.fonts.get(font_key) if font_key else self.pick_font(style, rng, norm)

        # ---- shaping with writer spacing ---- #
        n_clusters = max(len(segment_graphemes(norm)), 1)
        advances = None
        if cfg.writer.spacing_tight_max_frac > 0:
            # one plain shaping pass to learn each cluster's own width (shaping
            # order, the same order ``shape_with_spacing`` indexes extras by)
            plain = self.shaper.shape(font, norm)
            order: list[int] = []
            for g in plain.glyphs:
                if g.cluster_index not in order:
                    order.append(g.cluster_index)
            advances = np.array(
                [sum(plain.glyphs[i].x_advance for i in plain.cluster_glyphs[ci]) for ci in order],
                dtype=np.float64,
            )
            n_clusters = max(len(order), 1)
        extra = _spacing_offsets(
            style,
            rng,
            n_clusters,
            font.upem,
            cluster_advances=advances,
            tight_max_frac=cfg.writer.spacing_tight_max_frac,
        )
        shaped = self.shaper.shape_with_spacing(
            font,
            norm,
            cluster_spacing=style.character_spacing,
            word_spacing=style.word_spacing,
            per_cluster_extra=extra,
        )

        # ---- canvas + em size ---- #
        W, H = cfg.layout.canvas_width, cfg.layout.canvas_height
        # Randomised framing when configured: real crops are cut flush to the ink
        # (measured margins 0.016-0.042 of height) where a fixed 10 px margin on a
        # 96 px canvas gives 0.107.  See LayoutConfig.padding_px_range.
        if cfg.layout.padding_px_range is not None:
            lo, hi = cfg.layout.padding_px_range
            pad = int(round(uniform(rng, (float(lo), float(hi)))))
        else:
            pad = cfg.layout.padding_px
        em_px = uniform(rng, cfg.layout.em_fraction) * H
        advance_em = max(shaped.advance / float(font.upem), 1e-6)
        usable_w = max(W - 2 * pad, 16)
        if cfg.layout.auto_width and advance_em * em_px > usable_w:
            W = int(min(cfg.layout.max_canvas_width, np.ceil(advance_em * em_px + 2 * pad)))
            usable_w = max(W - 2 * pad, 16)
        if cfg.layout.fit_to_width and advance_em * em_px > usable_w:
            em_px = max(usable_w / advance_em, cfg.layout.min_em_px)

        # ---- strokes ---- #
        strokes = self._build_line(font, shaped, em_px, style, rng, debug)
        if not strokes:
            raise ValueError(f"no strokes produced for text {norm!r} with font {font.key}")
        if cfg.layout.min_cluster_gap_em is not None:
            strokes = _separate_clusters(strokes, em_px, cfg.layout.min_cluster_gap_em)
        if want_debug:
            debug["raw_strokes"] = [Stroke(s.points.copy(), [], s.component, s.cluster) for s in strokes]

        # ---- letterform abbreviation (before connection: a reduced letter has
        #      shorter gaps, so it also carries over more readily) ---- #
        cluster_keys = {i: c.text for i, c in enumerate(shaped.clusters) if getattr(c, "text", "")}
        strokes = abbreviate_strokes(
            strokes,
            style,
            cfg.abbreviation,
            em_px,
            cluster_keys=cluster_keys,
            allograph_seed=cfg.dataset.seed,
        )
        if not strokes:
            raise ValueError(f"abbreviation removed every stroke for {norm!r}")
        if want_debug:
            debug["abbreviated_strokes"] = [
                Stroke(s.points.copy(), [], s.component, s.cluster) for s in strokes
            ]

        # ---- stroke connection (before smoothing, so the carry-over is
        #      smoothed into one continuous curve with the strokes it joins) ---- #
        n_before = len(strokes)
        strokes = connect_strokes(
            strokes,
            rng,
            em_px,
            probability=style.connect_probability,
            max_gap_em=style.connect_max_gap,
            spacing_px=cfg.smoothing.point_spacing_px,
            max_chain=cfg.writer.connect_max_chain,
            max_rise_em=cfg.writer.connect_max_rise_em,
        )
        if want_debug:
            debug["connected_strokes"] = [
                Stroke(s.points.copy(), [], s.component, s.cluster, list(s.ligature_spans))
                for s in strokes
            ]
            debug["strokes_before_connect"] = n_before

        # ---- smoothing ---- #
        spacing = cfg.smoothing.point_spacing_px
        # a fast hand rounds off corners it "should" have drawn sharply, so the
        # angle above which a vertex is preserved as a hard corner rises with
        # the writer's fluency
        corner_angle = cfg.smoothing.corner_angle_deg + 46.0 * style.corner_rounding
        # savgol_window is specified for a 64 px em; keep it a fixed fraction of
        # the letter size so small writing is not smoothed out of shape.  It is
        # what removes the skeleton's raster kinks (the interpolating spline
        # alone passes through every one of them, giving wavy "trembling" lines).
        sg_win = int(round(cfg.smoothing.savgol_window * em_px / 64.0))
        smoothed: list[Stroke] = []
        for s in strokes:
            pts = smooth_stroke(
                s.points,
                cfg.smoothing,
                spacing=spacing,
                smoothness=style.stroke_smoothness,
                corner_angle_deg=corner_angle,
                savgol_window=sg_win,
            )
            if len(pts) == 0:
                continue
            smoothed.append(Stroke(pts, list(s.edges), s.component, s.cluster, list(s.ligature_spans)))
        strokes = smoothed
        if want_debug:
            debug["smoothed_strokes"] = [
                Stroke(s.points.copy(), [], s.component, s.cluster, list(s.ligature_spans))
                for s in strokes
            ]

        # ---- deformation ---- #
        strokes = deform_strokes(
            strokes,
            style,
            cfg.deformation,
            rng,
            em_px,
            cluster_keys=cluster_keys,
            allograph_seed=cfg.dataset.seed,
        )
        if want_debug:
            debug["deformed_strokes"] = [
                Stroke(s.points.copy(), [], s.component, s.cluster, list(s.ligature_spans))
                for s in strokes
            ]

        # ---- fit into the canvas ---- #
        if cfg.layout.auto_width:
            # cluster separation, slant and overshoot make the ink wider than the
            # shaped advance; grow the canvas instead of shrinking the writing
            xs = np.concatenate([s.points[:, 0] for s in strokes])
            need = int(np.ceil(float(xs.max() - xs.min()) + 2 * pad))
            W = int(min(cfg.layout.max_canvas_width, max(W, need)))
        strokes, em_px_final, fit = _fit_to_canvas(strokes, W, H, pad, cfg, rng, em_px)

        # ---- timing + pressure ---- #
        times, speeds, duration = synth_timing(strokes, style, cfg.timing, rng, em_px_final)
        pressures = synth_pressure(strokes, style, cfg.pressure, rng, speeds)

        traj = Trajectory(
            strokes=[
                TrajectoryStroke(s.points, pr, t, cluster=s.cluster, component=s.component)
                for s, pr, t in zip(strokes, pressures, times)
                if len(s.points)
            ],
            width=W,
            height=H,
            sample_id=sample_id,
            writer_id=style.name,
            text=norm,
            duration_ms=duration,
            meta={
                "font": font.key,
                "em_px": round(float(em_px_final), 3),
                "baseline_y": round(float(fit["baseline_y"]), 2),
                "pen_model": style.pen_model,
                "ink_color": style.ink_color,
                "seed": int(seed),
                "fit_scale": round(float(fit["scale"]), 4),
            },
        )

        problems = validate_trajectory(traj) if cfg.dataset.validate else []
        # A very small fit scale means the deformed line was far larger than the
        # canvas, i.e. something blew up upstream and the "fit" silently shrank
        # the glyphs to specks.  Surface it instead of writing a blank image.
        if cfg.dataset.validate and fit["scale"] < 0.35:
            problems.append(f"layout collapsed (fit scale {fit['scale']:.3f})")

        clean = None
        degraded = None
        fg_ratio = 0.0
        if render:
            bg, bg_kind = make_background(
                W, H, cfg.background, rng, baseline_y=fit["baseline_y"], em_px=em_px_final
            )
            clean = render_trajectory(traj, style, cfg.render, rng, em_px_final, background=bg)
            traj.meta["background"] = bg_kind
            grey = clean.mean(axis=2) / 255.0
            # Ink coverage measured *relative to the paper*, not against a fixed
            # threshold: a faint pencil on white paper is perfectly legible but
            # never crosses an absolute 0.6, so a fixed threshold reports it as
            # a blank image and the validator rejects a good sample.
            paper = float(np.median(grey))
            fg_ratio = float((grey < paper - 0.12).mean())
            do_degrade = cfg.degradation.enabled if degrade_image is None else degrade_image
            # The paper stage (Augraphy) is the ink/sheet half of the chain and
            # runs *before* the capture half - see config.PaperConfig.  It is off
            # unless a config turns it on, so nothing above changes for the
            # pre-existing presets (in particular it draws no random numbers).
            do_paper = cfg.paper.enabled and degrade_image is not False
            if do_paper and cfg.paper.stage == "replace_degradation":
                do_degrade = False
            if do_degrade or do_paper:
                work = clean
                pinfo: dict[str, Any] | None = None
                if do_paper and cfg.paper.stage != "after_degradation":
                    work, pinfo = apply_paper(work, cfg.paper, rng)
                if do_degrade:
                    work, dinfo = degrade(work, cfg.degradation, rng)
                    traj.meta["degradation"] = dinfo["ops"]
                    if want_debug:
                        debug["degradation_info"] = dinfo
                if do_paper and cfg.paper.stage == "after_degradation":
                    work, pinfo = apply_paper(work, cfg.paper, rng)
                if pinfo is not None:
                    traj.meta["paper"] = pinfo["ops"]
                    if want_debug:
                        debug["paper_info"] = pinfo
                degraded = work

        meta = SampleMetadata(
            sample_id=sample_id,
            text=text,
            normalized_text=norm,
            graphemes=[c.text for c in shaped.clusters],
            writer_id=style.name,
            font=font.key,
            font_size=round(float(em_px_final), 3),
            width=W,
            height=H,
            num_strokes=traj.num_strokes,
            num_points=traj.num_points,
            duration_ms=round(float(duration), 2),
            seed=int(seed),
            pen_model=style.pen_model,
            ink_color=style.ink_color,
            foreground_ratio=round(fg_ratio, 5),
            extra={
                "grapheme_kinds": [c.kind for c in shaped.clusters],
                "num_glyphs": len(shaped.glyphs),
                "has_notdef": shaped.has_notdef,
                "trajectory_length_px": round(traj.total_length_px(), 2),
                "generation_ms": round((time.perf_counter() - t_start) * 1000, 2),
            },
        )

        if want_debug:
            debug.update({"shaped": shaped, "font": font, "fit": fit, "em_px": em_px_final})

        return SampleResult(
            trajectory=traj,
            clean=clean,
            degraded=degraded,
            metadata=meta,
            style=style,
            problems=problems,
            debug=debug or {},
        )


# --------------------------------------------------------------------------- #


def _separate_clusters(strokes: list[Stroke], em_px: float, min_gap_em: float) -> list[Stroke]:
    """Guarantee a minimum horizontal gap between consecutive grapheme clusters.

    Clusters are taken in left-to-right order of their ink.  Whenever a cluster's
    ink starts before ``(right edge of everything before it) + min_gap``, that
    cluster and every later one are translated right by the deficit, so shapes
    and the gaps *after* it are untouched.  A negative ``min_gap_em`` allows a
    small overlap (real hands do touch neighbouring letters), but never a pile-up.
    """
    if not strokes:
        return strokes
    gap = float(min_gap_em) * em_px
    by_cluster: dict[int, list[Stroke]] = {}
    for s in strokes:
        if len(s.points):
            by_cluster.setdefault(s.cluster, []).append(s)
    order = sorted(by_cluster, key=lambda c: min(float(s.points[:, 0].min()) for s in by_cluster[c]))
    edge: float | None = None
    shift = 0.0
    for c in order:
        group = by_cluster[c]
        x0 = min(float(s.points[:, 0].min()) for s in group) + shift
        x1 = max(float(s.points[:, 0].max()) for s in group) + shift
        if edge is not None and x0 < edge + gap:
            d = edge + gap - x0
            shift += d
            x0 += d
            x1 += d
        if shift != 0.0:
            for s in group:
                s.points = s.points + np.array([shift, 0.0])
        edge = x1 if edge is None else max(edge, x1)
    return strokes


def _spacing_offsets(
    style: WriterStyle,
    rng: np.random.Generator,
    n_clusters: int,
    upem: int,
    cluster_advances: np.ndarray | None = None,
    tight_max_frac: float = 0.0,
) -> np.ndarray:
    """Extra advance (font units) to insert after each grapheme cluster.

    ``cluster_advances`` (font units, one per cluster in shaping order) together
    with ``tight_max_frac`` bound a collapse to a fraction of the preceding
    cluster's own width, so narrow clusters (digits, Latin) cannot be driven on
    top of each other.  Both default to off, which is the v1 behaviour.

    Three components, because typeset-looking spacing was the single biggest
    "this is synthetic" tell:

    * a **correlated drift** - writers bunch characters up for a few clusters and
      then spread out again, so the gap width wanders slowly along the line;
    * **i.i.d. jitter** - gap-to-gap randomness on top of the drift;
    * **occasional collapses** - a fraction of gaps close almost completely, so
      neighbouring characters touch.  Fast handwriting is full of these, and they
      never occur if the offsets are symmetric Gaussian noise.
    """
    n = max(int(n_clusters), 1)
    offsets = np.zeros(n, dtype=np.float64)

    if style.spacing_drift > 0:
        offsets += smooth_noise_1d(
            rng, n, scale=max(n / 4.0, 2.0), amplitude=style.spacing_drift * upem
        )
    if style.spacing_jitter > 0:
        offsets += rng.normal(0.0, style.spacing_jitter * upem, size=n)
    if style.spacing_tight_rate > 0:
        tight = rng.random(n) < style.spacing_tight_rate
        offsets[tight] -= rng.uniform(0.10, 0.26, size=int(tight.sum())) * upem

    # never let a gap collapse so far that clusters become unreadable
    if cluster_advances is not None and tight_max_frac > 0:
        adv = np.asarray(cluster_advances, dtype=np.float64)
        if len(adv) == n:
            offsets = np.maximum(offsets, -tight_max_frac * np.maximum(adv, 0.0))
    return np.clip(offsets, -0.30 * upem, 0.60 * upem)


def _fit_to_canvas(
    strokes: list[Stroke],
    W: int,
    H: int,
    pad: int,
    cfg: PipelineConfig,
    rng: np.random.Generator,
    em_px: float,
) -> tuple[list[Stroke], float, dict[str, float]]:
    """Scale + translate the deformed line so that it sits inside the canvas.

    Doing this *after* deformation is what guarantees that no amount of slant,
    baseline wobble or overshoot can push ink out of the image - a failure mode
    that silently corrupts synthetic OCR datasets.
    """
    pts = np.vstack([s.points for s in strokes])
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    bw, bh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)

    avail_w = max(W - 2 * pad, 8)
    avail_h = max(H - 2 * pad, 8)
    scale = min(1.0, avail_w / bw, avail_h / bh)

    free_w = max(avail_w - bw * scale, 0.0)  # can go a hair negative from rounding
    tx = pad - x0 * scale + (rng.uniform(0.0, free_w) if cfg.layout.random_x_placement else free_w * 0.5)

    baseline_y = uniform(rng, cfg.layout.baseline_fraction) * H
    ty = baseline_y
    top = y0 * scale + ty
    bot = y1 * scale + ty
    if top < pad:
        ty += pad - top
    if bot > H - pad:
        ty -= bot - (H - pad)
    baseline_y = ty

    m = make_affine(scale=(scale, scale), translate=(tx, ty))
    for s in strokes:
        s.points = transform_affine(s.points, m)
    return strokes, em_px * scale, {"scale": scale, "tx": tx, "ty": ty, "baseline_y": baseline_y}


def generate_sample(
    text: str,
    cfg: PipelineConfig | None = None,
    writer_id: int = 0,
    seed: int = 0,
    font_key: str | None = None,
    **kw,
) -> SampleResult:
    """Convenience one-shot entry point (builds a generator every call)."""
    cfg = cfg or PipelineConfig()
    return SampleGenerator(cfg).generate(text, writer_id=writer_id, seed=seed, font_key=font_key, **kw)
