"""Pure helpers behind the Streamlit playground (``streamlit_app.py``).

Everything here is UI-free so it can be unit-tested: turning a preset plus a
dict of on/off switches and slider ranges into an effective
:class:`~khmer_synth.config.PipelineConfig`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from .config import DegradationConfig, PipelineConfig

ROOT = Path(__file__).resolve().parents[2]

#: preset label -> config path (None = library defaults)
PRESETS: dict[str, str | None] = {
    "v2 realism (real_match_v2)": "configs/real_match_v2.json",
    "v1 real_match": "configs/real_match.json",
    "casual_fast": "configs/casual_fast.json",
    "careful": "configs/careful.json",
    "high_quality": "configs/high_quality.json",
    "clean_only": "configs/clean_only.json",
    "library defaults": None,
}

#: degradation switches: field -> human label
DEGRADE_OPS: dict[str, str] = {
    "p_blur": "optical blur",
    "p_motion_blur": "motion blur",
    "p_downscale": "downscale / upscale",
    "p_brightness": "brightness",
    "p_contrast": "contrast",
    "p_uneven_light": "uneven lighting",
    "p_shadow": "shadow",
    "p_perspective": "perspective warp",
    "p_rotate": "rotation",
    "p_noise": "sensor noise",
    "p_jpeg": "JPEG compression",
    "p_ink_fade": "ink fading",
    "p_stains": "stains",
    "p_sharpen": "sharpening",
}

#: writer range sliders exposed in the UI: field -> (label, min, max, step)
WRITER_RANGES: dict[str, tuple[str, float, float, float]] = {
    "fluency": ("fluency (speed / casualness)", 0.0, 1.0, 0.01),
    "slant": ("slant (shear)", -0.4, 0.5, 0.01),
    "stroke_width": ("stroke width (em)", 0.01, 0.12, 0.005),
    "pressure_mean": ("pressure / ink darkness", 0.1, 0.95, 0.01),
    "character_spacing": ("character spacing x", 0.6, 1.4, 0.01),
    "word_spacing": ("word spacing x", 0.5, 2.0, 0.05),
    "horizontal_scale": ("horizontal scale", 0.6, 1.4, 0.01),
    "vertical_scale": ("vertical scale", 0.6, 1.5, 0.01),
    "glyph_scale_var": ("letter size variance", 0.0, 0.3, 0.005),
    "glyph_rotation": ("letter rotation variance", 0.0, 0.15, 0.005),
    "baseline_amplitude": ("baseline wave (em)", 0.0, 0.15, 0.005),
    "baseline_walk": ("baseline walk (em)", 0.0, 0.12, 0.005),
    "local_jitter": ("stroke jitter (em)", 0.0, 0.03, 0.001),
    "tremor": ("tremor (em)", 0.0, 0.02, 0.001),
    "connect_probability": ("stroke connection probability", 0.0, 1.0, 0.01),
    "spacing_tight_rate": ("touching-letter rate", 0.0, 0.8, 0.01),
    "mark_shift_above": ("above-mark lift (em)", 0.0, 0.3, 0.005),
    "mark_drift_x": ("above-mark right drift (em)", 0.0, 0.15, 0.005),
    "below_stretch": ("subscript zone stretch", 0.0, 0.6, 0.01),
    "width_contrast": ("down/up-stroke width contrast", 0.0, 0.8, 0.01),
}

ARCHETYPES = ["careful", "neat", "average", "fast", "scrawl", "shaky"]


def preset_path(label: str) -> Path | None:
    rel = PRESETS.get(label)
    return None if rel is None else ROOT / rel


def load_preset(label: str) -> PipelineConfig:
    p = preset_path(label)
    return PipelineConfig() if p is None else PipelineConfig.load(p)


def _clone(cfg: PipelineConfig) -> PipelineConfig:
    return PipelineConfig.from_dict(cfg.to_dict())


def apply_toggles(cfg: PipelineConfig, t: dict[str, Any]) -> PipelineConfig:
    """Return a modified copy of ``cfg`` with the playground switches applied.

    Recognised keys (all optional):

    ``deform_global, deform_baseline, deform_cluster, deform_stroke,
    deform_point, deform_marks`` : bool
        the five deformation levels + v2 mark displacement
    ``connect, spacing_irregular, cluster_gap_guard, abbreviation`` : bool
    ``width_contrast, texture_bleed`` : bool
    ``degrade`` : bool  master switch;  ``degrade_ops`` : {p_field: bool}
    ``background`` : "random" | kind name
    ``pen`` / ``ink`` : "random" | name
    ``fonts_allowed`` : list[str] | None   (None = leave the preset's pool)
    ``all_font_keys`` : list[str]          (needed with fonts_allowed)
    ``archetypes`` : {name: weight}
    ``writer_ranges`` : {field: (lo, hi)}
    ``canvas_height`` : int, ``canvas_width`` : int, ``auto_width`` : bool
    """
    c = _clone(cfg)
    g = t.get

    # ---- deformation levels ---- #
    for key, field_ in (
        ("deform_global", "global_transform"),
        ("deform_baseline", "baseline"),
        ("deform_cluster", "cluster_level"),
        ("deform_stroke", "stroke_level"),
        ("deform_point", "point_level"),
        ("deform_marks", "mark_level"),
    ):
        if key in t:
            setattr(c.deformation, field_, bool(t[key]))

    # ---- writer behaviours ---- #
    if g("connect") is False:
        c.writer.connect_probability = (0.0, 0.0)
    if g("spacing_irregular") is False:
        c.writer.spacing_jitter = (0.0, 0.0)
        c.writer.spacing_drift = (0.0, 0.0)
        c.writer.spacing_tight_rate = (0.0, 0.0)
    if "cluster_gap_guard" in t:
        c.layout.min_cluster_gap_em = -0.02 if t["cluster_gap_guard"] else None
    if "abbreviation" in t:
        c.abbreviation.enabled = bool(t["abbreviation"])
    if g("width_contrast") is False:
        c.writer.width_contrast = (0.0, 0.0)
    if g("texture_bleed") is False:
        c.render.texture_strength = (0.0, 0.0)
        c.render.bleed_probability = 0.0
    if g("deform_marks") is False:
        c.writer.mark_shift_above = (0.0, 0.0)
        c.writer.mark_drift_x = (0.0, 0.0)
        c.writer.mark_scale_var = (0.0, 0.0)
        c.writer.below_stretch = (0.0, 0.0)

    # ---- degradation ---- #
    if "degrade" in t:
        c.degradation.enabled = bool(t["degrade"])
    for field_, on in (g("degrade_ops") or {}).items():
        if field_ in {f.name for f in dataclasses.fields(DegradationConfig)} and not on:
            setattr(c.degradation, field_, 0.0)

    # ---- background / pen / ink ---- #
    bg = g("background")
    if bg and bg != "random":
        c.background.kinds = {bg: 1.0}
    pen = g("pen")
    if pen and pen != "random":
        c.writer.pen_models = {pen: 1.0}
    ink = g("ink")
    if ink and ink != "random":
        c.writer.ink_colors = {ink: 1.0}

    # ---- fonts ---- #
    allowed = g("fonts_allowed")
    if allowed is not None:
        pool = list(g("all_font_keys") or [])
        allowed_set = set(allowed)
        c.fonts.exclude = sorted(set(c.fonts.exclude) | {k for k in pool if k not in allowed_set})
        # an explicitly allowed font must never stay excluded by the preset
        c.fonts.exclude = [k for k in c.fonts.exclude if k not in allowed_set]

    # ---- archetypes ---- #
    arch = g("archetypes")
    if arch:
        c.writer.archetype_weights = {k: float(v) for k, v in arch.items()}

    # ---- slider ranges ---- #
    for field_, rng in (g("writer_ranges") or {}).items():
        if hasattr(c.writer, field_):
            lo, hi = float(rng[0]), float(rng[1])
            setattr(c.writer, field_, (min(lo, hi), max(lo, hi)))

    # ---- canvas ---- #
    if "canvas_height" in t:
        c.layout.canvas_height = int(t["canvas_height"])
    if "canvas_width" in t:
        c.layout.canvas_width = int(t["canvas_width"])
    if "auto_width" in t:
        c.layout.auto_width = bool(t["auto_width"])

    c.dataset.use_disk_cache = True
    return c


def config_signature(cfg: PipelineConfig) -> str:
    """Stable string for caching a generator per effective config."""
    import json

    return json.dumps(cfg.to_dict(), sort_keys=True, ensure_ascii=False)


def random_corpus_lines(n: int, seed: int, path: Path | None = None) -> list[str]:
    import random

    p = path or (ROOT / "corpus_v2.txt")
    if not p.exists():
        p = ROOT / "corpus_demo.txt"
    lines = [l.strip() for l in p.read_text("utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    return rng.sample(lines, min(n, len(lines)))
