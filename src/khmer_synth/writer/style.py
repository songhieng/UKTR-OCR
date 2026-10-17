"""Procedural synthetic writer model.

A "writer" is a point in a ~25-dimensional style space.  Everything downstream
(layout, deformation, pressure, timing, pen model) reads its parameters from
here, so a writer is *persistent*: the same ``writer_id`` always produces the
same handwriting personality across every sample, corpus and run, because the
parameters are derived deterministically from ``(writer_id, seed)`` rather than
stored.  That makes ``--num-writers 10000`` free.

Per-sample variation (a person does not write identically twice) is added on top
as small jitter around the writer's mean, controlled by ``sample_jitter``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import WriterSamplingConfig
from ..utils.rng import choice_weighted, make_rng, triangular, uniform


@dataclass
class WriterStyle:
    """Persistent handwriting personality."""

    writer_id: int
    name: str

    #: master speed/casualness axis in [0, 1] (see WriterSamplingConfig.fluency)
    fluency: float = 0.5
    #: human-readable archetype the writer was drawn from
    archetype: str = "average"

    # --- global geometry --- #
    slant: float = 0.0
    horizontal_scale: float = 1.0
    vertical_scale: float = 1.0
    rotation: float = 0.0
    character_spacing: float = 1.0
    word_spacing: float = 1.0
    line_spacing: float = 1.0

    # --- baseline --- #
    baseline_amplitude: float = 0.0
    baseline_frequency: float = 0.4
    baseline_drift: float = 0.0
    baseline_walk: float = 0.0
    baseline_phase: float = 0.0

    # --- local deformation --- #
    local_jitter: float = 0.0
    tremor: float = 0.0
    local_elasticity: float = 0.0
    overshoot: float = 0.0
    glyph_rotation: float = 0.0
    glyph_translation: float = 0.0
    glyph_scale_var: float = 0.0
    slant_variation: float = 0.0

    # --- inter-character spacing irregularity --- #
    spacing_jitter: float = 0.02
    spacing_drift: float = 0.0
    spacing_tight_rate: float = 0.0

    # --- stroke connection --- #
    connect_probability: float = 0.0
    connect_max_gap: float = 0.12
    corner_rounding: float = 0.0

    # --- pen dynamics --- #
    stroke_smoothness: float = 1.0
    stroke_width: float = 0.05
    pressure_mean: float = 0.6
    pressure_variation: float = 0.1
    writing_speed: float = 1.0
    speed_variation: float = 0.15
    pen_lift_bias: float = 0.05

    # --- appearance --- #
    pen_model: str = "ballpoint"
    ink_color: str = "black"
    #: font preference: writers keep a stable letter-shape "hand"
    preferred_fonts: list[str] = field(default_factory=list)
    font_loyalty: float = 0.8

    def to_dict(self) -> dict:
        return asdict(self)

    def jittered(self, rng: np.random.Generator, amount: float = 0.06) -> WriterStyle:
        """Small per-sample deviation around this writer's mean style."""
        if amount <= 0:
            return self
        d = self.to_dict()

        def j(key: str, rel: float = 1.0, lo: float | None = None, hi: float | None = None):
            v = d[key]
            v = v * (1.0 + rng.normal(0.0, amount * rel)) if v != 0 else rng.normal(0.0, amount * rel * 0.02)
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            d[key] = float(v)

        for k in ("horizontal_scale", "vertical_scale", "character_spacing", "stroke_width"):
            j(k, 0.6, lo=0.05)
        for k in ("baseline_amplitude", "local_jitter", "tremor", "local_elasticity", "overshoot"):
            j(k, 1.0, lo=0.0)
        for k in ("pressure_mean", "writing_speed"):
            j(k, 0.5, lo=0.05)
        d["slant"] = float(self.slant + rng.normal(0.0, amount * 0.5))
        d["rotation"] = float(self.rotation + rng.normal(0.0, amount * 0.02))
        d["baseline_phase"] = float(rng.uniform(0.0, 2 * np.pi))
        d["baseline_drift"] = float(self.baseline_drift + rng.normal(0.0, amount * 0.2))
        d["pressure_mean"] = float(np.clip(d["pressure_mean"], 0.1, 0.95))
        return WriterStyle(**d)


def sample_writer(
    writer_id: int,
    cfg: WriterSamplingConfig | None = None,
    seed: int = 0,
    font_keys: list[str] | None = None,
) -> WriterStyle:
    """Deterministically materialise writer ``writer_id``."""
    cfg = cfg or WriterSamplingConfig()
    rng = make_rng("writer", seed, writer_id)

    # Coarse archetypes keep the population clustered rather than a featureless
    # uniform cloud, and each one fixes a *fluency* band.  Most real handwriting
    # in the wild is written at speed, so the mass sits well above the middle.
    names = ["careful", "neat", "average", "fast", "scrawl", "shaky"]
    probs = [0.10, 0.16, 0.28, 0.26, 0.12, 0.08]
    archetype = str(rng.choice(names, p=probs))
    fluency_band = {
        "careful": (0.03, 0.20),
        "neat": (0.15, 0.40),
        "average": (0.35, 0.65),
        "fast": (0.60, 0.88),
        "scrawl": (0.82, 1.00),
        "shaky": (0.30, 0.70),
    }[archetype]
    lo_f, hi_f = fluency_band
    fluency = float(np.clip(rng.uniform(lo_f, hi_f), cfg.fluency[0], cfg.fluency[1]))

    def by_fluency(rng_range, power: float = 1.0, spread: float = 0.22) -> float:
        """Sample a parameter that grows with fluency.

        The mean tracks ``fluency ** power`` across the configured range, with a
        little scatter so two writers of the same speed still differ.
        """
        lo, hi = float(rng_range[0]), float(rng_range[1])
        centre = lo + (hi - lo) * (fluency**power)
        return float(np.clip(rng.normal(centre, (hi - lo) * spread), lo, hi))

    style = WriterStyle(
        writer_id=writer_id,
        name=f"writer_{writer_id:04d}",
        fluency=fluency,
        archetype=archetype,
        slant=triangular(rng, cfg.slant),
        horizontal_scale=uniform(rng, cfg.horizontal_scale),
        vertical_scale=uniform(rng, cfg.vertical_scale),
        rotation=float(rng.normal(0.0, 0.012)),
        character_spacing=uniform(rng, cfg.character_spacing),
        word_spacing=uniform(rng, cfg.word_spacing),
        line_spacing=uniform(rng, cfg.line_spacing),
        baseline_amplitude=by_fluency(cfg.baseline_amplitude),
        baseline_frequency=uniform(rng, cfg.baseline_frequency),
        baseline_drift=triangular(rng, cfg.baseline_drift),
        baseline_walk=by_fluency(cfg.baseline_walk),
        baseline_phase=float(rng.uniform(0, 2 * np.pi)),
        local_jitter=by_fluency(cfg.local_jitter, power=0.8),
        # tremor is a slow-writing artefact: a confident fast hand does not shake
        tremor=uniform(rng, cfg.tremor) * (1.0 if archetype == "shaky" else (1.0 - fluency) ** 1.5),
        local_elasticity=by_fluency(cfg.local_elasticity),
        overshoot=by_fluency(cfg.overshoot, power=0.8),
        glyph_rotation=by_fluency(cfg.glyph_rotation),
        glyph_translation=by_fluency(cfg.glyph_translation),
        glyph_scale_var=by_fluency(cfg.glyph_scale_var),
        slant_variation=by_fluency(cfg.slant_variation),
        spacing_jitter=by_fluency(cfg.spacing_jitter, power=0.85),
        spacing_drift=by_fluency(cfg.spacing_drift),
        spacing_tight_rate=by_fluency(cfg.spacing_tight_rate, power=1.3),
        connect_probability=by_fluency(cfg.connect_probability, power=1.5, spread=0.16),
        connect_max_gap=by_fluency(cfg.connect_max_gap),
        corner_rounding=by_fluency(cfg.corner_rounding, power=1.2),
        stroke_smoothness=by_fluency(cfg.stroke_smoothness, power=0.7),
        stroke_width=uniform(rng, cfg.stroke_width),
        pressure_mean=uniform(rng, cfg.pressure_mean),
        pressure_variation=by_fluency(cfg.pressure_variation),
        writing_speed=by_fluency(cfg.writing_speed, power=0.9, spread=0.18),
        speed_variation=by_fluency(cfg.speed_variation),
        pen_lift_bias=by_fluency(cfg.pen_lift_bias, power=0.7),
        pen_model=choice_weighted(rng, cfg.pen_models),
        ink_color=choice_weighted(rng, cfg.ink_colors),
    )
    # a fast hand also writes tighter, narrower and leans forward more
    style.character_spacing *= 1.0 - 0.15 * fluency
    style.horizontal_scale *= 1.0 - 0.10 * fluency
    style.slant += 0.30 * fluency * float(rng.uniform(0.5, 1.0))
    if archetype == "shaky":
        style.writing_speed *= 0.75
    if rng.random() < 0.12:  # a bold, heavy-handed minority
        style.stroke_width *= 1.35
        style.pressure_mean = float(min(0.95, style.pressure_mean * 1.25))
    if font_keys:
        k = int(rng.integers(1, min(3, len(font_keys)) + 1))
        style.preferred_fonts = [str(x) for x in rng.choice(font_keys, size=k, replace=False)]
        style.font_loyalty = float(rng.uniform(0.6, 0.95))
    return style


def writer_table(
    num_writers: int,
    cfg: WriterSamplingConfig | None = None,
    seed: int = 0,
    font_keys: list[str] | None = None,
) -> dict[str, dict]:
    """Materialise a whole writer population (for ``writers.json``)."""
    return {
        f"writer_{i:04d}": sample_writer(i, cfg, seed, font_keys).to_dict() for i in range(num_writers)
    }


def save_writers(path: str | Path, table: dict[str, dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(table, indent=1, ensure_ascii=False), "utf-8")
