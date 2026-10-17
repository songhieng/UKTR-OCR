"""Pen / ink models.

Each pen is a small parameter set that tells the renderer how pressure and speed
map to stroke width and darkness, how grainy the deposit is, and how much the
ink spreads into the paper.  The relation requested in the spec,

    width = base_width * (0.6 + pressure * 0.8)

is the default (``width_pressure_bias`` / ``width_pressure_gain`` in
:class:`~khmer_synth.config.RenderConfig`); each pen scales and reshapes it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PenModel:
    name: str
    #: multiplier on the writer's stroke width
    width_scale: float = 1.0
    #: base opacity of the deposit
    alpha: float = 0.95
    #: how strongly pressure modulates width / opacity (0 = not at all)
    width_pressure: float = 1.0
    alpha_pressure: float = 1.0
    #: fountain-pen style thinning with speed (0 = none)
    speed_thinning: float = 0.0
    #: nib aspect ratio (1 = round, <1 = calligraphic/chisel)
    nib_ratio: float = 1.0
    #: grain / texture amount
    texture: float = 0.0
    #: ink spread into the paper (gaussian sigma in px)
    bleed: float = 0.0
    #: overlapping strokes darken (marker) instead of max-compositing
    overlap_darkening: bool = False
    #: hard edge (ballpoint) vs soft edge (pencil / marker)
    edge_softness: float = 0.0


PEN_MODELS: dict[str, PenModel] = {
    "ballpoint": PenModel(
        "ballpoint",
        width_scale=0.85,
        alpha=0.96,
        width_pressure=0.9,
        alpha_pressure=0.8,
        speed_thinning=0.2,
        texture=0.05,
    ),
    "gel": PenModel(
        "gel",
        width_scale=1.0,
        alpha=0.99,
        width_pressure=0.7,
        alpha_pressure=0.35,
        speed_thinning=0.15,
        bleed=0.35,
    ),
    "fountain": PenModel(
        "fountain",
        width_scale=1.15,
        alpha=0.97,
        width_pressure=1.2,
        alpha_pressure=0.6,
        speed_thinning=0.35,
        nib_ratio=0.55,
        bleed=0.6,
    ),
    "pencil": PenModel(
        "pencil",
        width_scale=0.95,
        alpha=0.88,
        width_pressure=0.9,
        alpha_pressure=0.9,
        texture=0.45,
        edge_softness=0.5,
    ),
    "marker": PenModel(
        "marker",
        width_scale=1.6,
        alpha=0.80,
        width_pressure=0.25,
        alpha_pressure=0.2,
        texture=0.12,
        bleed=1.0,
        overlap_darkening=True,
        edge_softness=0.25,
    ),
}


#: linear-RGB-ish ink colours in [0, 1]
INK_COLORS: dict[str, tuple[float, float, float]] = {
    "black": (0.07, 0.07, 0.08),
    "dark_blue": (0.09, 0.13, 0.36),
    "blue": (0.13, 0.22, 0.62),
    "faded": (0.32, 0.33, 0.38),
    "red": (0.55, 0.10, 0.12),
    "pencil_grey": (0.28, 0.29, 0.31),
}


def get_pen(name: str) -> PenModel:
    return PEN_MODELS.get(name, PEN_MODELS["ballpoint"])


def get_ink_color(name: str, pen: PenModel) -> tuple[float, float, float]:
    if pen.name == "pencil":
        return INK_COLORS["pencil_grey"]
    return INK_COLORS.get(name, INK_COLORS["black"])
