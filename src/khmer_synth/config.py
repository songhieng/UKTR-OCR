"""Central configuration objects for the whole pipeline.

Every stage of the pipeline is driven by a plain dataclass living here so that a
run can be reproduced exactly from ``config.json`` + a seed.  Dataclasses are
used (rather than pydantic models) because they round-trip trivially to JSON and
have zero overhead in the hot multiprocessing path; :func:`from_dict` performs
the recursive, type-aware construction that pydantic would otherwise give us.

Coordinate conventions used throughout the package
--------------------------------------------------
* **font space**  - y-UP, origin on the baseline, units = font units (``upem``).
* **em space**    - y-DOWN, origin on the baseline, 1.0 == 1 em.  All cached
  glyph/cluster skeletons are stored here so they are font-size independent.
* **canvas space**- y-DOWN, pixels, origin at the top-left of the image.
* **normalised**  - canvas space divided by (width, height) -> [0, 1].
"""

from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #


@dataclass
class TextConfig:
    """Unicode / Khmer text pre-processing."""

    unicode_form: str = "NFC"
    #: map deprecated code points (U+17A3, U+17A4, U+17B4, U+17B5 ...)
    fix_deprecated: bool = True
    #: re-order marks inside a Khmer orthographic syllable into canonical order
    reorder_marks: bool = True
    #: drop duplicated identical marks inside one cluster
    dedup_marks: bool = True
    #: 'keep' | 'space' | 'remove'
    zwsp_policy: str = "keep"
    #: strip ZWNJ (U+200C) / ZWJ (U+200D)
    strip_joiners: bool = False
    #: collapse runs of whitespace into a single space
    collapse_whitespace: bool = True


@dataclass
class ShapingConfig:
    """HarfBuzz shaping parameters."""

    script: str = "Khmr"
    language: str = "km"
    direction: str = "ltr"
    #: 0 = MONOTONE_GRAPHEMES, 1 = MONOTONE_CHARACTERS, 2 = CHARACTERS
    cluster_level: int = 0
    #: e.g. {"kern": True, "liga": True}
    features: dict[str, bool] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# fonts / rasterisation
# --------------------------------------------------------------------------- #


@dataclass
class FontConfig:
    """Where fonts come from and how they are weighted when sampling."""

    #: files and/or directories; directories are scanned for .ttf/.otf/.ttc
    paths: list[str] = field(default_factory=lambda: ["fonts"])
    #: optional per-font sampling weight, keyed by font stem (e.g. "Battambang-Regular")
    weights: dict[str, float] = field(default_factory=dict)
    #: Weight given to a font that is *not* named in ``weights``.
    #:
    #: WHY this exists.  MEASURED on the live tree: ``fonts.paths: ["fonts"]`` is
    #: rglob-ed, so it now also picks up the 232 faces under ``fonts/km``.  With
    #: unlisted fonts hard-coded to 1.0, only 17.9 of the pool's 243.9 total
    #: weight - **7.3%** - landed on the 17 faces `real_match_v2.json` tunes, and
    #: `Fasthand-Regular` drew 1.23% of samples where the config intends ~17%.
    #: Every tuning decision in a ``weights`` block was effectively inert.
    #: The default stays 1.0 so existing presets sample exactly as before; a
    #: preset that means its weights to be authoritative sets this lower (e.g.
    #: 0.05) or to 0.0 to restrict sampling to the listed faces alone.
    default_weight: float = 1.0
    #: Fonts to skip, by stem.  The defaults are decorative/display Khmer faces
    #: whose outlines are hollow, doubled or heavily ornamented; thinning them
    #: produces centrelines that trace the *decoration* rather than the letter
    #: (``scripts/inspect_font.py --skeleton-stats`` shows ~2x the edges per
    #: cluster for these).  Set ``fonts.exclude=[]`` to include them, e.g. for
    #: OCR robustness data where the *image* matters more than the trajectory.
    exclude: list[str] = field(
        default_factory=lambda: [
            "Bokor-Regular",
            "Moul-Regular",
            "Angkor-Regular",
            "Taprom-Regular",
            "PreahVihear-Regular",
            "Koulen-Regular",
            "Metal-Regular",
            "Bayon-Regular",
            "Dangrek-Regular",
            "OdorMeanChey-Regular",
        ]
    )
    #: How entries in ``exclude`` are matched against a font file.
    #:
    #: ``"exact"``   (default, = v1) the bare filename stem must match exactly,
    #:               except that an entry containing a glob metacharacter
    #:               (``* ? [``) is fnmatch-ed case-insensitively against the stem
    #:               and against every path component.  No shipped preset used a
    #:               glob before, so this leaves their font pools byte-identical.
    #: ``"smart"``   case-insensitive *substring* match against the stem and every
    #:               path component, so ``"Moul"`` also catches
    #:               ``KhmerOSMoulLight`` and ``fonts/km/Moul/…``.
    #:
    #: WHY: MEASURED - the exclude list names Google-style stems
    #: (``Moul-Regular``, ``Koulen-Regular``, …) but the ``fonts/km`` symlinks
    #: carry vendor filenames, so 26 decorative faces the config explicitly
    #: intends to drop (Moul, Koulen, Bokor, Dangrek, Bayon, Taprom, Metal-Chrieng
    #: and 19 more) survived exclusion and were sampled as handwriting -
    #: ``Kh-Metal-Chrieng`` was the single most-sampled font in a 400-sample run.
    #: Their skeletons trace the *decoration* rather than the letter (see the
    #: ``exclude`` docstring above), which feeds the substitution error mass.
    exclude_match: str = "exact"
    #: variable-font axis locations, keyed by font stem, e.g. {"Nokora-VF": {"wght": 400}}
    variations: dict[str, dict[str, float]] = field(default_factory=dict)
    #: require the font to cover this probe string, else it is dropped
    coverage_probe: str = "ក្នុំ"  # ក្នុំ


@dataclass
class RasterConfig:
    """Glyph/cluster rasterisation feeding the skeletoniser."""

    #: resolution of the analysis raster; higher = finer skeleton, slower
    px_per_em: int = 256
    #: anti-alias / winding supersampling factor (nonzero-winding scanline fill)
    supersample: int = 3
    #: coverage threshold when binarising the anti-aliased mask
    threshold: float = 0.5
    #: morphological closing radius in px (fills hairline gaps between components)
    closing_radius: int = 2
    #: morphological opening radius in px (kills salt noise / thin whiskers)
    opening_radius: int = 0
    #: drop connected components smaller than this fraction of the raster area
    min_component_area_ratio: float = 0.0008
    #: fill holes smaller than this fraction of the raster area (counter-space artefacts)
    max_hole_fill_ratio: float = 0.0


@dataclass
class SkeletonConfig:
    """Thinning + graph extraction + polyline simplification."""

    #: 'zhang' | 'lee' | 'guo_hall' | 'medial_axis'
    method: str = "zhang"
    #: remove terminal branches shorter than this many em (skeleton artefacts)
    prune_spur_em: float = 0.045
    #: drop whole components whose total skeleton length is below this many em
    min_component_len_em: float = 0.02
    #: Douglas-Peucker tolerance in em
    rdp_epsilon_em: float = 0.0035
    #: maximum number of pruning passes
    prune_iterations: int = 3
    #: 'cluster' (rasterise a whole grapheme cluster at once) | 'glyph'
    unit: str = "cluster"


# --------------------------------------------------------------------------- #
# strokes
# --------------------------------------------------------------------------- #


@dataclass
class PlannerConfig:
    """Stochastic stroke-order planner.

    The planner greedily/stochastically walks the skeleton graph and the whole
    resulting plan is scored with the configurable cost function

        C = w_pen_up_count      * (#pen lifts)
          + w_pen_up_distance   * (total pen-up travel, em)
          + w_direction_change  * (sum of sharp turn penalties)
          + w_reversal          * (path reversal penalty)
          + w_spatial_order     * (unnatural spatial-order penalty)

    ``num_candidates`` independent stochastic plans are sampled and the cheapest
    is kept, which turns the sampler into an anytime approximate minimiser.
    """

    w_pen_up_count: float = 1.0
    w_pen_up_distance: float = 2.0
    w_direction_change: float = 0.6
    w_reversal: float = 1.5
    w_spatial_order: float = 0.8
    #: 0 -> deterministic greedy, higher -> more diverse stroke orders
    temperature: float = 0.35
    #: number of stochastic plans sampled per cluster; best (lowest C) wins
    num_candidates: int = 4
    #: turn angle (deg) above which a continuation is considered "sharp"
    sharp_angle_deg: float = 75.0
    #: prefer to start strokes at the top / left of a component
    start_top_weight: float = 1.0
    start_left_weight: float = 0.7
    #: prefer travelling downward / rightward when starting a stroke
    prefer_downward: float = 0.5
    prefer_rightward: float = 0.35
    #: penalty for choosing a bridge edge while a non-bridge is available
    bridge_penalty: float = 0.8
    #: probability of gratuitously lifting the pen at a junction (writer-modulated)
    pen_lift_bias: float = 0.05
    #: order of the disconnected parts of a cluster: 'natural' | 'spatial' | 'random'
    component_order: str = "natural"
    #: how many distinct plans are pre-computed and cached per cluster
    #: (0 -> plan freshly for every sample; slower but maximally diverse)
    plan_variants: int = 6


@dataclass
class SmoothingConfig:
    """Polyline -> smooth, uniformly sampled trajectory."""

    #: 'catmull_rom' | 'bspline' | 'bezier' | 'none'
    method: str = "catmull_rom"
    #: centripetal Catmull-Rom exponent (0.5 = centripetal, 0 = uniform, 1 = chordal)
    alpha: float = 0.5
    #: turn angle (deg) above which a vertex is treated as a hard corner and preserved
    corner_angle_deg: float = 68.0
    #: target arc-length spacing of output points, in px
    point_spacing_px: float = 1.6
    #: optional Savitzky-Golay pass (0 window = disabled)
    savgol_window: int = 7
    savgol_poly: int = 2
    #: how strongly the Savitzky-Golay result is blended in (0..1)
    savgol_mix: float = 0.5
    #: never emit fewer points than this per stroke
    min_points: int = 3


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #


Range = tuple[float, float]


@dataclass
class WriterSamplingConfig:
    """Ranges from which a procedural :class:`~khmer_synth.writer.style.WriterStyle`
    is drawn.  A writer id + seed deterministically yields the same style."""

    #: Master "how fast / how casually does this person write" axis in [0, 1].
    #: 0 = slow, careful, evenly spaced printing; 1 = fast fluent scrawl.  It is
    #: NOT an independent knob - it drives spacing irregularity, stroke
    #: connection, corner rounding, per-character bounce and overshoot together,
    #: because in real handwriting those all covary.  See writer/style.py.
    fluency: Range = (0.05, 1.0)

    slant: Range = (-0.18, 0.38)            # shear x += slant * (baseline - y)
    horizontal_scale: Range = (0.86, 1.16)
    vertical_scale: Range = (0.84, 1.18)
    character_spacing: Range = (0.84, 1.14)  # mean multiplier on shaped advances
    word_spacing: Range = (0.8, 1.6)
    line_spacing: Range = (0.9, 1.4)

    # --- inter-character spacing irregularity (fluency-driven) --- #
    #: i.i.d. gap-to-gap variation, sigma in em.  This is what stops the line
    #: from looking typeset; at the top of the range some characters touch and
    #: others sit a third of an em apart, which is what fast writing looks like.
    spacing_jitter: Range = (0.005, 0.115)
    #: low-frequency correlated drift of the gap width along the line, em.
    #: Real writers bunch up and then spread out over several characters.
    spacing_drift: Range = (0.0, 0.085)
    #: fraction of gaps that collapse into a near-touch / overlap
    spacing_tight_rate: Range = (0.0, 0.34)
    #: v2: relative weight of each writer archetype (careful / neat / average /
    #: fast / scrawl / shaky).  The v1 mix is the default.  An adult-hand preset
    #: drops "shaky" and most of "careful": tremor and slow, separately placed
    #: letters are what read as a child's hand.
    archetype_weights: dict[str, float] = field(
        default_factory=lambda: {
            "careful": 0.10, "neat": 0.16, "average": 0.28, "fast": 0.26, "scrawl": 0.12, "shaky": 0.08
        }
    )
    #: v2: cap a collapsed gap at this fraction of the *preceding cluster's own
    #: advance* (0 = v1 behaviour, fixed 0.30 em cap).  A 0.26 em collapse is a
    #: touch after a wide consonant but piles narrow digits on top of each other.
    spacing_tight_max_frac: float = 0.0

    baseline_amplitude: Range = (0.0, 0.052)   # in em
    baseline_frequency: Range = (0.15, 1.3)    # cycles per em of advance
    baseline_drift: Range = (-0.06, 0.06)      # slope (dy/dx)
    baseline_walk: Range = (0.0, 0.038)        # random-walk amplitude, em
    local_jitter: Range = (0.0, 0.016)         # em, correlated point noise
    tremor: Range = (0.0, 0.010)               # em, high frequency noise
    stroke_smoothness: Range = (0.35, 1.0)     # 1 = very smooth
    stroke_width: Range = (0.030, 0.075)       # em
    pressure_mean: Range = (0.32, 0.78)
    pressure_variation: Range = (0.03, 0.22)
    writing_speed: Range = (0.55, 2.2)         # em / second-ish scale
    speed_variation: Range = (0.05, 0.32)
    pen_lift_bias: Range = (0.0, 0.22)
    glyph_rotation: Range = (0.0, 0.075)       # radians, per-cluster sigma
    glyph_translation: Range = (0.0, 0.032)    # em, per-cluster sigma
    glyph_scale_var: Range = (0.0, 0.085)      # per-cluster sigma
    local_elasticity: Range = (0.0, 0.030)     # em, low frequency stroke bending
    overshoot: Range = (0.0, 0.055)            # em, stroke start/end extension
    slant_variation: Range = (0.0, 0.085)      # per-cluster slant sigma

    #: how far the writer abbreviates the printed letterform: collapsing small
    #: loops to cusps, reducing subscripts, dropping fine detail.  Font skeletons
    #: give the full textbook shape of every letter, which is what a child
    #: copying from the board produces; a practised hand writes a *reduced* form.
    abbreviation: Range = (0.0, 0.85)
    #: directional stroke-width contrast: how much thicker the downstroke is
    #: than the upstroke.  A hand that presses on the pull stroke and lightens
    #: on the push stroke is the single strongest "adult" cue; a monoline hand
    #: reads as school printing regardless of how the letters are shaped.
    width_contrast: Range = (0.10, 0.55)

    # --- stroke connection (the main "fast handwriting" cue) --- #
    #: probability of carrying the pen through to the next stroke instead of
    #: lifting it, when the gap is short enough
    connect_probability: Range = (0.15, 0.95)
    #: longest pen-up gap (em) that may be turned into a connecting ligature
    connect_max_gap: Range = (0.30, 1.20)
    #: safety cap on how many strokes may be fused into one pen-down
    connect_max_chain: int = 16
    #: reject carry-overs that climb/drop more than this many em
    connect_max_rise_em: float = 0.60
    #: how much a fast hand rounds off corners it should have drawn sharply
    corner_rounding: Range = (0.0, 1.0)

    # --- v2 realism (measured on real Khmer handwriting; all off by default) --- #
    #: extra forward lean added as ``slant += slant_fluency_lean * fluency * U(0.3, 1)``.
    #: 0.10 reproduces the v1 behaviour.  Real slant is centred on zero, so v2
    #: presets set this to 0 and centre ``slant`` instead.
    slant_fluency_lean: float = 0.10
    #: detached above-marks (ិ ី ឹ ើ ំ ់ ...) are shifted *up* by this many em
    #: (times 0.6..1.4 per allograph).  Real hands float them well above the body.
    mark_shift_above: Range = (0.0, 0.0)
    #: ... and drift them to the right by this many em (times 0.4..1.6)
    mark_drift_x: Range = (0.0, 0.0)
    #: per-mark scale sigma (a hand draws marks smaller or larger than the font)
    mark_scale_var: Range = (0.0, 0.0)
    #: fractional vertical stretch of everything below the baseline zone (coeng,
    #: below-vowels): real hands hang subscripts deeper than a typeface does
    below_stretch: Range = (0.0, 0.0)

    # --- letter-shape variety (UKTR local; all off by default) --- #
    #: per-occurrence letter warp: amplitude (em) of a smooth displacement field
    #: whose correlation length (``glyph_warp_scale``, em) is below a letter's
    #: width, so every copy of every letter is bent a little differently.  A
    #: function of position, so junctions and attached marks move together.
    glyph_warp: Range = (0.0, 0.0)
    glyph_warp_scale: Range = (0.3, 0.3)
    #: per-writer size of tight curls / small loops relative to the font (1 =
    #: font); curls found by tangent turning, their ends stay put
    loop_scale: Range = (1.0, 1.0)
    #: per-writer stroke-end bias, em: > 0 lengthens every stroke end along its
    #: tangent, < 0 trims it (strokes shorter than 4x the trim are left alone)
    stroke_end_bias: Range = (0.0, 0.0)
    #: probability the writer uses a given pen model
    pen_models: dict[str, float] = field(
        default_factory=lambda: {
            "ballpoint": 0.34,
            "gel": 0.16,
            "fountain": 0.28,
            "pencil": 0.12,
            "marker": 0.10,
        }
    )
    ink_colors: dict[str, float] = field(
        default_factory=lambda: {"black": 0.5, "dark_blue": 0.3, "blue": 0.15, "faded": 0.05}
    )


@dataclass
class TimingConfig:
    """Curvature-driven synthetic timing (see writer/timing.py)."""

    #: base pen speed in em of arc length per second at writer speed 1.0
    base_speed_em_per_s: float = 7.0
    #: v = base / (1 + lambda * kappa_em) ** exponent
    #: kappa_em is curvature in radians per em of arc length.  The default
    #: exponent 1/3 reproduces the empirical "two-thirds power law" of human
    #: drawing (v ~ radius^(1/3)); set exponent = 1.0 for the plain 1/(1+lk) form,
    #: which slows the pen far too much on the tight curls of Khmer glyphs.
    curvature_lambda: float = 2.6
    curvature_exponent: float = 0.3333
    #: fraction of the stroke over which the pen accelerates / decelerates
    ramp_fraction: float = 0.18
    #: minimum speed multiplier at stroke ends
    end_speed_factor: float = 0.30
    #: multiplicative speed noise (log-normal sigma), smoothed along the stroke
    speed_noise_sigma: float = 0.12
    speed_noise_scale_px: float = 40.0
    #: pen-up travel speed multiplier and constant reaction time (ms)
    pen_up_speed_factor: float = 2.4
    pen_up_reaction_ms: float = 22.0
    #: clamp
    min_dt_ms: float = 0.5
    max_dt_ms: float = 120.0


@dataclass
class PressureConfig:
    """Synthetic pen pressure."""

    #: rise / fall fraction of the stroke for the pressure envelope
    ramp_fraction: float = 0.16
    envelope_exponent: float = 1.0
    #: correlated (low-frequency) pressure noise
    noise_scale_px: float = 55.0
    #: extra pressure when the pen slows down (curves, stroke ends)
    speed_coupling: float = 0.18
    #: low-pass width (in points) applied to the speed-coupling term - grip force
    #: has mass and cannot track instantaneous speed changes
    coupling_smooth_points: float = 6.0
    #: final low-pass (in points) applied to the pressure signal; bounds the
    #: per-sample derivative so the trace stays physically plausible
    smooth_points: float = 2.5
    #: pressure multiplier on connecting ligatures (see strokes/connect.py) -
    #: the pen is barely touching the paper while it carries over
    ligature_pressure: float = 0.42
    #: pressure floor for a pen that is down
    min_pressure: float = 0.05
    max_pressure: float = 1.0


@dataclass
class DeformationConfig:
    """Which deformation levels are active (all writer-parameterised)."""

    global_transform: bool = True
    baseline: bool = True
    cluster_level: bool = True
    stroke_level: bool = True
    point_level: bool = True
    #: v2: displace detached marks / stretch the below-baseline zone (see
    #: ``writer/deformation.py::displace_marks``).  A no-op unless the writer's
    #: ``mark_*`` / ``below_stretch`` parameters are non-zero.
    mark_level: bool = True
    #: below this many em under the baseline counts as the subscript zone
    below_zone_depth_em: float = 0.06
    #: spatial correlation length of the smooth noise field, in px
    noise_scale_px: float = 55.0
    tremor_scale_px: float = 6.0
    #: hard clamp on total displacement, in em (guards against exploded layouts)
    max_displacement_em: float = 0.22


# --------------------------------------------------------------------------- #
# layout / rendering / degradation
# --------------------------------------------------------------------------- #


@dataclass
class LayoutConfig:
    canvas_width: int = 1024
    canvas_height: int = 192
    #: target x-height-ish size: em size as a fraction of canvas height
    em_fraction: Range = (0.42, 0.60)
    #: baseline position as a fraction of canvas height
    baseline_fraction: Range = (0.62, 0.74)
    padding_px: int = 18
    #: Randomise the margin per sample over this inclusive range instead of using
    #: the fixed ``padding_px``.  ``None`` (default) = v1, fixed margin.
    #:
    #: WHY: MEASURED at the recogniser's own input height - real crops are cut
    #: flush to the ink (gemma_novel top/bottom margin 0.016/0.017, ink band
    #: 0.968; real_test 0.040/0.042) while a fixed ``padding_px: 10`` on a 96 px
    #: canvas gives 0.107/0.109 and an ink band of only 0.78.  That framing gap
    #: is 1.26x of the measured 1.49x deficit in advance-per-character, which is
    #: the single largest systematic real-vs-synthetic discrepancy: it directly
    #: sets how many CTC frames a fixed-height CRNN spends per character.
    padding_px_range: Range | None = None
    #: if the shaped line is wider than the canvas, shrink to fit (else clip)
    fit_to_width: bool = True
    min_em_px: float = 24.0
    #: random horizontal placement of the line inside the free space
    random_x_placement: bool = True
    #: 'auto' grows the canvas width to fit long text instead of shrinking the em
    auto_width: bool = False
    max_canvas_width: int = 4096
    #: v2: after placement, push each grapheme cluster right until its ink starts
    #: at least this many em after the previous cluster's ink ends (negative =
    #: a small overlap is allowed, which keeps real "touching" pairs).  None = off
    #: (v1).  Without it tight spacing on narrow clusters (digits!) piles letters
    #: on top of each other into an unreadable blob.
    min_cluster_gap_em: float | None = None


@dataclass
class RenderConfig:
    """Trajectory -> clean RGB image."""

    #: supersampling of the ink canvas
    supersample: int = 2
    #: stamp spacing along the path (px, at supersampled resolution)
    stamp_spacing_px: float = 0.45
    #: width = base_width * (0.5 + pressure * 1.0)  (see ink.py)
    width_pressure_gain: float = 1.0
    width_pressure_bias: float = 0.5
    #: alpha  = (0.55 + 0.45 * pressure) * pen alpha
    alpha_pressure_gain: float = 0.45
    alpha_pressure_bias: float = 0.55
    #: fountain-pen style width modulation by direction
    nib_angle_deg: float = -40.0
    #: direction the hand *pulls* the pen (canvas space, y down).  Motion along
    #: this axis is the downstroke and gets ``WriterStyle.width_contrast`` more
    #: width; motion against it is the upstroke and gets that much less.
    pull_angle_deg: float = 78.0
    #: how much of the writer's width_contrast the direction term actually uses
    direction_width_gain: float = 1.0
    #: opacity follows the same axis, but weakly (ink lays down thicker, not
    #: much darker, on the pull stroke)
    direction_alpha_gain: float = 0.30
    #: ink texture / bleed
    texture_strength: Range = (0.0, 0.25)
    bleed_probability: float = 0.35
    bleed_sigma_px: Range = (0.4, 1.4)
    #: draw pen-up segments faintly (debug only)
    draw_pen_up: bool = False


@dataclass
class BackgroundConfig:
    """Paper simulation."""

    kinds: dict[str, float] = field(
        default_factory=lambda: {
            "white": 0.25,
            "off_white": 0.3,
            "textured": 0.2,
            "ruled": 0.15,
            "grid": 0.1,
        }
    )
    base_brightness: Range = (0.86, 1.0)
    texture_strength: Range = (0.0, 0.06)
    ruled_line_alpha: Range = (0.06, 0.25)
    ruled_spacing_frac: Range = (0.5, 1.1)  # of canvas height
    #: Rule thickness as a fraction of canvas height.  ``None`` (default) = v1,
    #: a 1 px line (2 px with p = 0.2) regardless of canvas size.
    #:
    #: WHY: the config asks for a ruled or grid background on 24% of samples but
    #: MEASURED only 0.08% of crops carry a detectable full-width rule, against
    #: 10.3% of real ones.  Three compounding reasons: the alpha range tops out
    #: at ~27 gray levels of contrast, ``ruled_spacing_frac`` of 0.55-1.15 x
    #: height usually fits a single rule in frame, and that rule is drawn on the
    #: baseline where the ink covers it - then a 1 px line at h=96 is averaged
    #: away by the resize to h=48 the recogniser applies.  A height-relative
    #: width survives the resize.
    ruled_line_width_frac: Range | None = None


@dataclass
class DegradationConfig:
    """Document / camera degradation applied on top of the clean render."""

    enabled: bool = True
    #: each op fires with its own probability
    p_blur: float = 0.45
    blur_sigma: Range = (0.3, 1.6)
    p_motion_blur: float = 0.12
    motion_kernel: Range = (3, 9)
    p_downscale: float = 0.35
    downscale_factor: Range = (0.45, 0.9)
    p_brightness: float = 0.6
    brightness: Range = (0.75, 1.25)
    p_contrast: float = 0.5
    contrast: Range = (0.7, 1.35)
    p_uneven_light: float = 0.45
    uneven_light_strength: Range = (0.08, 0.42)
    p_shadow: float = 0.2
    shadow_strength: Range = (0.1, 0.4)
    p_perspective: float = 0.35
    perspective_strength: Range = (0.002, 0.02)
    p_rotate: float = 0.5
    rotate_deg: Range = (-2.5, 2.5)
    p_noise: float = 0.6
    noise_sigma: Range = (0.004, 0.045)
    p_jpeg: float = 0.55
    jpeg_quality: Range = (35, 92)
    p_ink_fade: float = 0.25
    ink_fade: Range = (0.15, 0.55)
    p_stains: float = 0.18
    stain_count: Range = (1, 4)
    p_sharpen: float = 0.15


@dataclass
class PaperConfig:
    """Physical paper / reprographic degradation, delegated to **Augraphy**.

    This is the "ink on real paper, then photocopied / scanned" stage.  It is a
    deliberately *narrow* wrapper: only the Augraphy augmentations that stay
    physically meaningful on a **text-line crop** are used (see
    ``augmentation/paper.py`` for the tier tables and for the list of
    augmentations that were evaluated and rejected).

    Composition with :class:`DegradationConfig`
    -------------------------------------------
    The two stages model different halves of the physical chain and are meant to
    run *together*, paper first::

        render (ink laid on paper)
          -> paper   : ink bleeding into the fibres, mottled ballpoint flow,
                       low-ink streaks, toner speckle, fax binarisation
          -> degrade : the *capture* chain - optical blur, page pose, uneven
                       illumination, sensor noise, downscale, JPEG

    That is the order a real sample goes through, so ``stage`` defaults to
    ``"before_degradation"``.  ``"after_degradation"`` and
    ``"replace_degradation"`` exist for experiments (the latter switches the
    ``degradation`` stage off for the run) but are not what the shipped presets
    use.

    Augraphy is an **optional** dependency (``pip install
    'khmer-handwriting-synth[paper]'``).  Nothing imports it unless
    ``enabled`` is true, and the error raised when it is missing names the
    extra to install.
    """

    #: off by default - every pre-existing preset keeps its exact behaviour
    enabled: bool = False
    #: intensity tier: ``'light'`` (subtle, realistic scan - use for the
    #: already-handwritten faces in ``fonts/km``) or ``'heavy'``
    #: (photocopy/fax grade - use for the printed Google faces in ``fonts/``)
    preset: str = "light"
    #: probability that a given sample gets the paper treatment at all, so one
    #: dataset can mix treated and untreated images
    apply_prob: float = 1.0
    #: global multiplier on every per-effect probability *and* on the amplitude
    #: of the effects that have a natural zero.  1.0 = the tier as tabulated.
    intensity: float = 1.0
    #: per-effect switches, e.g. ``{"faxify": false}``.  Keys not listed keep the
    #: tier default (on).  Unknown keys raise, so a typo fails loudly.
    effects: dict[str, bool] = field(default_factory=dict)
    #: ``'before_degradation'`` | ``'after_degradation'`` | ``'replace_degradation'``
    stage: str = "before_degradation"


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


@dataclass
class DatasetConfig:
    output_dir: str = "dataset"
    num_writers: int = 1000
    styles_per_text: int = 1
    #: image formats to write for clean/degraded ('png' | 'webp' | 'jpg')
    image_format: str = "png"
    save_clean: bool = True
    save_degraded: bool = True
    save_trajectory: bool = True
    #: also store the compact (dx, dy, pen, pressure, dt) array in the json
    save_compact: bool = True
    compact_only: bool = False
    workers: int = 0  # 0 -> cpu_count
    chunk_size: int = 8
    seed: int = 1234
    #: cap on samples (0 = all corpus lines x styles_per_text)
    limit: int = 0
    #: on-disk cache for cluster skeletons (shared between workers)
    cache_dir: str = ".cache/khmer_synth"
    use_disk_cache: bool = True
    #: skip samples whose validation fails instead of writing them
    validate: bool = True
    shard_size: int = 5000  # files per sub-directory


@dataclass
class AbbreviationConfig:
    """Reduction of the printed letterform toward a practised hand.

    Everything upstream of this stage produces the *textbook* shape of a letter,
    because the geometry is the medial axis of a font outline.  A practised hand
    does not draw that shape more sloppily - it draws a genuinely different,
    reduced shape: loops that a typeface closes become cusps, subscripts shrink
    toward the base, and fine detail disappears.  Deformation noise cannot
    produce that, which is why an unabbreviated trace reads as school printing
    however much it is wobbled.

    All lengths are in em so they are independent of the rendered size.

    **Disabled by default: evaluated on Khmer and rejected.**  The premise holds
    for Latin, where a descender or a closed bowl is largely ornament that a
    practised hand may reduce.  It does not hold here.  In Khmer the coeng below
    the baseline *is a consonant*, and the loops are what separate one letter
    from the next, so both reductions delete content rather than decoration:
    with the descender compression on, ខ្ញុំ loses its ញ and collapses toward a
    blob.  Side-by-side rendering of the three reductions in isolation put the
    damage on ``loop_max_collapse`` and ``subscript_max_shrink``;
    ``detail_epsilon_em`` alone is harmless but also buys almost nothing.

    Kept, configured and tested rather than deleted: the machinery is sound and
    the same idea may well pay off on a script whose below-baseline marks really
    are ornamental, or here at much lower strength.  Set ``enabled = True`` to
    experiment - and look at a stacked cluster before trusting the result.
    """

    enabled: bool = False

    # --- loop collapse --- #
    #: arc length of the shortest sub-path worth treating as a collapsible loop.
    #: Measured Khmer skeleton loops run 0.32-0.96 em of arc (median 0.45), so
    #: the window has to be wide; the old 0.055-0.42 window excluded the median
    #: loop entirely while admitting straight runs.
    loop_min_em: float = 0.12
    #: above this the "loop" is the body of the letter, not a flourish - a
    #: writer who collapsed those would be writing a different letter
    loop_max_em: float = 0.75
    #: how close the two ends of the sub-path must come to count as a loop
    loop_close_em: float = 0.14
    #: arc length must exceed this multiple of the closing distance.  Without a
    #: tortuosity test *every* nearly-straight run qualifies (a point 0.05 em
    #: along a straight line is 0.05 em away), which is not a loop at all.
    loop_min_ratio: float = 3.0
    #: strongest pinch toward the loop centroid, at abbreviation = 1.  The sin^2
    #: taper means the extreme point moves by less than this; measured on real
    #: skeletons, 0.85 here displaces the loop by ~0.058 em, about a third of a
    #: typical loop's width.
    loop_max_collapse: float = 0.85

    # --- subscript (coeng) reduction --- #
    #: where the descender zone starts, below the baseline (line space, y down,
    #: baseline at y = 0).  Points deeper than this are compressed toward it.
    subscript_min_depth_em: float = 0.05
    #: strongest size reduction of a subscript, at abbreviation = 1
    subscript_max_shrink: float = 0.32

    # --- detail softening --- #
    #: Douglas-Peucker tolerance at abbreviation = 1.  Kept well under the
    #: thinnest meaningful Khmer feature so it removes fussiness, not structure.
    detail_epsilon_em: float = 0.022
    #: never simplify a stroke below this many points
    min_points: int = 4
    #: per-letter scatter of the abbreviation amount (a hand is not uniform)
    amount_jitter: float = 0.18


@dataclass
class PipelineConfig:
    text: TextConfig = field(default_factory=TextConfig)
    shaping: ShapingConfig = field(default_factory=ShapingConfig)
    fonts: FontConfig = field(default_factory=FontConfig)
    raster: RasterConfig = field(default_factory=RasterConfig)
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    abbreviation: AbbreviationConfig = field(default_factory=AbbreviationConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    writer: WriterSamplingConfig = field(default_factory=WriterSamplingConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    pressure: PressureConfig = field(default_factory=PressureConfig)
    deformation: DeformationConfig = field(default_factory=DeformationConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # ---------------- serialisation ---------------- #

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=indent, ensure_ascii=False), "utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        return from_dict(cls, data)

    @classmethod
    def load(cls, path: str | Path | None) -> PipelineConfig:
        if path is None:
            return cls()
        p = Path(path)
        raw = p.read_text("utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml  # optional dependency

            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        return cls.from_dict(data or {})


# --------------------------------------------------------------------------- #
# generic recursive dataclass loader (small local stand-in for `dacite`)
# --------------------------------------------------------------------------- #


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def from_dict(cls: type, data: Any) -> Any:
    """Recursively build a dataclass ``cls`` from nested dicts, honouring type
    hints for tuples/lists/dicts/Optionals.  Unknown keys raise, so typos in a
    config file fail loudly instead of being silently ignored."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise KeyError(f"unknown config key(s) for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints[name], value, f"{cls.__name__}.{name}")
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if _is_dataclass_type(tp):
        return from_dict(tp, value)
    if origin is typing.Union or str(origin) == "types.UnionType":
        if value is None:
            return None
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        return _coerce(non_none[0], value, where)
    if origin in (list, list):
        return [_coerce(args[0], v, where) for v in value]
    if origin in (tuple, tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, where) for v in value)
        return tuple(_coerce(a, v, where) for a, v in zip(args, value))
    if origin in (dict, dict):
        return {k: _coerce(args[1], v, where) for k, v in value.items()}
    if tp in (int, float, str, bool):
        try:
            return tp(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError(f"cannot coerce {value!r} to {tp.__name__} at {where}") from exc
    return value


def merge_overrides(cfg: PipelineConfig, overrides: list[str]) -> PipelineConfig:
    """Apply ``a.b.c=value`` CLI overrides (value parsed as JSON, else string)."""
    data = cfg.to_dict()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like path.to.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(f"unknown config section {part!r} in override {item!r}")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {parts[-1]!r} in override {item!r}")
        node[parts[-1]] = value
    return PipelineConfig.from_dict(data)
