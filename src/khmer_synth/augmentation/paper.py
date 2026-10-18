"""Paper / reprographic degradation, delegated to Augraphy.

Where :mod:`~khmer_synth.augmentation.degradation` models the *capture* chain
(optical blur, page pose, illumination, sensor noise, resampling, JPEG), this
module models what happens to the ink and the sheet *before* anybody points a
camera at it::

    render (ink laid on paper)
      -> paper    : ink bleeding into the fibres, mottled ballpoint flow,
                    low-ink / dirty-roller streaks, toner speckle, fax
                    binarisation
      -> degrade  : the capture chain

so the paper stage runs **before** ``degrade`` (``PaperConfig.stage``).

Only augmentations that stay physically meaningful on a **text-line crop** are
used.  The engine emits line images (96 px tall in the v2 presets), not pages,
and most of Augraphy's catalogue silently assumes a full page.  What was
evaluated on real 96 px crops and *rejected*:

============================  ==========================================================
augmentation                  why not
============================  ==========================================================
``Brightness`` ``Gamma``      already in ``degradation`` (``p_brightness`` / ``p_contrast``)
``Jpeg``                      already in ``degradation`` (``p_jpeg``)
``Geometric`` ``Rescale``     already in ``degradation``; also changes the image size
``DPIMetrics``               changes the image size
``LightingGradient``          already in ``degradation`` (``p_uneven_light``)
``ShadowCast``                already in ``degradation`` (``p_shadow``)
``Stains``                    already in ``degradation`` (``p_stains``)
``SubtleNoise`` ``LowLightNoise``  already in ``degradation`` (``p_noise``)
``NoiseTexturize``            paper fibre is already in ``rendering/backgrounds.py``
``BrightnessTexturize``       paper grain is already in ``rendering/backgrounds.py``
``ColorPaper``                paper tint is already in ``rendering/backgrounds.py``
``PageBorder`` ``BookBinding``     page-scale: a line crop has no page edge or spine
``Folding`` ``SectionShift`` ``Squish``  page-scale geometry; shears a crop into nonsense
``BindingsAndFasteners``      staples/holes belong to a page margin
``BleedThrough``              pastes a mirrored copy of *this* line as the show-through,
                              i.e. it teaches the OCR model to read ghost text
``Letterpress``               measured: bleaches the whole crop, not just the ink -
                              the line became unreadable at every setting tried
``Hollow``                    turns a 2-3 px pen stroke into an outline; destroys it
``NoisyLines`` ``Markup``     draws full-width lines / highlighter over a line crop,
                              which reads as a strikethrough through the label text
``InkShifter``                large ink warping; the writer stage already deforms strokes
``InkColorSwap``              the writer already samples an ink colour per sample
``DirtyRollers``              plausible banding but ~1.8 s per call and it greys the
                              whole crop; ``photocopy_noise`` covers the same ground
``DirtyDrum``                 adds speckle that ``degradation``'s noise already adds
``LowInkPeriodicLines``       its bands are spaced for a page: on a 96 px crop every
                              period tried banded the whole line, and a high line
                              alpha bleached the text off the paper
``Dithering``                 colour dither over a colour crop; looks like screen noise
``Faxify(halftone=1)``        243 ms/call and it lays a dotted band across the crop
``Moire`` ``LCDScreenPattern``     screen artefacts, not paper
``DirtyScreen`` ``GlitchEffect``   screen artefacts, not paper
``DotMatrix`` ``DelaunayTessellation`` ``VoronoiTessellation``  pattern generators, not documents
``DepthSimulatedBlur`` ``LensFlare`` ``DoubleExposure`` ``ReflectedLight``  page/scene-scale optics
``Scribbles`` ``WaterMark`` ``ImageOverlay``  page-scale overlays that occlude the label
``PixelBleed``                sub-pixel/LCD effect, and it has no ``p`` parameter
============================  ==========================================================

Determinism
-----------
Augraphy draws from the *global* :mod:`random` and :mod:`numpy.random` state.
:func:`apply_paper` therefore seeds both from the engine's own per-sample
:class:`numpy.random.Generator` and restores the previous global state on the
way out, so the stage is reproducible from ``(seed, text, writer_id)`` alone and
does not perturb anything else in the process.  Every augmentation listed in the
tier tables was checked to be byte-identical across repeated runs at a fixed
seed (numba-jitted kernels included).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from ..config import PaperConfig

# --------------------------------------------------------------------------- #
# optional dependency
# --------------------------------------------------------------------------- #

_MISSING_HINT = (
    "the 'paper' augmentation stage needs Augraphy, which is an optional "
    "dependency.  Install it with\n"
    "    pip install 'khmer-handwriting-synth[paper]'\n"
    "or\n"
    "    pip install augraphy>=8.2\n"
    "or set paper.enabled=false in the config (e.g. --set paper.enabled=false)."
)


def augraphy_available() -> bool:
    """``True`` if the optional Augraphy dependency can be imported."""
    try:
        import augraphy  # noqa: F401
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


def _require_augraphy():
    """Import Augraphy lazily, with an actionable message when it is absent."""
    try:
        import augraphy
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_MISSING_HINT) from exc
    return augraphy


# --------------------------------------------------------------------------- #
# tiers
# --------------------------------------------------------------------------- #

#: Fixed application order, following the physical chain: what the pen did to
#: the paper, then what the copier did to the page, then what the sensor did to
#: the colours.
#:
#: ``faxify`` must come *before* ``photocopy_noise``: the copier binarises what
#: it scanned and only then lays down toner.  With the two the other way round,
#: Otsu sees the grey toner band as foreground and flips the whole band to solid
#: black, which buries part of the line (observed, not hypothetical).
EFFECTS: tuple[str, ...] = (
    "ink_bleed",
    "ink_mottling",
    "low_ink_lines",
    "lines_degradation",
    "faxify",
    "photocopy_noise",
    "color_shift",
)

#: Per-tier firing probability of each effect.
#:
#: ``light``  - a good scan or phone photo of pen on paper.  Meant for the
#:              ``fonts/km`` faces, whose outlines already carry the pen-stroke
#:              irregularity; more than this adds noise, not signal.
#: ``heavy``  - photocopy / fax grade.  Meant for the 31 printed Google faces at
#:              the top of ``fonts/``, where the glyphs are clean printed
#:              typefaces and the model would otherwise memorise them.
TIERS: dict[str, dict[str, float]] = {
    "light": {
        "ink_bleed": 0.50,
        "ink_mottling": 0.35,
        "low_ink_lines": 0.15,
        "lines_degradation": 0.10,
        "faxify": 0.0,
        "photocopy_noise": 0.08,
        "color_shift": 0.08,
    },
    "heavy": {
        "ink_bleed": 0.65,
        "ink_mottling": 0.55,
        "low_ink_lines": 0.45,
        "lines_degradation": 0.28,
        "faxify": 0.15,
        "photocopy_noise": 0.50,
        "color_shift": 0.14,
    },
}


#: ``BadPhotoCopy`` mask generators that are safe here: 1 = blobs, 2 = gaussian,
#: 3 = perlin, 5 = rectangular.  4 (worley) is excluded - its numba kernel fails
#: to type-infer on numba >= 0.61 and raises mid-sample.
_PHOTOCOPY_NOISE_TYPES: tuple[int, ...] = (1, 2, 3, 5)


def _scaled(rng_range: tuple[float, float], k: float) -> tuple[float, float]:
    """Scale an amplitude range toward zero by ``k`` (``k == 1`` -> unchanged)."""
    return (rng_range[0] * k, rng_range[1] * k)


def _make(name: str, tier: str, k: float, rng: np.random.Generator, ag) -> Any:
    """Build the Augraphy augmentation for ``name`` at ``tier``.

    ``k`` is ``PaperConfig.intensity``; it scales the amplitude of the effects
    that have a natural zero.  Every augmentation is constructed with ``p=1``:
    whether it fires is decided by the engine's own generator in
    :func:`apply_paper`, so the decision stream stays inside our RNG.
    """
    heavy = tier == "heavy"

    if name == "ink_bleed":
        # ink diffusing along the stroke edge into the paper fibres.  This is a
        # ragged, edge-following fringe; the renderer's own ``bleed_sigma_px``
        # is an isotropic gaussian on the alpha channel, so the two do not
        # duplicate each other.
        kernel = (5, 5) if heavy and rng.random() < 0.5 else (3, 3)
        return ag.InkBleed(
            intensity_range=_scaled((0.30, 0.70) if heavy else (0.15, 0.35), k),
            kernel_size=kernel,
            severity=_scaled((0.25, 0.45) if heavy else (0.10, 0.25), k),
            p=1,
        )

    if name == "ink_mottling":
        # uneven ink density inside the stroke - a ballpoint that skips and
        # pools.  Nothing upstream models within-stroke density variation.
        return ag.InkMottling(
            ink_mottling_alpha_range=_scaled((0.08, 0.28) if heavy else (0.04, 0.14), k),
            ink_mottling_noise_scale_range=(1, 2),
            ink_mottling_gaussian_kernel_range=(3, 5),
            p=1,
        )

    if name == "low_ink_lines":
        # a printer/copier running out of toner, or a pen skipping: pale
        # horizontal streaks across the line.
        #
        # ``use_consistent_lines=True`` (Augraphy's default) draws a solid pale
        # streak - the physical thing.  With ``False`` each streak is a row of
        # uniform noise, which at 96 px eats ~30 % of the rows and dissolves the
        # glyphs into dots.  ``LowInkPeriodicLines`` is rejected outright: its
        # bands are spaced for a page, so on a line crop every setting tried
        # covered the whole line, and a high alpha bleached the text away.
        return ag.LowInkRandomLines(
            count_range=(3, 8) if heavy else (2, 4),
            use_consistent_lines=True,
            noise_probability=0.1,
            p=1,
        )

    if name == "lines_degradation":
        # breaks a fraction of the stroke edges, i.e. the thin parts of a letter
        # drop out the way they do on a third-generation photocopy.
        return ag.LinesDegradation(
            line_gradient_range=(32, 255),
            line_gradient_direction=(0, 2),
            line_split_probability=_scaled((0.15, 0.35) if heavy else (0.08, 0.18), k),
            line_replacement_value=(235, 255),
            line_min_length=(8, 16),
            line_long_to_short_ratio=(3, 6),
            line_replacement_probability=_scaled((0.25, 0.45) if heavy else (0.12, 0.25), k),
            line_replacement_thickness=(1, 2),
            p=1,
        )

    if name == "photocopy_noise":
        # the canonical photocopier look: toner speckle that clumps toward one
        # edge of the sheet.  ``wave_pattern`` and ``edge_effect`` are page-scale
        # and are switched off - on a line crop they read as a torn edge.
        #
        # The noise type is drawn here rather than with Augraphy's ``-1``
        # (= "any of 1..5") for two reasons: type 4 (worley) fails to compile in
        # its numba kernel on numba >= 0.61 and would abort the sample, and
        # ``numba_jit=0`` makes type 3 (perlin) ~50x faster at this crop size.
        noise_type = int(rng.choice(_PHOTOCOPY_NOISE_TYPES))
        return ag.BadPhotoCopy(
            noise_type=noise_type,
            noise_side="random",
            noise_iteration=(1, 2) if heavy else (1, 1),
            noise_size=(1, 3) if heavy else (1, 2),
            # noise_value is the *darkness* floor of the toner speckle.  On a
            # 96 px crop the noise side band covers a large fraction of the
            # image, so a dark floor (Augraphy's default (32, 128)) drops a grey
            # slab over half the text; (140, 210) keeps it a speckle.
            noise_value=(140, 210) if heavy else (170, 225),
            noise_sparsity=(0.35, 0.85),
            noise_concentration=_scaled((0.03, 0.11) if heavy else (0.02, 0.06), k),
            blur_noise=-1,
            wave_pattern=0,
            edge_effect=0,
            numba_jit=0,
            p=1,
        )

    if name == "faxify":
        # hard binarisation: a fax or a high-contrast photocopy.  Colour ink is
        # gone, stroke ends thicken or drop out.  Only in the heavy tier, and
        # only occasionally - it is a large domain shift, which is exactly what
        # the printed faces need and what the handwritten faces do not.
        return ag.Faxify(
            scale_range=(1.0, 1.0),  # anything else resizes the crop
            monochrome=1,
            monochrome_method="threshold_otsu",
            halftone=0,
            p=1,
        )

    if name == "color_shift":
        # channel mis-registration of a cheap CIS scanner.  Kept to a single
        # 1 px iteration: measured on 96 px crops, two iterations or a 2-3 px
        # offset puts a red/cyan fringe on every stroke that reads as an
        # anaglyph, not as a scan.
        return ag.ColorShift(
            color_shift_offset_x_range=(1, 1),
            color_shift_offset_y_range=(1, 1),
            color_shift_iterations=(1, 1),
            color_shift_brightness_range=(0.97, 1.03),
            color_shift_gaussian_kernel_range=(3, 3),
            p=1,
        )

    raise ValueError(f"unknown paper effect {name!r}; known: {sorted(EFFECTS)}")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def apply_paper(
    image: np.ndarray, cfg: PaperConfig, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(paper-degraded uint8 RGB, info)``.

    Mirrors :func:`~khmer_synth.augmentation.degradation.degrade`: same input and
    output contract (contiguous ``uint8`` ``[H, W, 3]`` **RGB**, unchanged
    shape), and an ``info`` dict whose ``"ops"`` entry lists what actually fired.

    Unlike ``degrade`` this stage is purely photometric - no geometry - so it
    never invalidates the trajectory alignment and returns no homography.

    :raises ImportError: if ``cfg.enabled`` and Augraphy is not installed.
    :raises ValueError: on an unknown tier name or an unknown key in
        ``cfg.effects``.
    """
    info: dict[str, Any] = {"ops": []}
    if not cfg.enabled:
        return image, info

    tier = cfg.preset
    if tier not in TIERS:
        raise ValueError(f"unknown paper preset {tier!r}; known: {sorted(TIERS)}")
    unknown = set(cfg.effects) - set(EFFECTS)
    if unknown:
        raise ValueError(f"unknown paper.effects key(s): {sorted(unknown)}; known: {sorted(EFFECTS)}")

    if rng.random() >= float(cfg.apply_prob):
        info["skipped"] = True
        return image, info

    ag = _require_augraphy()

    img = np.asarray(image)
    if img.ndim == 2:  # defensive: the renderer always emits RGB
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]

    k = max(float(cfg.intensity), 0.0)
    probs = TIERS[tier]

    # Augraphy is written against OpenCV's BGR convention (its colour ops assume
    # it); the engine works in RGB, so swap on the way in and back on the way out.
    work = np.ascontiguousarray(img[:, :, ::-1])

    # Augraphy samples its parameters from the *global* random/numpy state.
    # Seed both from our generator so the stage is reproducible, and put the
    # previous global state back afterwards so we do not disturb the process.
    seed = int(rng.integers(0, 2**31 - 1))
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        for name in EFFECTS:
            if not cfg.effects.get(name, True):
                continue
            p = probs[name] * k
            if p <= 0.0 or rng.random() >= p:
                continue
            aug = _make(name, tier, k, rng, ag)
            out = aug(work)
            if isinstance(out, list):  # augraphy returns a list when given masks
                out = out[0]
            out = np.asarray(out)
            if out.ndim == 2:
                out = np.repeat(out[:, :, None], 3, axis=2)
            # every augmentation in the tables is shape-preserving; guard anyway
            # so a future addition can never silently reshape the dataset
            if out.shape[:2] != (h, w):
                continue
            work = np.ascontiguousarray(out[:, :, :3].astype(np.uint8))
            info["ops"].append(name)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    info["preset"] = tier
    return np.ascontiguousarray(work[:, :, ::-1]), info
