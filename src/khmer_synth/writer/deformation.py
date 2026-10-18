"""Multi-scale handwriting deformation.

Deformation is applied at five nested scales, because that is how handwriting
variability actually decomposes:

============  ===============================================================
global        slant, x/y scale, small rotation - constant for a writer
baseline      drift + sinusoid + random walk along the line
cluster       per-grapheme translation / rotation / scale / slant wobble
stroke        overshoot, shortening, low-frequency bending (elasticity)
point         correlated jitter (a smooth 2-D field) + high-frequency tremor
============  ===============================================================

Cluster- and stroke-level deformation is drawn from a *per-allograph* stream
keyed on ``(writer, cluster text)`` rather than from the shared sample stream.
That makes every occurrence of the same letter, for the same writer, deform
identically - which is what a motor program does.  Drawing it i.i.d. per
occurrence instead makes each copy of a letter wrong in a *different* direction,
and inconsistency of exactly that kind is what reads as an unpractised (child's)
hand rather than a distinctive adult one.  An adult hand is consistently
deviant; only the letter's *placement* is fresh each time.

The key modelling choice is that point-level noise is **spatially correlated**,
not i.i.d.:  a shared smooth 2-D displacement field is sampled at every point of
every stroke, so neighbouring strokes distort *together*.  Independent per-point
Gaussian noise produces the unmistakable "wobbly font" look and, worse, teaches a
downstream model that stroke geometry is high-frequency noise.

Coordinates here are "line space": pixels, x increasing left to right, y = 0 on
the baseline (y down).  Fitting into the final canvas happens afterwards, so
deformation can never push ink outside the image.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import DeformationConfig
from ..strokes.planner import Stroke
from ..text.khmer_chars import count_above_marks
from ..utils.geometry import make_affine, tangents, transform_affine
from ..utils.noise import smooth_noise_1d, smooth_noise_2d, smooth_noise_along
from ..utils.rng import make_rng
from .style import WriterStyle

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class _VectorField:
    """A smooth 2-D displacement field sampled with bilinear interpolation."""

    def __init__(self, rng: np.random.Generator, bbox, scale_px: float, amplitude: float, step: float = 4.0):
        x0, y0, x1, y1 = bbox
        self.x0, self.y0 = x0 - scale_px, y0 - scale_px
        self.step = max(step, 1.0)
        w = int(max(4, np.ceil((x1 - x0 + 2 * scale_px) / self.step)))
        h = int(max(4, np.ceil((y1 - y0 + 2 * scale_px) / self.step)))
        s = max(scale_px / self.step, 0.8)
        self.fx = smooth_noise_2d(rng, (h, w), scale=s, amplitude=amplitude, octaves=2)
        self.fy = smooth_noise_2d(rng, (h, w), scale=s, amplitude=amplitude, octaves=2)

    def sample(self, pts: np.ndarray) -> np.ndarray:
        gx = (pts[:, 0] - self.x0) / self.step
        gy = (pts[:, 1] - self.y0) / self.step
        h, w = self.fx.shape
        gx = np.clip(gx, 0, w - 1.001)
        gy = np.clip(gy, 0, h - 1.001)
        x0 = np.floor(gx).astype(int)
        y0 = np.floor(gy).astype(int)
        tx = (gx - x0)[:, None]
        ty = (gy - y0)[:, None]
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        f = np.stack([self.fx, self.fy], axis=-1)
        v = (
            f[y0, x0] * (1 - ty) * (1 - tx)
            + f[y0, x1] * (1 - ty) * tx
            + f[y1, x0] * ty * (1 - tx)
            + f[y1, x1] * ty * tx
        )
        return v


def _bbox_of(strokes: Sequence[Stroke]) -> tuple[float, float, float, float]:
    pts = np.vstack([s.points for s in strokes if len(s.points)])
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def _extend_endpoints(pts: np.ndarray, start: float, end: float) -> np.ndarray:
    """Overshoot (>0) or shorten (<0) a stroke along its end tangents."""
    if len(pts) < 3 or (abs(start) < 1e-6 and abs(end) < 1e-6):
        return pts
    out = pts.copy()
    t0 = pts[min(2, len(pts) - 1)] - pts[0]
    n0 = np.linalg.norm(t0)
    t1 = pts[-1] - pts[max(0, len(pts) - 3)]
    n1 = np.linalg.norm(t1)
    if n0 > 1e-9 and start != 0:
        out = np.vstack([pts[0] - t0 / n0 * start, out])
    if n1 > 1e-9 and end != 0:
        out = np.vstack([out, pts[-1] + t1 / n1 * end])
    return out


# --------------------------------------------------------------------------- #
# mark displacement (v2)
# --------------------------------------------------------------------------- #


def _allograph_rng_factory(style: WriterStyle, rng, cluster_keys, allograph_seed: int):
    def allograph_rng(cluster: int) -> np.random.Generator:
        """The writer's persistent hand for one letter (falls back to ``rng``)."""
        if not cluster_keys:
            return rng
        key = cluster_keys.get(cluster)
        if not key:
            return rng
        return make_rng("allograph", allograph_seed, style.writer_id, key)

    return allograph_rng


#: A component is only a *candidate* above-mark when its lowest point is no more
#: than this far below the body's highest point (the v1 test, kept as a
#: necessary - no longer sufficient - condition).
_ABOVE_SLACK_EM = 0.25
#: Fallback-only size gate: an above-component is accepted without text evidence
#: only when its skeleton is at most this fraction of the body's.  MEASURED over
#: 40 fonts: the detached horn/serif of a bare consonant has length ratio
#: p05 0.216 / p50 0.437, so 0.25 keeps 5.3% of horns instead of 100%, at the
#: cost of reaching only 18% of genuine marks.  The two distributions overlap
#: almost completely (marked-cluster components: p05 0.127 / p50 0.421), which is
#: precisely why the text gate - not a size gate - is the real fix.
_FALLBACK_MAX_SIZE_RATIO = 0.25


def _above_mark_components(
    strokes: list[Stroke],
    comps: dict[int, list[int]],
    em_px: float,
    n_above: int | None,
) -> list[list[int]]:
    """Pick the components of one cluster that are genuinely floating marks.

    ``n_above`` is how many above-rendering marks the cluster's source text
    contains (``None`` when no text was supplied).  Returns the stroke-index
    lists of the components that may be floated.

    WHY it is done this way.  The v1 test was "any non-body component sitting
    above the body top", which is purely geometric and therefore cannot tell a
    diacritic from a detached part of the letter.  MEASURED with `real_match_v2`
    over 40 fonts x 5 writers, it fired on 59.3% of renders of *bare, unmarked*
    ក គ ត ដ ភ ន ស and on 60.4% of below-/side-vowel clusters - manufacturing the
    ក->ត / គ->ត / ន->ដ / ស->ត confusions that dominate the real OCR error matrix.

    Two conditions now have to hold:

    1. the cluster's TEXT contains at least one above-rendering mark
       (``text/khmer_chars.ABOVE_RENDERING_MARKS``); and
    2. at most that many components are moved, taken TOPMOST first.

    (2) resolves the remaining ambiguity for a cluster like កិ, where both the
    ក horn and the ិ lie above the body.  Ground truth was obtained by rendering
    `base` and `base+mark` and identifying the component present only in the
    latter: over 20 bases x 9 above-marks x 30 fonts, the true mark was the
    TOPMOST above-component in **757/757 = 100%** of the resolvable cases, while
    "smallest" was right only 41.6% of the time.  In 3,617 further cases the mark
    was the only above-component at all, so the cap alone already settles them.
    """
    length = {c: sum(strokes[i].length for i in ii) for c, ii in comps.items()}
    body = max(length, key=length.get)
    body_top = float(np.vstack([strokes[i].points for i in comps[body]])[:, 1].min())
    body_len = max(length[body], 1e-9)

    cands: list[tuple[float, int, list[int]]] = []
    for c, ii in comps.items():
        if c == body:
            continue
        pts = np.vstack([strokes[i].points for i in ii])
        if float(pts[:, 1].max()) > body_top + _ABOVE_SLACK_EM * em_px:
            continue  # not above at all (side vowel, below part, coeng, ...)
        if n_above is None:
            # No text available.  Be conservative rather than reproducing the v1
            # damage: demand the component sit *entirely* above the body and be
            # small relative to it, which is what a real diacritic looks like.
            if float(pts[:, 1].max()) > body_top:
                continue
            if length[c] / body_len > _FALLBACK_MAX_SIZE_RATIO:
                continue
        cands.append((float(pts[:, 1].min()), c, ii))

    if n_above is not None:
        if n_above <= 0:
            return []  # the cluster carries no above-mark: nothing may float
        cands.sort(key=lambda t: t[0])  # topmost first (y grows downward)
        cands = cands[:n_above]
    return [ii for _, _c, ii in cands]


def displace_marks(
    strokes: list[Stroke],
    style: WriterStyle,
    cfg: DeformationConfig,
    rng: np.random.Generator,
    em_px: float,
    cluster_keys: dict[int, str] | None = None,
    allograph_seed: int = 0,
) -> list[Stroke]:
    """Move detached marks off their font positions and deepen the subscript zone.

    Measured on real Khmer handwriting (HF khmer-handwritten-4.2k): the dense
    x-height band holds only ~35 % of the text height, against ~46 % for a
    font-derived line.  Writers float the above-marks (ិ ី ឹ ើ ំ ់ ...) higher
    and to the right of where a typeface anchors them, and hang the coeng zone
    deeper.  Cluster-level deformation cannot produce that, because it moves the
    whole cluster together.

    Within one grapheme cluster the strokes are grouped by skeleton component.
    The component with the greatest arc length is the *body*.  Which of the
    remaining components actually is a mark is decided by
    :func:`_above_mark_components` from the cluster's **text** plus geometry -
    geometry alone cannot tell the ិ of កិ from the detached upper horn of a bare
    ក, and moving the horn is what produced the ក/គ/ត/ដ/ស confusions.  Selected
    components are translated up / right and scaled about their own centre.
    Everything below ``cfg.below_zone_depth_em`` is stretched downward with a
    continuous piecewise map (zero displacement at the boundary), so a coeng
    that touches its base stays attached.

    Amounts are drawn from the per-allograph stream keyed on (writer, cluster
    text): the same writer places the same letter's marks the same way every
    time.  Returns the input unchanged when every relevant style parameter is
    zero, which is the v1 behaviour.
    """
    if not strokes:
        return strokes
    if (
        style.mark_shift_above <= 0
        and style.mark_drift_x <= 0
        and style.mark_scale_var <= 0
        and style.below_stretch <= 0
    ):
        return strokes

    allograph_rng = _allograph_rng_factory(style, rng, cluster_keys, allograph_seed)
    by_cluster: dict[int, list[int]] = {}
    for i, s in enumerate(strokes):
        by_cluster.setdefault(s.cluster, []).append(i)

    depth = cfg.below_zone_depth_em * em_px
    max_disp = cfg.max_displacement_em * em_px

    for cluster, idx in by_cluster.items():
        crng = allograph_rng(cluster)
        comps: dict[int, list[int]] = {}
        for i in idx:
            comps.setdefault(strokes[i].component, []).append(i)

        # persistent amounts for this (writer, letter)
        u_shift, u_drift, u_stretch = crng.uniform(0.0, 1.0, size=3)
        shift = style.mark_shift_above * em_px * (0.6 + 0.8 * u_shift)
        drift = style.mark_drift_x * em_px * (0.4 + 1.2 * u_drift)
        stretch = style.below_stretch * (0.5 + 1.0 * u_stretch)

        if len(comps) > 1 and (shift > 0 or drift > 0 or style.mark_scale_var > 0):
            # How many above-rendering marks does this cluster's *text* have?
            # ``None`` means the caller gave us no text and we must fall back to
            # geometry alone.  See ``_above_mark_components`` for why.
            key = (cluster_keys or {}).get(cluster)
            n_above = count_above_marks(key) if key else None
            for ii in _above_mark_components(strokes, comps, em_px, n_above):
                pts = np.vstack([strokes[i].points for i in ii])
                sc = float(np.clip(1.0 + crng.normal(0.0, style.mark_scale_var), 0.6, 1.4)) if style.mark_scale_var > 0 else 1.0
                cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                dx = float(np.clip(drift, 0.0, max_disp))
                dy = -float(np.clip(shift, 0.0, max_disp))
                m = make_affine(scale=(sc, sc), translate=(dx, dy), center=(cx, cy))
                for i in ii:
                    strokes[i].points = transform_affine(strokes[i].points, m)

        if stretch > 0:
            for i in idx:
                p = strokes[i].points
                below = p[:, 1] > depth
                if np.any(below):
                    p = p.copy()
                    p[below, 1] = depth + (p[below, 1] - depth) * (1.0 + stretch)
                    strokes[i].points = p

    return strokes


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #


def deform_strokes(
    strokes: list[Stroke],
    style: WriterStyle,
    cfg: DeformationConfig,
    rng: np.random.Generator,
    em_px: float,
    cluster_keys: dict[int, str] | None = None,
    allograph_seed: int = 0,
) -> list[Stroke]:
    """Apply the full deformation stack in line space (baseline at y = 0)."""
    strokes = [
        Stroke(
            np.asarray(s.points, np.float64), list(s.edges), s.component, s.cluster, list(s.ligature_spans)
        )
        for s in strokes
    ]
    strokes = [s for s in strokes if len(s.points)]
    if not strokes:
        return strokes

    max_disp = cfg.max_displacement_em * em_px

    allograph_rng = _allograph_rng_factory(style, rng, cluster_keys, allograph_seed)

    # ---------------- mark level (v2) ---------------- #
    # First, in font coordinates, before anything moves the cluster as a whole:
    # the body/mark geometry is cleanest here.
    if getattr(cfg, "mark_level", True):
        strokes = displace_marks(
            strokes, style, cfg, rng, em_px, cluster_keys=cluster_keys, allograph_seed=allograph_seed
        )

    by_cluster: dict[int, list[int]] = {}
    for i, s in enumerate(strokes):
        by_cluster.setdefault(s.cluster, []).append(i)

    # ---------------- cluster level ---------------- #
    if cfg.cluster_level:
        for cluster, idx in by_cluster.items():
            crng = allograph_rng(cluster)
            pts = np.vstack([strokes[i].points for i in idx])
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            rot = float(crng.normal(0.0, style.glyph_rotation))
            sc = 1.0 + float(crng.normal(0.0, style.glyph_scale_var))
            scy = 1.0 + float(crng.normal(0.0, style.glyph_scale_var))
            # placement wobble is *not* part of the letterform, so it stays on
            # the shared stream: the same letter is shaped the same way but is
            # not parked in exactly the same spot twice.
            tx = float(rng.normal(0.0, style.glyph_translation * em_px))
            ty = float(rng.normal(0.0, style.glyph_translation * em_px))
            shear = float(crng.normal(0.0, style.slant_variation))
            m = make_affine(
                scale=(sc, scy), rotation=rot, shear_x=shear, translate=(tx, ty), center=(cx, cy)
            )
            for i in idx:
                strokes[i].points = transform_affine(strokes[i].points, m)

    # ---------------- stroke level ---------------- #
    # Also allograph-locked: overshoot and bending are part of how this writer
    # draws this letter, not per-occurrence noise.
    if cfg.stroke_level:
        for cluster, idx in by_cluster.items():
            srng = allograph_rng(cluster)
            for i in idx:
                s = strokes[i]
                p = s.points
                if len(p) < 3:
                    continue
                if style.overshoot > 0:
                    a = float(srng.normal(0.0, style.overshoot * em_px))
                    b = float(srng.normal(0.0, style.overshoot * em_px))
                    p = _extend_endpoints(
                        p, np.clip(a, -max_disp, max_disp), np.clip(b, -max_disp, max_disp)
                    )
                if style.local_elasticity > 0 and len(p) > 4:
                    seg = np.concatenate(
                        [[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
                    )
                    amp = style.local_elasticity * em_px
                    bend = smooth_noise_along(
                        srng, seg, scale=max(seg[-1] * 0.6, 1.0), amplitude=amp
                    )
                    # taper to zero at the endpoints so strokes stay attached
                    taper = np.sin(np.pi * np.clip(seg / max(seg[-1], 1e-9), 0, 1))
                    nrm = np.stack([-tangents(p)[:, 1], tangents(p)[:, 0]], axis=1)
                    p = p + nrm * (bend * taper)[:, None]
                s.points = p

    # ---------------- point level ---------------- #
    if cfg.point_level:
        bb = _bbox_of(strokes)
        if style.local_jitter > 0:
            field = _VectorField(
                rng, bb, scale_px=cfg.noise_scale_px, amplitude=style.local_jitter * em_px
            )
            for s in strokes:
                d = field.sample(s.points)
                s.points = s.points + np.clip(d, -max_disp, max_disp)
        if style.tremor > 0:
            for s in strokes:
                n = len(s.points)
                if n < 3:
                    continue
                amp = style.tremor * em_px
                dx = smooth_noise_1d(rng, n, scale=max(cfg.tremor_scale_px, 1.0), amplitude=amp)
                dy = smooth_noise_1d(rng, n, scale=max(cfg.tremor_scale_px, 1.0), amplitude=amp)
                s.points = s.points + np.clip(np.stack([dx, dy], axis=1), -max_disp, max_disp)

    # ---------------- baseline ---------------- #
    if cfg.baseline:
        bb = _bbox_of(strokes)
        x0, x1 = bb[0], bb[2]
        span = max(x1 - x0, 1e-6)
        walk_n = 128
        walk = (
            smooth_noise_1d(rng, walk_n, scale=18.0, amplitude=style.baseline_walk * em_px, octaves=2)
            if style.baseline_walk > 0
            else np.zeros(walk_n)
        )
        for s in strokes:
            x = s.points[:, 0]
            u = (x - x0) / em_px
            dy = style.baseline_amplitude * em_px * np.sin(
                2 * np.pi * style.baseline_frequency * u + style.baseline_phase
            )
            dy = dy + style.baseline_drift * (x - x0)
            if style.baseline_walk > 0:
                g = np.clip((x - x0) / span * (walk_n - 1), 0, walk_n - 1)
                dy = dy + np.interp(g, np.arange(walk_n), walk)
            s.points = s.points + np.stack([np.zeros_like(dy), dy], axis=1)

    # ---------------- global ---------------- #
    if cfg.global_transform:
        bb = _bbox_of(strokes)
        cx = 0.5 * (bb[0] + bb[2])
        m = make_affine(
            scale=(style.horizontal_scale, style.vertical_scale),
            rotation=style.rotation,
            shear_x=-style.slant,  # x' = x - slant * y ; y is negative above the baseline
            translate=(0.0, 0.0),
            center=(cx, 0.0),
        )
        for s in strokes:
            s.points = transform_affine(s.points, m)

    for s in strokes:
        s.points = np.asarray(s.points, dtype=np.float64)
    return strokes
