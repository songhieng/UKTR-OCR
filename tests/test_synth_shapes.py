"""Synthetic handwriting: letter-shape variety (glyph warp, curl scale, stroke-end bias) in writer/deformation.py."""
from dataclasses import replace

import numpy as np


def _deform(seed, **style_kw):
    from src.khmer_synth.config import DeformationConfig
    from src.khmer_synth.strokes.planner import Stroke
    from src.khmer_synth.utils.rng import make_rng
    from src.khmer_synth.writer.deformation import deform_strokes
    from src.khmer_synth.writer.style import WriterStyle

    t = np.linspace(0, 3 * np.pi, 120)  # a stem ending in a tight 0.08 em curl, plus a mark
    curl = np.stack([18 - 4 * np.cos(t), -40 + 4 * np.sin(t)], 1)
    stem = np.stack([np.full(40, 14.0), np.linspace(0, -40, 40)], 1)
    strokes = [Stroke(np.vstack([stem, curl]), cluster=0), Stroke(np.stack([np.linspace(40, 60, 20), np.full(20, -10.0)], 1), cluster=1)]
    style = replace(WriterStyle(writer_id=1, name="w"), **style_kw)
    rng = make_rng("test", seed)
    out = deform_strokes(strokes, style, DeformationConfig(), rng, em_px=50.0)
    return [s.points for s in out], rng.random()


def _same(a, b):
    return all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b)) and len(a) == len(b)


def test_shape_variety_deterministic_and_noop_when_off():
    on = dict(glyph_warp=0.05, glyph_warp_scale=0.3, loop_scale=1.4, stroke_end_bias=0.03)
    a, ra = _deform(7, **on)
    b, rb = _deform(7, **on)
    assert _same(a, b) and ra == rb                       # deterministic for a seed
    base, rbase = _deform(7)
    assert not _same(a, base)                             # changes the output when enabled
    for k, v in on.items():                               # each feature alone changes it ...
        if k != "glyph_warp_scale":                       # (only a length scale for glyph_warp)
            assert not _same(_deform(7, **{k: v})[0], base), k
    off, roff = _deform(7, glyph_warp=0.0, glyph_warp_scale=0.4, loop_scale=1.0, stroke_end_bias=0.0)
    assert _same(off, base) and roff == rbase             # ... and amplitude 0 is a bit-exact no-op (rng untouched)


def test_curl_scale_keeps_stroke_ends():
    from src.khmer_synth.writer.deformation import _scale_curls

    t = np.linspace(0, 3 * np.pi, 120)
    p = np.vstack([np.stack([np.full(40, 14.0), np.linspace(0, -40, 40)], 1),
                   np.stack([18 - 4 * np.cos(t), -40 + 4 * np.sin(t)], 1)])
    q = _scale_curls(p, 1.4, em_px=50.0)
    assert np.abs(q - p).max() > 1.0                      # the curl grew
    assert np.allclose(q[:20], p[:20])                    # the stem base (its attachment) did not move
