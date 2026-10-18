# Vendored: khmer_synth

Copied from the author's `khmer-synth` project (`khmer-handwriting-synth` 0.1.0, MIT) so the Stage-2
synthetic-handwriting pipeline travels with this repo. All imports are package-relative; it is used
here as `src.khmer_synth`. Synced with upstream on 2026-10-18 (v2/v3 realism: mark placement,
stroke-direction width contrast, randomised tight framing, cluster separation, smart font exclusion,
content-hashed skeleton cache, optional Augraphy paper stage). The `.bak-prefix` files were not copied.

UKTR-specific settings live outside this package: `configs/handwriting_synth.json` (preset, based on
upstream `real_match_v3`) and `scripts/generate_handwriting.py` (corpus + labels for Stage 2). See
`docs/SYNTHETIC_HANDWRITING.md`. Upstream tests are not vendored; run them in the original project
when changing this code.

## Local changes on top of upstream

Re-apply these when syncing again; `tests/test_synth_spacing.py` fails if the two bugs come back.

* `strokes/smoothing.py` (**bug**): the raw/smoothed blend matched points by index although the two
  curves have different lengths, which shears points along the stroke into waves and misplaces its
  end (5–7 px on a zigzag). Both are now resampled to the same count by normalised arc length.
  `smooth_stroke` takes a per-call `savgol_window`.
* `writer/style.py` (**bug**): tremor grew with fluency (`by_fluency`), so the fastest, most confident
  hands shook ~5x more than careful ones. It now uses upstream's own `by_care` (shrinks with fluency),
  the same fix upstream applied to baseline waviness. The shaky archetype is unchanged.
* `dataset/generator.py`: `savgol_window` is scaled with the em size (specified per 64 px em) so small
  writing is not smoothed out of shape; with `auto_width`, the canvas grows to the final ink width
  after cluster separation and deformation instead of shrinking the writing to fit.
* `writer/pressure.py`, `writer/timing.py`: fluent writers land quickly and lift while still moving
  (shorter entry ramp, longer exit ramp, lower end-pressure floor, speed kept at stroke ends), so fast
  strokes end in tapered tails instead of round blobs.

Earlier local changes now covered upstream and dropped: the row-profile overprint pass (upstream
`layout.min_cluster_gap_em` + `writer.spacing_tight_max_frac`), writer font weights
(`fonts.weights` + `fonts.default_weight`), and ballpoint/gel speed thinning (upstream's
`width_contrast` direction model covers stroke-width variation).
