# Vendored: khmer_synth

Copied from the author's `Khmer Gen` project (`khmer-handwriting-synth` 0.1.0, MIT, per its
README/pyproject) so the Stage-2 synthetic-handwriting pipeline travels with this repo.
All imports are package-relative; it is used here as `src.khmer_synth`.

UKTR-specific settings live outside this package: `configs/handwriting_synth.json` (preset) and
`scripts/generate_handwriting.py` (corpus + labels for Stage 2). See `docs/SYNTHETIC_HANDWRITING.md`.
Upstream tests (211) are not vendored; run them in the original project when changing this code.

## Local changes (fast adult handwriting)

The upstream writer model drew every hand like a slow, careful writer (blunt stroke ends, tremor that
*grew* with speed). Changed here:

* `writer/style.py`: tremor falls with fluency (shaky archetype unchanged); stroke smoothness rises
  with fluency; fast hands get tighter spacing (−15 %), narrower letters (−10 %) and more slant.
* `writer/pressure.py`: fluent writers land quickly and lift while moving: shorter entry ramp, longer
  exit ramp, lower end-pressure floor → tapered tails instead of round blobs.
* `writer/timing.py`: fluent writers keep speed at stroke ends (feeds speed thinning and pressure).
* `rendering/ink.py`: ballpoint/gel get speed thinning and stronger pressure→width coupling.
* `dataset/generator.py`: **no overprinting letters.** Writer spacing scales whole advances and adds
  negative "tight" offsets, so 38 % of neighbouring clusters overprinted. `_build_line` now pushes each
  cluster right until its row-by-row ink profile keeps a minimum gap (ink width + headroom for the
  later deformations) from everything already written; auto-width canvases grow to the final ink
  width instead of shrinking the text. Overprint is now 0 % before deformation and ≈5 % after
  (mostly the intentional joining strokes). Checked by `tests/test_synth_spacing.py`.
