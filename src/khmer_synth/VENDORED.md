# Vendored: khmer_synth

Copied unchanged from the author's `Khmer Gen` project (`khmer-handwriting-synth` 0.1.0, MIT, per its
README/pyproject) so the Stage-2 synthetic-handwriting pipeline travels with this repo.
All imports are package-relative; it is used here as `src.khmer_synth`.

UKTR-specific settings live outside this package: `configs/handwriting_synth.json` (preset) and
`scripts/generate_handwriting.py` (corpus + labels for Stage 2). See `docs/SYNTHETIC_HANDWRITING.md`.
Upstream tests (211) are not vendored; run them in the original project when changing this code.
