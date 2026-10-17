# Synthetic Khmer handwriting for Stage 2

`scripts/generate_handwriting.py` drives the vendored `src/khmer_synth` generator (from the
`Khmer Gen` project) with a preset tuned for OCR line images, and writes a label file that Stage 2's
data pipeline reads directly.

```bash
python scripts/fetch_fonts.py --out data/fonts             # once: 41 body + 16 display Khmer fonts (OFL/LGPL)
python scripts/generate_handwriting.py --num-texts 100000 --workers 16
#  -> data/raw/khmer_synth_hw/{images_degraded/, metadata.jsonl, labels.csv, writers.json, config.json}
python scripts/prepare_data.py   --config configs/stage2_data.yaml
python scripts/build_manifest.py --config configs/stage2_data.yaml
python scripts/build_vocab.py    --config configs/stage2_data.yaml --existing data/tokenizer/kcc_vocab.json
```

Run it on the CPU box of your choice; it never uses the GPU. Measured locally: ~13.6 images/s with
8 workers once the skeleton cache is warm (100k images ≈ 2 h), ~21 KB per JPEG (100k ≈ 2 GB).

## How an image is made

HarfBuzz-shaped font outline → skeleton (centreline) → planned pen strokes → a procedural writer
(≈30 persistent style parameters: slant, spacing drift, baseline wobble, stroke connection, tremor,
pressure, speed) → swept-disk ink rendering (ballpoint / gel / fountain / pencil / marker, 5 ink colours)
on paper backgrounds (white, off-white, textured, ruled, grid) → camera/scan degradation (blur, light,
shadow, perspective, rotation, noise, JPEG, ink fade, stains). Every sample is reproducible from
`(text, writer_id, seed, config)`.

## What `configs/handwriting_synth.json` changes vs the generator's `ocr_paddle` preset, and why

| Setting | Change | Reason |
|---|---|---|
| `fonts.exclude` | all 16 display faces (Moul, Bokor, Metal, Koulen, Bayon, Muol variants, …) | the skeleton traces their ornament, not the letter; no hand writes like that |
| `fonts.weights` | Fasthand / Freehand ×4 | the closest letter shapes to everyday Khmer handwriting |
| `layout.canvas_width` | 64 (with `auto_width`) | canvas = text width + padding → tight line crops instead of short words floating in empty canvas |
| `writer.fluency` | 0.4 – 0.92 (was 0.05 – 1.0) | mostly fast adult hands; no slow "child-like" writers, no illegible scrawl |
| `writer.stroke_width` / `character_spacing` / `connect_probability` | 0.025–0.06 / 0.8–1.08 / 0.1–0.85 | thinner pens, tighter letters, more joins (fast writing) |
| `render.width_pressure_bias/gain` | 0.3 / 1.15 (was 0.6 / 0.8) | pressure can thin a stroke to a tapered tail |
| `degradation` | shadow 0.3→0.15, ink fade 0.3→0.15, uneven light 0.55→0.4 | stacked heavy vignettes + fading erased whole syllables |
| `dataset` | degraded JPEG only, no trajectories | Stage 2 needs images; halves disk and time |

Code changes for fast, confident strokes (tapered exits, no tremor on fast hands, speed-dependent
width) are listed in `src/khmer_synth/VENDORED.md`.

## Text

Snippets of 2–30 Khmer clusters cut at cluster boundaries (preferring word starts) from the
**train split** of the Stage-1 manifest, so no validation/test text is reused. Snippets without Khmer
are dropped. Add your own text with `--corpus file.txt` (one line per sample), e.g. names, addresses
or form vocabulary that matches your real handwriting domain; `--manifest-texts 0` uses only it.

## Labels and splits

`labels.csv` = `image_path, text, writer_id, font`, using the generator's normalized text (what was
actually rendered). Samples with missing glyphs (`.notdef`) or failed layout are dropped. The Stage-2
config splits by `writer_id`, so validation writers never appear in training.

## Limits (from the generator's own documentation; keep them in mind when reading results)

* Letter shapes come from typefaces, deformed — not from real hands. This is the main domain gap.
* Writer styles are procedural, not sampled from real writers.
* Degradation approximates scans/phone photos; it is not a camera model.

So treat synthetic-handwriting CER as an upper bound and **select checkpoints on real handwriting**
(`best_metric: real_hw/tr_cer`). Mixing a few thousand real lines matters more than more synthetic ones.
