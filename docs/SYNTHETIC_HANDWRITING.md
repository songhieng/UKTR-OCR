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

Run it on the CPU box of your choice; it never uses the GPU. Measured locally: ~0.5 s per image on
one process (full-quality skeleton raster), so 100k images ≈ 1 h on 16 workers.

## How an image is made

HarfBuzz-shaped font outline → skeleton (centreline) → planned pen strokes → a procedural writer
(≈30 persistent style parameters: slant, spacing drift, baseline wobble, stroke connection, tremor,
pressure, speed) → swept-disk ink rendering (ballpoint / gel / fountain / pencil / marker, 5 ink colours)
on paper backgrounds (white, off-white, textured, ruled, grid) → camera/scan degradation (blur, light,
shadow, perspective, rotation, noise, JPEG, ink fade, stains). Every sample is reproducible from
`(text, writer_id, seed, config)`.

## The preset: `configs/handwriting_synth.json`

It is the generator's `real_match_v3` preset (tuned against real Khmer handwriting crops: tight
framing, dark navy/black ballpoint ink, ruled and grid paper, cluster separation, adult writer mix,
decorative fonts excluded by substring) with these UKTR changes:

| Setting | Change | Reason |
|---|---|---|
| `fonts.paths` | `data/fonts` | where `scripts/fetch_fonts.py` puts them |
| `fonts.weights` / `exclude` | + `KhmerOS_fasthand` 3.0, `KhmerOS_freehand` 2.5; exclude `muol` | our font set ships the KhmerOS handwriting faces; KhmerOS spells Moul "muol" |
| `writer.local_jitter`, `local_elasticity`, `glyph_rotation`, `glyph_translation`, `glyph_scale_var`, `slant_variation`, `overshoot` | back to the earlier preset's wider ranges | with v3's ranges the Stage-1 model read the synthetic lines at 10% CER vs 28% on real handwriting with the same texts (it looked like neat print); these ranges give 18% and similar exact-line accuracy (37% vs 41%), image realism unchanged |
| `dataset` | `data/raw/khmer_synth_hw`, degraded JPEG only, 1 style per text | Stage 2 needs images; halves disk and time |

Measured on 200 texts from the real `SoyVitou/khmer-handwritten-dataset-4.2k` **val** split (its
test split stays untouched for evaluation). Code changes kept on top of upstream are listed in
`src/khmer_synth/VENDORED.md`.

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
