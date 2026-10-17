# UKTR: Khmer OCR (Stage 1 pretraining + Stage 2 handwriting adaptation with MAFS)

An independent PyTorch re-implementation of *Towards Universal Khmer Text Recognition*
(Kong et al., arXiv:2603.00702).
**Stage 1** trains the general recognizer on public printed/synthetic Khmer and Latin text.
**Stage 2** adapts it to handwriting with MAFS: supervised (modality-labelled router), unsupervised
(as in the paper) or without MAFS, plus a synthetic Khmer handwriting generator.

The original code, weights, vocabulary and part of the data are unreleased. See
[docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) for which values come from the paper and which are assumptions,
[docs/STAGE2_MAFS.md](docs/STAGE2_MAFS.md) for Stage 2, and
[docs/SYNTHETIC_HANDWRITING.md](docs/SYNTHETIC_HANDWRITING.md) for the handwriting generator.

## Model

```
gray line image (H=32, variable width)
  -> ResNet (6 blocks, 32->512 ch, stride 4)          12.65M params   (paper 13.0M)
  -> Transformer encoder on the flattened 2D map      9.46M          (paper 9.5M)
       |- height-avg-pool -> [MAFS hook] -> linear -> CTC                (greedy decoding)
       '- [MAFS hook] -> Transformer decoder (3L, 30% input masking)     24.8M (paper 24.08M)
loss = CTC + cross-entropy, Adam, clip 50, cyclic LR 1e-5..1e-4, batch 32, 5 epochs
```

Tokenizer: Khmer Character Cluster (KCC) tokens + Latin, digits and punctuation, with a code-point fallback
(lossless for the whole Khmer block, ASCII, Latin-1/Extended-A and common punctuation). Built only from
**training** transcriptions.

## Data (Stage 1)

| Source | Size | Split used | License (as recorded) |
|---|---|---|---|
| [KhmerSynthetic1M](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M) | 1,000,000 lines, 12.7 GB | 99/0.5/0.5 by text hash | **conflicting**: apache-2.0 metadata vs "research and academic use only" in the card body; treated as research-only |
| [62k Khmer printed](https://huggingface.co/datasets/SoyVitou/62k-images-khmer-printed-dataset) | 50,143 / 6,268 / 6,268 | official train/valid/test | MIT (metadata; body placeholder) |
| [HierText](https://github.com/google-research-datasets/hiertext) | 8,281 / 1,724 / 1,634 images, cropped to lines | official train/validation/test | CC BY-SA 4.0 annotations; Open Images photos |

All three were verified downloadable on 2026-10-17 (HierText images over plain HTTPS from the CVDF S3
bucket; no AWS CLI needed). The paper's own phase-1 synthetic data (Buoy et al.) and the SynthText ID-card
set are not public. KHOB (real printed benchmark) was not found, so real-printed evaluation uses the printed
dataset's test split.

## Quick start (Linux GPU box, e.g. vast.ai)

```bash
git clone https://github.com/songhieng/UKTR-OCR.git && cd UKTR-OCR
SMOKE=1 bash scripts/vast_setup.sh   # ~1.2 GB download + 300-step smoke run (checks everything works)
bash scripts/vast_setup.sh           # full data (~80 GB disk) + preflight + full training in tmux
```

Or step by step:

```bash
pip install -r requirements.txt
python scripts/download_data.py            # --smoke for small subsets; resumable
python scripts/prepare_data.py --workers 16 # extract/crop images, corruption check, md5 + dHash
python scripts/build_manifest.py           # normalize, validate, dedup, split, leakage + coverage report
python scripts/build_vocab.py              # KCC vocab from the train split -> data/tokenizer/kcc_vocab.json
python -m pytest -q tests                  # unit tests (tokenizer, shapes, masking, checkpoint/Stage-2 loading)
python scripts/preflight.py --config configs/full.yaml   # GPU, shapes, fwd/bwd, CTC lengths, overfit, ckpt, memory
python scripts/train.py --config configs/full.yaml
tensorboard --logdir results --host 0.0.0.0 --port 6006
```

**Resume** after an interruption (checkpoints are written atomically every `save_every` steps):

```bash
python scripts/train.py --config configs/full.yaml --resume checkpoints/uktr_base_stage1_last.pt
```

**Smaller GPU**: keep the effective batch at 32 with
`--set train.batch_size=16 train.grad_accum=2`. Check `8_worst_case_batch` in `preflight.json`;
lowering `data.max_width` caps the attention cost of very long lines.

## Evaluation, inference, curves

```bash
python scripts/evaluate.py --checkpoint checkpoints/uktr_base_stage1_best.pt --splits val test
#   -> results/eval/eval_report.{md,json}, examples.html (image / GT / CTC / Transformer / CER)
#   CER, KCC error rate, exact-line accuracy, CTC vs Transformer, val loss, latency, peak GPU memory
python scripts/infer.py --checkpoint checkpoints/uktr_base_stage1_best_infer.pt line1.png line2.jpg
python scripts/plot_curves.py --results results/stage1_full     # -> curves.png
```

## Outputs

| Path | Content |
|---|---|
| `checkpoints/uktr_base_stage1_best.pt` | best weights + model config + tokenizer/vocab + normalization + optimizer/scheduler/scaler/RNG state + metrics + data provenance + parameter counts |
| `checkpoints/uktr_base_stage1_best_infer.pt` | same without the optimizer/resume state |
| `checkpoints/uktr_base_stage1_last.pt` | latest state for `--resume` |
| `data/manifest.csv` | image_path, transcription, source_dataset, language, split, width, height, license, handwritten, raw_transcription, md5, dhash |
| `data/tokenizer/kcc_vocab.json` | vocabulary + normalization rules (also embedded in every checkpoint) |
| `results/data/data_report.md` | counts, length stats, dropped rows, duplicates, leakage, coverage |
| `results/<exp>/train_log.csv`, `val_log.csv`, `tensorboard/` | curves |

Checkpoints load with `torch.load(..., weights_only=True)` (no pickled code).

## Stage 2 (handwriting, MAFS)

```bash
python scripts/fetch_fonts.py --out data/fonts
python scripts/generate_handwriting.py --num-texts 100000 --workers 16   # synthetic handwriting (CPU)
# real handwriting: data/raw/real_hw/labels.csv (image_path,text,writer_id) + images/
python scripts/prepare_data.py   --config configs/stage2_data.yaml
python scripts/build_manifest.py --config configs/stage2_data.yaml
python scripts/build_vocab.py    --config configs/stage2_data.yaml --existing data/tokenizer/kcc_vocab.json
python scripts/train.py --config configs/stage2.yaml               # supervised MAFS
python scripts/train.py --config configs/stage2_unsupervised.yaml  # paper's MAFS
python scripts/train.py --config configs/stage2_no_mafs.yaml       # ablation
```

Details, data format and assumptions: [docs/STAGE2_MAFS.md](docs/STAGE2_MAFS.md).

## Layout

```
configs/   data.yaml, full.yaml, smoke.yaml (inherits full.yaml)
           stage2_data.yaml, stage2.yaml, stage2_unsupervised.yaml, stage2_no_mafs.yaml,
           handwriting_synth.json (generator preset)
src/       tokenizer/kcc.py  models/uktr.py, mafs.py, checkpoint.py  datasets/ocr_dataset.py
           training/trainer.py, config.py  evaluation/metrics.py
           khmer_synth/ (vendored synthetic-handwriting generator)
scripts/   download_data, prepare_data, build_manifest, build_vocab, preflight, train,
           evaluate, infer, export, plot_curves, vast_setup.sh,
           fetch_fonts, generate_handwriting
tests/     pytest suite
docs/      ASSUMPTIONS.md, STAGE2_MAFS.md, SYNTHETIC_HANDWRITING.md
```

## Citation

```bibtex
@article{kong2026uktr,
  title   = {Towards Universal Khmer Text Recognition},
  author  = {Kong, Marry and Buoy, Rina and Chenda, Sovisal and Taing, Nguonly and Iwamura, Masakazu and Kise, Koichi},
  journal = {arXiv preprint arXiv:2603.00702},
  year    = {2026}
}
```
