# Stage 2: handwriting adaptation with supervised MAFS

Stage 2 starts from the Stage-1 checkpoint, inserts MAFS (Modality-Aware Adaptive Feature Selector)
between the encoder and both decoders, and fine-tunes on handwriting mixed with replayed printed data.
One config switch selects each variant you want to compare:

| Config | MAFS | Router training | Paper equivalent |
|---|---|---|---|
| `configs/stage2.yaml` | on, n = 3 (printed, scene, handwritten) | **supervised**: `+ 0.1 · CE(router, modality)` | your addition (paper lists it as a limitation) |
| `configs/stage2_unsupervised.yaml` | on, n = 5 | recognition loss only | UKTR (D + S&H) |
| `configs/stage2_no_mafs.yaml` | off | — | "w/o MAFS" ablation |

## Run

On a fresh GPU box, `scripts/vast_stage2.sh` does all of it (upload the Stage-1 checkpoint to
`checkpoints/uktr_base_stage1_best.pt` first). By hand:

```bash
# data (see docs/SYNTHETIC_HANDWRITING.md for the synthetic part); needs the Stage-1 data + manifest
python scripts/fetch_fonts.py --out data/fonts --all
python scripts/generate_handwriting.py --num-texts 100000 --workers 16
python scripts/download_data.py  --config configs/stage2_data.yaml --datasets real_hw
python scripts/prepare_data.py   --config configs/stage2_data.yaml
python scripts/build_manifest.py --config configs/stage2_data.yaml
python scripts/build_vocab.py    --config configs/stage2_data.yaml --existing data/tokenizer/kcc_vocab.json

# train: 24 GB (RTX 4090) runs the default batch 32 as is
python scripts/preflight.py --config configs/stage2.yaml
python scripts/train.py     --config configs/stage2.yaml
#   16 GB: --set train.batch_size=16 train.grad_accum=2 (on both commands)
python scripts/evaluate.py  --checkpoint checkpoints/stage2_supervised/uktr_stage2_best.pt --splits val test
```

## Real handwriting

`real_hw` is `SoyVitou/khmer-handwritten-dataset-4.2k` (MIT): 3,549 train / 346 val / 307 test line
crops with official splits; 99.3–99.7% of its clusters are in the Stage-1 vocabulary. It has no writer
ids, so validation writers may also appear in training. `repeat: {real_hw: 5}` in `configs/stage2.yaml`
makes the real lines ~16% of the handwriting the model sees per epoch.

To use your own lines instead, make `real_hw` a `type: labels` source: a csv with `image_path`
(relative to `images_root`), `text`, `writer_id`. Also accepted: tsv / headerless txt
(`header: false` → columns 0, 1), jsonl, an existing split column (`split_column`, `split_map`).
With `group_column` the split is writer-independent, so validation measures new writers.

## What happens in training

* **Init**: model config and tokenizer come from `init_checkpoint`; Stage-1 weights load with only
  `feature_selector.*` missing (anything else raises). Input height must match Stage 1.
* **MAFS** (`src/models/mafs.py`): z = mean of the height-pooled features over valid columns →
  router MLP → softmax r over n sources; n residual bottleneck adapters (512→128→512, up-projection
  zero-initialised) applied to both the 2D features (Transformer path) and the 1D features (CTC path);
  outputs are r-weighted sums. Zero-init makes a fresh MAFS an exact identity, so step 0 reproduces the
  Stage-1 model (checked by `preflight.py` check 9 and `tests/test_stage2.py`).
* **Loss**: `L = L_CTC + L_TR + route_loss_weight · CE(router_logits, modality)`; rows with unknown
  modality are ignored. Modality per row comes from the manifest (`printed` / `scene` /
  `handwritten`; HierText lines flagged handwritten count as handwritten).
* **Sampling**: every epoch = all `adapt_sources` rows (× `repeat`) + a fresh random draw of
  `replay_sources` rows, `replay_ratio` × as many ([paper] equal numbers, against forgetting printed text).
* **LR**: [paper] cyclic 1e-6 … 1e-5 for pretrained weights; MAFS gets `mafs_lr_scale` × that (default 10×).
* **Selection**: `best_metric: real_hw/tr_cer` (Transformer CER on real handwriting). Logged extras:
  `val_route_loss`, `route_acc`, and `route_p/<modality>/src<j>` (mean routing weight per modality —
  for the unsupervised variant this shows whether sources specialised at all).

## Assumptions (not specified by the paper)

Residual/zero-init adapters; adapters shared between the 2D and 1D paths; router = LayerNorm → 256 →
GELU → n; supervised weight 0.1; MAFS LR ×10; n = 3 for the supervised variant (the paper finds
n = 3 and n = 5 equivalent).

## Loading a Stage-2 checkpoint

`load_stage1(path)` rebuilds MAFS from the stored `model_config["mafs"]` and loads strictly, so
`scripts/evaluate.py` and `scripts/infer.py` work unchanged; `model.recognize(...)` also returns
`route_probs` per image.
