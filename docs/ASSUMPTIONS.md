# Stage-1 reproduction: what is from the paper, what is assumed

Reference: Kong et al., *Towards Universal Khmer Text Recognition* (arXiv:2603.00702, v1).
The authors' repository (rinabuoy/Towards-Universal-Khmer-Text-Recognition) contains only a README;
code, weights, vocabulary and the new datasets are "to be released upon acceptance"
(checked 2026-10-17). **This is an independent re-implementation, not an exact reproduction.**

## Taken from the paper

| Item | Paper value | Here |
|---|---|---|
| CNN | 6 ResNet blocks, channels 32 → 512, (2,2) downsampling in blocks 1 and 4 | same |
| CNN params | 13.0M | 12.65M |
| Transformer encoder | 3 layers, d=512, 8 heads, FFN 2048, dropout 0.1 | same; 9.46M params (paper 9.5M) |
| Height pooling | average over height → 1D features for CTC | same |
| CTC decoder | non-autoregressive, on height-pooled features | linear projection + CTC |
| Transformer decoder | 3 layers, d=512, 8 heads, FFN 2048, dropout 0.1, 30% masking ratio | same; 24.8M params (paper 24.08M, vocab-dependent) |
| Loss | L = L_CTC + L_TR (equal weights) | same |
| Optimizer | Adam, gradient clip 50 | same |
| Phase-1 schedule | 5 epochs, batch 32, cyclic LR 1e-5 … 1e-4 | same bounds |
| Tokenizer | KCC tokenizer (Buoy et al.) + case-sensitive Latin, digits, symbols; 11,899 entries | KCC re-implemented; vocab built from our training data |
| MAFS | after the encoder; n=5 sources, adapters p=128; trained in phase 2 only | not built in Stage 1; hook provided (see STAGE2_MAFS.md) |

## Assumptions (not specified by the paper)

1. **Input size**: grayscale, height 32 (the paper reports FLOPs on a 32×116 input), aspect ratio kept,
   width clamped to [16, 1600] px. Wider lines are squeezed horizontally.
2. **Residual block layout**: basic blocks (two 3×3 convs) with layers per block `[1,2,3,3,1,1]`,
   chosen to land near the 13.0M parameter count. Total stride is 4 in both directions.
3. **Encoder input**: the 2D feature map (H/4 × W/4) is flattened into a sequence with a 2D sinusoidal
   position encoding (half the channels for rows, half for columns). This matches the paper's encoder
   FLOPs (2.2G on 32×116 ⇒ 232 tokens). Padded columns are masked in attention.
   Attention cost grows with the square of line width, which dominates memory for long lines.
4. **Pre-norm** Transformer layers (with a final LayerNorm) for stability; the paper does not say which.
5. **Decoder masking (30%)**: interpreted as replacing 30% of the teacher-forced decoder input tokens
   (not BOS/padding) with a `<mask>` token during training.
6. **Cyclic LR**: `triangular` mode, 5,000 optimizer steps per half-cycle (the paper gives only the bounds).
7. **Augmentation**: light (small affine/shear, brightness/contrast, blur, down-up sampling); the
   paper does not describe any.
8. **Model selection**: best checkpoint = lowest mean of CTC and Transformer CER on a validation subset
   (2,000 lines per source).
9. **KCC definition**: a cluster is a base (consonant U+1780–17A2 or independent vowel U+17A3–17B3)
   followed by any number of COENG+base subscripts and combining marks (U+17B6–17D1, U+17D3, U+17DD).
   Orphan marks form their own token, and every other character is one token. Clusters missing from
   the vocabulary fall back to their code points, which are always in the vocabulary (whole Khmer block,
   ASCII, Latin-1/Extended-A, common punctuation), so decoding is reversible for that character set.
10. **Normalization** (saved in the vocabulary file and in every checkpoint): NFC; remove ZWSP/BOM/soft
    hyphen/ZWNJ/ZWJ and the inherent vowels U+17B4/17B5; replace deprecated U+17A3/17A4; canonical order
    inside a cluster (subscripts, COENG RO last, register shifters, robat, vowels, other signs); compose
    េ+ី→ើ and េ+ា→ោ; collapse whitespace. COENG DA vs COENG TA are **not** merged (configurable).

## Data deviations from the paper's Phase-1 ("D") group

The paper's group D is Buoy et al. synthetic (2.8M lines) + SynthText ID cards (70k) + HierText (518,726).

* **Buoy et al. synthetic and the SynthText ID-card set are not publicly available.** As the task requested,
  they are replaced by KhmerSynthetic1M (1M) and the 62k Khmer printed dataset.
* **HierText**: the paper crops train+validation lines and removes vertical text (518,726). By default we
  also drop *illegible* lines (their text is partial or empty, so the label would be wrong) and crops shorter
  than 8 px, and we keep HierText validation for evaluation. To get closer to the paper's count, set
  `keep_illegible: true`, `min_crop_height: 1` and `validation_to_train: true` in `configs/data.yaml`.
  HierText contains about 6% handwritten Latin lines; the paper keeps them, and so do we
  (`data.exclude_handwritten` drops them).
* **Evaluation**: KHOB (the paper's real printed benchmark) could not be found publicly. Real-printed
  evaluation therefore uses the printed dataset's official `testset`. Its card does not say whether
  the images are scanned or rendered, so treat it as "printed, possibly synthetic".

## Licensing (recorded in the manifest's `license` column and in DOWNLOAD_PROVENANCE.json)

| Source | Recorded license | Note |
|---|---|---|
| KhmerSynthetic1M | **conflicting** | card metadata says `apache-2.0`; card body says "Research and academic use only. Commercial use is not permitted." Treated as research-only, non-commercial. |
| 62k Khmer printed | MIT (metadata) | card body has an unfilled license placeholder |
| HierText | CC BY-SA 4.0 (annotations) | images from Open Images via CVDF (listed as CC BY 2.0) |
| UKTR paper repo | CC BY-NC-ND 4.0 | nothing released; no code or weights from it are used here |
