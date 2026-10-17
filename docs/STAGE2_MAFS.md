# Stage 2: loading the Stage-1 weights and inserting supervised MAFS

Stage 1 trains the backbone, encoder, CTC head and Transformer decoder **without MAFS**. In the paper,
MAFS is placed after the visual encoder and trained during phase 2 (modality adapting). This document
describes the interface Stage 2 plugs into. Nothing here is implemented yet.

## Where MAFS plugs in

```
image -> backbone -> encoder -> G (B,H',W',d), G1d = mean_H(G) (B,W',d)
                                   |
                        model.feature_selector(G, G1d, col_mask)  -> (H, U1d)
                                   |                  |
                        Transformer decoder(H)     CTC head(U1d)
```

`UKTRBase.feature_selector` is `None` in Stage 1 and has no parameters. Assigning a module adds only
`feature_selector.*` keys; every existing parameter keeps its name and shape.

Interface: `forward(G, G1d, col_mask) -> (H, U1d)` with the same shapes as the inputs. `col_mask` is
`True` on padded columns. Pool z only over valid columns (`~col_mask`).

## Loading

```python
from src.models.uktr import UKTRBase
from src.models.checkpoint import load_stage1

ck_path = "checkpoints/uktr_base_stage1_best.pt"
model, tok, ck = load_stage1(ck_path)          # plain Stage-1 model
model.feature_selector = SupervisedMAFS(d=512, n_sources=5, p=128)   # your Stage-2 module

# or build first, then load (raises on any unexpected key, missing non-MAFS key or shape mismatch):
model = UKTRBase(ck["model_config"]); model.feature_selector = SupervisedMAFS(...)
model, tok, ck = load_stage1(ck_path, model=model)
```

Reuse the tokenizer from the checkpoint (`tok`). Its vocabulary and normalization rules define the
output layer. If Stage-2 data contains clusters that are not in the vocabulary, they still encode
losslessly through the code-point fallback. Adding new vocabulary entries would change the shapes of
`ctc_head`, `tok_emb` and `dec_head`; expand those rows explicitly if you need that.

## Keeping pretrained behaviour at initialization

The paper's adapters project to p=128, and the decoders expect d=512. To keep the Stage-1 decoders
valid from step 0, make each adapter residual and zero-initialize its output projection:
`A_i(x) = x + W_up,i(act(W_down,i x))`, with `W_down: 512→128` and `W_up: 128→512` (initialized to 0).
Then `H = Σ_i r_i A_i(G) = G` at initialization for any router output r. Stage-2 training therefore starts
exactly from the Stage-1 model.

## Supervised routing (your Stage 2 addition; not in the paper)

The paper trains the router only through the recognition loss and lists the lack of modality
supervision as a limitation. To supervise it, add `L_route = CE(router_logits, modality_label)`
for samples with known modality (printed / scene / handwritten …), so that
`L = L_CTC + L_TR + λ L_route`. Stage 1 deliberately trains no router: the Stage-1 data is almost
entirely printed or synthetic and can't teach modality separation.

## Suggested phase-2 recipe from the paper

50 epochs, cyclic LR 1e-6 … 1e-5. Mix in an equal number of document (Stage-1) images with the
scene/handwritten data to avoid forgetting printed text.
