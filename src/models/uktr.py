"""UKTR-style base recognizer (Stage 1): ResNet -> Transformer encoder -> {CTC, Transformer decoder}.

Paper: "Towards Universal Khmer Text Recognition" (arXiv:2603.00702). Values the paper
states are marked [paper]; everything else is an assumption documented in docs/ASSUMPTIONS.md.

Stage-2 hook: `feature_selector` sits between the encoder and both decoders (where MAFS goes
in the paper). It is None in Stage 1 and owns no parameters; with cfg["mafs"] set it is a MAFS
module, and a Stage-1 state dict loads into it with only `feature_selector.*` keys missing.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mafs import MAFS

DEFAULT_MODEL_CONFIG = {
    "in_channels": 1,
    # six residual blocks, channels 32 -> 512 [paper]; (2,2) downsampling in blocks 1 and 4 [paper]
    "block_channels": [32, 64, 128, 256, 512, 512],
    "block_layers": [1, 2, 3, 3, 1, 1],   # assumption: chosen to land near the paper's 13.0M params
    "block_strides": [2, 1, 1, 2, 1, 1],
    "d_model": 512, "nhead": 8, "dim_ff": 2048, "dropout": 0.1,       # [paper]
    "enc_layers": 3, "dec_layers": 3,                                    # [paper]
    "max_dec_len": 256,
    "vocab_size": None,                   # set from the tokenizer
    "mafs": None,                         # Stage 2: dict of MAFS kwargs (n_sources, adapter_dim, ...)
}


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(x)) + idt, inplace=True)


class ResNetBackbone(nn.Module):
    """Width-preserving ResNet: total stride 4 in both height and width."""

    def __init__(self, in_channels, channels, layers, strides):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, channels[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        blocks, cin = [], channels[0]
        for cout, n, s in zip(channels, layers, strides):
            blocks += [BasicBlock(cin, cout, s)] + [BasicBlock(cout, cout, 1) for _ in range(n - 1)]
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.stride = math.prod(strides)

    def forward(self, x):
        return self.blocks(self.stem(x))


def sinusoid(n: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class UKTRBase(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        cfg = {**DEFAULT_MODEL_CONFIG, **cfg}
        assert cfg["vocab_size"], "model config needs vocab_size (from the tokenizer)"
        self.cfg = cfg
        d, V = cfg["d_model"], cfg["vocab_size"]
        self.backbone = ResNetBackbone(cfg["in_channels"], cfg["block_channels"],
                                       cfg["block_layers"], cfg["block_strides"])
        self.proj = nn.Identity() if cfg["block_channels"][-1] == d else nn.Linear(cfg["block_channels"][-1], d)
        enc_layer = nn.TransformerEncoderLayer(d, cfg["nhead"], cfg["dim_ff"], cfg["dropout"],
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg["enc_layers"], norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        self.feature_selector = MAFS(d, **cfg["mafs"]) if cfg.get("mafs") else None
        self.ctc_head = nn.Linear(d, V)
        self.tok_emb = nn.Embedding(V, d)
        dec_layer = nn.TransformerDecoderLayer(d, cfg["nhead"], cfg["dim_ff"], cfg["dropout"],
                                               batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg["dec_layers"], norm=nn.LayerNorm(d))
        self.dec_head = nn.Linear(d, V)
        self.dec_drop = nn.Dropout(cfg["dropout"])

    # ---- shapes -------------------------------------------------------------------
    def feature_widths(self, widths: torch.Tensor) -> torch.Tensor:
        """Valid encoder columns for each image width (each stride-2 conv: ceil(w/2))."""
        w = widths.clone()
        for s in self.cfg["block_strides"]:
            w = (w + s - 1) // s
        return w

    # ---- encoder ------------------------------------------------------------------
    def encode(self, images: torch.Tensor, widths: torch.Tensor):
        """images (B,C,H,W) padded on the right; widths (B,) valid pixel widths.

        Returns G (B,H',W',d) contextual 2D features, G1d (B,W',d) height-pooled features,
        col_mask (B,W') True on padded columns, feat_w (B,) valid column counts.
        """
        f = self.backbone(images)                              # B,C,H',W'
        B, C, Hp, Wp = f.shape
        f = self.proj(f.permute(0, 2, 3, 1))                   # B,H',W',d
        d = f.shape[-1]
        pe = torch.cat([sinusoid(Hp, d // 2, f.device)[:, None, :].expand(Hp, Wp, d // 2),
                        sinusoid(Wp, d // 2, f.device)[None, :, :].expand(Hp, Wp, d // 2)], -1)
        feat_w = self.feature_widths(widths).clamp(max=Wp)
        col_mask = torch.arange(Wp, device=f.device)[None, :] >= feat_w[:, None]     # B,W'
        key_mask = col_mask[:, None, :].expand(B, Hp, Wp).reshape(B, Hp * Wp)
        g = self.encoder((f + pe.to(f.dtype)).reshape(B, Hp * Wp, d), src_key_padding_mask=key_mask)
        G = g.reshape(B, Hp, Wp, d)
        G1d = G.mean(1)                                        # height pooling [paper]
        return G, G1d, col_mask, feat_w

    def select_features(self, G, G1d, col_mask):
        """-> (H, U1d, aux). aux may carry `route_logits` from MAFS."""
        if self.feature_selector is None:
            return G, G1d, {}
        out = self.feature_selector(G, G1d, col_mask)
        return out if len(out) == 3 else (*out, {})

    # ---- heads --------------------------------------------------------------------
    def ctc_logits(self, G1d):
        return self.ctc_head(G1d)                              # B,W',V

    def decode_logits(self, H, col_mask, tgt_in, tgt_pad_mask=None):
        B, Hp, Wp, d = H.shape
        mem = H.reshape(B, Hp * Wp, d)
        mem_mask = col_mask[:, None, :].expand(B, Hp, Wp).reshape(B, Hp * Wp)
        L = tgt_in.shape[1]
        x = self.tok_emb(tgt_in) * math.sqrt(d) + sinusoid(L, d, H.device).to(H.dtype)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=H.device), 1)
        y = self.decoder(self.dec_drop(x), mem, tgt_mask=causal, tgt_key_padding_mask=tgt_pad_mask,
                         memory_key_padding_mask=mem_mask)
        return self.dec_head(y)                                # B,L,V

    def forward(self, images, widths, tgt_in, tgt_pad_mask=None):
        G, G1d, col_mask, feat_w = self.encode(images, widths)
        H, U1d, aux = self.select_features(G, G1d, col_mask)
        return {"ctc_logits": self.ctc_logits(U1d), "feat_w": feat_w,
                "dec_logits": self.decode_logits(H, col_mask, tgt_in, tgt_pad_mask), **aux}

    # ---- inference ----------------------------------------------------------------
    @torch.no_grad()
    def recognize(self, images, widths, bos_id, eos_id, max_len=None, decoders=("ctc", "tr")):
        """Greedy decoding. Returns dict of id lists per decoder."""
        G, G1d, col_mask, feat_w = self.encode(images, widths)
        H, U1d, aux = self.select_features(G, G1d, col_mask)
        out = {}
        if "route_logits" in aux:
            out["route_probs"] = aux["route_logits"].float().softmax(-1).tolist()
        best = self.ctc_logits(U1d).argmax(-1)
        if "ctc" in decoders:
            out["ctc"] = [best[b, :feat_w[b]].tolist() for b in range(best.shape[0])]
        if "tr" in decoders:
            # cap each TR output at 2x its CTC read + 10 tokens: the decoder occasionally misses EOS and
            # runs to max_len, and one such line can double the CER of a small val set (blank id = 0)
            new_tok = torch.ones_like(best, dtype=torch.bool)
            new_tok[:, 1:] = best[:, 1:] != best[:, :-1]
            valid = torch.arange(best.shape[1], device=best.device)[None] < feat_w[:, None]
            cap = 2 * ((best != 0) & new_tok & valid).sum(1) + 10
            out["tr"] = self.greedy_decode(H, col_mask, bos_id, eos_id, max_len or self.cfg["max_dec_len"], cap)
        return out

    @torch.no_grad()
    def greedy_decode(self, H, col_mask, bos_id, eos_id, max_len, cap=None):
        """Incremental greedy decoding with cached K/V, using the nn.TransformerDecoderLayer weights.

        Equivalent to re-running decode_logits on the growing prefix (eval mode), but memory K/V are
        projected once and self-attention K/V are cached, so each step costs O(prefix + memory).
        """
        B, Hp, Wp, d = H.shape
        nh = self.cfg["nhead"]
        mem = H.reshape(B, Hp * Wp, d)
        attend = ~col_mask[:, None, :].expand(B, Hp, Wp).reshape(B, 1, 1, Hp * Wp)
        heads = lambda x: x.reshape(B, -1, nh, d // nh).transpose(1, 2)
        proj = lambda mha, x, i: F.linear(x, mha.in_proj_weight[i * d:(i + 1) * d], mha.in_proj_bias[i * d:(i + 1) * d])
        merge = lambda a: a.transpose(1, 2).reshape(B, 1, d)
        layers = self.decoder.layers
        cross = [(heads(proj(l.multihead_attn, mem, 1)), heads(proj(l.multihead_attn, mem, 2))) for l in layers]
        cache_k, cache_v = [None] * len(layers), [None] * len(layers)
        pe = sinusoid(max_len, d, H.device)
        tok = torch.full((B,), bos_id, dtype=torch.long, device=H.device)
        done = torch.zeros(B, dtype=torch.bool, device=H.device)
        seq = []
        for t in range(max_len - 1):
            x = self.tok_emb(tok)[:, None] * math.sqrt(d) + pe[t].to(H.dtype)
            for i, l in enumerate(layers):
                h = l.norm1(x)
                k, v = heads(proj(l.self_attn, h, 1)), heads(proj(l.self_attn, h, 2))
                cache_k[i] = k if t == 0 else torch.cat([cache_k[i], k], 2)
                cache_v[i] = v if t == 0 else torch.cat([cache_v[i], v], 2)
                a = F.scaled_dot_product_attention(heads(proj(l.self_attn, h, 0)), cache_k[i], cache_v[i])
                x = x + l.self_attn.out_proj(merge(a))
                a = F.scaled_dot_product_attention(heads(proj(l.multihead_attn, l.norm2(x), 0)), *cross[i],
                                                   attn_mask=attend)
                x = x + l.multihead_attn.out_proj(merge(a))
                x = x + l.linear2(l.activation(l.linear1(l.norm3(x))))
            nxt = self.dec_head(self.decoder.norm(x))[:, 0].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
            seq.append(nxt)
            tok = nxt
            done |= nxt == eos_id
            if cap is not None:
                done |= cap <= t + 1                               # per-sample length limit
            if done.all():
                break
        return torch.stack(seq, 1).tolist() if seq else [[] for _ in range(B)]


def count_parameters(model: UKTRBase) -> dict:
    n = lambda m: sum(p.numel() for p in m.parameters())
    return {"backbone": n(model.backbone), "encoder": n(model.encoder) + n(model.proj),
            "ctc_head": n(model.ctc_head),
            "transformer_decoder": n(model.decoder) + n(model.tok_emb) + n(model.dec_head),
            "mafs": n(model.feature_selector) if model.feature_selector is not None else 0,
            "total": n(model)}
