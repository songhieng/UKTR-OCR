"""Checkpoint format + Stage-2 loading.

A checkpoint is a dict with (at least):
  format_version, stage, model_config, model_state, tokenizer (KCCTokenizer.to_dict()),
  and for training resumption: optimizer_state, scheduler_state, scaler_state, epoch,
  global_step, batch_in_epoch, best_metric, rng_state. Plus metadata (metrics, data provenance,
  param counts, train config). The inference export drops the resumption fields.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from ..tokenizer.kcc import KCCTokenizer
from .uktr import UKTRBase

FORMAT_VERSION = 1
RESUME_KEYS = ("optimizer_state", "scheduler_state", "scaler_state", "rng_state")


def save_checkpoint(path, **payload) -> None:
    """Atomic save (write to tmp then rename) so a crash never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": FORMAT_VERSION, **payload}, tmp)
    os.replace(tmp, path)


def export_inference(src, dst) -> None:
    ck = torch.load(src, map_location="cpu", weights_only=True)
    save_checkpoint(dst, **{k: v for k, v in ck.items() if k not in RESUME_KEYS and k != "format_version"})


def extend_vocab(ck: dict, chars) -> dict:
    """Append `chars` missing from a checkpoint's vocabulary and grow the vocab-sized layers.

    Old token ids and their logits are unchanged. New output rows are drawn with the std of the existing
    rows and get the mean existing bias, so no new token starts out favoured. Returns a new checkpoint dict.
    """
    tok = KCCTokenizer.from_dict(ck["tokenizer"])
    new = [c for c in dict.fromkeys(chars) if c not in tok.stoi]
    if not new:
        return ck
    tok, st = KCCTokenizer(tok.vocab + new, tok.normalization), dict(ck["model_state"])
    g = torch.Generator().manual_seed(0)
    for k in ("ctc_head.weight", "tok_emb.weight", "dec_head.weight"):
        w = st[k]
        st[k] = torch.cat([w, torch.randn(len(new), w.shape[1], generator=g).to(w.dtype) * w.float().std()])
    for k in ("ctc_head.bias", "dec_head.bias"):
        st[k] = torch.cat([st[k], st[k].mean().expand(len(new))])
    return {**ck, "tokenizer": tok.to_dict(), "model_state": st,
            "model_config": {**ck["model_config"], "vocab_size": len(tok)}}


def load_stage1(path, model: torch.nn.Module | None = None, device="cpu",
                allowed_missing_prefixes=("feature_selector.",)):
    """Load a Stage-1 checkpoint.

    model=None  -> build a UKTRBase from the stored config and load strictly.
    model=<Stage-2 model> -> load the Stage-1 weights into it; only keys under
        `allowed_missing_prefixes` (the new MAFS module) may be missing, nothing may be
        unexpected and every shape must match. Raises otherwise.
    Returns (model, tokenizer, checkpoint_dict).
    """
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck.get("format_version") == FORMAT_VERSION, f"unknown checkpoint format {ck.get('format_version')}"
    tok = KCCTokenizer.from_dict(ck["tokenizer"])
    if model is None:
        model = UKTRBase(ck["model_config"])
        model.load_state_dict(ck["model_state"], strict=True)
    else:
        own = model.state_dict()
        bad = [k for k, v in ck["model_state"].items() if k in own and own[k].shape != v.shape]
        if bad:
            raise ValueError(f"shape mismatch for {bad[:10]}")
        res = model.load_state_dict(ck["model_state"], strict=False)
        missing = [k for k in res.missing_keys if not k.startswith(tuple(allowed_missing_prefixes))]
        if missing or res.unexpected_keys:
            raise ValueError(f"missing={missing[:10]} unexpected={res.unexpected_keys[:10]}")
    return model.to(device), tok, ck
