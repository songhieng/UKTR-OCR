"""Add the Thai block to a checkpoint's vocabulary, so Thai handwriting can be used as extra training data.

  python scripts/extend_vocab.py checkpoints/uktr_base_stage1_best.pt checkpoints/uktr_stage1_thai.pt

Khmer token ids are unchanged; the CTC head, token embedding and decoder head get one new row per Thai
character. Every Thai-experiment config (configs/thai_*.yaml) starts from the output, so they share one vocabulary.
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.checkpoint import extend_vocab, save_checkpoint  # noqa: E402

THAI = [chr(c) for c in range(0x0E00, 0x0E80) if unicodedata.category(chr(c)) != "Cn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=True)
    n0 = ck["model_config"]["vocab_size"]
    ck = extend_vocab(ck, THAI)
    save_checkpoint(a.dst, **{k: v for k, v in ck.items() if k != "format_version"})
    print(f"vocab {n0} -> {ck['model_config']['vocab_size']} -> {a.dst}")


if __name__ == "__main__":
    main()
