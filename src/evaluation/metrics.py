"""Corpus-level recognition metrics. CER = total edit distance / total reference characters."""
from __future__ import annotations

from rapidfuzz.distance import Levenshtein

from ..tokenizer.kcc import segment


def _rate(pairs) -> float:
    err = ref = 0
    for p, r in pairs:
        err += Levenshtein.distance(p, r)
        ref += len(r)
    return err / max(ref, 1)


def cer(preds, refs) -> float:
    return _rate(zip(preds, refs))


def kcc_er(preds, refs) -> float:
    """Khmer Character Cluster error rate: edit distance over KCC token sequences."""
    return _rate((segment(p), segment(r)) for p, r in zip(preds, refs))


def line_accuracy(preds, refs) -> float:
    return sum(p == r for p, r in zip(preds, refs)) / max(len(refs), 1)


def sample_cer(pred, ref) -> float:
    return Levenshtein.distance(pred, ref) / max(len(ref), 1)


def summarize(preds, refs) -> dict:
    return {"cer": cer(preds, refs), "kcc_er": kcc_er(preds, refs),
            "line_acc": line_accuracy(preds, refs), "n": len(refs)}
