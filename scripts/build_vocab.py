"""Build the KCC vocabulary from TRAIN-split transcriptions only.

  python scripts/build_vocab.py --min-freq 2
Writes data/tokenizer/kcc_vocab.json (vocab + normalization config) and kcc_vocab_stats.json.
The UKTR paper's vocabulary (11,899 entries) is unreleased, so the size here is data-dependent.

Stage 2 must REUSE the Stage-1 vocabulary (its size fixes the output layers). With --existing the
tokenizer is loaded instead of built, and only the coverage / round-trip report is written:
  python scripts/build_vocab.py --config configs/stage2_data.yaml --existing data/tokenizer/kcc_vocab.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tokenizer.kcc import DEFAULT_NORMALIZATION, SPECIALS, KCCTokenizer, segment  # noqa: E402
from src.training.config import load_config, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out", default="data/tokenizer/kcc_vocab.json")
    ap.add_argument("--min-freq", type=int, default=2, help="min train frequency for a multi-char cluster")
    ap.add_argument("--max-size", type=int, default=None)
    ap.add_argument("--existing", default=None, help="load this vocab and only report coverage (Stage 2)")
    a = ap.parse_args()
    cfg = load_config(resolve(a.config))
    df = pd.read_csv(resolve(cfg["manifest"]), keep_default_na=False, dtype={"transcription": str})
    train = df[df.split == "train"]
    assert len(train) and set(train.split) == {"train"}
    if a.existing:
        tok = KCCTokenizer.load(resolve(a.existing))
        counts = {c for t in train.transcription for c in segment(t)}
        out = resolve(cfg["report_dir"]) / "kcc_vocab_coverage.json"
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        tok, counts = KCCTokenizer.build(train.transcription, DEFAULT_NORMALIZATION, a.min_freq, a.max_size)
        out = resolve(a.out)
        tok.save(out)

    stats = {"vocab_size": len(tok), "n_specials": len(SPECIALS), "min_freq": a.min_freq,
             "train_rows": len(train), "train_unique_kcc": len(counts), "paper_vocab_size": 11899, "eval": {}}
    report_df = df if a.existing else df[df.split != "train"]    # Stage 2: also check the new train data
    for (src, sp), e in report_df.groupby(["source_dataset", "split"]):
        n_tok = n_fallback = n_unk = 0
        for t in e.transcription:
            for c in segment(t):
                n_tok += 1
                if c not in tok.stoi:
                    n_fallback += 1
                    n_unk += any(ch not in tok.stoi for ch in c)
        rt = sum(tok.decode(tok.encode(t, normalized=True)) == t for t in e.transcription) / max(len(e), 1)
        stats["eval"][f"{src}/{sp}"] = {"kcc_tokens": n_tok, "in_vocab_frac": round(1 - n_fallback / max(n_tok, 1), 5),
                                        "codepoint_fallback_tokens": n_fallback, "tokens_with_unk": n_unk,
                                        "roundtrip_exact_frac": round(rt, 5)}
    (out if a.existing else out.with_name(out.stem + "_stats.json")).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
