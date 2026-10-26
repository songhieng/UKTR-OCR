"""Merge interim CSVs into data/manifest.csv: normalize, validate, de-duplicate, split, check leakage.

Outputs: data/manifest.csv, data/manifest_meta.json, results/data/data_report.{json,md}
Manifest columns: image_path, transcription, source_dataset, language, split, width, height,
license, handwritten, modality, group, raw_transcription, md5, dhash

modality: the row's handwritten flag -> "handwritten", else the source's `modality` (default "printed").
Sources without an official split are split by md5 of `group` (e.g. writer id) when given, else of the
normalized text, so a writer / a text never straddles train and eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tokenizer.kcc import DEFAULT_NORMALIZATION, detect_language, normalize, segment  # noqa: E402
from src.training.config import load_config, resolve  # noqa: E402

SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def supported_char(c: str) -> bool:
    """Khmer, Thai, Latin (incl. Latin-1/Extended-A/B), general punctuation, currency, common symbols."""
    o = ord(c)
    return (o < 0x250 or 0x0E00 <= o < 0x0E80 or 0x1780 <= o < 0x1800 or 0x19E0 <= o < 0x1A00 or 0x2000 <= o < 0x2070
            or 0x20A0 <= o < 0x20D0 or 0x2100 <= o < 0x2190)


def invalid_reason(t: str, drop_unsupported: bool) -> str | None:
    if not t:
        return "empty_after_normalization"
    for c in t:
        cat = unicodedata.category(c)
        if c == "�" or cat in ("Cc", "Co", "Cn", "Cs"):
            return "invalid_codepoint"
    if drop_unsupported and not all(supported_char(c) for c in t):
        return "unsupported_script"
    return None


def assign_split(row, src_cfg) -> str:
    if row.source_dataset == "hiertext" and row.official_split == "validation":
        return "train" if src_cfg.get("validation_to_train") else "val"
    if row.official_split in ("train", "val", "test"):
        return row.official_split
    m = src_cfg.get("split_map", {})
    if row.official_split in m:
        return m[row.official_split]
    # no official split: bucket by group (writer) or text hash (deterministic)
    key = row.group or row.transcription
    b = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    v, te = src_cfg.get("val_fraction", 0.005), src_cfg.get("test_fraction", 0.005)
    return "val" if b < v else "test" if b < v + te else "train"


def by_source_split(df) -> dict:
    return {f"{a}/{b}": int(n) for (a, b), n in df.groupby(["source_dataset", "split"]).size().items()}


def describe(x) -> dict:
    x = np.asarray(x)
    if not len(x):
        return {}
    return {"mean": round(float(x.mean()), 2), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "max": int(x.max()), "min": int(x.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    a = ap.parse_args()
    cfg = load_config(resolve(a.config))
    vcfg, interim = cfg["validation"], resolve(cfg["interim_dir"])
    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "normalization": DEFAULT_NORMALIZATION,
              "dropped": {}, "prepare_stats": {}}

    frames = []
    for prio, source in enumerate(cfg["sources"]):
        f = interim / f"{source}.csv"
        if not f.exists():
            print(f"[{source}] no interim CSV, skipped"); continue
        df = pd.read_csv(f, keep_default_na=False, dtype={"raw_text": str, "official_split": str, "group": str})
        if "group" not in df:
            df["group"] = ""
        df["_prio"] = prio
        frames.append(df)
        st = interim / f"{source}_stats.json"
        if st.exists():
            report["prepare_stats"][source] = json.loads(st.read_text(encoding="utf-8"))
    df = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(df)} rows from {len(frames)} sources")

    # ---- normalize + validate
    df["transcription"] = [normalize(t) for t in df.raw_text]
    reasons = [invalid_reason(t, vcfg["drop_unsupported_script"]) for t in df.transcription]
    df["_bad"] = reasons
    report["dropped"]["validation"] = (df[df._bad.notna()].groupby(["source_dataset", "_bad"]).size()
                                       .unstack(fill_value=0).to_dict("index"))
    report["normalization_changed_rows"] = int((df.transcription != df.raw_text.str.strip()).sum())
    df = df[df._bad.isna()].copy()
    df["language"] = [detect_language(t) for t in df.transcription]
    df["modality"] = ["handwritten" if str(hw).lower() == "true" else cfg["sources"][s].get("modality", "printed")
                      for s, hw in zip(df.source_dataset, df.handwritten)]
    df["split"] = [assign_split(r, cfg["sources"][r.source_dataset]) for r in df.itertuples()]

    # sources that re-host another source's lines (KiteAether contains most of SoyVitou, test split included):
    # drop their copies first, else the dedup below keeps the train copy and removes the original's eval row
    for s, c in cfg["sources"].items():
        if c.get("drop_duplicates_of"):
            ref = df[df.source_dataset.isin(c["drop_duplicates_of"])]
            m = (df.source_dataset == s) & (df.md5.isin(ref.md5) | df.dhash.isin(ref.dhash)
                                            | df.transcription.isin(ref.transcription))
            report["dropped"][f"{s}_duplicate_of_{'+'.join(c['drop_duplicates_of'])}"] = int(m.sum())
            df = df[~m]

    # ---- de-duplicate: train copies first so eval copies are the ones removed (no leakage)
    df["_so"] = df.split.map(SPLIT_ORDER)
    df = df.sort_values(["_so", "_prio", "image_path"], kind="stable")
    exact = df.duplicated("md5")
    report["dropped"]["exact_duplicate"] = by_source_split(df[exact])
    df = df[~exact]
    if vcfg["perceptual_dedup"]:
        near = df.duplicated(["dhash", "transcription"])
        report["dropped"]["perceptual_duplicate"] = by_source_split(df[near])
        df = df[~near]
    # same perceptual hash but different text: possible label noise, reported only
    g = df.groupby("dhash").transcription.nunique()
    report["dhash_conflicts"] = int((g > 1).sum())

    # ---- leakage checks
    train = df[df.split == "train"]
    tr_md5, tr_ph = set(train.md5), set(zip(train.dhash, train.transcription))
    tr_text = set(train.transcription)
    leak = {}
    for (src, sp), e in df[df.split != "train"].groupby(["source_dataset", "split"]):
        leak[f"{src}/{sp}"] = {
            "n": len(e), "image_md5_in_train": int(e.md5.isin(tr_md5).sum()),
            "dhash+text_in_train": int(sum(k in tr_ph for k in zip(e.dhash, e.transcription))),
            "exact_text_in_train": int(e.transcription.isin(tr_text).sum()),
            "exact_text_in_train_frac": round(float(e.transcription.isin(tr_text).mean()), 4)}
    report["leakage"] = leak
    assert all(v["image_md5_in_train"] == 0 for v in leak.values()), leak
    if vcfg["perceptual_dedup"]:
        assert all(v["dhash+text_in_train"] == 0 for v in leak.values()), leak
    if vcfg["drop_text_overlap_from_eval"]:
        m = (df.split != "train") & df.transcription.isin(tr_text)
        report["dropped"]["eval_text_overlap"] = int(m.sum())
        df = df[~m]

    # ---- statistics
    df["n_chars"] = df.transcription.str.len()
    df["n_kcc"] = [len(segment(t)) for t in df.transcription]
    stats = {}
    for (src, sp), e in df.groupby(["source_dataset", "split"]):
        stats[f"{src}/{sp}"] = {"n": len(e), "chars": describe(e.n_chars), "kcc": describe(e.n_kcc),
                                "width": describe(e.width), "height": describe(e.height),
                                "language": e.language.value_counts().to_dict(),
                                "modality": e.modality.value_counts().to_dict(),
                                "handwritten": int(e.handwritten.astype(str).str.lower().eq("true").sum())}
    report["stats"] = stats
    tr_clusters = Counter(c for t in df[df.split == "train"].transcription for c in segment(t))
    tr_chars = Counter(c for t in df[df.split == "train"].transcription for c in t)
    cov = {}
    for (src, sp), e in df[df.split != "train"].groupby(["source_dataset", "split"]):
        cl = [c for t in e.transcription for c in segment(t)]
        ch = [c for t in e.transcription for c in t]
        cov[f"{src}/{sp}"] = {"kcc_token_coverage": round(sum(c in tr_clusters for c in cl) / max(len(cl), 1), 5),
                              "char_token_coverage": round(sum(c in tr_chars for c in ch) / max(len(ch), 1), 5),
                              "unseen_kcc_types": len({c for c in cl if c not in tr_clusters})}
    report["coverage_vs_train"] = cov
    report["train_unique_kcc"] = len(tr_clusters)
    report["train_unique_chars"] = len(tr_chars)
    report["train_top_kcc"] = tr_clusters.most_common(40)
    report["kcc_freq_buckets"] = {f">={k}": sum(1 for v in tr_clusters.values() if v >= k)
                                  for k in (1, 2, 5, 10, 100)}

    # ---- write
    cols = ["image_path", "transcription", "source_dataset", "language", "split", "width", "height",
            "license", "handwritten", "modality", "group", "raw_text", "md5", "dhash"]
    out = df.sort_values(["source_dataset", "split", "image_path"])[cols].rename(
        columns={"raw_text": "raw_transcription"})
    man = resolve(cfg["manifest"])
    out.to_csv(man, index=False, encoding="utf-8")
    meta = {"generated_at": report["generated_at"], "rows": len(out),
            "counts": by_source_split(out),
            "normalization": DEFAULT_NORMALIZATION,
            "licenses": {s: c["license"] for s, c in cfg["sources"].items()},
            "split_policy": "official splits where provided; khmer_synthetic by md5(normalized text) buckets; "
                            "hiertext validation held out unless validation_to_train"}
    (man.parent / "manifest_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    rep_dir = resolve(cfg["report_dir"]); rep_dir.mkdir(parents=True, exist_ok=True)
    report["counts"] = meta["counts"]
    (rep_dir / "data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                              encoding="utf-8")
    write_md(report, rep_dir / "data_report.md")
    print(json.dumps(meta["counts"], indent=1))
    print(f"manifest -> {man} ({len(out)} rows); report -> {rep_dir}")


def write_md(r, path: Path):
    L = ["# Stage-1 data report", "", f"Generated {r['generated_at']}", "", "## Counts", "",
         "| source/split | n | chars p50/p95/max | KCC p50/p95/max | height p50 | width p50 | languages |",
         "|---|---|---|---|---|---|---|"]
    for k, s in r["stats"].items():
        c, q = s["chars"], s["kcc"]
        L.append(f"| {k} | {s['n']} | {c['p50']:.0f}/{c['p95']:.0f}/{c['max']} | {q['p50']:.0f}/{q['p95']:.0f}/{q['max']} "
                 f"| {s['height']['p50']:.0f} | {s['width']['p50']:.0f} | {s['language']} |")
    L += ["", "## Dropped rows", "", "```json", json.dumps(r["dropped"], indent=1, default=str), "```",
          "", f"dHash conflicts (same hash, different text; kept): {r['dhash_conflicts']}",
          "", "## Train/eval leakage", "", "| eval set | n | md5 in train | dHash+text in train | exact text in train |",
          "|---|---|---|---|---|"]
    for k, v in r["leakage"].items():
        L.append(f"| {k} | {v['n']} | {v['image_md5_in_train']} | {v['dhash+text_in_train']} | "
                 f"{v['exact_text_in_train']} ({v['exact_text_in_train_frac']:.1%}) |")
    L += ["", "## Coverage of eval tokens by the training set", "",
          "| eval set | KCC token coverage | char coverage | unseen KCC types |", "|---|---|---|---|"]
    for k, v in r["coverage_vs_train"].items():
        L.append(f"| {k} | {v['kcc_token_coverage']:.4%} | {v['char_token_coverage']:.4%} | {v['unseen_kcc_types']} |")
    L += ["", f"Unique KCC in train: {r['train_unique_kcc']}; unique chars: {r['train_unique_chars']}; "
          f"frequency buckets: {r['kcc_freq_buckets']}", "",
          "Prepare-stage stats (corrupt images, skipped HierText lines):", "", "```json",
          json.dumps(r["prepare_stats"], indent=1), "```"]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
