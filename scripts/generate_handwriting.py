"""Generate synthetic Khmer handwriting line images for Stage 2 (vendored khmer_synth generator).

  python scripts/fetch_fonts.py --out data/fonts                     # once (OFL/LGPL Khmer fonts)
  python scripts/generate_handwriting.py --num-texts 100000 --workers 16
  python scripts/generate_handwriting.py --corpus my_lines.txt --manifest-texts 0   # your own text only

Pipeline (src/khmer_synth, see its README section in docs/SYNTHETIC_HANDWRITING.md): HarfBuzz-shaped
font outlines -> skeleton -> planned pen strokes -> procedural writer deformation/timing/pressure ->
swept-disk ink rendering on paper backgrounds -> camera/scan degradation.

Text comes from the Stage-1 TRAIN split only (so no validation/test text leaks into Stage 2), cut at
Khmer-cluster boundaries into handwriting-length snippets, plus any --corpus files.
Writes <out>/labels.csv (image_path, text, writer_id, font). Use writer_id as the split group so
validation writers are never seen in training (configs/stage2_data.yaml does this).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.khmer_synth.config import PipelineConfig, merge_overrides  # noqa: E402
from src.khmer_synth.dataset.build import build_dataset  # noqa: E402
from src.tokenizer.kcc import has_khmer, normalize, segment  # noqa: E402
from src.training.config import resolve  # noqa: E402


def manifest_snippets(manifest, sources, n, min_k, max_k, seed) -> list[str]:
    """n distinct snippets of min_k..max_k Khmer clusters from train-split Khmer lines."""
    df = pd.read_csv(manifest, keep_default_na=False, usecols=["transcription", "split", "source_dataset", "language"])
    df = df[(df.split == "train") & df.source_dataset.isin(sources) & df.language.isin(["km", "mixed"])]
    texts = sorted(df.transcription.unique())
    rng = random.Random(seed)
    rng.shuffle(texts)
    out, seen = [], set()
    for t in texts * 3:                      # long lines can yield several different snippets
        toks = segment(t)
        k = rng.randint(min_k, max_k)
        if len(toks) > k:                    # random window; prefer starting right after a space
            starts = [i + 1 for i, c in enumerate(toks[:-k]) if c == " "] or list(range(len(toks) - k + 1))
            s = rng.choice(starts)
            toks = toks[s:s + k]
        snip = "".join(toks).strip()
        if min_k <= len(segment(snip)) and has_khmer(snip) and snip not in seen:
            seen.add(snip); out.append(snip)
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="configs/handwriting_synth.json")
    ap.add_argument("--out", default=None, help="output dir (default: preset dataset.output_dir)")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--sources", nargs="+", default=["khmer_printed", "khmer_synthetic"])
    ap.add_argument("--manifest-texts", "--num-texts", dest="n", type=int, default=100_000,
                    help="snippets taken from the Stage-1 train split (0 = none)")
    ap.add_argument("--min-clusters", type=int, default=2)
    ap.add_argument("--max-clusters", type=int, default=30)
    ap.add_argument("--corpus", nargs="*", default=[], help="extra .txt files, one line per sample")
    ap.add_argument("--num-writers", type=int, default=None)
    ap.add_argument("--styles-per-text", type=int, default=None, help="renderings per text (different writers)")
    ap.add_argument("--workers", type=int, default=None, help="default: all cores")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="generator overrides, e.g. degradation.p_blur=0.3")
    a = ap.parse_args()

    cfg = merge_overrides(PipelineConfig.load(resolve(a.preset)), a.set)
    ds = cfg.dataset
    out = resolve(a.out or ds.output_dir)
    ds.output_dir, ds.cache_dir = str(out), str(resolve(ds.cache_dir))
    cfg.fonts.paths = [str(resolve(p)) for p in cfg.fonts.paths]
    if a.num_writers:
        ds.num_writers = a.num_writers
    if a.styles_per_text:
        ds.styles_per_text = a.styles_per_text
    if not any(Path(p).glob("*.[ot]tf") for p in cfg.fonts.paths):
        sys.exit(f"no fonts in {cfg.fonts.paths}: run  python scripts/fetch_fonts.py --out data/fonts")

    texts = manifest_snippets(resolve(a.manifest), a.sources, a.n, a.min_clusters, a.max_clusters, a.seed) if a.n else []
    for f in a.corpus:
        texts += [normalize(t) for t in Path(f).read_text(encoding="utf-8").splitlines() if normalize(t)]
    if not texts:
        sys.exit("empty corpus")
    print(f"corpus: {len(texts)} texts -> {len(texts) * ds.styles_per_text} images, "
          f"{ds.num_writers} writers -> {out}")
    summary = build_dataset(texts, cfg, progress=True, workers=a.workers)

    # labels.csv for scripts/prepare_data.py (type: labels); drop samples with missing glyphs
    n_ok = n_bad = 0
    with (out / "metadata.jsonl").open(encoding="utf-8") as f, \
            (out / "labels.csv").open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["image_path", "text", "writer_id", "font"])
        for line in f:
            m = json.loads(line)
            if not m.get("degraded_image") or m.get("extra", {}).get("has_notdef"):
                n_bad += 1; continue
            w.writerow([m["degraded_image"], m["normalized_text"], m["writer_id"], m["font"]])
            n_ok += 1
    print(json.dumps({k: summary.get(k) for k in ("written", "failed", "elapsed_s", "samples_per_s")}),
          f"labels.csv: {n_ok} rows ({n_bad} dropped: missing glyphs / no image)")


if __name__ == "__main__":
    main()
