"""Extract / crop images to data/images/<source>/<split>/ and write data/interim/<source>.csv.

Per row: image_path, raw_text, source_dataset, official_split, width, height, license, md5,
dhash, handwritten, group. Corrupt or undecodable images are counted in <source>_stats.json and skipped.

Source types (configs/*data.yaml `type:`):
  parquet   parquet files with embedded images (HF datasets; local ones via `path:`)
  hiertext  HierText annotations + images (line crops)
  labels    your own data: a csv/tsv/txt/jsonl label file + an image folder (see configs/stage2_data.yaml)

  python scripts/prepare_data.py                  # all downloaded sources
  python scripts/prepare_data.py --sources hiertext --workers 16
  python scripts/prepare_data.py --config configs/stage2_data.yaml   # Stage-2 handwriting sources
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.config import load_config, resolve  # noqa: E402

COLUMNS = ["image_path", "raw_text", "source_dataset", "official_split", "width", "height",
           "license", "md5", "dhash", "handwritten", "group"]


def dhash(im: Image.Image) -> str:
    """64-bit difference hash (perceptual)."""
    a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def check_image(data: bytes):
    """Decode fully; returns (PIL image, ext) or raises on corruption."""
    im = Image.open(io.BytesIO(data))
    fmt = (im.format or "PNG").lower()
    im.load()
    if im.width < 2 or im.height < 2:
        raise ValueError("degenerate image")
    return im, {"jpeg": "jpg"}.get(fmt, fmt)


# ---------------------------------------------------------------- parquet sources
def parquet_task(task, data_root: Path, source: str, license_: str, text_column=None, image_column="image",
                 handwritten=False, max_save_height=None):
    pq_path, rg, split, stem = task
    pf = pq.ParquetFile(pq_path)
    tbl = pf.read_row_group(rg).to_pylist()
    rows, stats = [], Counter()
    out_dir = data_root / "images" / source / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(tbl):
        text = r.get(text_column) if text_column else r.get("label", r.get("text"))
        img = r.get(image_column)
        data = img.get("bytes") if isinstance(img, dict) else img
        if not data or text is None:
            stats["missing_field"] += 1; continue
        try:
            im, ext = check_image(data)
        except Exception:
            stats["corrupt_image"] += 1; continue
        if max_save_height and im.height > max_save_height:   # big scans: store downscaled (training uses 32 px)
            im = im.convert("RGB").resize((max(2, round(im.width * max_save_height / im.height)), max_save_height),
                                    Image.LANCZOS)
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=95); data, ext = buf.getvalue(), "jpg"
            stats["downscaled"] += 1
        name = f"{stem}_{rg:04d}_{i:04d}.{ext}"
        (out_dir / name).write_bytes(data)          # original bytes unless downscaled above
        rows.append([f"images/{source}/{split}/{name}", text, source, split, im.width, im.height, license_,
                     hashlib.md5(data).hexdigest(), dhash(im), handwritten, ""])
        stats["ok"] += 1
    return rows, stats


def parquet_tasks(source, src, raw: Path):
    tasks, root = [], source_root(source, src, raw)
    for f in ([root] if root.is_file() else sorted(root.rglob("*.parquet"))):
        split = src.get("split_map", {}).get(f.stem, "")   # khmer_synthetic: no official split
        stem = f.stem.replace("-of-", "_").replace("train-", "s") if source == "khmer_synthetic" else f.stem
        tasks += [(str(f), rg, split or "all", stem) for rg in range(pq.ParquetFile(f).num_row_groups)]
    return tasks


def source_root(source, src, raw: Path) -> Path:
    return resolve(src["path"]) if "path" in src else resolve(src["labels"]) if "labels" in src else raw / source


# ---------------------------------------------------------------- own label files
def read_labels(src) -> pd.DataFrame:
    path = resolve(src["labels"])
    fmt = src.get("format") or path.suffix.lstrip(".").lower()
    if fmt == "jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    tsv = fmt in ("tsv", "txt")
    return pd.read_csv(path, sep="\t" if tsv else ",", header=0 if src.get("header", True) else None,
                       dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE if tsv else csv.QUOTE_MINIMAL)


def labels_tasks(src, chunk=256):
    df = read_labels(src)
    headerless = not src.get("header", True)
    ic, tc = src.get("image_column", 0 if headerless else "image_path"), src.get("text_column", 1 if headerless else "text")
    root = resolve(src.get("images_root", resolve(src["labels"]).parent))
    sc, gc = src.get("split_column"), src.get("group_column")
    rows = [(str(root / str(r[ic])), str(r[tc]), str(r[sc]) if sc else "all", str(r[gc]) if gc else "")
            for _, r in df.iterrows()]
    return [rows[i:i + chunk] for i in range(0, len(rows), chunk)]


def labels_task(task, data_root: Path, source: str, license_: str, handwritten: bool):
    rows, stats = [], Counter()
    out_dir = data_root / "images" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    for src_path, text, split, group in task:
        try:
            data = Path(src_path).read_bytes()
        except OSError:
            stats["image_missing"] += 1; continue
        try:
            im, ext = check_image(data)
        except Exception:
            stats["corrupt_image"] += 1; continue
        if not text.strip():
            stats["empty_text"] += 1; continue
        name = f"{hashlib.md5(src_path.encode('utf-8')).hexdigest()[:20]}.{ext}"   # unique per source path
        (out_dir / name).write_bytes(data)
        rows.append([f"images/{source}/{name}", text, source, split, im.width, im.height, license_,
                     hashlib.md5(data).hexdigest(), dhash(im), handwritten, group])
        stats["ok"] += 1
    return rows, stats


# ---------------------------------------------------------------- HierText
def crop_line(img, vertices):
    """Perspective-rectify a 4-point line box ordered TL, TR, BR, BL (HierText convention)."""
    p = np.asarray(vertices, dtype=np.float32)
    if len(p) != 4:
        p = cv2.boxPoints(cv2.minAreaRect(p)).astype(np.float32)
    w = int(round(max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3]))))
    h = int(round(max(np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1]))))
    if w < 2 or h < 2:
        return None
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(p, dst), (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def hiertext_task(ann, img_dir: Path, data_root: Path, split: str, cfg: dict, license_: str):
    rows, stats = [], Counter()
    path = img_dir / f"{ann['image_id']}.jpg"
    if not path.exists():
        stats["image_not_downloaded"] += 1
        return rows, stats
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        stats["corrupt_image"] += 1
        return rows, stats
    out_dir = data_root / "images" / "hiertext" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for pi, para in enumerate(ann["paragraphs"]):
        for li, line in enumerate(para["lines"]):
            text = line.get("text", "")
            if line.get("vertical") and not cfg["keep_vertical"]:
                stats["skip_vertical"] += 1; continue
            if not line.get("legible", True) and not cfg["keep_illegible"]:
                stats["skip_illegible"] += 1; continue
            if not text.strip():
                stats["skip_empty_text"] += 1; continue
            crop = crop_line(img, line["vertices"])
            if crop is None or crop.shape[0] < cfg["min_crop_height"]:
                stats["skip_too_small"] += 1; continue
            if crop.shape[0] > cfg["max_save_height"]:
                s = cfg["max_save_height"] / crop.shape[0]
                crop = cv2.resize(crop, (max(2, round(crop.shape[1] * s)), cfg["max_save_height"]),
                                  interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            data = buf.tobytes()
            name = f"{ann['image_id']}_{pi:03d}_{li:03d}.jpg"
            (out_dir / name).write_bytes(data)
            im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            rows.append([f"images/hiertext/{split}/{name}", text, "hiertext", split, crop.shape[1],
                         crop.shape[0], license_, hashlib.md5(data).hexdigest(), dhash(im),
                         bool(line.get("handwritten", False)), ""])
            stats["ok"] += 1
    return rows, stats


def run(pool, fn, tasks, desc):
    rows, stats = [], Counter()
    for k, (r, s) in enumerate(pool.imap_unordered(fn, tasks, chunksize=4), 1):
        rows += r; stats.update(s)
        if k % max(1, len(tasks) // 20) == 0 or k == len(tasks):
            print(f"  {desc}: {k}/{len(tasks)} tasks, {stats['ok']} rows", flush=True)
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-run sources that already have an interim CSV")
    a = ap.parse_args()
    cfg = load_config(resolve(a.config))
    raw, data_root, interim = resolve(cfg["raw_dir"]), resolve(cfg["data_root"]), resolve(cfg["interim_dir"])
    interim.mkdir(parents=True, exist_ok=True)
    workers = a.workers or cfg.get("num_workers", 8)
    with Pool(workers) as pool:
        for source in a.sources or list(cfg["sources"]):
            src, out_csv = cfg["sources"][source], interim / f"{source}.csv"
            if out_csv.exists() and not a.force:
                print(f"[{source}] {out_csv} exists, skipping (use --force)"); continue
            if not source_root(source, src, raw).exists():
                print(f"[{source}] not found at {source_root(source, src, raw)}, skipping"); continue
            print(f"[{source}]")
            kind = src.get("type", "hiertext" if source == "hiertext" else "parquet")
            hw = src.get("modality") == "handwritten"
            if kind == "labels":
                fn = partial(labels_task, data_root=data_root, source=source, license_=src["license"], handwritten=hw)
                rows, stats = run(pool, fn, labels_tasks(src), source)
            elif kind == "hiertext":
                rows, stats = [], Counter()
                for split in src["splits"]:
                    gt = raw / "hiertext" / "gt" / f"{split}.jsonl.gz"
                    img_dir = raw / "hiertext" / "images" / split
                    if not gt.exists() or not img_dir.exists():
                        print(f"  {split}: annotations or images missing, skipping"); continue
                    anns = json.load(gzip.open(gt, "rt", encoding="utf-8"))["annotations"]
                    fn = partial(hiertext_task, img_dir=img_dir, data_root=data_root, split=split,
                                 cfg=src, license_=src["license"])
                    r, s = run(pool, fn, anns, f"hiertext/{split}")
                    rows += r; stats.update(s)
            else:
                fn = partial(parquet_task, data_root=data_root, source=source, license_=src["license"],
                             text_column=src.get("text_column"), image_column=src.get("image_column", "image"),
                             handwritten=hw, max_save_height=src.get("max_save_height"))
                rows, stats = run(pool, fn, parquet_tasks(source, src, raw), source)
            df = pd.DataFrame(rows, columns=COLUMNS).sort_values("image_path", kind="stable")
            df.to_csv(out_csv, index=False, encoding="utf-8")
            (interim / f"{source}_stats.json").write_text(json.dumps(dict(stats), indent=2), encoding="utf-8")
            print(f"[{source}] {len(df)} rows -> {out_csv}  stats={dict(stats)}")


if __name__ == "__main__":
    main()
