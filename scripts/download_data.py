"""Download the Stage-1 pretraining datasets.

  python scripts/download_data.py                       # everything (~12.7 GB + 2.4 GB + ~3.9 GB)
  python scripts/download_data.py --smoke               # small subsets for an end-to-end check
  python scripts/download_data.py --datasets hiertext   # one source

Writes data/raw/<source>/... and data/raw/DOWNLOAD_PROVENANCE.json (urls, revisions, sizes,
licenses). Downloads resume if interrupted.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.config import load_config, resolve  # noqa: E402

UA = {"User-Agent": "uktr-stage1-downloader"}


def http_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        n = r.headers.get("Content-Length")
        return int(n) if n else None


def download(url: str, dst: Path, retries: int = 5) -> int:
    """Resumable download with HTTP Range. Returns final size; verifies against Content-Length."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = http_size(url)
    if dst.exists() and total and dst.stat().st_size == total:
        print(f"  exists {dst.name} ({total / 1e9:.2f} GB)")
        return total
    part = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        hdr = dict(UA, **({"Range": f"bytes={have}-"} if have else {}))
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r, \
                    part.open("ab" if have and r.status == 206 else "wb") as f:
                got, t0 = (have if r.status == 206 else 0), time.time()
                while chunk := r.read(1 << 20):
                    f.write(chunk); got += len(chunk)
                    if total and time.time() - t0 > 10:
                        print(f"  {dst.name}: {got / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True); t0 = time.time()
            break
        except OSError as e:
            print(f"  retry {attempt + 1}/{retries} after error: {e}")
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"failed to download {url}")
    size = part.stat().st_size
    if total and size != total:
        raise RuntimeError(f"size mismatch for {url}: {size} != {total}")
    part.replace(dst)
    return size


def extract_jpgs(fileobj, dest: Path, limit: int | None = None) -> int:
    """Stream a .tgz and write *.jpg members flat into dest (basename only: no path traversal)."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
        for m in tar:
            if not (m.isfile() and m.name.lower().endswith(".jpg")):
                continue
            out = dest / Path(m.name).name
            if not out.exists():
                out.write_bytes(tar.extractfile(m).read())
            n += 1
            if limit and n >= limit:
                break
    return n


def get_hf(name, src, raw, smoke, smoke_cfg):
    from huggingface_hub import HfApi, snapshot_download
    patterns = smoke_cfg[f"{name}_files"] if smoke else src["files"]
    info = HfApi().dataset_info(src["hf_repo"], revision=src.get("revision", "main"))
    print(f"[{name}] {src['hf_repo']} @ {info.sha} files={patterns}")
    local = Path(snapshot_download(src["hf_repo"], repo_type="dataset", revision=info.sha,
                                   local_dir=raw / name, allow_patterns=patterns + ["README.md"]))
    files = {str(p.relative_to(local)).replace("\\", "/"): p.stat().st_size
             for p in local.rglob("*.parquet")}
    return {"hf_repo": src["hf_repo"], "revision": info.sha, "files": files, "license": src["license"]}


def get_hiertext(src, raw, smoke, smoke_cfg):
    out = {"annotations": {}, "images": {}, "license": src["license"]}
    for split in src["splits"]:
        url = src["annotations_url"].format(split=split)
        dst = raw / "hiertext" / "gt" / f"{split}.jsonl.gz"
        out["annotations"][split] = {"url": url, "bytes": download(url, dst)}
    for split in (smoke_cfg["hiertext_splits"] if smoke else src["splits"]):
        url = src["images_url"].format(split=split)
        dest = raw / "hiertext" / "images" / split
        if smoke:  # stream only the first N images; no full tarball on disk
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                n = extract_jpgs(r, dest, smoke_cfg["hiertext_max_images"])
            out["images"][split] = {"url": url, "n_images": n, "partial": True}
        else:
            tgz = raw / "hiertext" / f"{split}.tgz"
            size = download(url, tgz)
            marker = dest / ".extracted"
            if not marker.exists():
                print(f"  extracting {tgz.name} ...")
                with tgz.open("rb") as f:
                    n = extract_jpgs(f, dest)
                marker.write_text(str(n))
            out["images"][split] = {"url": url, "bytes": size, "n_images": int(marker.read_text()),
                                    "partial": False}
        print(f"[hiertext] {split}: {out['images'][split]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--datasets", nargs="+", default=["khmer_printed", "khmer_synthetic", "hiertext"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(resolve(a.config))
    raw = resolve(cfg["raw_dir"]); raw.mkdir(parents=True, exist_ok=True)
    prov_path = raw / "DOWNLOAD_PROVENANCE.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {}
    for name in a.datasets:
        src = cfg["sources"][name]
        print(f"\n=== {name}\nLICENSE: {src['license']}")
        try:
            rec = (get_hiertext(src, raw, a.smoke, cfg["smoke"]) if name == "hiertext"
                   else get_hf(name, src, raw, a.smoke, cfg["smoke"]))
            rec.update(status="ok", mode="smoke" if a.smoke else "full")
        except Exception as e:  # record and continue with the other sources
            print(f"!! {name} unavailable: {e}")
            rec = {"status": "failed", "error": repr(e), "license": src["license"]}
        rec["downloaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        prov[name] = rec
        prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nprovenance -> {prov_path}")


if __name__ == "__main__":
    main()
