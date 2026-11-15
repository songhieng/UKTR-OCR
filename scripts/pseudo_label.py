"""Self-training: label unlabeled handwriting lines with the current model, keep only confident reads.

A line is kept when the CTC and TR reads agree exactly and the joint NLL per token is <= --max-nll.

  # 1. calibrate --max-nll on labeled val lines (CER of the lines you would keep, per threshold)
  python scripts/pseudo_label.py --checkpoint CKPT --calibrate real_hw kiteaether
  # 2. label a folder of line crops (one sub-folder per writer -> writer_id)
  python scripts/pseudo_label.py --checkpoint CKPT --images data/raw/unlabeled --max-nll 0.05 \
      --out data/raw/pseudo_hw/labels.csv

Then add `pseudo_hw` (configs/stage2_data.yaml), rebuild the manifest and train configs/stage2_pseudo.yaml.
Never point --images at writers you test on.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.ocr_dataset import load_manifest, target_width, to_tensor  # noqa: E402
from src.evaluation.metrics import cer  # noqa: E402
from src.models.checkpoint import load_stage1  # noqa: E402
from src.training.config import resolve  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@torch.no_grad()
def read_all(model, tok, ck, paths, dev, batch_size=32):
    """-> DataFrame path, text (joint read), agree (CTC == TR), nll_per_tok."""
    d, c = ck["train_config"]["data"], ck["model_config"]["in_channels"]
    ims = [Image.open(p).convert("L" if c == 1 else "RGB") for p in paths]
    tws = [int(target_width(im.width, im.height, d["img_height"], d["min_width"], d["max_width"])) for im in ims]
    rows = []
    order = np.argsort(tws)                                   # similar widths per batch: less padding
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        W = -(-max(tws[i] for i in idx) // 4) * 4
        x = torch.zeros(len(idx), c, d["img_height"], W)      # zero padding, as the training collate
        for b, i in enumerate(idx):
            x[b, :, :, :tws[i]] = to_tensor(ims[i], tws[i], d["img_height"])
        out = model.recognize(x.to(dev), torch.tensor([tws[i] for i in idx], device=dev),
                              tok.bos_id, tok.eos_id, decoders=("joint",))
        for b, i in enumerate(idx):
            t_ids = out["tr"][b]
            t_ids = t_ids[:t_ids.index(tok.eos_id)] if tok.eos_id in t_ids else t_ids
            text = tok.decode(out["joint"][b])
            rows.append({"path": str(paths[i]), "text": text,
                         "agree": tok.decode_ctc(out["ctc"][b]) == tok.decode(t_ids),
                         "nll_per_tok": out["joint_nll"][b] / (len(out["joint"][b]) + 1)})
    return pd.DataFrame(rows).set_index("path").loc[[str(p) for p in paths]].reset_index()


def calibrate(df, refs):
    df = df.assign(ref=refs)
    print(f"all lines: n={len(df)} CER={cer(df.text, df.ref):.2%}")
    print("max_nll   kept    CER(kept)  CER(dropped)")
    for th in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, float("inf")):
        k = df.agree & (df.nll_per_tok <= th)
        dropped = cer(df.text[~k], df.ref[~k]) if (~k).any() else float("nan")
        print(f"{th:<8} {k.mean():6.1%}   {cer(df.text[k], df.ref[k]):8.2%}   {dropped:8.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calibrate", nargs="+", metavar="SOURCE", help="manifest val sources with true labels")
    ap.add_argument("--manifest", default="data/manifest_stage2.csv")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--images", help="folder of unlabeled line crops; sub-folder name = writer_id")
    ap.add_argument("--out", default="data/raw/pseudo_hw/labels.csv")
    ap.add_argument("--max-nll", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, ck = load_stage1(resolve(a.checkpoint), device=dev)
    model.eval()

    if a.calibrate:
        va = load_manifest(resolve(a.manifest), ["val"], a.calibrate)
        root = resolve(a.data_root)
        df = read_all(model, tok, ck, [root / p for p in va.image_path], dev, a.batch_size)
        calibrate(df, list(va.transcription))
        sys.exit()

    assert a.images, "--images or --calibrate"
    root, out = Path(a.images).resolve(), Path(a.out).resolve()
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    df = read_all(model, tok, ck, paths, dev, a.batch_size)
    keep = df[df.agree & (df.nll_per_tok <= a.max_nll) & (df.text.str.strip() != "")]
    rel = [Path(p).relative_to(root) for p in keep.path]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_path": [p.as_posix() for p in rel], "text": keep.text.values,
                  "writer_id": [p.parent.as_posix() if p.parent != Path(".") else p.stem for p in rel],
                  "nll_per_tok": keep.nll_per_tok.round(4).values}).to_csv(out, index=False)
    print(f"kept {len(keep)}/{len(df)} lines ({len(keep) / max(len(df), 1):.1%}) -> {out}")
