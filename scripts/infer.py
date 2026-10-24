"""Recognize text-line images with a Stage-1 checkpoint.

  python scripts/infer.py --checkpoint checkpoints/uktr_base_stage1_best_infer.pt img1.png img2.jpg
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.ocr_dataset import target_width, to_tensor  # noqa: E402
from src.models.checkpoint import load_stage1  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--decoder", choices=["ctc", "tr", "joint", "all"], default="all",
                    help="joint = the better of the CTC and TR reads under both heads (most accurate)")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, ck = load_stage1(a.checkpoint, device=dev)
    model.eval()
    d, c = ck["train_config"]["data"], ck["model_config"]["in_channels"]
    decs = ("ctc", "tr", "joint") if a.decoder == "all" else (a.decoder,)
    for p in a.images:
        im = Image.open(p).convert("L" if c == 1 else "RGB")
        tw = int(target_width(im.width, im.height, d["img_height"], d["min_width"], d["max_width"]))
        x = to_tensor(im, tw, d["img_height"])[None].to(dev)
        out = model.recognize(x, torch.tensor([tw], device=dev), tok.bos_id, tok.eos_id, decoders=decs)
        res = {k: (tok.decode_ctc(out[k][0]) if k == "ctc" else tok.decode(out[k][0])) for k in decs}
        if "route_probs" in out:  # Stage-2 MAFS modality routing
            res["route"] = " ".join(f"{p:.2f}" for p in out["route_probs"][0])
        print(p, "\t".join(f"{k}: {v}" for k, v in res.items()))
