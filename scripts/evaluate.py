"""Evaluate a Stage-1 checkpoint on held-out data (per source and split).

  python scripts/evaluate.py --checkpoint checkpoints/uktr_base_stage1_best.pt --splits val test

Reports CER, KCC error rate, exact-line accuracy for CTC and Transformer decoders, validation
loss, latency, peak GPU memory; writes results/eval/eval_report.{json,md} and examples.html.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.ocr_dataset import BucketBatchSampler, LineDataset, load_manifest, make_collate  # noqa: E402
from src.evaluation.metrics import sample_cer  # noqa: E402
from src.models.checkpoint import load_stage1  # noqa: E402
from src.training.config import load_config, resolve  # noqa: E402
from src.training.trainer import amp_dtype, evaluate  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def latency(model, ds, collate, tok, dev, dtype, n=50, batch_size=32):
    model.eval()
    items = [ds[i] for i in range(min(n, len(ds)))]
    out = {}
    for dec in ("ctc", "tr"):
        for _ in range(3):  # warm-up
            b = collate(items[:1])
            model.recognize(b["images"].to(dev), b["widths"].to(dev), tok.bos_id, tok.eos_id, decoders=(dec,))
        sync(); t0 = time.perf_counter()
        for it in items:
            b = collate([it])
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                model.recognize(b["images"].to(dev), b["widths"].to(dev), tok.bos_id, tok.eos_id, decoders=(dec,))
        sync()
        out[f"{dec}_ms_per_line_bs1"] = round(1000 * (time.perf_counter() - t0) / len(items), 2)
        sync(); t0 = time.perf_counter()
        for s in range(0, len(items), batch_size):
            b = collate(items[s:s + batch_size])
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                model.recognize(b["images"].to(dev), b["widths"].to(dev), tok.bos_id, tok.eos_id, decoders=(dec,))
        sync()
        out[f"{dec}_ms_per_line_bs{batch_size}"] = round(1000 * (time.perf_counter() - t0) / len(items), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="train config for paths/data (default: the one in the checkpoint)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--max-per-source", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--examples", type=int, default=10, help="examples per source/split")
    ap.add_argument("--out", default="results/eval")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, ck = load_stage1(resolve(a.checkpoint), device=dev)
    cfg = load_config(resolve(a.config)) if a.config else ck["train_config"]
    dtype = amp_dtype(cfg)
    out_dir = resolve(a.out); (out_dir / "examples").mkdir(parents=True, exist_ok=True)
    d, root = cfg["data"], resolve(cfg["paths"]["data_root"])
    collate = make_collate(tok)
    report = {"checkpoint": str(a.checkpoint), "global_step": ck.get("global_step"), "epoch": ck.get("epoch"),
              "param_counts": ck.get("param_counts"), "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
              "amp_dtype": str(dtype), "groups": {}}
    mods = (model.cfg.get("mafs") or {}).get("modalities")   # Stage-2 MAFS: report routing stats
    rng = random.Random(0)
    rows_html = []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    df_all = load_manifest(resolve(cfg["paths"]["manifest"]), a.splits, a.sources, a.max_per_source, cfg["seed"])
    for (src, split), df in df_all.groupby(["source_dataset", "split"]):
        df = df.reset_index(drop=True)
        ds = LineDataset(df, root, tok, d["img_height"], d["min_width"], d["max_width"],
                         model.cfg["in_channels"], augment=False, filter_infeasible=False, modalities=mods)
        loader = DataLoader(ds, batch_sampler=BucketBatchSampler(ds.widths, a.batch_size, shuffle=False),
                            collate_fn=collate, num_workers=min(4, d.get("num_workers", 4)))
        refs = df.transcription.values[ds.df_index]
        t0 = time.time()
        m, preds = evaluate(model, loader, tok, dict(enumerate(refs)), dtype, modalities=mods)
        m = {k.replace("all/", ""): v for k, v in m.items()}
        m["seconds"] = round(time.time() - t0, 1)
        report["groups"][f"{src}/{split}"] = m
        print(f"{src}/{split}: n={m['ctc_n']} CER ctc {m['ctc_cer']:.4f} tr {m['tr_cer']:.4f} "
              f"KCC-ER ctc {m['ctc_kcc_er']:.4f} tr {m['tr_kcc_er']:.4f} line-acc tr {m['tr_line_acc']:.4f}"
              + (f" route-acc {m['route_acc']:.3f}" if "route_acc" in m else ""))
        for i in rng.sample(range(len(ds)), min(a.examples, len(ds))):
            p = root / ds.paths[i].decode()
            dst = out_dir / "examples" / f"{src}_{split}_{p.name}"
            shutil.copy(p, dst)
            ref = refs[i]
            cells = [f'<img src="examples/{dst.name}" style="max-height:48px">', html.escape(ref)]
            for dec in ("ctc", "tr"):
                cells.append(f"{html.escape(preds[dec][i])}<br><small>CER {sample_cer(preds[dec][i], ref):.3f}</small>")
            rows_html.append(f"<tr><td>{src}/{split}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        if "latency" not in report and len(ds):
            report["latency"] = latency(model, ds, collate, tok, dev, dtype)
    if dev.type == "cuda":
        report["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "examples.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>UKTR Stage-1 examples</title>"
        "<style>body{font-family:'Khmer OS','Noto Sans Khmer',sans-serif}td{border:1px solid #ccc;padding:4px}</style>"
        "<table><tr><th>set</th><th>image</th><th>ground truth</th><th>CTC</th><th>Transformer</th></tr>"
        + "\n".join(rows_html) + "</table>", encoding="utf-8")
    L = ["# Stage-1 evaluation", "", f"checkpoint `{a.checkpoint}` (step {report['global_step']}), "
         f"device {report['device']}", "",
         "| set | n | CER CTC | CER Tr | KCC-ER CTC | KCC-ER Tr | line-acc CTC | line-acc Tr | val loss | route acc |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in report["groups"].items():
        L.append(f"| {k} | {m['ctc_n']} | {m['ctc_cer']:.4f} | {m['tr_cer']:.4f} | {m['ctc_kcc_er']:.4f} | "
                 f"{m['tr_kcc_er']:.4f} | {m['ctc_line_acc']:.4f} | {m['tr_line_acc']:.4f} | {m['val_loss']:.4f} | "
                 f"{round(m['route_acc'], 3) if 'route_acc' in m else '-'} |")
    L += ["", f"Latency: {report.get('latency')}", f"Peak GPU memory: {report.get('peak_gpu_mem_gb')} GB", "",
          "Examples: see examples.html"]
    (out_dir / "eval_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"report -> {out_dir}")


if __name__ == "__main__":
    main()
