"""Pre-training checklist on the target machine (run before the full training).

  python scripts/preflight.py --config configs/full.yaml

1 GPU + memory  2 tensor shapes  3 forward/backward  4 CTC input lengths  5 overfit a small set
6 checkpoint save/load equality  7 Khmer Unicode output  8 worst-case batch memory
Writes results/preflight.json; exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.ocr_dataset import ctc_min_frames, make_collate  # noqa: E402
from src.evaluation.metrics import cer  # noqa: E402
from src.models.checkpoint import load_stage1, save_checkpoint  # noqa: E402
from src.models.uktr import UKTRBase, count_parameters  # noqa: E402
from src.tokenizer.kcc import KCCTokenizer, has_khmer  # noqa: E402
from src.training.config import load_config, resolve  # noqa: E402
from src.training.trainer import amp_dtype, build_datasets, compute_losses, seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--overfit-samples", type=int, default=16)
    ap.add_argument("--overfit-steps", type=int, default=600)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(resolve(a.config), a.set)
    seed_everything(cfg["seed"])
    rep, ok = {}, True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(cfg)

    # 1 GPU
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        rep["gpu"] = {"name": p.name, "total_mem_gb": round(p.total_memory / 1e9, 2), "count": torch.cuda.device_count(),
                      "bf16": torch.cuda.is_bf16_supported(), "amp_dtype": str(dtype)}
    else:
        rep["gpu"] = "none (CPU only: full training is impractical)"
    print("1 GPU:", rep["gpu"])

    tok = KCCTokenizer.load(resolve(cfg["paths"]["tokenizer"]))
    cfg["data"]["max_val_per_source"] = 64
    train_ds, val_ds, tr_df, _ = build_datasets(cfg, tok)
    collate = make_collate(tok)
    model = UKTRBase({**cfg["model"], "vocab_size": len(tok)}).to(dev)
    rep["param_counts"] = count_parameters(model)

    # 2 shapes
    batch = collate([train_ds[i] for i in range(min(8, len(train_ds)))])
    model.train()
    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        out = model(batch["images"].to(dev), batch["widths"].to(dev), batch["tgt_in"].to(dev))
    B, _, H, W = batch["images"].shape
    s = math.prod(cfg["model"]["block_strides"])
    shapes = {"images": list(batch["images"].shape), "ctc_logits": list(out["ctc_logits"].shape),
              "dec_logits": list(out["dec_logits"].shape), "feat_w": out["feat_w"].tolist()}
    good = (out["ctc_logits"].shape == (B, math.ceil(W / s), len(tok))
            and out["dec_logits"].shape == (B, batch["tgt_in"].shape[1], len(tok)))
    rep["2_shapes"] = {"ok": bool(good), **shapes}; ok &= good
    print("2 shapes:", rep["2_shapes"])

    # 3 forward/backward
    loss, lc, le = compute_losses(model, batch, tok, dtype, 0.3)
    loss.backward()
    gfinite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    good = bool(torch.isfinite(loss) and gfinite)
    rep["3_forward_backward"] = {"ok": good, "loss": loss.item(), "ctc": float(lc), "ce": float(le)}; ok &= good
    model.zero_grad(set_to_none=True)
    print("3 fwd/bwd:", rep["3_forward_backward"])

    # 4 CTC lengths over the whole (filtered) training set
    frames = -(-train_ds.widths // s)
    need = np.array([ctc_min_frames(train_ds.label(i)) for i in range(len(train_ds))])
    good = bool((frames >= need).all())
    rep["4_ctc_lengths"] = {"ok": good, "train_samples": len(train_ds), "dropped_infeasible_or_too_long": train_ds.n_dropped,
                            "min_slack_frames": int((frames - need).min()) if len(need) else None}
    ok &= good
    print("4 CTC lengths:", rep["4_ctc_lengths"])

    # 5 overfit + 7 Khmer output
    idx = [i for i in range(len(train_ds)) if has_khmer(tr_df.transcription.values[train_ds.df_index[i]])]
    idx = idx[:a.overfit_samples] or list(range(a.overfit_samples))
    small = Subset(train_ds, idx)
    train_ds.aug = None
    loader = DataLoader(small, batch_size=len(idx), collate_fn=collate)
    refs = [tr_df.transcription.values[train_ds.df_index[i]] for i in idx]
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    fb = next(iter(loader))
    hist = []
    for step in range(1, a.overfit_steps + 1):
        model.train()
        loss, _, _ = compute_losses(model, fb, tok, dtype, 0.0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 50); opt.step()
        if step % 50 == 0 or step == a.overfit_steps:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                r = model.recognize(fb["images"].to(dev), fb["widths"].to(dev), tok.bos_id, tok.eos_id)
            pc = [tok.decode_ctc(x) for x in r["ctc"]]; pt = [tok.decode(x) for x in r["tr"]]
            hist.append({"step": step, "loss": round(float(loss), 4), "cer_ctc": round(cer(pc, refs), 4),
                         "cer_tr": round(cer(pt, refs), 4)})
            print("  overfit", hist[-1], flush=True)
            if hist[-1]["cer_ctc"] < 0.02 and hist[-1]["cer_tr"] < 0.02:
                break
    good = hist[-1]["cer_ctc"] < 0.05 and hist[-1]["cer_tr"] < 0.05
    rep["5_overfit"] = {"ok": good, "samples": len(idx), "history": hist}; ok &= good
    khmer = any(has_khmer(p) for p in pt)
    rep["7_khmer_output"] = {"ok": khmer, "examples": [{"ref": r_, "ctc": c_, "tr": t_}
                                                       for r_, c_, t_ in list(zip(refs, pc, pt))[:4]]}
    ok &= khmer
    print("7 Khmer output:", rep["7_khmer_output"]["examples"][:2])

    # 6 checkpoint save/load equality (eval mode, fp32)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ck.pt"
        save_checkpoint(path, stage=1, model_config=model.cfg, model_state=model.state_dict(), tokenizer=tok.to_dict())
        m2, tok2, _ = load_stage1(path, device=dev)
        model.eval(); m2.eval()
        with torch.no_grad():
            o1 = model(fb["images"].to(dev), fb["widths"].to(dev), fb["tgt_in"].to(dev))
            o2 = m2(fb["images"].to(dev), fb["widths"].to(dev), fb["tgt_in"].to(dev))
        diff = max(float((o1[k] - o2[k]).abs().max()) for k in ("ctc_logits", "dec_logits"))
        good = diff < 1e-5 and tok2.vocab == tok.vocab
        rep["6_checkpoint_roundtrip"] = {"ok": good, "max_abs_diff": diff}; ok &= good
    print("6 checkpoint:", rep["6_checkpoint_roundtrip"])

    # 8 worst-case batch memory (widest images, longest labels) at the configured batch size
    if dev.type == "cuda":
        del opt
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        bs = cfg["train"]["batch_size"]
        widest = np.argsort(-train_ds.widths)[:bs].tolist()
        wb = collate([train_ds[i] for i in widest])
        model.train()
        try:
            loss, _, _ = compute_losses(model, wb, tok, dtype, 0.3)
            loss.backward(); model.zero_grad(set_to_none=True)
            peak, oom = round(torch.cuda.max_memory_allocated() / 1e9, 2), False
        except torch.cuda.OutOfMemoryError:
            peak, oom = "OOM", True
            ok = False
        rep["8_worst_case_batch"] = {"ok": not oom, "batch_size": bs, "width": int(wb["images"].shape[-1]),
                                     "label_len": int(wb["tgt_in"].shape[1]), "peak_mem_gb": peak,
                                     "note": "if this is close to GPU memory, lower data.max_width or use "
                                             "train.batch_size=16 train.grad_accum=2"}
        print("8 worst-case batch:", rep["8_worst_case_batch"])

    rep["all_ok"] = bool(ok)
    out = resolve(cfg["paths"]["results_dir"]) / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nALL OK: {ok} -> {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
