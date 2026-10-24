"""Training loop for both stages: L = L_CTC + L_Transformer (+ route_loss_weight * L_route in Stage 2).

Stage 1 (cfg stage: 1): fresh model, tokenizer from paths.tokenizer.
Stage 2 (cfg stage: 2): Stage-1 weights + tokenizer from `init_checkpoint`, optional MAFS module,
replay sampling of printed data, separate LR group for MAFS, optional supervised routing loss.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..datasets.ocr_dataset import BucketBatchSampler, LineDataset, ReplayBucketSampler, load_manifest, make_collate
from ..evaluation.metrics import summarize
from ..models.checkpoint import export_inference, load_stage1, save_checkpoint
from ..models.uktr import UKTRBase, count_parameters
from ..tokenizer.kcc import KCCTokenizer
from .config import resolve


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def amp_dtype(cfg):
    if not cfg["train"].get("amp", True) or not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def mask_decoder_input(tgt_in, tok: KCCTokenizer, ratio: float):
    """Random 30% decoder-input masking (paper, following Buoy et al.); BOS and padding are kept."""
    if ratio <= 0:
        return tgt_in
    m = (torch.rand(tgt_in.shape, device=tgt_in.device) < ratio) & (tgt_in != tok.pad_id) & (tgt_in != tok.bos_id)
    return tgt_in.masked_fill(m, tok.mask_id)


def compute_losses(model, batch, tok, dtype=None, mask_ratio=0.0, ctc_w=1.0, ce_w=1.0, route_w=0.0):
    """-> (total, ctc, ce, route). The routing loss (supervised MAFS) is only computed when route_w > 0."""
    dev = next(model.parameters()).device
    b = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
    tgt_in = mask_decoder_input(b["tgt_in"], tok, mask_ratio) if model.training else b["tgt_in"]
    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        out = model(b["images"], b["widths"], tgt_in, tgt_pad_mask=b["tgt_in"] == tok.pad_id)
    logp = F.log_softmax(out["ctc_logits"].float(), -1).transpose(0, 1)          # T,B,V
    l_ctc = F.ctc_loss(logp, b["ctc_targets"], out["feat_w"], b["ctc_lengths"], blank=tok.blank_id,
                       reduction="mean", zero_infinity=True)
    V = out["dec_logits"].shape[-1]
    l_ce = F.cross_entropy(out["dec_logits"].float().reshape(-1, V), b["tgt_out"].reshape(-1),
                           ignore_index=tok.pad_id)
    l_route = torch.zeros((), device=dev)
    if route_w > 0 and "route_logits" in out and (b["modality"] >= 0).any():
        l_route = F.cross_entropy(out["route_logits"].float(), b["modality"], ignore_index=-1)
    return ctc_w * l_ctc + ce_w * l_ce + route_w * l_route, l_ctc.detach(), l_ce.detach(), l_route.detach()


@torch.no_grad()
def evaluate(model, loader, tok, refs_by_index, dtype=None, max_len=None, sources=None, modalities=None,
             route_w=0.0):
    """Validation loss + greedy CTC/Transformer metrics, overall and per source; MAFS routing stats."""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    tot = {"loss": 0.0, "ctc": 0.0, "ce": 0.0, "route": 0.0, "n": 0}
    preds = {"ctc": {}, "tr": {}, "joint": {}}
    route = {}                                                     # index -> (probs, modality id)
    for batch in loader:
        loss, lc, le, lr_ = compute_losses(model, batch, tok, dtype, route_w=route_w)
        n = len(batch["widths"])
        for k, v in (("loss", loss), ("ctc", lc), ("ce", le), ("route", lr_)):
            tot[k] += float(v) * n
        tot["n"] += n
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            out = model.recognize(batch["images"].to(dev), batch["widths"].to(dev), tok.bos_id, tok.eos_id,
                                  max_len=max_len, decoders=("ctc", "tr", "joint"))
        for i, c, t, j in zip(batch["index"].tolist(), out["ctc"], out["tr"], out["joint"]):
            preds["ctc"][i], preds["tr"][i], preds["joint"][i] = tok.decode_ctc(c), tok.decode(t), tok.decode(j)
        for i, pr, m in zip(batch["index"].tolist(), out.get("route_probs", []), batch["modality"].tolist()):
            route[i] = (pr, m)
    model.train(was_training)
    idx = sorted(preds["ctc"])
    res = {"val_loss": tot["loss"] / max(tot["n"], 1), "val_ctc_loss": tot["ctc"] / max(tot["n"], 1),
           "val_ce_loss": tot["ce"] / max(tot["n"], 1)}
    if route:
        res["val_route_loss"] = tot["route"] / max(tot["n"], 1)
        res.update(routing_stats(route, modalities or []))
    groups = {"all": idx}
    if sources is not None:
        for i in idx:
            groups.setdefault(sources[i], []).append(i)
    for g, ii in groups.items():
        refs = [refs_by_index[i] for i in ii]
        for dec in ("ctc", "tr", "joint"):
            for k, v in summarize([preds[dec][i] for i in ii], refs).items():
                res[f"{g}/{dec}_{k}"] = v
    res["cer_ctc"], res["cer_tr"], res["cer_joint"] = res["all/ctc_cer"], res["all/tr_cer"], res["all/joint_cer"]
    res["cer_mean"] = (res["cer_ctc"] + res["cer_tr"]) / 2
    return res, preds


def routing_stats(route: dict, modalities: list) -> dict:
    """Router accuracy vs modality labels (meaningful when n_sources == len(modalities)) and the mean
    routing distribution per labelled modality (useful for the unsupervised router too)."""
    res = {}
    probs = np.array([p for p, _ in route.values()])
    mods = np.array([m for _, m in route.values()])
    lab = mods >= 0
    if lab.any() and probs.shape[1] == len(modalities):
        res["route_acc"] = float((probs[lab].argmax(1) == mods[lab]).mean())
    for k, name in enumerate(modalities):
        if (mods == k).any():
            for j, v in enumerate(probs[mods == k].mean(0)):
                res[f"route_p/{name}/src{j}"] = float(v)
    return res


def build_model(cfg):
    """-> (model, tokenizer). Stage 2 starts from the Stage-1 checkpoint (+ MAFS if enabled)."""
    if cfg.get("stage", 1) == 1:
        tok = KCCTokenizer.load(resolve(cfg["paths"]["tokenizer"]))
        return UKTRBase({**cfg["model"], "vocab_size": len(tok)}), tok
    init = resolve(cfg["init_checkpoint"])
    ck = torch.load(init, map_location="cpu", weights_only=True)
    d0, m = ck["train_config"]["data"], cfg.get("mafs", {})
    assert cfg["data"]["img_height"] == d0["img_height"], "Stage 2 must use the Stage-1 input height"
    mafs = None
    if m.get("enabled", True):
        mafs = {k: m[k] for k in ("n_sources", "adapter_dim", "router_hidden", "dropout", "modalities") if k in m}
        if cfg["train"].get("route_loss_weight", 0) > 0:
            assert m["n_sources"] == len(m["modalities"]), "supervised routing needs n_sources == len(modalities)"
    model, tok, _ = load_stage1(init, model=UKTRBase({**ck["model_config"], "mafs": mafs}))
    return model, tok


def modalities_of(cfg):
    m = cfg.get("mafs") or {}
    return m.get("modalities") if cfg.get("stage", 1) == 2 else None


def build_datasets(cfg, tok):
    d, p = cfg["data"], cfg["paths"]
    common = dict(data_root=resolve(p["data_root"]), tokenizer=tok, img_height=d["img_height"],
                  min_width=d["min_width"], max_width=d["max_width"], in_channels=cfg["model"]["in_channels"],
                  max_label_len=cfg["model"]["max_dec_len"] - 2, modalities=modalities_of(cfg))
    stage2 = cfg.get("stage", 1) == 2
    train_sources = d["adapt_sources"] + d.get("replay_sources", []) if stage2 else d.get("train_sources")
    tr_df = load_manifest(resolve(p["manifest"]), ["train"], train_sources,
                          d.get("max_train_per_source"), cfg["seed"])
    if d.get("exclude_handwritten") and "handwritten" in tr_df:
        tr_df = tr_df[tr_df.handwritten.astype(str).str.lower() != "true"].reset_index(drop=True)
    va_df = load_manifest(resolve(p["manifest"]), ["val"], d.get("val_sources"),
                          d.get("max_val_per_source"), cfg["seed"])
    train = LineDataset(tr_df, augment=d.get("augment", False), neighbor_p=d.get("neighbor_p", 0.0), **common)
    val = LineDataset(va_df, augment=False, filter_infeasible=False, **common)
    return train, val, tr_df, va_df


def _rng_state():
    st = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    torch.set_rng_state(st["torch"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _append_csv(path: Path, row: dict):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _provenance(cfg) -> dict:
    out = {}
    for name in ("raw/DOWNLOAD_PROVENANCE.json", "manifest_meta.json"):
        f = resolve(cfg["paths"]["data_root"]) / name
        if f.exists():
            out[name] = f.read_text(encoding="utf-8")   # stored as text: keeps the checkpoint weights_only-safe
    return out


def train(cfg: dict, resume: str | None = None) -> dict:
    t_cfg, p = cfg["train"], cfg["paths"]
    seed_everything(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(cfg)
    model, tok = build_model(cfg)
    model = model.to(dev)
    stage, route_w = cfg.get("stage", 1), t_cfg.get("route_loss_weight", 0.0)
    out_dir, res_dir = resolve(p["out_dir"]), resolve(p["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, tr_df, va_df = build_datasets(cfg, tok)
    print(f"train={len(train_ds)} (dropped {train_ds.n_dropped} infeasible/too long)  "
          f"val={len(val_ds)} (dropped {val_ds.n_dropped})  vocab={len(tok)}")
    collate = make_collate(tok)
    nw = cfg["data"].get("num_workers", 4)
    skw = dict(batch_size=t_cfg["batch_size"], seed=cfg["seed"], bucket_mult=cfg["data"].get("bucket_mult", 50))
    if stage == 2:
        d = cfg["data"]
        sampler = ReplayBucketSampler(train_ds.widths, train_ds.sources, d["adapt_sources"],
                                      d.get("replay_sources", []), d.get("replay_ratio", 1.0), d.get("repeat"), **skw)
        print(f"per epoch: {len(sampler.adapt)} adapt + {sampler.n_replay} replay samples")
    else:
        sampler = BucketBatchSampler(train_ds.widths, **skw)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=nw,
                        pin_memory=dev.type == "cuda", persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_sampler=BucketBatchSampler(val_ds.widths, t_cfg.get("eval_batch_size", 64),
                                                                     shuffle=False),
                            collate_fn=collate, num_workers=min(nw, 4))
    val_refs = va_df.transcription.values[val_ds.df_index]
    val_src = va_df.source_dataset.values[val_ds.df_index]

    model_cfg = model.cfg
    params = count_parameters(model)
    print("params:", {k: f"{v / 1e6:.2f}M" for k, v in params.items()})
    # newly initialised MAFS gets its own (higher) LR range: mafs_lr_scale x the pretrained-layer range
    new = [p_ for n, p_ in model.named_parameters() if n.startswith("feature_selector.")]
    old = [p_ for n, p_ in model.named_parameters() if not n.startswith("feature_selector.")]
    groups, scale = [{"params": old}] + ([{"params": new}] if new else []), t_cfg.get("mafs_lr_scale", 1.0)
    lr_lo = [t_cfg["lr_min"]] + [t_cfg["lr_min"] * scale] * (len(groups) - 1)
    lr_hi = [t_cfg["lr_max"]] + [t_cfg["lr_max"] * scale] * (len(groups) - 1)
    opt = torch.optim.Adam(groups, lr=t_cfg["lr_min"], betas=tuple(t_cfg.get("betas", (0.9, 0.999))),
                           weight_decay=t_cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr_lo, max_lr=lr_hi,
                                              step_size_up=t_cfg["cyclic_step_size_up"],
                                              mode=t_cfg.get("cyclic_mode", "triangular"), cycle_momentum=False)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    accum = t_cfg.get("grad_accum", 1)
    steps_per_epoch = math.ceil(len(sampler) / accum)
    total_steps = t_cfg.get("max_steps") or steps_per_epoch * t_cfg["epochs"]

    epoch, step, batch_in_epoch, best = 0, 0, 0, math.inf
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optimizer_state"])
        sched.load_state_dict(ck["scheduler_state"]); scaler.load_state_dict(ck["scaler_state"])
        epoch, step, batch_in_epoch = ck["epoch"], ck["global_step"], ck["batch_in_epoch"]
        best = ck.get("best_metric", math.inf)
        _set_rng_state(ck["rng_state"])
        print(f"resumed from {resume}: epoch {epoch} step {step} batch {batch_in_epoch}")

    def payload(metrics):
        return dict(stage=stage, init_checkpoint=str(cfg.get("init_checkpoint", "")), model_config=model_cfg, model_state=model.state_dict(), tokenizer=tok.to_dict(),
                    optimizer_state=opt.state_dict(), scheduler_state=sched.state_dict(),
                    scaler_state=scaler.state_dict(), epoch=epoch, global_step=step,
                    batch_in_epoch=batch_in_epoch, best_metric=best, metrics=metrics, train_config=cfg,
                    param_counts=params, data_provenance=_provenance(cfg), rng_state=_rng_state(),
                    torch_version=str(torch.__version__), saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    n_train=len(train_ds), n_val=len(val_ds))

    last_path = out_dir / t_cfg.get("last_name", "uktr_base_stage1_last.pt")
    best_path = out_dir / t_cfg.get("best_name", "uktr_base_stage1_best.pt")
    best_key = t_cfg.get("best_metric", "cer_mean")
    metrics = {}
    # TensorBoard: `tensorboard --logdir results` ; purge_step drops events past a resumed step
    tb = SummaryWriter(res_dir / "tensorboard", purge_step=step or None)

    def run_eval():
        nonlocal best, metrics
        t0 = time.time()
        metrics, _ = evaluate(model, val_loader, tok, dict(enumerate(val_refs)), dtype,
                              t_cfg.get("max_eval_decode_len"), sources=dict(enumerate(val_src)),
                              modalities=modalities_of(cfg), route_w=route_w)
        if best_key not in metrics:
            raise KeyError(f"best_metric '{best_key}' not in validation metrics: {sorted(metrics)}")
        metrics.update(step=step, epoch=epoch, eval_seconds=round(time.time() - t0, 1))
        _append_csv(res_dir / "val_log.csv", metrics)
        for k, v in metrics.items():
            if k not in ("step", "epoch") and isinstance(v, (int, float)):
                tb.add_scalar(f"val/{k}", v, step)
        tb.flush()
        print(f"[eval step {step}] loss {metrics['val_loss']:.4f} CER ctc {metrics['cer_ctc']:.4f} "
              f"tr {metrics['cer_tr']:.4f} joint {metrics['cer_joint']:.4f} {best_key} {metrics[best_key]:.4f}"
              + (f" route_acc {metrics['route_acc']:.3f}" if "route_acc" in metrics else ""))
        if metrics[best_key] < best:
            best = metrics[best_key]
            save_checkpoint(best_path, **payload(metrics))
            print(f"  new best {best_key}={best:.4f} -> {best_path}")
        save_checkpoint(last_path, **payload(metrics))

    model.train()
    t_log, done = time.time(), step >= total_steps
    while not done and epoch < t_cfg["epochs"]:
        sampler.set_epoch(epoch, skip_batches=batch_in_epoch)
        agg = {"loss": 0.0, "ctc": 0.0, "ce": 0.0, "route": 0.0, "n": 0}
        for batch in loader:
            loss, lc, le, lr_ = compute_losses(model, batch, tok, dtype, t_cfg.get("dec_mask_ratio", 0.3),
                                               t_cfg.get("ctc_weight", 1.0), t_cfg.get("ce_weight", 1.0), route_w)
            scaler.scale(loss / accum).backward()
            batch_in_epoch += 1
            agg["loss"] += loss.item(); agg["ctc"] += float(lc); agg["ce"] += float(le); agg["n"] += 1
            agg["route"] += float(lr_)
            if batch_in_epoch % accum:
                continue
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get("grad_clip", 50.0))
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1
            if step % t_cfg.get("log_every", 50) == 0:
                dt = time.time() - t_log
                row = {"step": step, "epoch": epoch, "lr": sched.get_last_lr()[0],
                       "loss": agg["loss"] / agg["n"], "ctc_loss": agg["ctc"] / agg["n"], "ce_loss": agg["ce"] / agg["n"],
                       "route_loss": agg["route"] / agg["n"],
                       "grad_norm": float(gnorm), "sec_per_step": dt / t_cfg.get("log_every", 50),
                       "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0}
                _append_csv(res_dir / "train_log.csv", row)
                for k, v in row.items():
                    if k != "step":
                        tb.add_scalar(f"train/{k}", v, step)
                print(f"ep {epoch} step {step}/{total_steps} loss {row['loss']:.4f} (ctc {row['ctc_loss']:.4f} "
                      f"ce {row['ce_loss']:.4f}) lr {row['lr']:.2e} {row['sec_per_step']:.3f}s/step "
                      f"mem {row['gpu_mem_gb']:.2f}GB", flush=True)
                agg, t_log = {"loss": 0.0, "ctc": 0.0, "ce": 0.0, "route": 0.0, "n": 0}, time.time()
            if step % t_cfg.get("eval_every", 5000) == 0:
                run_eval()
            elif step % t_cfg.get("save_every", 1000) == 0:
                save_checkpoint(last_path, **payload(metrics))
            if step >= total_steps:
                done = True
                break
        if not done:
            epoch, batch_in_epoch = epoch + 1, 0
    if metrics.get("step") != step:
        run_eval()
    tb.close()
    infer_path = best_path.with_name(best_path.stem + "_infer.pt")
    if best_path.exists():
        export_inference(best_path, infer_path)
    summary = {"completed_steps": step, "planned_steps": total_steps, "epochs_done": epoch,
               "finished": step >= total_steps or epoch >= t_cfg["epochs"], "best_metric": best_key,
               "best_value": best, "last_metrics": metrics, "param_counts": params,
               "best_checkpoint": str(best_path), "inference_export": str(infer_path)}
    (res_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
