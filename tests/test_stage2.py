import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model import SMALL, TOK, batch  # noqa: E402

from src.datasets.ocr_dataset import ReplayBucketSampler  # noqa: E402
from src.models.checkpoint import load_stage1, save_checkpoint  # noqa: E402
from src.models.uktr import UKTRBase  # noqa: E402
from src.training.trainer import compute_losses, routing_stats  # noqa: E402

MAFS = {"n_sources": 3, "adapter_dim": 8, "router_hidden": 16, "modalities": ["printed", "scene", "handwritten"]}


def test_fresh_mafs_is_identity_on_stage1_weights(tmp_path):
    torch.manual_seed(0)
    s1 = UKTRBase(SMALL).eval()
    p = tmp_path / "s1.pt"
    save_checkpoint(p, stage=1, model_config=s1.cfg, model_state=s1.state_dict(), tokenizer=TOK.to_dict())
    s2, _, _ = load_stage1(p, model=UKTRBase({**SMALL, "mafs": MAFS}))
    s2.eval()
    b = batch()
    with torch.no_grad():
        o1 = s1(b["images"], b["widths"], b["tgt_in"])
        o2 = s2(b["images"], b["widths"], b["tgt_in"])
    assert torch.allclose(o1["ctc_logits"], o2["ctc_logits"], atol=1e-5)
    assert torch.allclose(o1["dec_logits"], o2["dec_logits"], atol=1e-5)
    assert o2["route_logits"].shape == (3, 3)
    probs = s2.recognize(b["images"], b["widths"], TOK.bos_id, TOK.eos_id, max_len=4)["route_probs"]
    assert np.allclose(np.sum(probs, 1), 1, atol=1e-5)


def test_supervised_routing_loss_trains_router():
    torch.manual_seed(0)
    m = UKTRBase({**SMALL, "mafs": MAFS})
    b = batch()
    b["modality"] = torch.tensor([2, 0, -1])                       # -1 = unlabelled, ignored
    _, _, _, r0 = compute_losses(m, b, TOK, None, route_w=0.0)
    assert float(r0) == 0.0                                          # unsupervised: no routing loss
    opt = torch.optim.Adam(m.feature_selector.parameters(), 1e-2)
    first = None
    for _ in range(30):
        loss, _, _, lr_ = compute_losses(m, b, TOK, None, route_w=1.0)
        first = first if first is not None else lr_.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert lr_.item() < first * 0.5
    # unlabelled modality only -> no routing loss, no NaN
    b["modality"] = torch.tensor([-1, -1, -1])
    total, *_, lr_ = compute_losses(m, b, TOK, None, route_w=1.0)
    assert float(lr_) == 0.0 and torch.isfinite(total)


def test_stage2_checkpoint_roundtrip(tmp_path):
    m = UKTRBase({**SMALL, "mafs": MAFS}).eval()
    p = tmp_path / "s2.pt"
    save_checkpoint(p, stage=2, model_config=m.cfg, model_state=m.state_dict(), tokenizer=TOK.to_dict())
    m2, _, ck = load_stage1(p)                                       # strict: MAFS rebuilt from config
    assert m2.feature_selector is not None and ck["stage"] == 2
    for k, v in m.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k])


def test_replay_sampler_mix_and_resume():
    src = np.array(["hw"] * 100 + ["real"] * 20 + ["printed"] * 1000)
    w = np.random.default_rng(0).integers(16, 800, len(src))
    s = ReplayBucketSampler(w, src, ["hw", "real"], ["printed"], replay_ratio=1.0, repeat={"real": 3},
                            batch_size=16, seed=1)
    s.set_epoch(0)
    e0 = [i for b_ in s for i in b_]
    n_adapt = 100 + 20 * 3
    assert len(e0) == 2 * n_adapt
    assert sum(src[i] == "real" for i in e0) == 60 and sum(src[i] == "printed" for i in e0) == n_adapt
    s.set_epoch(1)
    e1 = {i for b_ in s for i in b_ if src[i] == "printed"}
    assert e1 != {i for i in e0 if src[i] == "printed"}             # fresh replay draw every epoch
    s.set_epoch(0)
    full = list(s)
    s.set_epoch(0, skip_batches=3)
    assert list(s) == full[3:]


def test_routing_stats():
    route = {0: ([0.9, 0.05, 0.05], 0), 1: ([0.1, 0.2, 0.7], 2), 2: ([0.6, 0.3, 0.1], 2), 3: ([0.3, 0.3, 0.4], -1)}
    r = routing_stats(route, ["printed", "scene", "handwritten"])
    assert r["route_acc"] == 2 / 3
    assert abs(r["route_p/handwritten/src2"] - 0.4) < 1e-9
