import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.ocr_dataset import BucketBatchSampler, ctc_min_frames, make_collate  # noqa: E402
from src.models.checkpoint import load_stage1, save_checkpoint  # noqa: E402
from src.models.uktr import UKTRBase, count_parameters  # noqa: E402
from src.tokenizer.kcc import KCCTokenizer  # noqa: E402
from src.training.trainer import compute_losses  # noqa: E402

TOK, _ = KCCTokenizer.build(["ការ abc", "ស្ត្រី 42"])
SMALL = {"vocab_size": len(TOK), "block_channels": [8, 16, 16, 32, 32, 32], "block_layers": [1] * 6,
         "d_model": 32, "nhead": 4, "dim_ff": 64, "enc_layers": 1, "dec_layers": 1}


def batch(n=3):
    items = []
    for i in range(n):
        w = 40 + 24 * i
        items.append((torch.randn(1, 32, w), w, np.asarray(TOK.encode("ការ abc"[: 3 + i]), np.int32), i))
    return make_collate(TOK)(items)


def test_paper_sized_param_counts():
    pc = count_parameters(UKTRBase({"vocab_size": 11899}))
    assert abs(pc["backbone"] - 13.0e6) / 13.0e6 < 0.05          # paper: 13.0M
    assert abs(pc["encoder"] - 9.5e6) / 9.5e6 < 0.02             # paper: 9.5M
    assert abs(pc["transformer_decoder"] - 24.08e6) / 24.08e6 < 0.05  # paper: 24.08M


def test_shapes_and_ctc_lengths():
    m = UKTRBase(SMALL)
    b = batch()
    out = m(b["images"], b["widths"], b["tgt_in"], b["tgt_in"] == TOK.pad_id)
    W = b["images"].shape[-1]
    assert out["ctc_logits"].shape == (3, math.ceil(W / 4), len(TOK))
    assert out["dec_logits"].shape == (*b["tgt_in"].shape, len(TOK))
    assert out["feat_w"].tolist() == [math.ceil(w / 4) for w in b["widths"].tolist()]
    assert ctc_min_frames(np.array([1, 1, 2])) == 4


def test_padding_does_not_change_outputs():
    torch.manual_seed(0)
    m = UKTRBase(SMALL).eval()
    b = batch()
    with torch.no_grad():
        full = m(b["images"], b["widths"], b["tgt_in"])
        one = m(b["images"][:1, :, :, : b["widths"][0]], b["widths"][:1], b["tgt_in"][:1])
        nomask = m(b["images"], torch.full_like(b["widths"], b["images"].shape[-1]), b["tgt_in"])
    T = full["feat_w"][0]
    # CNN receptive field leaks a little across the pad boundary; attention masking must remove the rest
    err_masked = (full["ctc_logits"][0, :T - 2] - one["ctc_logits"][0, :T - 2]).abs().max()
    err_unmasked = (nomask["ctc_logits"][0, :T - 2] - one["ctc_logits"][0, :T - 2]).abs().max()
    assert err_masked < 1e-2 and err_masked * 20 < err_unmasked   # measured over 20 seeds: >100x apart


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    m = UKTRBase(SMALL)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    b = batch()
    losses = []
    for _ in range(30):
        loss, *_ = compute_losses(m, b, TOK, None, 0.3)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.7


class FakeMAFS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.router = nn.Linear(d, 5)

    def forward(self, G, G1d, mask):
        return G, G1d


def test_checkpoint_roundtrip_and_stage2_loading(tmp_path):
    m = UKTRBase(SMALL).eval()
    p = tmp_path / "ck.pt"
    save_checkpoint(p, stage=1, model_config=m.cfg, model_state=m.state_dict(), tokenizer=TOK.to_dict(),
                    rng_state={"torch": torch.get_rng_state()}, metrics={"cer": 0.1})
    m2, tok2, ck = load_stage1(p)                         # weights_only=True load
    for k, v in m.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k])
    assert tok2.vocab == TOK.vocab
    # Stage 2: a model with a MAFS module loads Stage-1 weights; only MAFS keys are missing
    s2 = UKTRBase(SMALL)
    s2.feature_selector = FakeMAFS(SMALL["d_model"])
    load_stage1(p, model=s2)
    assert torch.equal(s2.ctc_head.weight, m.ctc_head.weight)
    # a shape-incompatible model must be rejected
    bad = UKTRBase({**SMALL, "vocab_size": len(TOK) + 1})
    with pytest.raises(ValueError):
        load_stage1(p, model=bad)


def test_sampler_deterministic_and_resumable():
    w = np.random.default_rng(0).integers(16, 800, 1000)
    s = BucketBatchSampler(w, 32, seed=7)
    s.set_epoch(3)
    full = list(s)
    s.set_epoch(3, skip_batches=10)
    assert list(s) == full[10:] and len(s) == len(full) - 10
    assert sorted(i for b in full for i in b) == list(range(1000))
    s.set_epoch(4)
    assert list(s) != full


def test_incremental_greedy_matches_naive():
    torch.manual_seed(0)
    m = UKTRBase(SMALL)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    b = batch()
    for _ in range(40):  # train a little so greedy outputs are non-trivial
        loss, *_ = compute_losses(m, b, TOK, None, 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        G, G1d, cm, _ = m.encode(b["images"], b["widths"])
        seq = torch.full((3, 1), TOK.bos_id)
        for _ in range(15):  # naive: re-run the full decoder on the prefix
            nxt = m.decode_logits(G, cm, seq)[:, -1].argmax(-1)
            seq = torch.cat([seq, nxt[:, None]], 1)
        fast = m.greedy_decode(G, cm, TOK.bos_id, TOK.eos_id, 16)
    for f, n in zip(fast, seq[:, 1:].tolist()):
        k = n.index(TOK.eos_id) + 1 if TOK.eos_id in n else len(n)
        assert f[:k] == n[:k]


def test_pad_margins_keeps_text_and_widens_line(tmp_path):
    import pandas as pd
    from PIL import Image
    from src.datasets.ocr_dataset import LineDataset, _pad_margins
    a = np.full((32, 100), 200, np.uint8)
    a[8:24, 10:90] = 0                                  # "ink" on grey paper
    out = np.asarray(_pad_margins(Image.fromarray(a)))
    assert out.shape[0] == 32 and out.shape[1] >= 100 + 16
    assert (out[:, 0] == 200).all() and (out[:, -1] == 200).all()
    assert (out == 0).sum() == (a == 0).sum()           # text pasted intact
    Image.fromarray(a).save(tmp_path / "x.png")
    df = pd.DataFrame({"image_path": ["x.png"], "transcription": ["ក"], "width": [100], "height": [32],
                       "source_dataset": ["s"]})
    ds = LineDataset(df, tmp_path, TOK, augment=True)
    ds.aug = [(1.0, _pad_margins)]
    x, w, *_ = ds[0]
    assert x.shape[-1] == w > 100


def test_neighbor_lines_keep_text_and_label(tmp_path):
    import pandas as pd
    from PIL import Image
    from src.datasets.ocr_dataset import LineDataset
    a = np.full((32, 100), 200, np.uint8)
    a[8:24, 10:90] = 0                                  # this line's "ink"
    b = np.full((40, 80), 180, np.uint8)
    b[:, 20:60] = 30                                    # donor line: full-height strokes
    Image.fromarray(a).save(tmp_path / "a.png"); Image.fromarray(b).save(tmp_path / "b.png")
    df = pd.DataFrame({"image_path": ["a.png", "b.png"], "transcription": ["ក", "ការ"], "width": [100, 80],
                       "height": [32, 40], "source_dataset": ["s", "s"], "modality": ["handwritten"] * 2})
    ds = LineDataset(df, tmp_path, TOK, augment=True, neighbor_p=1.0)
    ds.aug, ds.hw_idx = [(1.0, lambda im: im)], np.array([1])
    out = np.asarray(ds._neighbors(Image.fromarray(a)))
    assert out.shape[0] > 32 and out.shape[1] == 100
    top = next(r for r in range(out.shape[0]) if (out[r:r + 32] == a).all())
    assert (out[:top] < 100).any() or (out[top + 32:] < 100).any()   # donor strokes in the bands
    assert out.max() <= 200                                         # donor paper matched to ours
    x, w, y, *_ = ds[0]
    assert x.shape[-2] == 32 and w < 100 and (y == ds.label(0)).all()

def test_greedy_decode_respects_per_sample_cap():
    torch.manual_seed(0)
    m = UKTRBase(SMALL).eval()
    b = batch()
    with torch.no_grad():
        G, G1d, cm, _ = m.encode(b["images"], b["widths"])
        out = m.greedy_decode(G, cm, TOK.bos_id, TOK.eos_id, 32, cap=torch.tensor([2, 5, 32]))
    for seq, c in zip(out, [2, 5]):
        assert all(t == TOK.eos_id for t in seq[c:])


def test_joint_rescoring_scores_and_picks_one_of_the_reads():
    torch.manual_seed(0)
    m = UKTRBase(SMALL).eval()
    b = batch()
    with torch.no_grad():
        G, G1d, cm, fw = m.encode(b["images"], b["widths"])
        H, U1d, _ = m.select_features(G, G1d, cm)
        labels = [b["tgt_out"][i, :n].tolist() for i, n in enumerate(b["ctc_lengths"].tolist())]
        got = m.joint_nll(H, cm, U1d, fw, labels, TOK.bos_id, TOK.eos_id)
        # reference: per-sample CTC NLL and summed teacher-forced cross-entropy from the training forward
        out = m(b["images"], b["widths"], b["tgt_in"], b["tgt_in"] == TOK.pad_id)
        ctc = torch.nn.functional.ctc_loss(out["ctc_logits"].float().log_softmax(-1).transpose(0, 1), b["ctc_targets"],
                                           out["feat_w"], b["ctc_lengths"], blank=TOK.blank_id, reduction="none")
        ce = torch.nn.functional.cross_entropy(out["dec_logits"].transpose(1, 2), b["tgt_out"],
                                               ignore_index=TOK.pad_id, reduction="none").sum(1)
        assert torch.allclose(got, 0.5 * ctc + 0.5 * ce, atol=1e-4)
        r = m.recognize(b["images"], b["widths"], TOK.bos_id, TOK.eos_id, max_len=8, decoders=("joint",))
    for c, t, j in zip(r["ctc"], r["tr"], r["joint"]):
        cc = [i for n, i in enumerate(c) if i != 0 and (n == 0 or i != c[n - 1])]
        assert j in (cc, t[:t.index(TOK.eos_id)] if TOK.eos_id in t else t)
