"""Line-image dataset over the unified manifest, plus width-bucketed batching."""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms as T

from ..tokenizer.kcc import KCCTokenizer


def load_manifest(path, splits=None, sources=None, max_per_source=None, seed=0) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, dtype={"transcription": str})
    if splits:
        df = df[df.split.isin(splits)]
    if sources:
        df = df[df.source_dataset.isin(sources)]
    if max_per_source:  # deterministic subsample (e.g. for periodic validation / smoke runs)
        df = (df.groupby("source_dataset", group_keys=False)
                .sample(frac=1.0, random_state=seed).groupby("source_dataset").head(max_per_source))
    return df.reset_index(drop=True)


def target_width(w, h, img_height, min_width, max_width):
    return np.clip(np.round(np.asarray(w) * img_height / np.asarray(h)), min_width, max_width).astype(np.int64)


def to_tensor(im: Image.Image, width: int, height: int) -> torch.Tensor:
    """Resize to (width, height) and scale to [-1, 1]; returns (C,H,W)."""
    im = im.resize((int(width), height), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).float().div_(127.5).sub_(1.0)
    return x[None] if x.ndim == 2 else x.permute(2, 0, 1)


def ctc_min_frames(ids: np.ndarray) -> int:
    """Frames CTC needs: one per label plus a blank between each repeated pair."""
    return len(ids) + int((ids[1:] == ids[:-1]).sum()) if len(ids) else 0


# augmentations are module-level functions so DataLoader workers can pickle them (Windows spawn)
def _affine(im):
    return T.RandomAffine(degrees=1.5, shear=(-6, 6, -1, 1), scale=(0.95, 1.02), fill=im.getpixel((0, 0)))(im)


def _blur(im):
    return im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))


def _downup(im):
    return im.resize((max(1, im.width // 2), max(1, im.height // 2))).resize(im.size)


def _edge_fill(a):
    """Paper colour of an image array: median of its left/right 2-pixel edges (scalar or per channel)."""
    return np.median(np.concatenate([a[:, :2], a[:, -2:]], axis=1).reshape(-1, *a.shape[2:]), axis=0)


def _pad_margins(im):
    """Blank paper left/right of the text, as loose real crops have: teaches the decoder to emit EOS
    over empty space instead of hallucinating (Stage-1 crops are tight). Fill = median edge colour."""
    fill = _edge_fill(np.asarray(im)).astype(int)
    fill = int(fill) if fill.ndim == 0 else tuple(fill.tolist())
    left, right = int(random.uniform(0, 0.5) * im.height), int(random.uniform(0.5, 4.0) * im.height)
    out = Image.new(im.mode, (im.width + left + right, im.height), fill)
    out.paste(im, (left, 0))
    return out


class LineDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root, tokenizer: KCCTokenizer, img_height=32,
                 min_width=16, max_width=1600, in_channels=1, augment=False, max_label_len=254,
                 feat_stride=4, filter_infeasible=True, modalities=None, neighbor_p=0.0, stretch=None):
        self.root = Path(data_root)
        self.tok, self.h, self.c, self.augment, self.neighbor_p = tokenizer, img_height, in_channels, augment, neighbor_p
        self.max_width = max_width
        tw = target_width(df.width.values, df.height.values, img_height, min_width, max_width)
        ids = [np.asarray(tokenizer.encode(t, normalized=True), dtype=np.int32) for t in df.transcription]
        # stretch={source: max factor}: lines too dense for CTC (Thai: one token per mark) are widened just
        # enough, +20% slack for augmentation, instead of dropped; lines needing more than the cap still drop
        cap = np.array([(stretch or {}).get(s, 1.0) for s in df.source_dataset], dtype=float)
        want = np.minimum(np.ceil(np.array([ctc_min_frames(a) for a in ids]) * 1.2) * feat_stride, max_width)
        s = (cap > 1) & (want > tw) & (want <= tw * cap)
        tw, self.n_stretched = np.where(s, want, tw).astype(np.int64), int(s.sum())
        frames = -(-tw // feat_stride)
        # training drops labels CTC cannot align / the decoder cannot hold; evaluation keeps them
        keep = np.array([0 < len(a) and (not filter_infeasible or (len(a) <= max_label_len
                                                                   and ctc_min_frames(a) <= f))
                         for a, f in zip(ids, frames)], dtype=bool)
        self.n_dropped = int((~keep).sum())
        ids = [a for a, k in zip(ids, keep) if k]
        # flat arrays instead of per-row Python objects: avoids copy-on-read blowup in workers
        self.ids_flat = np.concatenate(ids) if ids else np.zeros(0, np.int32)
        self.offsets = np.concatenate([[0], np.cumsum([len(a) for a in ids])]).astype(np.int64)
        kept = df[keep]
        self.paths = np.array(kept.image_path.tolist(), dtype="S")
        self.widths = tw[keep]
        self.sources = np.array(kept.source_dataset.tolist(), dtype="S")
        # Stage 2: modality class per sample for the supervised router (-1 = unknown / not routed)
        mod = {m: i for i, m in enumerate(modalities or [])}
        col = kept.modality.tolist() if "modality" in kept else [""] * len(kept)
        self.modality_ids = np.array([mod.get(m, -1) for m in col], dtype=np.int64)
        self.hw = np.array([m == "handwritten" for m in col], dtype=bool)
        self.hw_idx = np.flatnonzero(self.hw)  # neighbour-line donors
        self.df_index = np.flatnonzero(keep)    # row in the source dataframe for each sample
        self.aug = self._build_aug() if augment else None

    def __len__(self):
        return len(self.widths)

    def label(self, i) -> np.ndarray:
        return self.ids_flat[self.offsets[i]:self.offsets[i + 1]]

    @staticmethod
    def _build_aug():
        # Paper does not specify augmentation; light, text-preserving perturbations (assumption).
        return [(0.3, _affine), (0.4, T.ColorJitter(brightness=0.35, contrast=0.35)), (0.15, _blur), (0.15, _downup),
                (0.3, _pad_margins)]

    def _open(self, path):
        return Image.open(path).convert("L" if self.c == 1 else "RGB")

    def _neighbors(self, im):
        """Strokes of the line above/below cut into the crop, as loose real handwriting crops have, so
        the model learns to ignore them (label unchanged). Donor = random handwritten training line at
        this line's height: its bottom 10-35% goes above, its top 10-35% below (or both)."""
        h, a = im.height, np.asarray(im, dtype=np.float32)
        fill, r = _edge_fill(a), random.random()
        bands = []
        for above in [True, False] if r < 0.2 else [r < 0.6]:
            n = self._open(self.root / self.paths[random.choice(self.hw_idx)].decode())
            n = np.asarray(n.resize((max(1, round(n.width * h / n.height)), h), Image.BILINEAR), dtype=np.float32)
            n = n * (fill / np.maximum(_edge_fill(n), 1))         # match its paper to ours, ink stays dark
            s = max(1, int(random.uniform(0.1, 0.35) * h))
            band = np.broadcast_to(fill, (s, *a.shape[1:])).copy()
            x = random.randint(-n.shape[1] // 2, a.shape[1] // 2)  # random offset; donor width varies too
            src = n[h - s:] if above else n[:s]
            lo, hi = max(0, x), min(a.shape[1], x + n.shape[1])
            band[:, lo:hi] = np.minimum(band[:, lo:hi], src[:, lo - x:hi - x])
            bands.append((above, band))
        out = [b for up, b in bands if up] + [a] + [b for up, b in bands if not up]
        return Image.fromarray(np.clip(np.concatenate(out), 0, 255).astype(np.uint8), im.mode)

    def load_image(self, path: Path, width: int, hw=False) -> torch.Tensor:
        im = self._open(path)
        if self.aug:
            w0, h0 = im.size
            if hw and self.neighbor_p and random.random() < self.neighbor_p:
                im = self._neighbors(im)
            for p, f in self.aug:
                if random.random() < p:
                    im = f(im)
            # margins widen the line; neighbour bands make it taller (keep aspect, text shrinks)
            width = min(self.max_width, max(1, round(width * im.width / w0 * h0 / im.height)))
        return to_tensor(im, width, self.h)

    def __getitem__(self, i):
        x = self.load_image(self.root / self.paths[i].decode(), self.widths[i], self.hw[i])
        return x, int(x.shape[-1]), self.label(i), i, int(self.modality_ids[i])


class Collate:
    """Pads images to a common width and builds CTC / decoder targets (a class so workers can pickle it)."""

    def __init__(self, tokenizer: KCCTokenizer, pad_multiple=4):
        self.tok, self.pad_multiple = tokenizer, pad_multiple

    def __call__(self, batch):
        tok = self.tok
        xs, ws, labels, idx, *mod = zip(*batch)
        C, H = xs[0].shape[:2]
        W = math.ceil(max(ws) / self.pad_multiple) * self.pad_multiple
        images = torch.zeros(len(xs), C, H, W)
        for b, x in enumerate(xs):
            images[b, :, :, :x.shape[-1]] = x
        L = max(len(l) for l in labels) + 1
        tgt_in = torch.full((len(xs), L), tok.pad_id, dtype=torch.long)
        tgt_out = torch.full((len(xs), L), tok.pad_id, dtype=torch.long)
        for b, l in enumerate(labels):
            t = torch.from_numpy(l.astype(np.int64))
            tgt_in[b, 0], tgt_in[b, 1:len(t) + 1] = tok.bos_id, t
            tgt_out[b, :len(t)], tgt_out[b, len(t)] = t, tok.eos_id
        return {
            "images": images, "widths": torch.tensor(ws), "tgt_in": tgt_in, "tgt_out": tgt_out,
            "ctc_targets": torch.from_numpy(np.concatenate(labels).astype(np.int64)),
            "ctc_lengths": torch.tensor([len(l) for l in labels]), "index": torch.tensor(idx),
            "modality": torch.tensor(mod[0] if mod else [-1] * len(xs)),
        }


make_collate = Collate


class BucketBatchSampler(torch.utils.data.Sampler):
    """Shuffle, then sort by width inside chunks of `batch_size * bucket_mult` to limit padding.

    Deterministic per (seed, epoch); `skip_batches` resumes mid-epoch without loading data.
    """

    def __init__(self, widths, batch_size, seed=0, shuffle=True, bucket_mult=50, drop_last=False):
        self.widths, self.bs, self.seed = np.asarray(widths), batch_size, seed
        self.shuffle, self.mult, self.drop_last = shuffle, bucket_mult, drop_last
        self.epoch, self.skip_batches = 0, 0

    def set_epoch(self, epoch, skip_batches=0):
        self.epoch, self.skip_batches = epoch, skip_batches

    def batches(self):
        n = len(self.widths)
        if not self.shuffle:
            order = np.argsort(self.widths, kind="stable")
            return [order[i:i + self.bs].tolist() for i in range(0, n, self.bs)]
        rng = np.random.default_rng([self.seed, self.epoch])
        order, chunk, out = self.epoch_order(rng), self.bs * self.mult, []
        for s in range(0, len(order), chunk):
            c = order[s:s + chunk]
            c = c[np.argsort(self.widths[c], kind="stable")]
            out += [c[i:i + self.bs].tolist() for i in range(0, len(c), self.bs)]
        if self.drop_last:
            out = [b for b in out if len(b) == self.bs]
        rng.shuffle(out)
        return out

    def epoch_order(self, rng):
        return rng.permutation(len(self.widths))

    def __iter__(self):
        yield from self.batches()[self.skip_batches:]

    def __len__(self):
        return max(0, len(self.batches()) - self.skip_batches)


class ReplayBucketSampler(BucketBatchSampler):
    """Stage 2: every epoch = all adapt-source samples (each repeated `repeat[source]` times) plus a
    fresh random draw of replay samples, `replay_ratio` x as many (paper: equal number of document
    images, to avoid forgetting printed text). Deterministic per (seed, epoch), so resume works."""

    def __init__(self, widths, sources, adapt, replay, replay_ratio=1.0, repeat=None, **kw):
        super().__init__(widths, **kw)
        src = np.asarray(sources).astype(str)
        repeat = repeat or {}
        self.adapt = np.concatenate([np.tile(np.flatnonzero(src == a), int(repeat.get(a, 1))) for a in adapt]
                                    + [np.zeros(0, np.int64)])
        self.replay = np.flatnonzero(np.isin(src, list(replay)))
        self.n_replay = min(len(self.replay), int(round(replay_ratio * len(self.adapt))))
        assert len(self.adapt), f"no training samples for adapt sources {adapt}"

    def epoch_order(self, rng):
        pick = rng.choice(self.replay, self.n_replay, replace=False) if self.n_replay else self.replay[:0]
        return rng.permutation(np.concatenate([self.adapt, pick]))
