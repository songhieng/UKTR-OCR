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


class LineDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root, tokenizer: KCCTokenizer, img_height=32,
                 min_width=16, max_width=1600, in_channels=1, augment=False, max_label_len=254,
                 feat_stride=4, filter_infeasible=True):
        self.root = Path(data_root)
        self.tok, self.h, self.c, self.augment = tokenizer, img_height, in_channels, augment
        tw = target_width(df.width.values, df.height.values, img_height, min_width, max_width)
        ids = [np.asarray(tokenizer.encode(t, normalized=True), dtype=np.int32) for t in df.transcription]
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
        self.df_index = np.flatnonzero(keep)    # row in the source dataframe for each sample
        self.aug = self._build_aug() if augment else None

    def __len__(self):
        return len(self.widths)

    def label(self, i) -> np.ndarray:
        return self.ids_flat[self.offsets[i]:self.offsets[i + 1]]

    @staticmethod
    def _build_aug():
        # Paper does not specify augmentation; light, text-preserving perturbations (assumption).
        return [(0.3, _affine), (0.4, T.ColorJitter(brightness=0.35, contrast=0.35)), (0.15, _blur), (0.15, _downup)]

    def load_image(self, path: Path, width: int) -> torch.Tensor:
        im = Image.open(path).convert("L" if self.c == 1 else "RGB")
        if self.aug:
            for p, f in self.aug:
                if random.random() < p:
                    im = f(im)
        return to_tensor(im, width, self.h)

    def __getitem__(self, i):
        x = self.load_image(self.root / self.paths[i].decode(), self.widths[i])
        return x, int(self.widths[i]), self.label(i), i


class Collate:
    """Pads images to a common width and builds CTC / decoder targets (a class so workers can pickle it)."""

    def __init__(self, tokenizer: KCCTokenizer, pad_multiple=4):
        self.tok, self.pad_multiple = tokenizer, pad_multiple

    def __call__(self, batch):
        tok = self.tok
        xs, ws, labels, idx = zip(*batch)
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
        order, chunk, out = rng.permutation(n), self.bs * self.mult, []
        for s in range(0, n, chunk):
            c = order[s:s + chunk]
            c = c[np.argsort(self.widths[c], kind="stable")]
            out += [c[i:i + self.bs].tolist() for i in range(0, len(c), self.bs)]
        if self.drop_last:
            out = [b for b in out if len(b) == self.bs]
        rng.shuffle(out)
        return out

    def __iter__(self):
        yield from self.batches()[self.skip_batches:]

    def __len__(self):
        return max(0, len(self.batches()) - self.skip_batches)
