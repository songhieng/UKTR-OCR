"""PyTorch datasets.

Two modes:

``SyntheticKhmerHandwritingDataset``
    Generates samples *on the fly*.  ``__len__`` is virtual (``epoch_size``) and
    every index maps to a deterministic ``(text, writer, seed)`` triple, so the
    dataset is reproducible, shard-safe under ``num_workers > 0`` and effectively
    infinite - the model never sees the same rendering twice.

``KhmerHandwritingDiskDataset``
    Reads a dataset previously written by ``scripts/generate_dataset.py``.

Both return the same dict, so training code does not care which one is used::

    {"image": FloatTensor[C, H, W], "text": str, "writer_id": int,
     "trajectory": FloatTensor[N, 5], "metadata": dict}

The trajectory tensor is the compact ``[dx, dy, pen_state, pressure, dt]``
format documented in ``schema.py`` - directly consumable by an LSTM+MDN, a
trajectory Transformer or a 1-D diffusion model (see :func:`collate_trajectories`
for padding + masks).

``torch`` is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..utils.rng import stable_hash
from .corpus import DEFAULT_CORPUS
from .generator import SampleGenerator
from .schema import Trajectory

try:  # torch is optional
    import torch
    from torch.utils.data import Dataset

    _TORCH = True
except Exception:  # pragma: no cover - torch not installed
    torch = None  # type: ignore

    class Dataset:  # type: ignore
        pass

    _TORCH = False


def _require_torch() -> None:
    if not _TORCH:  # pragma: no cover
        raise ImportError("PyTorch is required for khmer_synth.dataset.pytorch_dataset")


def image_to_tensor(image: np.ndarray, grayscale: bool = True) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if grayscale and arr.ndim == 3:
        arr = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(arr))


class SyntheticKhmerHandwritingDataset(Dataset):
    """Infinite, deterministic, on-the-fly synthetic handwriting."""

    def __init__(
        self,
        corpus: Sequence[str] | None = None,
        config: PipelineConfig | None = None,
        epoch_size: int = 100_000,
        seed: int = 0,
        grayscale: bool = True,
        degraded: bool = True,
        return_trajectory: bool = True,
        return_image: bool = True,
        transform: Callable | None = None,
        max_retries: int = 4,
    ):
        _require_torch()
        self.cfg = config or PipelineConfig()
        self.corpus = list(corpus) if corpus else list(DEFAULT_CORPUS)
        if not self.corpus:
            raise ValueError("empty corpus")
        self.epoch_size = int(epoch_size)
        self.seed = int(seed)
        self.grayscale = grayscale
        self.degraded = degraded
        self.return_trajectory = return_trajectory
        self.return_image = return_image
        self.transform = transform
        self.max_retries = max_retries
        self._gen: SampleGenerator | None = None

    # a generator holds unpicklable HarfBuzz handles -> build it lazily, per worker
    @property
    def generator(self) -> SampleGenerator:
        if self._gen is None:
            self._gen = SampleGenerator(self.cfg)
        return self._gen

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        for attempt in range(self.max_retries):
            h = stable_hash(self.seed, index, attempt)
            text = self.corpus[h % len(self.corpus)]
            writer_id = int((h >> 17) % max(1, self.cfg.dataset.num_writers))
            sample_seed = int((h >> 33) % (2**31))
            try:
                result = self.generator.generate(
                    text,
                    writer_id=writer_id,
                    seed=sample_seed,
                    sample_id=f"dyn_{index:09d}",
                    render=self.return_image,
                    degrade_image=self.degraded and self.return_image,
                )
            except Exception:
                continue
            if result.problems:
                continue

            item: dict[str, Any] = {
                "text": result.trajectory.text,
                "writer_id": writer_id,
                "metadata": result.metadata.to_dict(),
            }
            if self.return_image:
                img = result.degraded if (self.degraded and result.degraded is not None) else result.clean
                tensor = image_to_tensor(img, self.grayscale)
                item["image"] = self.transform(tensor) if self.transform else tensor
            if self.return_trajectory:
                item["trajectory"] = torch.from_numpy(result.trajectory.to_compact())
            return item
        raise RuntimeError(f"could not generate a valid sample for index {index}")


class KhmerHandwritingDiskDataset(Dataset):
    """Reads a materialised dataset directory (``metadata.jsonl``)."""

    def __init__(
        self,
        root: str | Path,
        split_file: str = "metadata.jsonl",
        use_degraded: bool = True,
        grayscale: bool = True,
        return_trajectory: bool = True,
        transform: Callable | None = None,
    ):
        _require_torch()
        self.root = Path(root)
        self.records: list[dict[str, Any]] = []
        with (self.root / split_file).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.use_degraded = use_degraded
        self.grayscale = grayscale
        self.return_trajectory = return_trajectory
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        rec = self.records[index]
        rel = rec.get("degraded_image") if self.use_degraded else rec.get("clean_image")
        rel = rel or rec.get("clean_image") or rec.get("degraded_image")
        item: dict[str, Any] = {
            "text": rec.get("normalized_text", rec.get("text", "")),
            "writer_id": int(str(rec.get("writer_id", "writer_0")).split("_")[-1]),
            "metadata": rec,
        }
        if rel:
            img = np.array(Image.open(self.root / rel).convert("RGB"))
            tensor = image_to_tensor(img, self.grayscale)
            item["image"] = self.transform(tensor) if self.transform else tensor
        if self.return_trajectory and rec.get("trajectory_file"):
            data = json.loads((self.root / rec["trajectory_file"]).read_text("utf-8"))
            traj = Trajectory.from_dict(data)
            item["trajectory"] = torch.from_numpy(traj.to_compact())
        return item


# --------------------------------------------------------------------------- #
# collation
# --------------------------------------------------------------------------- #


def collate_trajectories(batch: Sequence[dict[str, Any]], pad_value: float = 0.0) -> dict[str, Any]:
    """Pad variable-length trajectories and stack images.

    Returns ``trajectory`` ``[B, T, 5]``, ``trajectory_mask`` ``[B, T]`` (True =
    real point) and ``lengths`` ``[B]``, which is what an autoregressive
    (LSTM/MDN or Transformer) or a diffusion trajectory model expects.
    """
    _require_torch()
    out: dict[str, Any] = {}
    if "image" in batch[0]:
        out["image"] = torch.stack([b["image"] for b in batch])
    out["text"] = [b["text"] for b in batch]
    out["writer_id"] = torch.tensor([b["writer_id"] for b in batch], dtype=torch.long)
    out["metadata"] = [b.get("metadata", {}) for b in batch]

    if "trajectory" in batch[0]:
        lengths = torch.tensor([len(b["trajectory"]) for b in batch], dtype=torch.long)
        t_max = int(lengths.max().item()) if len(lengths) else 0
        dim = batch[0]["trajectory"].shape[1] if len(batch[0]["trajectory"]) else 5
        padded = torch.full((len(batch), t_max, dim), pad_value, dtype=torch.float32)
        mask = torch.zeros((len(batch), t_max), dtype=torch.bool)
        for i, b in enumerate(batch):
            n = len(b["trajectory"])
            if n:
                padded[i, :n] = b["trajectory"]
                mask[i, :n] = True
        out["trajectory"] = padded
        out["trajectory_mask"] = mask
        out["lengths"] = lengths
    return out
