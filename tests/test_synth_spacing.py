"""Synthetic handwriting: neighbouring letter clusters must never overprint (spacing collision pass)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not any((ROOT / "data/fonts").glob("*.[ot]tf")),
                                reason="fonts not fetched (scripts/fetch_fonts.py)")

TEXTS = ["សម្លៀកបំពាក់សមរម្យ ទៅតាមពេលវេលា", "ឆ្នាំដើម្បីបញ្ចប់", "ប្រទេសកម្ពុជា", "ស្ត្រីៗដែលស្រឡាញ់"]


def test_no_overprint_between_clusters():
    from src.khmer_synth.config import PipelineConfig, merge_overrides
    from src.khmer_synth.dataset.generator import SampleGenerator

    off = ["cluster_level", "stroke_level", "point_level", "global_transform", "baseline"]
    cfg = merge_overrides(PipelineConfig.load(ROOT / "configs/handwriting_synth.json"),
                          [f"deformation.{k}=false" for k in off] + ["writer.connect_probability=[0,0]"])
    cfg.fonts.paths = [str(ROOT / p) for p in cfg.fonts.paths]
    cfg.dataset.use_disk_cache = False
    gen = SampleGenerator(cfg)
    for i, text in enumerate(TEXTS * 3):
        t = gen.generate(text, writer_id=i * 37, seed=i, render=False).trajectory
        masks = {}
        for s in t.strokes:
            m = masks.setdefault(s.cluster, np.zeros((t.height, t.width), np.uint8))
            cv2.polylines(m, [np.rint(s.points).astype(np.int32)], False, 1, 1)
        ks = sorted(masks)
        for a, b in zip(ks, ks[1:]):
            assert int((masks[a] & masks[b]).sum()) == 0, (text, a, b)
