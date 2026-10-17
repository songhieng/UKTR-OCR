"""Stage-1 pretraining.

  python scripts/train.py --config configs/smoke.yaml
  python scripts/train.py --config configs/full.yaml
  python scripts/train.py --config configs/full.yaml --resume checkpoints/uktr_base_stage1_last.pt
  python scripts/train.py --config configs/full.yaml --set train.batch_size=16 train.grad_accum=2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.config import load_config, resolve  # noqa: E402
from src.training.trainer import train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override config values, e.g. train.batch_size=16")
    a = ap.parse_args()
    cfg = load_config(resolve(a.config), a.set)
    print(json.dumps(train(cfg, a.resume), indent=2))
