#!/usr/bin/env bash
# One-shot Stage-1 pipeline for a fresh Linux GPU box (e.g. vast.ai, PyTorch image).
#   bash scripts/vast_setup.sh            # full data + preflight, then start full training in tmux
#   SMOKE=1 bash scripts/vast_setup.sh    # small download + smoke run only
# Needs ~80 GB free disk for the full data (raw parquet/tgz ~19 GB + extracted images ~25 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q -r requirements.txt
W=${WORKERS:-$(nproc)}
if [ "${SMOKE:-0}" = "1" ]; then
  python scripts/download_data.py --smoke
  python scripts/prepare_data.py --workers "$W"
  python scripts/build_manifest.py
  python scripts/build_vocab.py
  python -m pytest -q tests
  python scripts/preflight.py --config configs/smoke.yaml
  python scripts/train.py --config configs/smoke.yaml
  exit 0
fi
python scripts/download_data.py
python scripts/prepare_data.py --workers "$W"
python scripts/build_manifest.py
python scripts/build_vocab.py
python -m pytest -q tests
python scripts/preflight.py --config configs/full.yaml
# training runs inside tmux so it survives SSH disconnects; TensorBoard on port 6006
tmux new-session -d -s train "python scripts/train.py --config configs/full.yaml 2>&1 | tee -a results/train_full.log"
tmux new-session -d -s tb "tensorboard --logdir results --host 0.0.0.0 --port 6006"
echo "training started: tmux attach -t train   |   TensorBoard: http://<host>:6006"
