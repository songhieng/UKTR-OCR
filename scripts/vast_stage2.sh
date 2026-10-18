#!/usr/bin/env bash
# One-shot Stage-2 pipeline for a fresh Linux GPU box (e.g. vast.ai RTX 4090 24 GB, PyTorch image).
#   1. upload the Stage-1 checkpoint to checkpoints/uktr_base_stage1_best.pt
#   2. bash scripts/vast_stage2.sh                              # data + preflight, then train in tmux
#      CONFIG=configs/stage2_no_mafs.yaml bash scripts/vast_stage2.sh   # another variant (data is reused)
#      NUM_TEXTS=100000 WORKERS=32 bash scripts/vast_stage2.sh          # synthetic texts / CPU workers
# Needs ~100 GB free disk (Stage-1 replay data ~80 GB, synthetic handwriting ~2 GB). Synthetic
# generation is CPU-bound (~0.5 s per image per core): pick a box with many vCPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=checkpoints/uktr_base_stage1_best.pt
CONFIG=${CONFIG:-configs/stage2.yaml}
# vast.ai rents a share of a big host: nproc shows all host cores (e.g. 256) but the cgroup quota is
# ~25. Use the quota, and cap torch/OpenMP threads, or they busy-wait the quota away (GPU at ~20%).
cpus() { read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ] && echo $(( q / p )) || nproc; }
W=${WORKERS:-$(cpus)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
[ -f "$CKPT" ] || { echo "missing $CKPT - upload the Stage-1 checkpoint first"; exit 1; }
# the Python that has the GPU torch: images differ (python, python3, /venv/main, /opt/conda); PYTHON=... overrides
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python python3 /venv/main/bin/python /opt/conda/bin/python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch" >/dev/null 2>&1; then PY=$(command -v "$c"); break; fi
  done
fi
[ -n "$PY" ] || { echo "no Python with torch found - run: PYTHON=/path/to/python bash scripts/vast_stage2.sh"; exit 1; }
echo "using $PY ($("$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'))"
"$PY" -m pip install -q -r requirements.txt

# Stage-1 data: printed + scene replay, and the text source for synthetic handwriting
if [ ! -f data/manifest.csv ]; then
  "$PY" scripts/download_data.py
  "$PY" scripts/prepare_data.py --workers "$W"
  "$PY" scripts/build_manifest.py
fi

# Stage 2 must reuse the Stage-1 vocabulary exactly: write it from the checkpoint
"$PY" - "$CKPT" <<'EOF'
import sys, torch
from src.tokenizer.kcc import KCCTokenizer
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
KCCTokenizer.from_dict(ck["tokenizer"]).save("data/tokenizer/kcc_vocab.json")
print("vocab from checkpoint -> data/tokenizer/kcc_vocab.json")
EOF

# synthetic handwriting (--all: the preset's exclude list drops the decorative faces)
"$PY" scripts/fetch_fonts.py --out data/fonts --all || echo "some fonts failed to download - continuing with the rest"
if [ ! -f data/raw/khmer_synth_hw/labels.csv ]; then
  "$PY" scripts/generate_handwriting.py --num-texts "${NUM_TEXTS:-100000}" --workers "$W"
fi

# real handwriting + Stage-2 manifest
"$PY" scripts/download_data.py --config configs/stage2_data.yaml --datasets real_hw
"$PY" scripts/prepare_data.py   --config configs/stage2_data.yaml --workers "$W"
"$PY" scripts/build_manifest.py --config configs/stage2_data.yaml
"$PY" scripts/build_vocab.py    --config configs/stage2_data.yaml --existing data/tokenizer/kcc_vocab.json

"$PY" -m pytest -q tests
"$PY" scripts/preflight.py --config "$CONFIG" || {
  echo "preflight failed: see results/*/preflight.json (start training by hand if only the overfit check failed)"; exit 1; }

# training runs inside tmux so it survives SSH disconnects; TensorBoard on port 6006
NAME=$(basename "$CONFIG" .yaml)
tmux new-session -d -s "$NAME" "OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS $PY scripts/train.py --config $CONFIG --set data.num_workers=$(( W > 6 ? W - 4 : 2 )) 2>&1 | tee -a results/train_$NAME.log"
tmux has-session -t tb 2>/dev/null || tmux new-session -d -s tb "$PY -m tensorboard.main --logdir results --host 0.0.0.0 --port 6006"
echo "training started: tmux attach -t $NAME   |   TensorBoard: http://<host>:6006"
