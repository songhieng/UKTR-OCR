#!/usr/bin/env bash
# Thai-transfer experiment (configs/thai_*.yaml) on a fresh Linux GPU box; fits a 64 GB disk.
#   1. upload the Stage-1 checkpoint to checkpoints/uktr_base_stage1_best.pt
#   2. bash scripts/vast_thai.sh                                    # data + preflight, then A, C, B1, B2 in tmux `thai`
#      SET="train.batch_size=16 train.grad_accum=2" bash scripts/vast_thai.sh   # 16 GB GPU (same effective batch)
#      RUNS="thai_a_khmer thai_c_mixed" bash scripts/vast_thai.sh             # a subset (data is reused)
# Downloads printed Khmer (replay), SoyVitou + KiteAether (Khmer handwriting) and iapp Thai handwriting (~9 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=checkpoints/uktr_base_stage1_best.pt
THAI_CKPT=checkpoints/uktr_stage1_thai.pt
D=configs/stage2_data.yaml
SOURCES="khmer_printed real_hw kiteaether thai_hw"
cpus() { read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ] && echo $(( q / p > 0 ? q / p : 1 )) || nproc; }
W=${WORKERS:-$(cpus)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
[ -f "$CKPT" ] || { echo "missing $CKPT - upload the Stage-1 checkpoint first"; exit 1; }
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python python3 /venv/main/bin/python /opt/conda/bin/python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch" >/dev/null 2>&1; then PY=$(command -v "$c"); break; fi
  done
fi
[ -n "$PY" ] || { echo "no Python with torch found - run: PYTHON=/path/to/python bash scripts/vast_thai.sh"; exit 1; }
echo "using $PY ($("$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())')), $W CPUs"
"$PY" -m pip install -q -r requirements.txt

# one vocabulary for every run: Stage 1 + the Thai block
[ -f "$THAI_CKPT" ] || "$PY" scripts/extend_vocab.py "$CKPT" "$THAI_CKPT"
"$PY" - "$THAI_CKPT" <<'EOF'
import sys, torch
from src.tokenizer.kcc import KCCTokenizer
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
KCCTokenizer.from_dict(ck["tokenizer"]).save("data/tokenizer/kcc_vocab.json")
EOF

"$PY" scripts/download_data.py  --config $D --datasets $SOURCES
"$PY" scripts/prepare_data.py   --config $D --sources $SOURCES --workers "$W"
"$PY" scripts/build_manifest.py --config $D
"$PY" scripts/build_vocab.py    --config $D --existing data/tokenizer/kcc_vocab.json

"$PY" -m pytest -q tests/test_model.py tests/test_stage2.py tests/test_tokenizer.py
"$PY" scripts/preflight.py --config configs/thai_c_mixed.yaml --set ${SET:-} || {
  echo "preflight failed: see results/thai_c/preflight.json"; exit 1; }

RUNS=${RUNS:-thai_a_khmer thai_c_mixed thai_b1_thai thai_b2_khmer}
NW=$(( W > 6 ? W - 4 : 2 ))
tmux new-session -d -s thai "for r in $RUNS; do OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS $PY scripts/train.py --config configs/\$r.yaml --set data.num_workers=$NW ${SET:-} 2>&1 | tee -a results/train_\$r.log; done"
echo "training started: tmux attach -t thai   |   logs: results/train_thai_*.log"
