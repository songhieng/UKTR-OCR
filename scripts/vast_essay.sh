#!/usr/bin/env bash
# Essay self-training round on a 64 GB vast.ai box, after the essay pipeline (segment_lines -> crop_lines ->
# make_test_set -> split_phrases) has filled data/raw/unlabeled_phrases.
#   STEP=prep  bash scripts/vast_essay.sh        # data + manifest + pseudo-label calibration table, then stop
#   MAX_NLL=0.05 bash scripts/vast_essay.sh      # pseudo-label, cap, rebuild manifest, preflight, train in tmux
#   CKPT=checkpoints/essay_r1/uktr_stage2_best_infer.pt CONFIG=... bash scripts/vast_essay.sh   # next round
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=${CKPT:-checkpoints/thai_b2/uktr_stage2_best_infer.pt}
CONFIG=${CONFIG:-configs/essay_pseudo_r1.yaml}
D=configs/stage2_data.yaml
PY=${PYTHON:-/venv/main/bin/python}
cpus() { read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ] && echo $(( q / p > 0 ? q / p : 1 )) || nproc; }
W=${WORKERS:-$(cpus)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }

if [ "${STEP:-train}" = prep ]; then
  "$PY" - "$CKPT" <<'EOF'
import sys, torch
from src.tokenizer.kcc import KCCTokenizer
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
KCCTokenizer.from_dict(ck["tokenizer"]).save("data/tokenizer/kcc_vocab.json")
EOF
  "$PY" scripts/download_data.py  --config $D --datasets khmer_printed real_hw kiteaether
  "$PY" scripts/prepare_data.py   --config $D --sources khmer_printed real_hw kiteaether essay_test_extra --workers "$W"
  "$PY" scripts/build_manifest.py --config $D
  "$PY" scripts/pseudo_label.py --checkpoint "$CKPT" --calibrate real_hw kiteaether
  exit 0
fi

# pseudo-label the phrases, keep at most as many as the real training lines (x2 repeat), most confident first
"$PY" scripts/pseudo_label.py --checkpoint "$CKPT" --images data/raw/unlabeled_phrases \
  --max-nll "${MAX_NLL:?set MAX_NLL from the STEP=prep calibration table}" --out data/raw/pseudo_hw/phrases.csv
"$PY" - <<'EOF'
import pandas as pd
m = pd.read_csv("data/manifest_stage2.csv", usecols=["source_dataset", "split"], low_memory=False)
cap = 2 * int(((m.split == "train") & m.source_dataset.isin(["real_hw", "kiteaether"])).sum())
p = pd.read_csv("data/raw/pseudo_hw/phrases.csv")
p.sort_values("nll_per_tok").head(cap).to_csv("data/raw/pseudo_hw/phrases.csv", index=False)
print(f"pseudo phrases: kept {min(len(p), cap)} of {len(p)} confident (cap {cap})")
EOF
"$PY" scripts/prepare_data.py   --config $D --sources pseudo_hw_phrases --workers "$W" --force
"$PY" scripts/build_manifest.py --config $D
"$PY" scripts/preflight.py --config "$CONFIG" --set train.batch_size=16 train.grad_accum=2 ${SET:-} || {
  echo "preflight failed: see results/*/preflight.json"; exit 1; }
NAME=$(basename "$CONFIG" .yaml)
tmux new-session -d -s "$NAME" "$PY scripts/train.py --config $CONFIG --set data.num_workers=$(( W > 6 ? W - 4 : 2 )) train.batch_size=16 train.grad_accum=2 ${SET:-} 2>&1 | tee -a results/train_$NAME.log"
echo "training started: tmux attach -t $NAME"
