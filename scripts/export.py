"""Write an inference-only copy of a checkpoint (drops optimizer/scheduler/scaler/RNG state).

  python scripts/export.py checkpoints/uktr_base_stage1_best.pt checkpoints/uktr_base_stage1_best_infer.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.checkpoint import export_inference  # noqa: E402

if __name__ == "__main__":
    export_inference(sys.argv[1], sys.argv[2])
    print(f"-> {sys.argv[2]}")
