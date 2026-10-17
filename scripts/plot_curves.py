"""Plot training / validation curves from results/<experiment>/{train,val}_log.csv.

  python scripts/plot_curves.py --results results/stage1_full
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.config import resolve  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    d = resolve(ap.parse_args().results)
    tr = pd.read_csv(d / "train_log.csv")
    va = pd.read_csv(d / "val_log.csv") if (d / "val_log.csv").exists() else None
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k in ("loss", "ctc_loss", "ce_loss"):
        ax[0].plot(tr.step, tr[k], label=f"train {k}", lw=1)
    if va is not None:
        for k in ("val_loss", "val_ctc_loss", "val_ce_loss"):
            ax[0].plot(va.step, va[k], "o-", label=k, ms=3)
        for k in ("cer_ctc", "cer_tr"):
            ax[1].plot(va.step, va[k], "o-", label=k, ms=3)
    ax[0].set(title="loss", xlabel="optimizer step", yscale="log"); ax[0].legend(fontsize=8)
    ax[1].set(title="validation CER (subset)", xlabel="optimizer step"); ax[1].legend()
    ax[2].plot(tr.step, tr.lr); ax[2].set(title="learning rate", xlabel="optimizer step")
    fig.tight_layout()
    fig.savefig(d / "curves.png", dpi=120)
    print(f"-> {d / 'curves.png'}")
