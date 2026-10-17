"""Console-script entry points (thin wrappers around ``scripts/*.py``).

Installing the package exposes:

    khmer-synth-generate   -> scripts/generate_dataset.py
    khmer-synth-debug      -> scripts/debug_pipeline.py
    khmer-synth-validate   -> scripts/validate_dataset.py
    khmer-synth-fonts      -> scripts/fetch_fonts.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(name: str) -> int:
    path = _SCRIPTS / name
    if not path.exists():  # installed without the repo checkout
        raise SystemExit(
            f"{name} is only available from a source checkout (looked in {_SCRIPTS})"
        )
    sys.argv[0] = str(path)
    runpy.run_path(str(path), run_name="__main__")
    return 0


def generate_dataset_main() -> int:
    return _run("generate_dataset.py")


def debug_pipeline_main() -> int:
    return _run("debug_pipeline.py")


def validate_dataset_main() -> int:
    return _run("validate_dataset.py")


def fetch_fonts_main() -> int:
    return _run("fetch_fonts.py")
