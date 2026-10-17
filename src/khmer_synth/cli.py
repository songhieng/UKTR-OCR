"""Shared CLI plumbing: config loading, ``--set`` overrides, common flags."""

from __future__ import annotations

import argparse

from .config import PipelineConfig, merge_overrides


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="JSON/YAML pipeline config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="path.key=value",
        help="override any config value, e.g. --set skeleton.method=lee "
        "--set layout.canvas_height=128 (value is parsed as JSON)",
    )
    parser.add_argument("--fonts", default=None, help="font file or directory (repeatable)", action="append")
    parser.add_argument("--width", type=int, default=None, help="canvas width in px")
    parser.add_argument("--height", type=int, default=None, help="canvas height in px")
    parser.add_argument("--seed", type=int, default=None, help="global RNG seed")
    parser.add_argument("--no-degrade", action="store_true", help="disable the degradation stage")
    return parser


def build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig.load(getattr(args, "config", None))
    if getattr(args, "fonts", None):
        cfg.fonts.paths = list(args.fonts)
    if getattr(args, "width", None):
        cfg.layout.canvas_width = int(args.width)
    if getattr(args, "height", None):
        cfg.layout.canvas_height = int(args.height)
    if getattr(args, "seed", None) is not None:
        cfg.dataset.seed = int(args.seed)
    if getattr(args, "no_degrade", False):
        cfg.degradation.enabled = False
    if getattr(args, "overrides", None):
        cfg = merge_overrides(cfg, args.overrides)
    return cfg
