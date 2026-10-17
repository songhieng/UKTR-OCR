"""Glyph/cluster  ->  cached centreline graph.

This is the expensive part of the whole system (rasterise, clean, thin, trace,
prune, simplify), so its result is cached on
``(font, glyph signature, raster config, skeleton config)``.  A Khmer corpus
re-uses a few thousand distinct clusters millions of times, which is what makes
large-scale generation cheap.

The unit of work is configurable:

``unit='cluster'``  the whole grapheme cluster (base + coeng stack + vowel +
                    signs) is composited into ONE mask before thinning.  This is
                    the default because Khmer glyphs *overlap*: a subscript
                    often touches its base, and skeletonising the union produces
                    a single connected centreline exactly where the ink is
                    connected, instead of two skeletons crossing each other.
``unit='glyph'``    every glyph is thinned separately (faster, higher cache hit
                    rate, but overlapping ink is modelled less faithfully).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..config import RasterConfig, SkeletonConfig
from ..fonts.glyphs import extract_outline_cached
from ..fonts.raster import RasterFrame, rasterize_contours
from ..utils.caching import TwoLevelCache
from .graph import SkeletonGraph, build_graph, drop_small_components, merge_chains, prune_spurs
from .simplify import simplify_graph, smooth_graph
from .thinning import binarize, clean_mask, skeletonize_mask, stroke_width_map

if TYPE_CHECKING:  # pragma: no cover
    from ..fonts.loader import LoadedFont


#: glyph placement relative to the cluster origin, in font units
GlyphPlacement = tuple[int, float, float]


@dataclass
class ClusterSkeleton:
    """Centreline graph of one grapheme cluster, in em space (y-DOWN)."""

    graph: SkeletonGraph
    #: (x_min, y_min, x_max, y_max) in em
    bbox: tuple[float, float, float, float] | None
    #: mean half stroke width of the source outline, in em (renderer prior)
    stroke_halfwidth_em: float
    #: diagnostics for the debug pipeline / validator
    info: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.graph.num_edges == 0


_CACHE: TwoLevelCache | None = None


def get_cache(capacity: int = 8192, disk_root: str | None = None) -> TwoLevelCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = TwoLevelCache(capacity=capacity, disk_root=disk_root)
    return _CACHE


def reset_cache() -> None:
    global _CACHE
    _CACHE = None


def _debug_stages(
    font: LoadedFont,
    placements: Sequence[GlyphPlacement],
    raster_cfg: RasterConfig,
    skel_cfg: SkeletonConfig,
) -> dict:
    """Compute a cluster skeleton, returning every intermediate stage.

    Used by ``scripts/debug_pipeline.py``; the fast path below shares the code.
    """
    contours: list[np.ndarray] = []
    for gid, dx, dy in placements:
        outline = extract_outline_cached(font, gid)
        if outline.is_empty:
            continue
        contours.extend(outline.transformed(dx, dy))

    if not contours:
        frame = RasterFrame(0.0, 0.0, raster_cfg.px_per_em / font.upem, 1, 1, font.upem)
        empty = SkeletonGraph(np.zeros((0, 2), np.float32), [], [], "em")
        return {
            "contours": [],
            "coverage": np.zeros((1, 1), np.float32),
            "mask": np.zeros((1, 1), bool),
            "skeleton": np.zeros((1, 1), bool),
            "frame": frame,
            "graph_px": empty,
            "graph_em": empty,
            "halfwidth_em": 0.0,
        }

    coverage, frame = rasterize_contours(
        contours,
        upem=font.upem,
        px_per_em=raster_cfg.px_per_em,
        supersample=raster_cfg.supersample,
        margin_em=0.03,
    )
    mask = binarize(coverage, raster_cfg.threshold)
    mask = clean_mask(
        mask,
        closing_radius=raster_cfg.closing_radius,
        opening_radius=raster_cfg.opening_radius,
        min_component_area_ratio=raster_cfg.min_component_area_ratio,
        max_hole_fill_ratio=raster_cfg.max_hole_fill_ratio,
    )
    skel = skeletonize_mask(mask, skel_cfg.method)

    dist = stroke_width_map(mask)
    halfwidth_px = float(dist[skel].mean()) if skel.any() else 0.0
    halfwidth_em = halfwidth_px / max(raster_cfg.px_per_em, 1e-6)

    graph_px = build_graph(skel)
    px_per_em = raster_cfg.px_per_em
    graph_px = prune_spurs(
        graph_px, min_length=skel_cfg.prune_spur_em * px_per_em, iterations=skel_cfg.prune_iterations
    )
    graph_px = merge_chains(graph_px)
    graph_px = drop_small_components(graph_px, min_length=skel_cfg.min_component_len_em * px_per_em)
    graph_px = smooth_graph(graph_px, window=3, passes=1)

    def to_em(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return frame.px_to_em(pts[:, 0], pts[:, 1])

    graph_em = graph_px.transformed(to_em, space="em")
    graph_em = simplify_graph(graph_em, epsilon=skel_cfg.rdp_epsilon_em)

    return {
        "contours": contours,
        "coverage": coverage,
        "mask": mask,
        "skeleton": skel,
        "frame": frame,
        "graph_px": graph_px,
        "graph_em": graph_em,
        "halfwidth_em": halfwidth_em,
    }


def _compute(
    font: LoadedFont,
    placements: Sequence[GlyphPlacement],
    raster_cfg: RasterConfig,
    skel_cfg: SkeletonConfig,
) -> ClusterSkeleton:
    stages = _debug_stages(font, placements, raster_cfg, skel_cfg)
    graph = stages["graph_em"]
    bb = graph.bbox()
    return ClusterSkeleton(
        graph=graph,
        bbox=bb,
        stroke_halfwidth_em=float(stages["halfwidth_em"]),
        info={
            "mask_shape": tuple(int(v) for v in stages["mask"].shape),
            "ink_ratio": float(stages["mask"].mean()),
            **graph.stats(),
        },
    )


def cluster_skeleton(
    font: LoadedFont,
    placements: Sequence[GlyphPlacement],
    raster_cfg: RasterConfig,
    skel_cfg: SkeletonConfig,
    cache: TwoLevelCache | None = None,
) -> ClusterSkeleton:
    """Cached centreline graph of a positioned glyph group (em space)."""
    if not placements:
        return ClusterSkeleton(
            SkeletonGraph(np.zeros((0, 2), np.float32), [], [], "em"), None, 0.0, {}
        )
    cache = cache or get_cache()
    # quantise offsets so that near-identical placements share a cache entry
    sig = tuple(sorted((int(g), round(float(dx), 1), round(float(dy), 1)) for g, dx, dy in placements))
    key = (
        "cluster_skeleton_v1",
        font.key,
        tuple(sorted(font.variations.items())),
        font.upem,
        sig,
        (
            raster_cfg.px_per_em,
            raster_cfg.supersample,
            raster_cfg.threshold,
            raster_cfg.closing_radius,
            raster_cfg.opening_radius,
            raster_cfg.min_component_area_ratio,
            raster_cfg.max_hole_fill_ratio,
        ),
        (
            skel_cfg.method,
            skel_cfg.prune_spur_em,
            skel_cfg.min_component_len_em,
            skel_cfg.rdp_epsilon_em,
            skel_cfg.prune_iterations,
        ),
    )
    return cache.get_or_compute(key, lambda: _compute(font, placements, raster_cfg, skel_cfg))
