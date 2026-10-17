"""Polyline simplification / resampling for skeleton graphs.

The traced pixel paths are massively over-sampled (one point per pixel) and
staircased.  Douglas-Peucker removes the redundancy while bounding the geometric
error, which is exactly the guarantee we want: the tolerance is expressed in em,
so it can be set well below the thinnest meaningful Khmer feature and therefore
cannot destroy glyph structure.
"""

from __future__ import annotations

import numpy as np

from ..utils.geometry import rdp, resample_uniform
from .graph import SkeletonEdge, SkeletonGraph


def simplify_graph(graph: SkeletonGraph, epsilon: float, min_points: int = 2) -> SkeletonGraph:
    """Douglas-Peucker every edge (endpoints always preserved)."""
    if epsilon <= 0:
        return graph
    edges = []
    for e in graph.edges:
        pts = e.points
        if len(pts) > min_points:
            simplified = rdp(pts.astype(np.float64), epsilon)
            if len(simplified) < min_points:
                simplified = pts[:: max(1, len(pts) // min_points)]
            pts = simplified.astype(np.float32)
        edges.append(SkeletonEdge(e.u, e.v, np.asarray(pts, dtype=np.float32)))
    return SkeletonGraph(graph.nodes, list(graph.node_kind), edges, graph.space, dict(graph.meta))


def resample_graph(graph: SkeletonGraph, spacing: float) -> SkeletonGraph:
    """Uniform arc-length resampling of every edge."""
    if spacing <= 0:
        return graph
    edges = []
    for e in graph.edges:
        pts = e.points
        if e.length > 0:
            pts = resample_uniform(pts.astype(np.float64), spacing).astype(np.float32)
        edges.append(SkeletonEdge(e.u, e.v, pts))
    return SkeletonGraph(graph.nodes, list(graph.node_kind), edges, graph.space, dict(graph.meta))


def smooth_graph(graph: SkeletonGraph, window: int = 3, passes: int = 1) -> SkeletonGraph:
    """Light moving-average smoothing that keeps edge endpoints fixed.

    Removes the 1-pixel staircase of the thinning output before the (much
    stronger) spline smoothing in ``strokes/smoothing.py``.
    """
    if window < 3 or passes < 1:
        return graph
    k = np.ones(window, dtype=np.float64) / window
    edges = []
    for e in graph.edges:
        p = e.points.astype(np.float64).copy()
        if len(p) > window + 2:
            for _ in range(passes):
                pad = window // 2
                padded = np.vstack([np.repeat(p[:1], pad, 0), p, np.repeat(p[-1:], pad, 0)])
                sm = np.stack(
                    [np.convolve(padded[:, 0], k, "valid"), np.convolve(padded[:, 1], k, "valid")],
                    axis=1,
                )
                sm[0] = p[0]
                sm[-1] = p[-1]
                p = sm
        edges.append(SkeletonEdge(e.u, e.v, p.astype(np.float32)))
    return SkeletonGraph(graph.nodes, list(graph.node_kind), edges, graph.space, dict(graph.meta))
