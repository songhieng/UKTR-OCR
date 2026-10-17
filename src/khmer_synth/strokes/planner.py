"""Skeleton graph -> ordered, plausible strokes.

SYNTHETIC-BIAS WARNING
----------------------
The order produced here is a *heuristic reconstruction*, not observed Khmer
stroke order.  Real Khmer handwriting order is taught explicitly (and varies by
writer and by school), and a font-derived skeleton contains no information about
it whatsoever.  What this planner guarantees is only that the resulting sequence
is (a) geometrically consistent with the glyph, (b) cheap under an explicit,
inspectable cost function, and (c) diverse.  Use it for pretraining, not for
claims about how Khmer is written.

Method
------
The problem is a route-inspection ("Chinese postman") variant: cover every edge
of a small multigraph while minimising pen lifts and awkward motion.  We use a
stochastic greedy walk with Fleury-style bridge avoidance, then score the whole
plan with the configurable cost

    C = w1 * pen_up_count
      + w2 * pen_up_distance
      + w3 * sharp_direction_changes
      + w4 * path_reversals
      + w5 * unnatural_spatial_order

and keep the best of ``num_candidates`` samples.  ``temperature -> 0`` makes the
walk deterministic (useful for tests and for ablations); higher temperatures give
several plausible orders for the same grapheme, which is exactly the variation a
downstream trajectory model should learn to tolerate.

Component ordering (``component_order='natural'``) additionally encodes one weak
piece of real-world prior knowledge that is safe to assert: writers draw the main
body of a syllable before its detached marks, and generally work top-to-bottom /
left-to-right between disconnected parts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..config import PlannerConfig
from ..skeleton.graph import SkeletonGraph
from ..utils.rng import softmax_sample


@dataclass
class Stroke:
    """One pen-down excursion."""

    points: np.ndarray  # (N, 2) float32
    edges: list[int] = field(default_factory=list)
    component: int = 0
    #: index of the grapheme cluster this stroke belongs to (-1 = unknown);
    #: set by the layout stage and carried through to the exported metadata
    cluster: int = -1
    #: half-open intervals of *normalised arc length* that are connecting
    #: ligatures rather than glyph ink (see strokes/connect.py).  Stored as
    #: fractions so they survive resampling, smoothing and deformation.
    ligature_spans: list[tuple[float, float]] = field(default_factory=list)

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    @property
    def is_dot(self) -> bool:
        return self.length < 1e-6


@dataclass
class StrokePlan:
    strokes: list[Stroke]
    cost: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        return int(sum(len(s.points) for s in self.strokes))

    def total_length(self) -> float:
        return float(sum(s.length for s in self.strokes))


# --------------------------------------------------------------------------- #
# bridge detection (Tarjan, iterative, multigraph-safe)
# --------------------------------------------------------------------------- #


def find_bridges(num_nodes: int, edges: Sequence[tuple[int, int]], active: Iterable[int]) -> set[int]:
    """Bridges of the sub-multigraph induced by ``active`` edge indices.

    Parallel edges are never bridges, which matters here: two skeleton arcs
    joining the same pair of junctions (a loop split in two) must not be treated
    as separating.
    """
    active = list(active)
    adj: dict[int, list[tuple[int, int]]] = {}
    for ei in active:
        u, v = edges[ei]
        adj.setdefault(u, []).append((v, ei))
        adj.setdefault(v, []).append((u, ei))

    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    bridges: set[int] = set()
    timer = 0

    for root in list(adj.keys()):
        if root in disc:
            continue
        stack: list[tuple[int, int, int]] = [(root, -1, 0)]  # node, parent edge, child ptr
        disc[root] = low[root] = timer
        timer += 1
        while stack:
            node, pedge, ptr = stack[-1]
            neigh = adj.get(node, [])
            if ptr < len(neigh):
                stack[-1] = (node, pedge, ptr + 1)
                nxt, ei = neigh[ptr]
                if ei == pedge or nxt == node:
                    continue
                if nxt in disc:
                    low[node] = min(low[node], disc[nxt])
                else:
                    disc[nxt] = low[nxt] = timer
                    timer += 1
                    stack.append((nxt, ei, 0))
            else:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    low[parent] = min(low[parent], low[node])
                    if low[node] > disc[parent]:
                        bridges.add(pedge)
    return bridges


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def _edge_points(graph: SkeletonGraph, ei: int, from_node: int) -> np.ndarray:
    e = graph.edges[ei]
    if e.u == from_node:
        return e.points
    return e.reversed_points()


def _dir_at_start(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.zeros(2)
    d = pts[min(3, len(pts) - 1)] - pts[0]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.zeros(2)


def _dir_at_end(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.zeros(2)
    d = pts[-1] - pts[max(0, len(pts) - 4)]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.zeros(2)


def _component_order(graph: SkeletonGraph, comps: list[list[int]], mode: str, rng, temperature: float) -> list[list[int]]:
    """Order the disconnected parts of a cluster.

    ``natural``: the largest (body) component first, then marks sorted by
    vertical position (above marks before below marks) and x - a crude but
    defensible model of "write the syllable, then decorate it".
    """
    if len(comps) <= 1:
        return comps
    stats = []
    for ci, comp in enumerate(comps):
        pts = np.vstack([graph.edges[i].points for i in comp])
        length = float(sum(graph.edges[i].length for i in comp))
        stats.append(
            {
                "i": ci,
                "len": length,
                "cx": float(pts[:, 0].mean()),
                "cy": float(pts[:, 1].mean()),
                "ymin": float(pts[:, 1].min()),
                "ymax": float(pts[:, 1].max()),
            }
        )
    if mode == "spatial":
        keys = [(s["cx"], s["cy"]) for s in stats]
    elif mode == "random":
        keys = [(float(rng.random()),) for _ in stats]
    else:  # natural
        main = max(stats, key=lambda s: s["len"])
        keys = []
        for s in stats:
            if s is main:
                rank = 0
            elif s["ymax"] <= main["cy"]:
                rank = 1  # above the body
            else:
                rank = 2  # below the body
            jitter = float(rng.normal(0.0, 0.35 * max(temperature, 0.0)))
            keys.append((rank + jitter, s["cx"], s["cy"]))
    order = [c for _, c in sorted(zip(keys, comps), key=lambda kv: kv[0])]
    return order


def _sample_plan(graph: SkeletonGraph, cfg: PlannerConfig, rng: np.random.Generator) -> StrokePlan:
    edges_uv = [(e.u, e.v) for e in graph.edges]
    bb = graph.bbox() or (0.0, 0.0, 1.0, 1.0)
    span = max(bb[2] - bb[0], bb[3] - bb[1], 1e-6)

    comps = _component_order(
        graph, graph.components(), getattr(cfg, "component_order", "natural"), rng, cfg.temperature
    )

    strokes: list[Stroke] = []
    pen = None  # current pen position (end of last stroke)

    for comp_idx, comp in enumerate(comps):
        unvisited: set[int] = set(comp)
        while unvisited:
            # ---- choose where to put the pen down ---- #
            deg: dict[int, int] = {}
            for ei in unvisited:
                u, v = edges_uv[ei]
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1

            cands: list[tuple[int, int]] = []  # (node, edge)
            for ei in unvisited:
                u, v = edges_uv[ei]
                cands.append((u, ei))
                if u != v:
                    cands.append((v, ei))
            costs = []
            for node, ei in cands:
                pts = _edge_points(graph, ei, node)
                start = pts[0]
                c = 0.0
                c += cfg.start_top_weight * (start[1] - bb[1]) / span
                c += cfg.start_left_weight * (start[0] - bb[0]) / span
                if deg.get(node, 0) > 1:
                    c += 0.6  # prefer starting from a free end
                d = _dir_at_start(pts)
                c -= cfg.prefer_downward * float(d[1])
                c -= cfg.prefer_rightward * float(d[0])
                if pen is not None:
                    c += cfg.w_pen_up_distance * float(np.linalg.norm(start - pen)) / span * 0.5
                costs.append(c)
            k = softmax_sample(rng, costs, cfg.temperature)
            node, ei = cands[k]

            # ---- walk ---- #
            pts = _edge_points(graph, ei, node)
            path = [pts]
            used = [ei]
            unvisited.discard(ei)
            cur_node = edges_uv[ei][1] if edges_uv[ei][0] == node else edges_uv[ei][0]
            cur_dir = _dir_at_end(pts)

            while unvisited:
                incident = [e for e in unvisited if cur_node in edges_uv[e]]
                if not incident:
                    break
                if cfg.pen_lift_bias > 0 and rng.random() < cfg.pen_lift_bias:
                    break
                bridges = (
                    find_bridges(graph.num_nodes, edges_uv, unvisited)
                    if cfg.bridge_penalty > 0
                    else set()
                )
                non_bridge_exists = any(e not in bridges for e in incident)
                ccosts = []
                for e in incident:
                    npts = _edge_points(graph, e, cur_node)
                    d = _dir_at_start(npts)
                    ang = 0.0
                    if np.any(cur_dir) and np.any(d):
                        ang = float(np.arccos(np.clip(float(np.dot(cur_dir, d)), -1.0, 1.0)))
                    c = cfg.w_direction_change * (ang / np.pi)
                    if np.degrees(ang) > cfg.sharp_angle_deg:
                        c += cfg.w_direction_change * 0.5
                    if np.degrees(ang) > 150.0:
                        c += cfg.w_reversal
                    if cfg.bridge_penalty > 0 and non_bridge_exists and e in bridges:
                        c += cfg.bridge_penalty
                    ccosts.append(c)
                k = softmax_sample(rng, ccosts, cfg.temperature)
                e = incident[k]
                npts = _edge_points(graph, e, cur_node)
                path.append(npts[1:] if len(npts) > 1 else npts)
                used.append(e)
                unvisited.discard(e)
                cur_node = edges_uv[e][1] if edges_uv[e][0] == cur_node else edges_uv[e][0]
                cur_dir = _dir_at_end(npts)

            pts_all = np.vstack(path).astype(np.float32)
            strokes.append(Stroke(points=pts_all, edges=used, component=comp_idx))
            pen = pts_all[-1]

    plan = StrokePlan(strokes=strokes)
    plan.cost, plan.breakdown = score_plan(plan, cfg, span)
    return plan


def score_plan(plan: StrokePlan, cfg: PlannerConfig, span: float = 1.0) -> tuple[float, dict[str, float]]:
    """Evaluate the explicit stroke-planning cost function."""
    n_up = max(len(plan.strokes) - 1, 0)
    up_dist = 0.0
    spatial = 0.0
    for a, b in zip(plan.strokes[:-1], plan.strokes[1:]):
        up_dist += float(np.linalg.norm(b.points[0] - a.points[-1]))
        dx = float(b.points[0][0] - a.points[0][0])
        if dx < 0:
            spatial += -dx / max(span, 1e-6)

    sharp = 0.0
    reversal = 0.0
    thr = np.radians(cfg.sharp_angle_deg)
    for s in plan.strokes:
        p = s.points
        if len(p) < 3:
            continue
        v = np.diff(p, axis=0)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.where(n < 1e-9, 1.0, n)
        cosang = np.clip(np.einsum("ij,ij->i", v[:-1], v[1:]), -1.0, 1.0)
        ang = np.arccos(cosang)
        sharp += float(np.sum(np.maximum(ang - thr, 0.0)) / np.pi)
        reversal += float(np.sum(ang > np.radians(150.0)))

    up_dist_n = up_dist / max(span, 1e-6)
    breakdown = {
        "pen_up_count": float(n_up),
        "pen_up_distance": up_dist_n,
        "sharp_direction_changes": sharp,
        "reversals": reversal,
        "spatial_order": spatial,
    }
    cost = (
        cfg.w_pen_up_count * n_up
        + cfg.w_pen_up_distance * up_dist_n
        + cfg.w_direction_change * sharp
        + cfg.w_reversal * reversal
        + cfg.w_spatial_order * spatial
    )
    return float(cost), breakdown


def plan_strokes(
    graph: SkeletonGraph, cfg: PlannerConfig | None = None, rng: np.random.Generator | None = None
) -> StrokePlan:
    """Sample ``cfg.num_candidates`` plans and return the cheapest one."""
    cfg = cfg or PlannerConfig()
    rng = rng if rng is not None else np.random.default_rng(0)
    if graph.num_edges == 0:
        return StrokePlan(strokes=[])
    best: StrokePlan | None = None
    for _ in range(max(1, cfg.num_candidates)):
        plan = _sample_plan(graph, cfg, rng)
        if best is None or plan.cost < best.cost:
            best = plan
    assert best is not None
    return best


def plan_variants(
    graph: SkeletonGraph, cfg: PlannerConfig | None = None, n: int = 8, seed: int = 0
) -> list[StrokePlan]:
    """Pre-compute ``n`` distinct plans for a cluster (cached by the generator).

    Planning is cheap but not free, and a corpus re-uses the same clusters
    constantly; sampling from a small pool of pre-planned orders keeps the
    per-sample cost near zero while preserving stroke-order diversity.
    """
    cfg = cfg or PlannerConfig()
    rng = np.random.default_rng(seed)
    out: list[StrokePlan] = []
    seen: set[tuple] = set()
    for _ in range(max(1, n) * 3):
        if len(out) >= max(1, n):
            break
        plan = plan_strokes(graph, cfg, rng)
        sig = tuple(tuple(s.edges) for s in plan.strokes)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(plan)
    if not out:
        out.append(StrokePlan(strokes=[]))
    return out
