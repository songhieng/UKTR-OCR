"""Skeleton -> graph  G = (V, E).

A thinned bitmap is an *unordered* set of pixels.  To plan strokes we need
topology: which arcs exist, where they meet, which are loops, which are isolated
dots.  This module extracts

    V : endpoints (degree 1), junctions (degree >= 3), loop anchors, dots
    E : the pixel path (a polyline) joining two vertices

and then cleans the result (spur pruning, degree-2 chain merging).

Diagonal redundancy
-------------------
Naive 8-neighbour degree counting produces *false junctions*: for the L-shaped
skeleton ``(0,0) (0,1) (1,1) (2,1)`` the corner pixel ``(1,1)`` has three
8-neighbours and would be classified as a junction.  We therefore drop a
diagonal link whenever one of the two orthogonal links that "cut the corner" is
also present.  The rule is symmetric (both pixels test the same two cells), so
the resulting adjacency is a proper undirected graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

# N, NE, E, SE, S, SW, W, NW  (dy, dx) with y down
DIRS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
_DIAGONALS = (1, 3, 5, 7)
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass
class SkeletonEdge:
    """One arc of the skeleton graph."""

    u: int
    v: int
    points: np.ndarray  # (M, 2) float32, ordered u -> v

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    @property
    def is_loop(self) -> bool:
        return self.u == self.v

    def reversed_points(self) -> np.ndarray:
        return self.points[::-1]


@dataclass
class SkeletonGraph:
    """Undirected multigraph with polyline edges (coordinates are (x, y))."""

    nodes: np.ndarray  # (N, 2) float32
    node_kind: list[str]
    edges: list[SkeletonEdge]
    space: str = "px"  # 'px' or 'em'
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def degree(self) -> np.ndarray:
        deg = np.zeros(self.num_nodes, dtype=np.int32)
        for e in self.edges:
            deg[e.u] += 1
            deg[e.v] += 1  # a self-loop correctly contributes 2
        return deg

    def adjacency(self) -> dict[int, list[tuple[int, int]]]:
        """node -> list of (edge_index, other_node)."""
        adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(self.num_nodes)}
        for i, e in enumerate(self.edges):
            adj[e.u].append((i, e.v))
            if e.u != e.v:
                adj[e.v].append((i, e.u))
            else:
                adj[e.u].append((i, e.u))  # loop is incident twice
        return adj

    def total_length(self) -> float:
        return float(sum(e.length for e in self.edges))

    def components(self) -> list[list[int]]:
        """Connected components as lists of edge indices."""
        parent = list(range(self.num_nodes))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for e in self.edges:
            union(e.u, e.v)
        groups: dict[int, list[int]] = {}
        for i, e in enumerate(self.edges):
            groups.setdefault(find(e.u), []).append(i)
        return [v for _, v in sorted(groups.items(), key=lambda kv: kv[0])]

    def transformed(self, fn, space: str | None = None) -> SkeletonGraph:
        """Apply ``fn(points (N,2)) -> (N,2)`` to nodes and every edge."""
        nodes = np.asarray(fn(self.nodes), dtype=np.float32) if len(self.nodes) else self.nodes
        edges = [
            SkeletonEdge(e.u, e.v, np.asarray(fn(e.points), dtype=np.float32)) for e in self.edges
        ]
        return SkeletonGraph(nodes, list(self.node_kind), edges, space or self.space, dict(self.meta))

    def bbox(self) -> tuple[float, float, float, float] | None:
        pts = [e.points for e in self.edges if len(e.points)]
        if not pts:
            if len(self.nodes) == 0:
                return None
            pts = [self.nodes]
        a = np.vstack(pts)
        return (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max()))

    def stats(self) -> dict:
        deg = self.degree()
        return {
            "nodes": int(self.num_nodes),
            "edges": int(self.num_edges),
            "endpoints": int((deg == 1).sum()),
            "junctions": int((deg >= 3).sum()),
            "loops": int(sum(1 for e in self.edges if e.is_loop)),
            "components": len(self.components()),
            "length": self.total_length(),
        }


# --------------------------------------------------------------------------- #
# adjacency with diagonal-redundancy removal
# --------------------------------------------------------------------------- #


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Value of ``mask`` at (y+dy, x+dx), zero-padded."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys_dst = slice(max(0, -dy), h - max(0, dy))
    ys_src = slice(max(0, dy), h - max(0, -dy))
    xs_dst = slice(max(0, -dx), w - max(0, dx))
    xs_src = slice(max(0, dx), w - max(0, -dx))
    out[ys_dst, xs_dst] = mask[ys_src, xs_src]
    return out


def adjacency_code(skel: np.ndarray) -> np.ndarray:
    """8-bit neighbour bitmask per pixel, with redundant diagonals removed."""
    skel = np.asarray(skel, dtype=bool)
    present = [_shift(skel, dy, dx) for dy, dx in DIRS]
    code = np.zeros(skel.shape, dtype=np.uint8)
    for i in (0, 2, 4, 6):
        code |= (present[i].astype(np.uint8) << i)
    for i in _DIAGONALS:
        eff = present[i] & ~(present[(i - 1) % 8] | present[(i + 1) % 8])
        code |= (eff.astype(np.uint8) << i)
    code[~skel] = 0
    return code


def degree_map(code: np.ndarray) -> np.ndarray:
    return _POPCOUNT[code]


# --------------------------------------------------------------------------- #
# graph construction
# --------------------------------------------------------------------------- #


def build_graph(skel: np.ndarray, dot_min_pixels: int = 1) -> SkeletonGraph:
    """Trace a thinned bitmap into a :class:`SkeletonGraph` (pixel space)."""
    skel = np.asarray(skel, dtype=bool)
    if not skel.any():
        return SkeletonGraph(np.zeros((0, 2), np.float32), [], [], "px")

    code = adjacency_code(skel)
    deg = degree_map(code)

    node_mask = skel & (deg != 2)
    labels, n_labels = ndi.label(node_mask, structure=np.ones((3, 3), dtype=int))

    nodes: list[tuple[float, float]] = []
    node_kind: list[str] = []
    node_id = -np.ones(skel.shape, dtype=np.int32)

    if n_labels:
        objects = ndi.find_objects(labels)
        for lab in range(1, n_labels + 1):
            sl = objects[lab - 1]
            sub = labels[sl] == lab
            ys, xs = np.nonzero(sub)
            ys = ys + sl[0].start
            xs = xs + sl[1].start
            nid = len(nodes)
            node_id[ys, xs] = nid
            nodes.append((float(xs.mean()), float(ys.mean())))
            d = deg[ys, xs]
            if d.max() == 0:
                node_kind.append("dot")
            elif d.max() >= 3:
                node_kind.append("junction")
            else:
                node_kind.append("end")

    edges: list[SkeletonEdge] = []
    visited_starts: set[tuple[int, int, int, int]] = set()

    def neighbours(y: int, x: int) -> list[tuple[int, int]]:
        c = int(code[y, x])
        out = []
        for i, (dy, dx) in enumerate(DIRS):
            if c & (1 << i):
                out.append((y + dy, x + dx))
        return out

    # --- edges incident to a node --- #
    for nid in range(len(nodes)):
        ys, xs = np.nonzero(node_id == nid)
        for y, x in zip(ys.tolist(), xs.tolist()):
            for ny, nx in neighbours(y, x):
                if node_id[ny, nx] == nid:
                    continue  # inside the same junction blob
                key = (y, x, ny, nx)
                if key in visited_starts:
                    continue
                path = [(y, x), (ny, nx)]
                prev = (y, x)
                cur = (ny, nx)
                guard = 0
                while node_id[cur] < 0 and guard < skel.size:
                    guard += 1
                    nxt = None
                    for cand in neighbours(*cur):
                        if cand != prev:
                            nxt = cand
                            break
                    if nxt is None:
                        break
                    path.append(nxt)
                    prev, cur = cur, nxt
                visited_starts.add(key)
                visited_starts.add((path[-1][0], path[-1][1], path[-2][0], path[-2][1]))
                end_nid = int(node_id[cur]) if node_id[cur] >= 0 else nid
                pts = _path_to_points(path, nodes[nid], nodes[end_nid])
                edges.append(SkeletonEdge(nid, end_nid, pts))

    # --- pure loops: components in which every pixel has degree 2, so no
    #     vertex was created above.  They still need an (arbitrary) anchor. --- #
    used = node_mask.copy()
    for e in edges:
        pts = np.rint(e.points).astype(int)
        ys = np.clip(pts[:, 1], 0, skel.shape[0] - 1)
        xs = np.clip(pts[:, 0], 0, skel.shape[1] - 1)
        used[ys, xs] = True

    remaining = skel & ~used
    while remaining.any():
        ys, xs = np.nonzero(remaining)
        y, x = int(ys[0]), int(xs[0])
        nid = len(nodes)
        nodes.append((float(x), float(y)))
        node_kind.append("loop")
        node_id[y, x] = nid
        nb = neighbours(y, x)
        if not nb:
            edges.append(SkeletonEdge(nid, nid, np.array([[x, y], [x, y]], dtype=np.float32)))
            remaining[y, x] = False
            continue
        prev = (y, x)
        cur = nb[0]
        path = [(y, x), cur]
        guard = 0
        while cur != (y, x) and guard < skel.size:
            guard += 1
            nxt = None
            for cand in neighbours(*cur):
                if cand != prev:
                    nxt = cand
                    break
            if nxt is None:
                break
            path.append(nxt)
            prev, cur = cur, nxt
        pts = _path_to_points(path, nodes[nid], nodes[nid])
        edges.append(SkeletonEdge(nid, nid, pts))
        for py, px in path:
            remaining[py, px] = False
        remaining[y, x] = False

    # --- dots: a component so small that it collapses into a single vertex
    #     (isolated pixels, 2-3 px blobs).  Khmer is full of tiny diacritics, so
    #     these must still produce a (degenerate) stroke rather than vanish. --- #
    incident = np.zeros(len(nodes), dtype=np.int32)
    for e in edges:
        incident[e.u] += 1
        incident[e.v] += 1
    for nid in range(len(nodes)):
        if incident[nid] == 0:
            x, y = nodes[nid]
            node_kind[nid] = "dot"
            edges.append(
                SkeletonEdge(nid, nid, np.array([[x, y], [x, y]], dtype=np.float32))
            )

    graph = SkeletonGraph(
        np.asarray(nodes, dtype=np.float32).reshape(-1, 2), node_kind, edges, "px"
    )
    return graph


def _path_to_points(
    path: Sequence[tuple[int, int]], start_node: tuple[float, float], end_node: tuple[float, float]
) -> np.ndarray:
    pts = [(float(x), float(y)) for y, x in path]
    pts[0] = (float(start_node[0]), float(start_node[1]))
    pts[-1] = (float(end_node[0]), float(end_node[1]))
    arr = np.asarray(pts, dtype=np.float32)
    if len(arr) > 2:
        keep = np.concatenate([[True], np.linalg.norm(np.diff(arr, axis=0), axis=1) > 1e-6])
        keep[-1] = True
        arr = arr[keep]
    if len(arr) < 2:
        arr = np.vstack([arr, arr[-1:]])
    return arr


# --------------------------------------------------------------------------- #
# cleanup
# --------------------------------------------------------------------------- #


def prune_spurs(graph: SkeletonGraph, min_length: float, iterations: int = 3) -> SkeletonGraph:
    """Remove short terminal branches (thinning artefacts at stroke ends).

    A branch is only removed when its far end is a real junction, so genuine
    short strokes (Khmer diacritics!) that form their own component survive.
    """
    if min_length <= 0:
        return graph
    edges = list(graph.edges)
    for _ in range(max(1, iterations)):
        deg = np.zeros(graph.num_nodes, dtype=np.int32)
        for e in edges:
            deg[e.u] += 1
            deg[e.v] += 1
        victims = []
        for i, e in enumerate(edges):
            if e.is_loop:
                continue
            terminal = (deg[e.u] == 1) or (deg[e.v] == 1)
            anchored = (deg[e.u] >= 3) or (deg[e.v] >= 3)
            if terminal and anchored and e.length < min_length:
                victims.append(i)
        if not victims:
            break
        drop = set(victims)
        edges = [e for i, e in enumerate(edges) if i not in drop]
    return _rebuild(graph, edges)


def merge_chains(graph: SkeletonGraph) -> SkeletonGraph:
    """Collapse degree-2 vertices, concatenating their two edges.

    After pruning, many junctions degenerate into plain corners; merging keeps
    the planner's decision space small and gives longer, more natural arcs.
    """
    edges = list(graph.edges)
    if not edges:
        return graph
    changed = True
    while changed:
        changed = False
        adj: dict[int, list[int]] = {}
        for i, e in enumerate(edges):
            adj.setdefault(e.u, []).append(i)
            adj.setdefault(e.v, []).append(i)
        for node, inc in adj.items():
            if len(inc) != 2:
                continue
            i, j = inc
            if i == j:
                continue
            e1, e2 = edges[i], edges[j]
            if e1.is_loop or e2.is_loop:
                continue
            p1 = e1.points if e1.v == node else e1.reversed_points()
            u = e1.u if e1.v == node else e1.v
            p2 = e2.points if e2.u == node else e2.reversed_points()
            v = e2.v if e2.u == node else e2.u
            pts = np.vstack([p1, p2[1:]]).astype(np.float32)
            new_edges = [e for k, e in enumerate(edges) if k not in (i, j)]
            new_edges.append(SkeletonEdge(u, v, pts))
            edges = new_edges
            changed = True
            break
    return _rebuild(graph, edges)


def drop_small_components(graph: SkeletonGraph, min_length: float) -> SkeletonGraph:
    """Drop whole components shorter than ``min_length`` (isolated speckles)."""
    if min_length <= 0:
        return graph
    keep: list[SkeletonEdge] = []
    for comp in graph.components():
        total = sum(graph.edges[i].length for i in comp)
        is_dot = all(graph.edges[i].length <= 1e-6 for i in comp)
        if total >= min_length or is_dot:
            keep.extend(graph.edges[i] for i in comp)
    return _rebuild(graph, keep)


def _rebuild(graph: SkeletonGraph, edges: list[SkeletonEdge]) -> SkeletonGraph:
    """Re-index nodes so only referenced vertices survive."""
    used: dict[int, int] = {}
    new_nodes: list[np.ndarray] = []
    new_kind: list[str] = []
    out: list[SkeletonEdge] = []
    for e in edges:
        for n in (e.u, e.v):
            if n not in used:
                used[n] = len(new_nodes)
                new_nodes.append(graph.nodes[n])
                new_kind.append(graph.node_kind[n] if n < len(graph.node_kind) else "end")
        out.append(SkeletonEdge(used[e.u], used[e.v], e.points))
    nodes = (
        np.asarray(new_nodes, dtype=np.float32).reshape(-1, 2)
        if new_nodes
        else np.zeros((0, 2), np.float32)
    )
    return SkeletonGraph(nodes, new_kind, out, graph.space, dict(graph.meta))
