"""Sampling methods + multiplicity-cap support.

Every method returns a list of *picked* node IDs in source-relabeled space
(i.e., indices into the SplitSource's `g_nx`). The driver then calls
`make_partition` to build the (sub_nx, sub_node_data, orig_indices) tuple.

Per-subgraph connectivity invariant — every emitted partition has exactly
one connected component, ≥ 2 nodes, and minimum degree ≥ 1. Methods that
don't produce a connected set by construction (uniform_random, metis_partition)
get LCC-trimmed inside `make_partition`.

The multiplicity cap (`MultiplicityCap`) is owned by the driver and shared
across draws; methods consult it via `is_blocked(node)`. Methods that are
disjoint by construction (METIS, BFS partition) ignore it.

`bfs_snowball` accepts both `target_size` (size-bound) and `max_depth`
(depth-bound from the seed); at least one must be set. With both set the
walk halts at whichever is hit first.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import networkx as nx
import torch


# ---------------------------------------------------------------------------
# Multiplicity cap — counts how many partitions a source node has appeared in
# ---------------------------------------------------------------------------

class MultiplicityCap:
    """Counts how many times each source-space node has been emitted."""

    def __init__(self, M: Optional[int], n_source: int):
        self.cap = M  # None = unlimited
        self.counts = [0] * n_source

    def is_blocked(self, node: int) -> bool:
        return self.cap is not None and self.counts[node] >= self.cap

    def visit(self, nodes) -> None:
        for n in nodes:
            self.counts[int(n)] += 1


def _blocked(counter: Optional[MultiplicityCap], n: int) -> bool:
    return counter is not None and counter.is_blocked(n)


def _valid_seed_pool(g_nx: nx.Graph, counter: Optional[MultiplicityCap]) -> List[int]:
    """Return source nodes that are non-isolated and not multiplicity-capped."""
    return [n for n in g_nx.nodes() if not _blocked(counter, n) and g_nx.degree(n) > 0]


# ---------------------------------------------------------------------------
# Forest fire — vendored from bigg/extension/preprocessing.py with M-cap support
# ---------------------------------------------------------------------------

def forest_fire(
    g_nx: nx.Graph,
    target_size: int,
    burn_prob: float,
    counter: Optional[MultiplicityCap] = None,
    seed_node: Optional[int] = None,
) -> Optional[List[int]]:
    """One forest-fire draw. Returns burned nodes in burn order, or None if no
    valid seed was available.

    Identical to bigg's `forest_fire_subsample` when `counter is None`."""
    seeds = _valid_seed_pool(g_nx, counter)
    if not seeds:
        return None
    if seed_node is None:
        seed_node = random.choice(seeds)
    elif _blocked(counter, seed_node):
        return None

    burned = [seed_node]
    burned_set = {seed_node}
    stack = [seed_node]

    while len(burned) < target_size:
        if not stack:
            candidates = [
                n for n in burned
                if any(nb not in burned_set and not _blocked(counter, nb)
                       for nb in g_nx.neighbors(n))
            ]
            if not candidates:
                break
            stack.append(random.choice(candidates))

        node = stack.pop()
        neighbors = [
            n for n in g_nx.neighbors(node)
            if n not in burned_set and not _blocked(counter, n)
        ]
        for n in neighbors:
            if len(burned) >= target_size:
                break
            if random.random() < burn_prob:
                burned_set.add(n)
                burned.append(n)
                stack.append(n)
    return burned


# ---------------------------------------------------------------------------
# Random walk with restart
# ---------------------------------------------------------------------------

def rwr(
    g_nx: nx.Graph,
    target_size: int,
    restart_prob: float,
    counter: Optional[MultiplicityCap] = None,
    seed_node: Optional[int] = None,
) -> Optional[List[int]]:
    """Random walk with restart. Walks until `target_size` unique nodes are
    visited or a step budget is exhausted. The induced subgraph on visited
    nodes is connected by construction (each visit follows an edge from a
    previously visited node)."""
    seeds = _valid_seed_pool(g_nx, counter)
    if not seeds:
        return None
    if seed_node is None:
        seed_node = random.choice(seeds)
    elif _blocked(counter, seed_node):
        return None

    visited_order: List[int] = [seed_node]
    visited_set = {seed_node}
    current = seed_node
    max_steps = max(10_000, target_size * 200)
    steps = 0

    while len(visited_set) < target_size and steps < max_steps:
        steps += 1
        if random.random() < restart_prob:
            current = seed_node
            continue
        nbrs = [n for n in g_nx.neighbors(current) if not _blocked(counter, n)]
        if not nbrs:
            current = seed_node
            continue
        nxt = random.choice(nbrs)
        if nxt not in visited_set:
            visited_set.add(nxt)
            visited_order.append(nxt)
        current = nxt

    return visited_order


# ---------------------------------------------------------------------------
# BFS snowball — bounded BFS from a seed, capping per-node neighbor expansion
# ---------------------------------------------------------------------------

def bfs_snowball(
    g_nx: nx.Graph,
    target_size: Optional[int],
    max_neighbors: int,
    counter: Optional[MultiplicityCap] = None,
    seed_node: Optional[int] = None,
    max_depth: Optional[int] = None,
) -> Optional[List[int]]:
    """BFS where each node expands at most `max_neighbors` un-visited children.

    Stop conditions:
      * `target_size`: stop once that many unique nodes are visited (size-bound).
      * `max_depth`:   children at depth > max_depth are not enqueued (depth-bound).
    At least one of the two must be set; if both are set the walk stops at
    whichever is hit first."""
    if target_size is None and max_depth is None:
        raise ValueError('bfs_snowball requires at least one of target_size or max_depth')

    seeds = _valid_seed_pool(g_nx, counter)
    if not seeds:
        return None
    if seed_node is None:
        seed_node = random.choice(seeds)
    elif _blocked(counter, seed_node):
        return None

    visited_order: List[int] = [seed_node]
    visited_set = {seed_node}
    queue: List[Tuple[int, int]] = [(seed_node, 0)]

    def _size_ok() -> bool:
        return target_size is None or len(visited_order) < target_size

    while queue and _size_ok():
        current, depth = queue.pop(0)
        if max_depth is not None and depth >= max_depth:
            continue
        nbrs = [n for n in g_nx.neighbors(current)
                if n not in visited_set and not _blocked(counter, n)]
        random.shuffle(nbrs)
        for n in nbrs[:max_neighbors]:
            if not _size_ok():
                break
            visited_set.add(n)
            visited_order.append(n)
            queue.append((n, depth + 1))
    return visited_order


# ---------------------------------------------------------------------------
# Uniform random + induced subgraph
# ---------------------------------------------------------------------------

def uniform_random(
    g_nx: nx.Graph,
    target_size: int,
    counter: Optional[MultiplicityCap] = None,
    seed_node: Optional[int] = None,  # unused, kept for signature symmetry
) -> Optional[List[int]]:
    """Pick `target_size` random source nodes (degree-unbiased). Induced
    subgraph + LCC will be applied by `make_partition` — typically yields a
    much smaller subgraph than `target_size` because the induced subgraph
    of a random node set is rarely connected."""
    pool = [n for n in g_nx.nodes() if not _blocked(counter, n)]
    if not pool:
        return None
    k = min(target_size, len(pool))
    return random.sample(pool, k)


# ---------------------------------------------------------------------------
# METIS partitioning — disjoint by construction (M = 1)
# ---------------------------------------------------------------------------

def metis_partition(g_nx: nx.Graph, num_parts: int) -> List[List[int]]:
    """K disjoint partitions of `g_nx` minimizing edge cuts. Returns a list
    of `num_parts` lists, each holding the source-space node IDs in that part.
    Connectivity within each part is *not* guaranteed by METIS; the caller
    runs LCC trimming via `make_partition`."""
    import pymetis
    nodes_sorted = sorted(g_nx.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_sorted)}
    adj = [[node_to_idx[nb] for nb in g_nx.neighbors(v)] for v in nodes_sorted]
    _, membership = pymetis.part_graph(num_parts, adjacency=adj)
    parts: List[List[int]] = [[] for _ in range(num_parts)]
    for v, p in zip(nodes_sorted, membership):
        parts[p].append(v)
    return parts


# ---------------------------------------------------------------------------
# BFS partitioning — vendored from bigg, kept disjoint
# ---------------------------------------------------------------------------

def bfs_partition(
    g_nx: nx.Graph,
    target_size: int,
    max_neighbors: int,
) -> List[List[int]]:
    """Disjoint BFS partitioning. Repeatedly BFS from a random unvisited
    seed (degree ≥ 1 in the not-yet-assigned subgraph) until all assignable
    nodes are in some partition. The previous SynBench default before forest
    fire replaced it; revived here as a structural baseline."""
    remaining = set(g_nx.nodes())
    parts: List[List[int]] = []

    while remaining:
        # Seed must have remaining neighbors so the BFS can grow at all
        sub = g_nx.subgraph(remaining)
        candidates = [n for n in remaining if sub.degree(n) > 0]
        if not candidates:
            # All remaining nodes are isolated within `remaining`. Drop them.
            break
        seed = random.choice(candidates)
        visited = [seed]; vset = {seed}; queue = [seed]
        while queue and len(visited) < target_size:
            cur = queue.pop(0)
            nbrs = [n for n in g_nx.neighbors(cur) if n in remaining and n not in vset]
            random.shuffle(nbrs)
            for n in nbrs[:max_neighbors]:
                if len(visited) >= target_size:
                    break
                vset.add(n); visited.append(n); queue.append(n)
        parts.append(visited)
        remaining -= vset
    return parts


# ---------------------------------------------------------------------------
# Partition assembly — induced subgraph + LCC + relabeling
# ---------------------------------------------------------------------------

Partition = Tuple[nx.Graph, torch.Tensor, torch.Tensor]


def make_partition(
    g_nx: nx.Graph,
    node_data: torch.Tensor,
    picked_nodes: List[int],
) -> Optional[Partition]:
    """Build a (sub_nx, sub_node_data, orig_indices_in_source_space) tuple.

    Steps: induced subgraph on `picked_nodes` → LCC (for connectivity invariant)
    → relabel `0..n-1` in the picked-node order intersected with the LCC.
    Returns None if the resulting LCC has < 2 nodes."""
    if not picked_nodes:
        return None
    induced = g_nx.subgraph(picked_nodes).copy()
    if induced.number_of_nodes() < 2:
        return None
    if not nx.is_connected(induced):
        lcc_nodes = max(nx.connected_components(induced), key=len)
        induced = induced.subgraph(lcc_nodes).copy()
    if induced.number_of_nodes() < 2:
        return None

    # Preserve the picked-node order (burn order, BFS order, walk order, etc.)
    # but filter to those that survived the LCC step.
    in_lcc = set(induced.nodes())
    ordered = [n for n in picked_nodes if n in in_lcc]
    relabel = {old: new for new, old in enumerate(ordered)}
    sub_nx = nx.relabel_nodes(induced, relabel)
    orig_indices = torch.tensor(ordered, dtype=torch.long)
    sub_node_data = node_data[orig_indices]
    return sub_nx, sub_node_data, orig_indices
