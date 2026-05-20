"""Sampling dispatch — exposes `sample_partitions` (validation, fixed forest fire
on bigg's primitive) and `sample_with_method` (grid driver, multi-method)."""
from __future__ import annotations

import os
import random
import sys
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

# Make bigg's preprocessing importable for the validation reproduction path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
_BIGG_EXT = os.path.join(_PROJECT_ROOT, 'bigg', 'bigg', 'extension')
if _BIGG_EXT not in sys.path:
    sys.path.insert(0, _BIGG_EXT)

from preprocessing import forest_fire_subsample  # noqa: E402

from sampling_methods import (
    MultiplicityCap,
    forest_fire as _ff,
    rwr as _rwr,
    bfs_snowball as _bfs_snowball,
    uniform_random as _uniform,
    metis_partition as _metis,
    bfs_partition as _bfs_partition,
    make_partition,
)


Partition = Tuple[nx.Graph, torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Validation reproduction path — bigg's forest fire, no M-cap
# ---------------------------------------------------------------------------

def sample_partitions(
    g_nx: nx.Graph,
    node_data: torch.Tensor,
    K: int,
    target_size: int,
    burn_prob: float,
    seed: int,
) -> List[Partition]:
    """K fresh forest-fire draws via bigg's primitive (validation gate)."""
    random.seed(seed)
    torch.manual_seed(seed)

    parts: List[Partition] = []
    for k in range(K):
        sub_nx, sub_nd, orig_idx_src = forest_fire_subsample(
            g_nx, node_data, target_size=target_size, burn_prob=burn_prob)

        n = sub_nx.number_of_nodes()
        m = sub_nx.number_of_edges()
        if n < 2:
            print(f'  WARNING: subgraph {k} has {n} node(s); skipping.')
            continue

        assert nx.is_connected(sub_nx), f'subgraph {k} not connected'
        assert min(deg for _, deg in sub_nx.degree()) >= 1, \
            f'subgraph {k} has isolated node(s)'
        parts.append((sub_nx, sub_nd, orig_idx_src))

    _print_summary('forest_fire (bigg)', parts, K)
    return parts


# ---------------------------------------------------------------------------
# Grid path — dispatches to sampling_methods.* by method name
# ---------------------------------------------------------------------------

def sample_with_method(
    method: str,
    g_nx: nx.Graph,
    node_data: torch.Tensor,
    K: int,
    seed: int,
    multiplicity_cap: Optional[int] = None,
    *,
    target_size: Optional[int] = None,
    burn_prob: Optional[float] = None,
    restart_prob: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    max_depth: Optional[int] = None,
) -> List[Partition]:
    """Run the requested sampling method K times (or partition once for METIS /
    bfs_partition) and return the assembled partition list.

    method ∈ {'forest_fire', 'rwr', 'bfs_snowball', 'uniform',
              'metis', 'bfs_partition'}.
    For partition methods (`metis`, `bfs_partition`), `K` is interpreted as the
    number of partitions. `multiplicity_cap` is ignored (M = 1 by construction)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_source = g_nx.number_of_nodes()

    # ------- Disjoint, partition-once methods -------
    if method == 'metis':
        raw = _metis(g_nx, num_parts=K)
        parts = _assemble(g_nx, node_data, raw)
        _print_summary('metis', parts, K)
        return parts

    if method == 'bfs_partition':
        if target_size is None or max_neighbors is None:
            raise ValueError('bfs_partition requires target_size and max_neighbors')
        raw = _bfs_partition(g_nx, target_size=target_size, max_neighbors=max_neighbors)
        parts = _assemble(g_nx, node_data, raw)
        _print_summary('bfs_partition', parts, len(raw))
        return parts

    # ------- Iterative seed-based methods -------
    counter = (MultiplicityCap(multiplicity_cap, n_source)
               if multiplicity_cap is not None else None)

    parts: List[Partition] = []
    for _ in range(K):
        if method == 'forest_fire':
            picked = _ff(g_nx, target_size=target_size, burn_prob=burn_prob,
                         counter=counter)
        elif method == 'rwr':
            picked = _rwr(g_nx, target_size=target_size, restart_prob=restart_prob,
                          counter=counter)
        elif method == 'bfs_snowball':
            picked = _bfs_snowball(g_nx, target_size=target_size,
                                   max_neighbors=max_neighbors, counter=counter,
                                   max_depth=max_depth)
        elif method == 'uniform':
            picked = _uniform(g_nx, target_size=target_size, counter=counter)
        else:
            raise ValueError(f'Unknown method: {method!r}')

        if picked is None:
            continue  # exhausted (M-cap closed off seed pool)
        p = make_partition(g_nx, node_data, picked)
        if p is None:
            continue
        if counter is not None:
            _, _, orig_indices = p
            counter.visit(orig_indices.tolist())
        parts.append(p)

    _print_summary(method, parts, K)
    return parts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assemble(g_nx, node_data, raw_lists) -> List[Partition]:
    parts: List[Partition] = []
    for picked in raw_lists:
        p = make_partition(g_nx, node_data, picked)
        if p is None:
            continue
        sub_nx, _, _ = p
        # Per-subgraph connectivity invariant (post-LCC)
        assert nx.is_connected(sub_nx), 'partition has multiple components after LCC'
        assert min(d for _, d in sub_nx.degree()) >= 1, 'partition has isolated node(s)'
        parts.append(p)
    return parts


def _print_summary(label: str, parts: List[Partition], K: int) -> None:
    if not parts:
        print(f'  {label}: 0/{K} subgraphs (all draws produced empty/singleton output)')
        return
    sizes = [sg.number_of_nodes() for sg, _, _ in parts]
    edges = [sg.number_of_edges() for sg, _, _ in parts]
    sm = sorted(sizes); em = sorted(edges)
    print(f'  {label}: {len(parts)}/{K} subgraphs '
          f'(n: min={sm[0]} med={sm[len(sm)//2]} max={sm[-1]} | '
          f'm: min={em[0]} med={em[len(em)//2]} max={em[-1]})')
