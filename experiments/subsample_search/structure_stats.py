"""Structural fidelity stats per (config, split, seed)."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from splitsource import SplitSource


Partition = Tuple[Any, torch.Tensor, torch.Tensor]


def compute_subgraph_stats(
    partitions: List[Partition],
    source: SplitSource,
) -> Dict[str, Any]:
    """Aggregate structural statistics across the K partitions.

    Compared against `source` (LCC of train+val) so the prior_shift /
    deg_ratio columns are interpretable per dataset."""
    if not partitions:
        return {
            'num_subgraphs': 0,
            'total_nodes': 0,
            'total_edges': 0,
            'unique_origs': 0,
            'coverage': 0.0,
            'multiplicity_max': 0,
            'multiplicity_p95': 0,
            'mean_size': 0.0,
            'size_min': 0,
            'size_p25': 0.0,
            'size_p75': 0.0,
            'size_max': 0,
            'size_std': 0.0,
            'mean_avg_deg': 0.0,
            'pos_rate_sub': float('nan'),
            'pos_rate_source': float('nan'),
            'prior_shift': float('nan'),
            'deg_ratio': float('nan'),
            'train_unique': 0,
            'val_unique': 0,
        }

    sizes = [sg.number_of_nodes() for sg, _, _ in partitions]
    edges = [sg.number_of_edges() for sg, _, _ in partitions]
    avg_degs = [2 * m / max(n, 1) for n, m in zip(sizes, edges)]

    # Multiplicity (in source-space — counts how often each kept original appears)
    flat_orig_src: List[int] = []
    for _, _, orig_idx in partitions:
        flat_orig_src.extend(orig_idx.tolist())
    cnt = Counter(flat_orig_src)
    mult_values = list(cnt.values())
    mult_max = max(mult_values) if mult_values else 0
    mult_p95 = int(np.percentile(mult_values, 95)) if mult_values else 0
    unique_origs = len(cnt)

    # Source-side reference quantities
    n_source = source.orig_ids.numel()
    src_pos_rate = float(source.labels.float().mean())
    src_avg_deg = (2 * source.g_nx.number_of_edges() / max(n_source, 1))

    # Subgraph-side label rate (across all batched copies — same as what the
    # downstream classifier sees as supervision)
    sub_labels: List[int] = []
    for _, sub_nd, orig_idx in partitions:
        # sub_nd carries [features..., label] from the source; pull label column
        # via orig_idx into source.labels for the cleanest read
        sub_labels.extend(source.labels[orig_idx].tolist())
    sub_pos_rate = float(np.mean(sub_labels)) if sub_labels else float('nan')

    # Train / val coverage (unique originals across the K partitions)
    unique_set = set(flat_orig_src)
    train_unique = sum(int(source.is_train[i]) for i in unique_set)
    val_unique = sum(int(source.is_val[i]) for i in unique_set)

    return {
        'num_subgraphs': len(partitions),
        'total_nodes': sum(sizes),
        'total_edges': sum(edges),
        'unique_origs': unique_origs,
        'coverage': unique_origs / max(n_source, 1),
        'multiplicity_max': mult_max,
        'multiplicity_p95': mult_p95,
        'mean_size': float(np.mean(sizes)),
        'size_min': int(np.min(sizes)),
        'size_p25': float(np.percentile(sizes, 25)),
        'size_p75': float(np.percentile(sizes, 75)),
        'size_max': int(np.max(sizes)),
        'size_std': float(np.std(sizes)),
        'mean_avg_deg': float(np.mean(avg_degs)),
        'pos_rate_sub': sub_pos_rate,
        'pos_rate_source': src_pos_rate,
        'prior_shift': (sub_pos_rate / src_pos_rate - 1.0) if src_pos_rate > 0 else float('nan'),
        'deg_ratio': float(np.mean(avg_degs)) / src_avg_deg if src_avg_deg > 0 else float('nan'),
        'train_unique': train_unique,
        'val_unique': val_unique,
    }
