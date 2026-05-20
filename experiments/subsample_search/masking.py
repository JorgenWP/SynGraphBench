"""Leak-safe assembly: dgl.batch + first-occurrence singular masks."""
from __future__ import annotations

from typing import List, Tuple

import dgl
import networkx as nx
import numpy as np
import torch

from splitsource import SplitSource


Partition = Tuple[nx.Graph, torch.Tensor, torch.Tensor]


def build_combined_graph(
    partitions: List[Partition],
    source: SplitSource,
) -> dgl.DGLGraph:
    """Batch K subgraphs into one DGL graph with leak-safe singular masks.

    Each original node's *first occurrence* in the batched node ordering carries
    train/val supervision (based on source membership); every later duplicate
    copy has all masks zeroed. test_mask is all zeros on the syn side — testing
    happens on the full original graph.
    """
    sub_dgls = []
    for k, (sub_nx, _sub_nd, orig_idx_src) in enumerate(partitions):
        n = sub_nx.number_of_nodes()
        if k == 0:
            assert sorted(sub_nx.nodes()) == list(range(n)), \
                'forest fire is expected to relabel nodes 0..n-1'

        sd = dgl.from_networkx(sub_nx)
        sd.ndata['feature'] = source.features[orig_idx_src].float()
        sd.ndata['label']   = source.labels[orig_idx_src].long()
        sd.ndata['_orig_id_full'] = source.orig_ids[orig_idx_src].long()
        sd.ndata['_orig_idx_src'] = orig_idx_src.long()
        sub_dgls.append(sd)

    combined = dgl.batch(sub_dgls)
    n_total = combined.num_nodes()

    # First-occurrence detection (vectorized via np.unique with return_index)
    flat = combined.ndata['_orig_id_full'].cpu().numpy()
    _, first_pos = np.unique(flat, return_index=True)
    is_first = torch.zeros(n_total, dtype=torch.bool)
    is_first[torch.from_numpy(first_pos).long()] = True

    # Source-space membership lookup
    s = combined.ndata['_orig_idx_src']
    train_at_node = source.is_train[s]
    val_at_node   = source.is_val[s]

    train_mask = (is_first & train_at_node).bool()
    val_mask   = (is_first & val_at_node).bool()
    test_mask  = torch.zeros(n_total, dtype=torch.bool)

    # Stamp public masks + original_indices, drop the underscored helpers
    combined.ndata['train_mask'] = train_mask
    combined.ndata['val_mask']   = val_mask
    combined.ndata['test_mask']  = test_mask
    combined.ndata['original_indices'] = combined.ndata['_orig_id_full']
    del combined.ndata['_orig_id_full']
    del combined.ndata['_orig_idx_src']

    # --- Asserts: leak-safety + non-empty supervision ---
    assert not (train_mask & val_mask).any(), 'train and val masks overlap'
    assert not (train_mask & test_mask).any(), 'train and test masks overlap'
    assert not (val_mask & test_mask).any(),   'val and test masks overlap'

    train_orig = combined.ndata['original_indices'][train_mask].cpu().tolist()
    val_orig   = combined.ndata['original_indices'][val_mask].cpu().tolist()
    assert set(train_orig).isdisjoint(set(val_orig)), \
        'an original node ID appears in both train_mask and val_mask'

    assert int(train_mask.sum()) > 0, 'train_mask is empty'
    assert int(val_mask.sum())   > 0, 'val_mask is empty'

    print(f'  Combined: N={n_total} M={combined.num_edges()} '
          f'train={int(train_mask.sum())} val={int(val_mask.sum())} '
          f'unique_origs={len(set(train_orig) | set(val_orig))}')
    return combined
