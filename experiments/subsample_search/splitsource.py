"""Per-split sampling source: train+val induced subgraph, restricted to its LCC."""
from __future__ import annotations

import os
from dataclasses import dataclass

import dgl
import networkx as nx
import torch


@dataclass
class SplitSource:
    """Sampling source for one GADBench split.

    Indices in `g_nx`/`features`/`labels`/`is_train`/`is_val` are in source-relabeled
    space (0..N_lcc-1, ordered as the LCC subset of the kept train+val nodes).
    `orig_ids[i]` returns the fully-original (full-graph) node id of source node i.
    """
    g_nx: nx.Graph
    orig_ids: torch.Tensor      # (N_lcc,) long
    features: torch.Tensor      # (N_lcc, D) float32
    labels: torch.Tensor        # (N_lcc,) long
    is_train: torch.Tensor      # (N_lcc,) bool
    is_val: torch.Tensor        # (N_lcc,) bool
    orig_dgl: dgl.DGLGraph      # full original graph; ndata['test_mask'] = split column


def load_split_source(dataset: str, split_id: int, data_dir: str) -> SplitSource:
    """Load `data_dir/dataset`, pick `split_id`'s columns, build LCC of (train|val)."""
    path = os.path.join(data_dir, dataset)
    g = dgl.load_graphs(path)[0][0]

    train_col = g.ndata['train_masks'][:, split_id].bool()
    val_col   = g.ndata['val_masks'][:, split_id].bool()
    test_col  = g.ndata['test_masks'][:, split_id]

    tv_keep = (train_col | val_col).nonzero(as_tuple=True)[0]   # (N_tv,)
    n_tv = tv_keep.numel()

    # Some saved DGL datasets expect int32 for node IDs, others int64. Try both.
    try:
        sub = dgl.node_subgraph(g, tv_keep.int())                # nodes 0..N_tv-1
    except dgl._ffi.base.DGLError:
        sub = dgl.node_subgraph(g, tv_keep.long())
    g_nx_tv = dgl.to_networkx(sub).to_undirected()
    g_nx_tv.remove_edges_from(nx.selfloop_edges(g_nx_tv))

    # Largest connected component in the (train+val)-induced subgraph
    components = list(nx.connected_components(g_nx_tv))
    lcc = max(components, key=len)
    lcc_sorted = sorted(lcc)
    n_lcc = len(lcc_sorted)
    print(f'  Split {split_id}: kept {n_tv} train+val nodes; '
          f'LCC = {n_lcc} ({n_lcc/n_tv:.1%}); dropped {n_tv - n_lcc} '
          f'({len(components)} components)')

    g_nx_lcc = g_nx_tv.subgraph(lcc_sorted).copy()
    relabel = {old: new for new, old in enumerate(lcc_sorted)}
    g_nx_lcc = nx.relabel_nodes(g_nx_lcc, relabel)
    assert nx.is_connected(g_nx_lcc), 'LCC must be connected by construction'

    lcc_in_tv = torch.tensor(lcc_sorted, dtype=torch.long)        # tv-space indices
    orig_ids  = tv_keep[lcc_in_tv].long()                          # full-graph ids

    feats = g.ndata['feature'][orig_ids].float()
    labels = g.ndata['label'][orig_ids].long()
    is_train = train_col[orig_ids]
    is_val   = val_col[orig_ids]

    # Install split's test column on the full original graph for the cross-graph detector
    g.ndata['test_mask'] = test_col

    return SplitSource(
        g_nx=g_nx_lcc,
        orig_ids=orig_ids,
        features=feats,
        labels=labels,
        is_train=is_train,
        is_val=is_val,
        orig_dgl=g,
    )
