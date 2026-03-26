"""Preprocessing utilities for BiGG pipelines.

Standalone functions for loading, converting, normalising, and
post-processing graphs so that the pipeline scripts stay focused on
training and generation.
"""

import dgl
import torch
import networkx as nx


# ---------------------------------------------------------------------------
# Loading & conversion
# ---------------------------------------------------------------------------

def load_dgl_graph(dataset, base_path='../datasets/original/'):
    """Load first DGL graph from *base_path/dataset*."""
    graphs, _ = dgl.load_graphs(base_path + dataset)
    return graphs[0]


def dgl_to_networkx(graph):
    """Convert a DGL graph to an undirected NetworkX graph without self-loops."""
    graph_nx = nx.Graph(graph.to_networkx().to_undirected())
    graph_nx.remove_edges_from(nx.selfloop_edges(graph_nx))
    return graph_nx


# ---------------------------------------------------------------------------
# BFS reordering
# ---------------------------------------------------------------------------

def bfs_reorder(graph_nx, node_data):
    """Reorder *graph_nx* and *node_data* by BFS from the highest-degree node.

    Returns the reordered (graph_nx, node_data) tuple.
    """
    start_node = max(graph_nx.degree(), key=lambda x: x[1])[0]
    bfs_order = list(nx.bfs_tree(graph_nx, source=start_node).nodes())
    # Include disconnected nodes not reached by BFS
    remaining = [n for n in graph_nx.nodes() if n not in set(bfs_order)]
    bfs_order += remaining

    mapping = {old: new for new, old in enumerate(bfs_order)}
    graph_nx = nx.relabel_nodes(graph_nx, mapping)

    perm = torch.tensor(bfs_order, dtype=torch.long)
    node_data = node_data[perm]

    print(f'Applied BFS ordering from node {start_node} '
          f'(degree {graph_nx.degree(mapping[start_node])})')
    return graph_nx, node_data


# ---------------------------------------------------------------------------
# Feature normalisation
# ---------------------------------------------------------------------------

NORMALIZATION_METHODS = ('zscore', 'minmax', 'row')


def normalize_features(features, method):
    """Normalise *features* tensor in-place and return it.

    Parameters
    ----------
    features : torch.Tensor
        Node feature matrix of shape (N, D).
    method : str
        One of ``'zscore'`` (zero mean, unit variance per feature),
        ``'minmax'`` ([0, 1] scaling), or ``'row'`` (L2 row normalisation).

    Returns
    -------
    torch.Tensor
        The normalised feature tensor.
    """
    if method not in NORMALIZATION_METHODS:
        raise ValueError(
            f"Unknown normalisation method '{method}'. "
            f"Choose from {NORMALIZATION_METHODS}."
        )

    if method == 'zscore':
        mean = features.mean(dim=0)
        std = features.std(dim=0)
        std[std == 0] = 1.0  # avoid division by zero for constant columns
        features = (features - mean) / std

    elif method == 'minmax':
        fmin = features.min(dim=0).values
        fmax = features.max(dim=0).values
        denom = fmax - fmin
        denom[denom == 0] = 1.0
        features = (features - fmin) / denom

    elif method == 'row':
        norms = features.norm(p=2, dim=1, keepdim=True)
        norms[norms == 0] = 1.0
        features = features / norms

    return features


# ---------------------------------------------------------------------------
# Post-generation: masks & DGL assembly
# ---------------------------------------------------------------------------

def create_split_masks(original_graph, num_nodes):
    """Create random train/val/test masks matching the split sizes of *original_graph*.

    Returns (train_masks, val_masks, test_masks) each of shape (num_nodes, num_splits).
    """
    num_splits = original_graph.ndata['train_masks'].shape[1]

    train_masks = torch.zeros(num_nodes, num_splits, dtype=torch.uint8)
    val_masks   = torch.zeros(num_nodes, num_splits, dtype=torch.uint8)
    test_masks  = torch.zeros(num_nodes, num_splits, dtype=torch.uint8)

    for col in range(num_splits):
        n_train = int(original_graph.ndata['train_masks'][:, col].sum().item())
        n_val   = int(original_graph.ndata['val_masks'][:, col].sum().item())
        perm = torch.randperm(num_nodes)
        train_masks[perm[:n_train],              col] = 1
        val_masks  [perm[n_train:n_train+n_val], col] = 1

    return train_masks, val_masks, test_masks


def build_generated_dgl(gen_nx, original_graph, features=None, labels=None):
    """Assemble a DGL graph from a generated NetworkX graph with masks.

    If *features* / *labels* are ``None`` placeholder zeros are used (for
    structure-only generation).
    """
    num_nodes = gen_nx.number_of_nodes()
    gen_dgl = dgl.from_networkx(gen_nx)

    if features is not None:
        gen_dgl.ndata['feature'] = features.cpu()
    else:
        feat_dim = original_graph.ndata['feature'].shape[1]
        gen_dgl.ndata['feature'] = torch.zeros(num_nodes, feat_dim)

    if labels is not None:
        gen_dgl.ndata['label'] = labels.squeeze().long().cpu()
    else:
        gen_dgl.ndata['label'] = torch.zeros(num_nodes, dtype=torch.long)

    train_masks, val_masks, test_masks = create_split_masks(original_graph, num_nodes)
    gen_dgl.ndata['train_masks'] = train_masks
    gen_dgl.ndata['val_masks']   = val_masks
    gen_dgl.ndata['test_masks']  = test_masks

    return gen_dgl
