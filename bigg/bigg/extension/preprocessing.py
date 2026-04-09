"""Preprocessing utilities for BiGG pipelines.

Standalone functions for loading, converting, normalising, and
post-processing graphs so that the pipeline scripts stay focused on
training and generation.
"""

import random
from collections import deque

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

    Returns the reordered (graph_nx, node_data, perm) tuple where *perm* is the
    permutation tensor used for reordering (can apply to other per-node tensors).
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
    return graph_nx, node_data, perm


# ---------------------------------------------------------------------------
# BFS subsampling
# ---------------------------------------------------------------------------

def bfs_subsample(graph_nx, node_data, target_size, max_neighbors, seed_node=None):
    """Sample a connected subgraph via BFS with a per-node neighbor cap.

    Starting from *seed_node* (random if None), expands outward via BFS.
    At each node, at most *max_neighbors* unvisited neighbors are randomly
    selected and enqueued. Stops once *target_size* nodes are collected.

    Parameters
    ----------
    graph_nx : nx.Graph
        Source graph. Nodes must be integers 0..N-1.
    node_data : torch.Tensor
        Per-node feature/label tensor of shape (N, D).
    target_size : int
        Maximum number of nodes to collect.
    max_neighbors : int
        Maximum neighbors to enqueue from each BFS node.
    seed_node : int or None
        Starting node. Chosen uniformly at random if None.

    Returns
    -------
    sub_graph_nx : nx.Graph
        Induced subgraph with nodes reindexed 0..n-1 in BFS-visit order.
    sub_node_data : torch.Tensor
        Node data rows corresponding to the sampled nodes.
    original_indices : torch.Tensor
        1-D tensor of original node IDs in BFS-visit order.
    """
    nodes = list(graph_nx.nodes())
    if seed_node is None:
        seed_node = random.choice(nodes)

    visited = [seed_node]
    visited_set = {seed_node}
    queue = deque([seed_node])

    while queue and len(visited) < target_size:
        node = queue.popleft()
        neighbors = [n for n in graph_nx.neighbors(node) if n not in visited_set]
        remaining = target_size - len(visited)
        if len(neighbors) > max_neighbors:
            neighbors = random.sample(neighbors, min(max_neighbors, remaining))
        else:
            neighbors = neighbors[:remaining]
        for n in neighbors:
            visited_set.add(n)
            visited.append(n)
            queue.append(n)

    # Induced subgraph reindexed 0..n-1
    subgraph = graph_nx.subgraph(visited).copy()
    mapping = {old: new for new, old in enumerate(visited)}
    subgraph = nx.relabel_nodes(subgraph, mapping)

    original_indices = torch.tensor(visited, dtype=torch.long)
    sub_node_data = node_data[original_indices]

    return subgraph, sub_node_data, original_indices


def partition_graph_bfs(graph_nx, node_data, target_size, max_neighbors):
    """Exhaustively partition *graph_nx* into non-overlapping BFS subgraphs.

    Every node in the original graph appears in exactly one subgraph.
    Subgraphs are formed by successive BFS expansions from random unvisited
    seeds. The final partition may contain a smaller residual subgraph if
    the node count is not evenly divisible by *target_size*.

    Parameters
    ----------
    graph_nx : nx.Graph
        Source graph.
    node_data : torch.Tensor
        Per-node feature/label tensor of shape (N, D).
    target_size : int
        Target number of nodes per subgraph.
    max_neighbors : int
        Per-node BFS neighbor cap (controls edge density).

    Returns
    -------
    list of (sub_graph_nx, sub_node_data, original_indices)
        One tuple per subgraph, in sampling order.
    """
    min_size = max(2, target_size // 4)

    unvisited = set(graph_nx.nodes())
    partitions = []
    n_skipped = 0

    while unvisited:
        seed = random.choice(list(unvisited))
        sub_view = graph_nx.subgraph(unvisited)
        sg, sub_nd, orig_idx = bfs_subsample(
            sub_view, node_data, target_size, max_neighbors, seed_node=seed
        )
        collected = set(orig_idx.tolist())
        unvisited -= collected

        if len(collected) >= min_size:
            partitions.append((sg, sub_nd, orig_idx))
        else:
            # Node(s) are stranded — all their neighbours consumed by earlier
            # partitions. They can't form a meaningful subgraph, so skip them.
            n_skipped += len(collected)

    if n_skipped:
        print(f'  Skipped {n_skipped} stranded node(s) '
              f'(neighborhood fully consumed by other partitions)')

    return partitions


# ---------------------------------------------------------------------------
# Forest fire subsampling
# ---------------------------------------------------------------------------

def forest_fire_subsample(graph_nx, node_data, target_size, burn_prob, seed_node=None):
    """Sample a subgraph via forest fire with burn probability *burn_prob*.

    Starting from *seed_node* (random if None), each unburned neighbor of a
    burning node is independently ignited with probability *burn_prob*.
    Burning propagates recursively until *target_size* nodes are collected.
    If the fire dies before reaching *target_size*, a new random unburned
    seed is chosen and burning continues.

    Parameters
    ----------
    graph_nx : nx.Graph
        Source graph. Nodes must be integers 0..N-1.
    node_data : torch.Tensor
        Per-node feature/label tensor of shape (N, D).
    target_size : int
        Maximum number of nodes to collect.
    burn_prob : float
        Probability of burning each unburned neighbor (controls density).
    seed_node : int or None
        Starting node. Chosen uniformly at random if None.

    Returns
    -------
    sub_graph_nx : nx.Graph
        Induced subgraph with nodes reindexed 0..n-1 in burn order.
    sub_node_data : torch.Tensor
        Node data rows corresponding to the sampled nodes.
    original_indices : torch.Tensor
        1-D tensor of original node IDs in burn order.
    """
    nodes = list(graph_nx.nodes())
    if seed_node is None:
        seed_node = random.choice(nodes)

    burned = [seed_node]
    burned_set = {seed_node}
    stack = [seed_node]

    while len(burned) < target_size:
        if not stack:
            # Fire died — restart from a random unburned node
            unburned = [n for n in nodes if n not in burned_set]
            if not unburned:
                break
            new_seed = random.choice(unburned)
            burned.append(new_seed)
            burned_set.add(new_seed)
            stack.append(new_seed)

        node = stack.pop()
        neighbors = [n for n in graph_nx.neighbors(node) if n not in burned_set]
        for n in neighbors:
            if len(burned) >= target_size:
                break
            if random.random() < burn_prob:
                burned_set.add(n)
                burned.append(n)
                stack.append(n)

    # Induced subgraph reindexed 0..n-1
    subgraph = graph_nx.subgraph(burned).copy()
    mapping = {old: new for new, old in enumerate(burned)}
    subgraph = nx.relabel_nodes(subgraph, mapping)

    original_indices = torch.tensor(burned, dtype=torch.long)
    sub_node_data = node_data[original_indices]

    return subgraph, sub_node_data, original_indices


# ---------------------------------------------------------------------------
# Feature normalisation
# ---------------------------------------------------------------------------

NORMALIZATION_METHODS = ('zscore', 'minmax', 'row')


def normalize_features(features, method):
    """Normalise *features* tensor and return it together with stats.

    Parameters
    ----------
    features : torch.Tensor
        Node feature matrix of shape (N, D).
    method : str
        One of ``'zscore'`` (zero mean, unit variance per feature),
        ``'minmax'`` ([0, 1] scaling), or ``'row'`` (L2 row normalisation).

    Returns
    -------
    tuple[torch.Tensor, dict]
        The normalised feature tensor and a stats dict that can be passed to
        :func:`apply_normalization` to transform other data identically.
        For ``'row'`` normalization the stats dict is empty (transform is
        per-sample, no global stats).
    """
    if method not in NORMALIZATION_METHODS:
        raise ValueError(
            f"Unknown normalisation method '{method}'. "
            f"Choose from {NORMALIZATION_METHODS}."
        )

    stats = {'method': method}

    if method == 'zscore':
        mean = features.mean(dim=0)
        std = features.std(dim=0)
        std[std == 0] = 1.0  # avoid division by zero for constant columns
        features = (features - mean) / std
        stats['mean'] = mean
        stats['std'] = std

    elif method == 'minmax':
        fmin = features.min(dim=0).values
        fmax = features.max(dim=0).values
        denom = fmax - fmin
        denom[denom == 0] = 1.0
        features = (features - fmin) / denom
        stats['min'] = fmin
        stats['denom'] = denom

    elif method == 'row':
        norms = features.norm(p=2, dim=1, keepdim=True)
        norms[norms == 0] = 1.0
        features = features / norms

    return features, stats


def apply_normalization(features, stats):
    """Apply a previously computed normalization to *features*.

    Parameters
    ----------
    features : torch.Tensor
        Node feature matrix of shape (N, D).
    stats : dict
        Stats dict returned by :func:`normalize_features`.

    Returns
    -------
    torch.Tensor
        The normalised feature tensor.
    """
    method = stats['method']

    if method == 'zscore':
        features = (features - stats['mean']) / stats['std']
    elif method == 'minmax':
        features = (features - stats['min']) / stats['denom']
    elif method == 'row':
        norms = features.norm(p=2, dim=1, keepdim=True)
        norms[norms == 0] = 1.0
        features = features / norms

    return features


# ---------------------------------------------------------------------------
# Post-generation: masks & DGL assembly
# ---------------------------------------------------------------------------

def create_split_masks(original_graph, num_nodes):
    """Create random train/val/test masks matching the split ratios of *original_graph*.

    Uses proportions rather than absolute counts so that subgraphs smaller than
    the original graph receive the correct fraction of train/val nodes.

    Returns (train_masks, val_masks, test_masks) each of shape (num_nodes, num_splits).
    """
    num_splits = original_graph.ndata['train_masks'].shape[1]
    orig_n = original_graph.num_nodes()

    train_masks = torch.zeros(num_nodes, num_splits, dtype=torch.uint8)
    val_masks   = torch.zeros(num_nodes, num_splits, dtype=torch.uint8)
    test_masks  = torch.zeros(num_nodes, num_splits, dtype=torch.uint8)

    for col in range(num_splits):
        train_frac = original_graph.ndata['train_masks'][:, col].sum().item() / orig_n
        val_frac   = original_graph.ndata['val_masks'][:, col].sum().item()   / orig_n
        n_train = max(1, round(train_frac * num_nodes))
        n_val   = max(1, round(val_frac   * num_nodes))
        if n_train + n_val > num_nodes:
            n_val = num_nodes - n_train
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
        gen_dgl.ndata['label'] = labels.reshape(-1).long().cpu()
    else:
        gen_dgl.ndata['label'] = torch.zeros(num_nodes, dtype=torch.long)

    train_masks, val_masks, test_masks = create_split_masks(original_graph, num_nodes)
    gen_dgl.ndata['train_masks'] = train_masks
    gen_dgl.ndata['val_masks']   = val_masks
    gen_dgl.ndata['test_masks']  = test_masks

    return gen_dgl
