"""
Computation graph data utilities for CGT-style evaluation.

Provides datasets and helpers for building batched computation graph
trees from either original graph data or CGT-generated synthetic
cluster-center sequences.

These are shared across tasks (anomaly detection, link prediction, etc.)
that need to operate on CGT computation graph trees.
"""

import torch
import numpy as np
import dgl
from torch.utils.data import Dataset
from collections import defaultdict


def compute_tree_adj(step_num, sample_num, self_connection=False):
    """Compute the fixed tree adjacency matrix for computation graphs.

    All computation graphs share the same tree topology determined by
    step_num (depth) and sample_num (branching factor).

    Returns:
        adj: dense adjacency matrix [num_tree_nodes, num_tree_nodes]
             where adj[parent][child] = 1
    """
    sampled_nodes = [0]
    curr_targets = [0]
    edges = defaultdict(list)

    for _ in range(step_num):
        new_targets = []
        for target in curr_targets:
            children = list(range(len(sampled_nodes),
                                  len(sampled_nodes) + sample_num))
            sampled_nodes.extend(children)
            new_targets.extend(children)
            edges[target].extend(children)
        curr_targets = new_targets

    n = len(sampled_nodes)
    rows, cols = [], []
    for parent, children in edges.items():
        for child in children:
            rows.append(parent)
            cols.append(child)

    indices = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)])
    adj = torch.sparse_coo_tensor(
        indices, torch.ones(len(cols)), (n, n)).to_dense()
    if self_connection:
        adj = adj + torch.eye(n)
    return adj


def compute_template_edges(tree_adj):
    """Pre-compute DGL edge arrays from the fixed tree adjacency.

    Extracts child→parent edges (reversed for message passing) and adds
    self-loops (skipped if tree_adj already contains them via
    self_connection=True). Returns tensors that can be offset-replicated
    per batch to avoid redundant per-sample DGL graph construction.

    Returns:
        (src, dst, num_nodes): LongTensors of edge endpoints + node count
    """
    n = tree_adj.shape[0]

    # adj[parent][child] = 1 → reverse to child→parent for message passing
    parent_idx, child_idx = tree_adj.nonzero(as_tuple=True)
    src = child_idx
    dst = parent_idx

    # Add self-loops only if not already present from self_connection
    has_self_loops = tree_adj.diagonal().any()
    if not has_self_loops:
        self_loops = torch.arange(n, dtype=torch.long)
        src = torch.cat([src, self_loops])
        dst = torch.cat([dst, self_loops])

    return src, dst, n


def _build_csr_adj(adj_list, num_nodes):
    """Build CSR-style flat adjacency from a list of neighbor lists.

    Includes a sentinel row at index `num_nodes` (the empty_id) with
    degree 0, so empty parents propagate empty children naturally
    through deeper sampling levels.

    Returns:
        adj_flat:    int64[E] concatenated neighbor IDs.
        adj_offsets: int64[N+2] prefix-sum of degrees (incl. empty_id row).
        degrees:     int64[N+1] degree per node; degrees[empty_id]=0.
    """
    n = num_nodes
    degrees = np.zeros(n + 1, dtype=np.int64)
    for i, nbrs in enumerate(adj_list):
        degrees[i] = len(nbrs)
    adj_offsets = np.zeros(n + 2, dtype=np.int64)
    np.cumsum(degrees, out=adj_offsets[1:])
    adj_flat = np.empty(int(adj_offsets[-1]), dtype=np.int64)
    for i, nbrs in enumerate(adj_list):
        if nbrs:
            start = adj_offsets[i]
            adj_flat[start:start + len(nbrs)] = nbrs
    return adj_flat, adj_offsets, degrees


def _floyd_indices(degrees, k):
    """Return [P, k] of distinct indices in [0, degrees[p]) per parent.

    Vectorized Floyd's algorithm: O(P*k^2) comparisons + O(P*k) PRNG
    draws, independent of max degree. For parents with degree < k, the
    returned values may exceed degrees[p]; the caller masks those rows.
    """
    P = degrees.shape[0]
    eff = np.maximum(degrees, k).astype(np.int64)
    upper = eff[:, None] - k + np.arange(k, dtype=np.int64)[None, :] + 1
    t = (np.random.random((P, k)) * upper).astype(np.int64)
    out = np.empty((P, k), dtype=np.int64)
    out[:, 0] = t[:, 0]
    for j in range(1, k):
        collision = (out[:, :j] == t[:, j:j + 1]).any(axis=1)
        replacement = eff - k + j
        out[:, j] = np.where(collision, replacement, t[:, j])
    return out


def _sample_neighbors_csr(parents, k, adj_flat, adj_offsets, degrees,
                          empty_id):
    """Per parent, sample k neighbor IDs uniformly without replacement.

    Pads with empty_id when degree < k. Three branches:
      A. degree >= k:    Floyd's index sampler + CSR gather.
      B. 0 < degree < k: take all real neighbors, pad rest with empty_id.
      C. degree == 0:    all empty_id (default).
    """
    P = parents.shape[0]
    d = degrees[parents]
    out = np.full((P, k), empty_id, dtype=np.int64)

    rich = d >= k
    if rich.any():
        rd = d[rich]
        idx = _floyd_indices(rd, k)
        flat = adj_offsets[parents[rich]][:, None] + idx
        out[rich] = adj_flat[flat]

    short = (d > 0) & (d < k)
    if short.any():
        sd = d[short]
        max_sd = int(sd.max())
        col = np.arange(max_sd, dtype=np.int64)[None, :]
        in_bounds = col < sd[:, None]
        flat = adj_offsets[parents[short]][:, None] + np.where(
            in_bounds, col, 0)
        gathered = adj_flat[flat]
        gathered = np.where(in_bounds, gathered, empty_id)
        out[short, :max_sd] = gathered

    return out


def _sample_tree_batch(seeds, step_num, sample_num, noise_num, node_num,
                       adj_flat, adj_offsets, degrees, empty_id):
    """Batched tree sampling. Returns [B, T] node IDs per seed.

    Each parent produces (sample_num + noise_num) children per level.
    Output ordering matches the v3 per-edge sampler (BFS, siblings of
    one parent contiguous), so it lines up with the template tree
    adjacency from `compute_tree_adj` / `compute_merged_tree_adj`.
    """
    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
    B = seeds.shape[0]
    total = sample_num + noise_num
    nodes = [seeds.reshape(B, 1)]
    parents = seeds
    parents_per_tree = 1
    for _ in range(step_num):
        P = parents.shape[0]
        children = _sample_neighbors_csr(
            parents, sample_num, adj_flat, adj_offsets, degrees, empty_id)
        if noise_num > 0:
            # noise_num is 0 in active link/AD paths; with-replacement
            # noise here vs v3's permutation-based without-replacement is
            # statistically negligible for noise_num << node_num.
            noise = np.random.randint(
                0, node_num, size=(P, noise_num), dtype=np.int64)
            children = np.concatenate([children, noise], axis=1)
        nodes.append(children.reshape(B, parents_per_tree * total))
        parents = children.reshape(-1)
        parents_per_tree *= total
    return np.concatenate(nodes, axis=1)


class OriginalCompGraphDataset(Dataset):
    """Build computation graph trees from original graph data.

    For each node, samples a fixed-structure tree of neighbors and
    returns the tree's features, adjacency, and the root node's label.
    """

    def __init__(self, adj_list, features, labels, node_ids,
                 step_num, sample_num, noise_num=0, self_connection=False):
        self.labels = labels
        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.node_num = features.shape[0]
        self.step_num = step_num
        self.sample_num = sample_num
        self.noise_num = noise_num
        self.total_sample = sample_num + noise_num
        self.self_connection = self_connection

        # Pad features with a zero row for empty/missing neighbors
        self.features = np.concatenate(
            [features, np.zeros((1, features.shape[1]), dtype=features.dtype)])
        self.empty_id = features.shape[0]

        # CSR-style adjacency for vectorized batch sampling.
        self.adj_flat, self.adj_offsets, self.degrees = _build_csr_adj(
            adj_list, self.node_num)

        tree_adj = compute_tree_adj(
            step_num, self.total_sample, self_connection)
        self.template_src, self.template_dst, self.num_tree_nodes = \
            compute_template_edges(tree_adj)

    def __len__(self):
        return len(self.node_ids)

    def get_labels(self):
        return self.labels[self.node_ids]

    def __getitems__(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        seeds = self.node_ids[idx]
        trees = _sample_tree_batch(
            seeds, self.step_num, self.sample_num, self.noise_num,
            self.node_num, self.adj_flat, self.adj_offsets, self.degrees,
            self.empty_id)
        feats = self.features[trees]
        feats_t = torch.from_numpy(np.ascontiguousarray(feats))
        return [{
            "feat": feats_t[i],
            "label": torch.LongTensor([self.labels[int(seeds[i])]]),
        } for i in range(len(idx))]

    def __getitem__(self, index):
        return self.__getitems__([index])[0]


class SyntheticCompGraphDataset(Dataset):
    """CGT synthetic computation graph dataset.

    Maps generated cluster ID sequences to feature vectors via cluster
    centers, paired with a fixed tree adjacency matrix. Equivalent to
    CGT's QuantizedDataset.
    """

    def __init__(self, sequences, labels, cluster_centers,
                 step_num, sample_num, self_connection=False):
        self.sequences = sequences
        self.labels = labels
        # Add a zero row for potential empty_id entries in sequences
        self.cluster_centers = torch.cat([
            cluster_centers.float(),
            torch.zeros(1, cluster_centers.shape[1]),
        ], dim=0)
        tree_adj = compute_tree_adj(step_num, sample_num, self_connection)
        self.template_src, self.template_dst, self.num_tree_nodes = \
            compute_template_edges(tree_adj)

    def __len__(self):
        return len(self.sequences)

    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        return {
            "feat": self.cluster_centers[self.sequences[index]],
            "label": torch.LongTensor([self.labels[index]]),
        }


def make_comp_graph_collate(template_src, template_dst, num_tree_nodes):
    """Create a collate function using pre-computed template edges.

    Instead of building separate DGL graphs per sample then batching,
    offsets the template edges for each sample and constructs a single
    DGL graph for the whole batch.
    """
    num_edges = len(template_src)

    def collate(items):
        B = len(items)
        all_feats = torch.cat([item['feat'] for item in items], dim=0)
        all_labels = torch.cat([item['label'] for item in items])

        # Offset template edges for each sample in the batch
        offsets = torch.arange(B, dtype=torch.long) * num_tree_nodes
        src = (template_src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst = (template_dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)

        g = dgl.graph((src, dst), num_nodes=B * num_tree_nodes)
        g.ndata['feature'] = all_feats
        g.set_batch_num_nodes(
            torch.full((B,), num_tree_nodes, dtype=torch.long))
        g.set_batch_num_edges(
            torch.full((B,), num_edges, dtype=torch.long))

        return g, all_labels

    return collate


def extract_root_logits(batched_graph, all_logits):
    """Extract root node (node 0 of each sub-graph) logits."""
    num_nodes = batched_graph.batch_num_nodes()
    root_ids = torch.zeros(
        len(num_nodes), dtype=torch.long, device=all_logits.device)
    root_ids[1:] = torch.cumsum(num_nodes[:-1], dim=0)
    return all_logits[root_ids]


def compute_merged_tree_adj(step_num, sample_num, self_connection=False):
    """Paper-style merged computation graph for a pair of endpoint nodes.

    Two disjoint trees share a root-root edge: root_u at index 0, root_v
    at index T, bidirectional edge between them. Message passing can flow
    from v's neighborhood into h_u (and vice versa) through that edge.
    """
    single = compute_tree_adj(step_num, sample_num, self_connection=False)
    T = single.shape[0]
    merged = torch.zeros(2 * T, 2 * T, dtype=single.dtype)
    merged[:T, :T] = single
    merged[T:, T:] = single
    merged[0, T] = 1
    merged[T, 0] = 1
    if self_connection:
        merged = merged + torch.eye(2 * T)
    return merged


class MergedOriginalCompGraphDataset(Dataset):
    """Per-edge merged computation graphs built from an original graph.

    For each edge (u, v): samples a depth-`step_num`, fanout-`sample_num`
    tree rooted at u, a second tree rooted at v, and stacks their features
    as `[2 * T, feat_dim]`. The merged-tree adjacency (from
    `compute_merged_tree_adj`) wires the two root nodes together so the
    GNN forward pass sees both neighborhoods in a single message-passing
    graph.
    """

    def __init__(self, adj_list, features, edges, step_num, sample_num,
                 noise_num=0, self_connection=False):
        if torch.is_tensor(edges):
            edges = edges.cpu().numpy()
        self.edges = np.asarray(edges, dtype=np.int64)
        self.step_num = step_num
        self.sample_num = sample_num
        self.noise_num = noise_num
        self.total_sample = sample_num + noise_num
        self.node_num = features.shape[0]
        self.self_connection = self_connection

        # Pad features with a zero row for empty/missing neighbors
        self.features = np.concatenate(
            [features, np.zeros((1, features.shape[1]), dtype=features.dtype)])
        self.empty_id = features.shape[0]

        # CSR-style adjacency for vectorized batch sampling.
        self.adj_flat, self.adj_offsets, self.degrees = _build_csr_adj(
            adj_list, self.node_num)

        merged = compute_merged_tree_adj(
            step_num, self.total_sample, self_connection)
        self.template_src, self.template_dst, self.num_merged_nodes = \
            compute_template_edges(merged)
        self.per_tree_num_nodes = self.num_merged_nodes // 2

    def __len__(self):
        return len(self.edges)

    def __getitems__(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        edges_batch = self.edges[idx]                  # [B, 2]
        B = edges_batch.shape[0]
        # Concatenate u and v seeds so both trees sample in one call.
        seeds = np.concatenate([edges_batch[:, 0], edges_batch[:, 1]])
        trees = _sample_tree_batch(
            seeds, self.step_num, self.sample_num, self.noise_num,
            self.node_num, self.adj_flat, self.adj_offsets, self.degrees,
            self.empty_id)                             # [2B, T]
        nodes = np.concatenate([trees[:B], trees[B:]], axis=1)  # [B, 2T]
        feats = self.features[nodes]                   # [B, 2T, F]
        feats_t = torch.from_numpy(np.ascontiguousarray(feats))
        return [{"feat": feats_t[i]} for i in range(B)]

    def __getitem__(self, index):
        return self.__getitems__([index])[0]


def make_merged_comp_graph_collate(template_src, template_dst, num_merged_nodes):
    """Collate per-edge merged CGs into one DGL batch graph.

    Mirrors `make_comp_graph_collate`; the only difference is that each
    sample contributes `num_merged_nodes = 2 * T` nodes and the template
    edge set already includes the root-root edge.
    """
    num_edges = len(template_src)

    def collate(items):
        B = len(items)
        all_feats = torch.cat([item['feat'] for item in items], dim=0)

        offsets = torch.arange(B, dtype=torch.long) * num_merged_nodes
        src = (template_src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst = (template_dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)

        g = dgl.graph((src, dst), num_nodes=B * num_merged_nodes)
        g.ndata['feature'] = all_feats
        g.set_batch_num_nodes(
            torch.full((B,), num_merged_nodes, dtype=torch.long))
        g.set_batch_num_edges(
            torch.full((B,), num_edges, dtype=torch.long))
        return g

    return collate


def extract_edge_root_embeddings(batched_graph, all_logits, per_tree_num_nodes):
    """Return (h_u, h_v) at positions 0 and T of each merged sub-graph."""
    num_nodes = batched_graph.batch_num_nodes()
    starts = torch.zeros(
        len(num_nodes), dtype=torch.long, device=all_logits.device)
    starts[1:] = torch.cumsum(num_nodes[:-1], dim=0)
    h_u = all_logits[starts]
    h_v = all_logits[starts + per_tree_num_nodes]
    return h_u, h_v


def dgl_to_adj_list(graph):
    """Convert a DGL graph to an adjacency list (list of neighbor lists).

    For each node, collects all nodes with an edge pointing to it.
    Suitable for undirected DGL graphs where edges appear in both directions.
    """
    num_nodes = graph.num_nodes()
    src, dst = graph.edges()
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()

    adj_list = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adj_list[int(d)].append(int(s))
    return adj_list
