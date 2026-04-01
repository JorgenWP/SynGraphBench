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


class OriginalCompGraphDataset(Dataset):
    """Build computation graph trees from original graph data.

    For each node, samples a fixed-structure tree of neighbors and
    returns the tree's features, adjacency, and the root node's label.
    """

    def __init__(self, adj_list, features, labels, node_ids,
                 step_num, sample_num, noise_num=0, self_connection=False):
        self.adj_list = adj_list
        self.labels = labels
        self.node_ids = node_ids
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

        tree_adj = compute_tree_adj(
            step_num, self.total_sample, self_connection)
        self.template_src, self.template_dst, self.num_tree_nodes = \
            compute_template_edges(tree_adj)

    def __len__(self):
        return len(self.node_ids)

    def get_labels(self):
        return self.labels[self.node_ids]

    def __getitem__(self, index):
        seed_id = self.node_ids[index]
        sampled = [seed_id]
        curr_targets = [seed_id]

        for _ in range(self.step_num):
            new_targets = []
            for tid in curr_targets:
                if tid == self.empty_id:
                    neighbors = []
                else:
                    neighbors = self.adj_list[tid]

                if len(neighbors) == 0:
                    picked = [self.empty_id] * self.sample_num
                elif len(neighbors) < self.sample_num:
                    picked = neighbors + [self.empty_id] * (
                        self.sample_num - len(neighbors))
                else:
                    picked = np.random.choice(
                        neighbors, self.sample_num, replace=False).tolist()

                if self.noise_num > 0:
                    noise = np.random.permutation(
                        self.node_num)[:self.noise_num].tolist()
                    picked = picked + noise

                sampled.extend(picked)
                new_targets.extend(picked)
            curr_targets = new_targets

        return {
            "feat": torch.FloatTensor(self.features[sampled]),
            "label": torch.LongTensor([self.labels[seed_id]]),
        }


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
