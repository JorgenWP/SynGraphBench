import logging
import random
import torch
import dgl
import numpy as np
import networkx as nx
from dgl.data.utils import load_graphs
from models.link_prediction.link_predictor import BaseGNNLinkPredictor

logger = logging.getLogger(__name__)


class LinkDataset:
    def __init__(self, name='tfinance', prefix='datasets/'):
        graph = load_graphs(prefix + name)[0][0]
        self.name = name
        self.graph = graph.long()

    def split(self, val_ratio=0.05, test_ratio=0.1, trial_id=0,
              neg_sampling='random'):
        torch.manual_seed(3407 + trial_id * 10)

        src, dst = self.graph.edges()
        E = src.shape[0]

        # Find spanning tree edges to protect graph connectivity
        nx_graph = dgl.to_networkx(self.graph).to_undirected()
        tree_edges_nx = set(nx.minimum_spanning_tree(nx_graph).edges())

        # Mark each DGL edge as tree or non-tree
        tree_mask = torch.zeros(E, dtype=torch.bool)
        for i in range(E):
            u, v = src[i].item(), dst[i].item()
            if (u, v) in tree_edges_nx or (v, u) in tree_edges_nx:
                tree_mask[i] = True

        tree_idx = torch.where(tree_mask)[0]
        candidate_idx = torch.where(~tree_mask)[0]

        # Shuffle candidates and split into val/test/train
        n_candidates = candidate_idx.shape[0]
        perm = torch.randperm(n_candidates)
        candidate_idx = candidate_idx[perm]

        n_test_target = int(E * test_ratio)
        n_val_target = int(E * val_ratio)

        n_test = min(n_test_target, n_candidates)
        n_val = min(n_val_target, n_candidates - n_test)

        if n_test < n_test_target or n_val < n_val_target:
            logger.warning(
                "Dataset '%s': graph too sparse for requested split ratios. "
                "Only %d/%d non-tree edges available (need %d test + %d val). "
                "Using %d test, %d val edges instead.",
                self.name, n_candidates, E,
                n_test_target, n_val_target, n_test, n_val)

        test_idx = candidate_idx[:n_test]
        val_idx = candidate_idx[n_test:n_test + n_val]
        extra_train_idx = candidate_idx[n_test + n_val:]

        # Training edges = spanning tree + remaining non-tree edges
        train_idx = torch.cat([tree_idx, extra_train_idx])

        self.test_pos_edges = torch.stack([src[test_idx], dst[test_idx]], dim=1)
        self.val_pos_edges = torch.stack([src[val_idx], dst[val_idx]], dim=1)
        self.train_pos_edges = torch.stack([src[train_idx], dst[train_idx]], dim=1)

        # Build training graph without val/test edges
        eids = torch.cat([test_idx, val_idx])
        self.train_graph = dgl.add_self_loop(dgl.remove_edges(self.graph, eids))

        # Build edge set for filtering false negatives
        N = self.graph.num_nodes()
        self.edge_set = self._build_edge_set(self.graph, N)

        # Sample fixed negative edges for val/test (consistent evaluation)
        # Training negatives are re-sampled each epoch in the detector
        if neg_sampling == 'hard':
            self.val_neg_edges = self._sample_hard_negatives(
                self.val_pos_edges, self.graph)
            self.test_neg_edges = self._sample_hard_negatives(
                self.test_pos_edges, self.graph)
        else:
            self.val_neg_edges = self._sample_negatives(
                self.val_pos_edges.shape[0], N)
            self.test_neg_edges = self._sample_negatives(
                self.test_pos_edges.shape[0], N)

        n_train = self.train_pos_edges.shape[0]
        n_val_actual = self.val_pos_edges.shape[0]
        n_test_actual = self.test_pos_edges.shape[0]
        logger.info(
            "Dataset '%s' trial %d: %d total edges -> "
            "train %d (%.1f%%), val %d (%.1f%%), test %d (%.1f%%)",
            self.name, trial_id, E,
            n_train, 100 * n_train / E,
            n_val_actual, 100 * n_val_actual / E,
            n_test_actual, 100 * n_test_actual / E)

    @staticmethod
    def _build_edge_set(graph, num_nodes):
        """Build a sorted tensor of edge hashes for vectorized collision checking."""
        src, dst = graph.edges()
        hashes = src.long() * num_nodes + dst.long()
        return hashes.sort()[0]

    def _filter_collisions(self, neg_edges, num_nodes):
        """Replace negative edges that collide with real edges."""
        src, dst = neg_edges[:, 0], neg_edges[:, 1]
        for attempt in range(10):
            hashes = src.long() * num_nodes + dst.long()
            # Check collisions: existing edge or self-loop
            collision = torch.isin(hashes, self.edge_set) | (src == dst)
            if not collision.any():
                break
            # Resample only colliding destinations
            n_bad = collision.sum().item()
            dst[collision] = torch.randint(0, num_nodes, (n_bad,))
        return neg_edges

    def _sample_negatives(self, n, num_nodes):
        """Sample random negative edges guaranteed to be non-edges."""
        neg_src = torch.randint(0, num_nodes, (n,))
        neg_dst = torch.randint(0, num_nodes, (n,))
        neg_edges = torch.stack([neg_src, neg_dst], dim=1)
        return self._filter_collisions(neg_edges, num_nodes)

    def _sample_hard_negatives(self, pos_edges, graph):
        """Sample hard negatives via 2-hop random walks, guaranteed non-edges."""
        src = pos_edges[:, 0]
        N = graph.num_nodes()

        walk_nodes, _ = dgl.sampling.random_walk(graph, src, metapath=[None, None])
        hard_dst = walk_nodes[:, 2]

        # Replace failed walks (-1) with random nodes
        failed = hard_dst == -1
        if failed.any():
            hard_dst[failed] = torch.randint(0, N, (failed.sum(),))

        neg_edges = torch.stack([src, hard_dst], dim=1)
        return self._filter_collisions(neg_edges, N)


model_lp_dict = {
    'GCN': BaseGNNLinkPredictor,
    'GIN': BaseGNNLinkPredictor,
    'GraphSAGE': BaseGNNLinkPredictor,
}


def save_results(results, file_id):
    import os
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if file_id is None:
        file_id = 0
        while os.path.exists('results/lp_{}.xlsx'.format(file_id)):
            file_id += 1
    results.transpose().to_excel('results/lp_{}.xlsx'.format(file_id))
    print('save to file ID: {}'.format(file_id))
    return file_id
