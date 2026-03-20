"""
CompGraphLinkPredictor: Link prediction using CGT computation graph trees.

Uses the same GNN architectures as GADBench but generates node embeddings
via computation graph trees (like CompGraphDetector) and scores edges
using dot product or MLP decoder (like BaseGNNLinkPredictor).
"""

import torch
import torch.nn.functional as F
import numpy as np
import dgl
from torch.utils.data import DataLoader

from data.comp_graph import (
    OriginalCompGraphDataset, comp_graph_collate,
    extract_root_logits, dgl_to_adj_list,
)
from models.link_prediction.link_predictor import BaseDetector, MLPDecoder

CG_LP_SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE']


class CompGraphLinkPredictor(BaseDetector):
    """Link prediction using CGT computation graph trees for node embeddings.

    Instead of running GNN on the full graph, builds a computation graph
    tree for each node (sampling neighbors at each hop), extracts the
    root embedding, and scores edges via dot product or MLP decoder.
    """

    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        self.device = train_config['device']

        model_config['output_emb'] = True
        if model_config['model'] == 'GraphSAGE':
            model_config.setdefault('agg', 'mean')

        import models.gnn as gnn_module
        model_name = model_config['model']
        if model_name not in CG_LP_SUPPORTED_MODELS:
            raise ValueError(
                f"'{model_name}' not supported for computation graph link "
                f"prediction. Supported: {CG_LP_SUPPORTED_MODELS}")
        gnn_cls = getattr(gnn_module, model_name)
        self.model = gnn_cls(**model_config).to(self.device)

        h_feats = model_config.get('h_feats', 32)
        decoder = train_config.get('decoder', 'dot')
        if decoder == 'mlp':
            self.decoder = MLPDecoder(
                h_feats, model_config.get('drop_rate', 0)
            ).to(self.device)
        else:
            self.decoder = None

        # CGT computation graph parameters
        self.step_num = train_config.get('step_num', 2)
        self.sample_num = train_config.get('sample_num', 5)
        self.batch_size = train_config.get('batch_size', 256)

        # Build adjacency list from train graph (no test/val edge leakage)
        self.train_adj_list = dgl_to_adj_list(data.train_graph)
        self.features = data.graph.ndata['feature'].cpu().numpy()
        self.dummy_labels = np.zeros(self.num_nodes, dtype=np.int64)

    def _compute_all_embeddings(self):
        """Compute embeddings for all nodes via batched computation graph trees."""
        all_node_ids = np.arange(self.num_nodes)
        dataset = OriginalCompGraphDataset(
            self.train_adj_list, self.features, self.dummy_labels,
            all_node_ids, self.step_num, self.sample_num)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=comp_graph_collate, num_workers=0)

        all_embs = []
        for batched_g, _ in loader:
            batched_g = batched_g.to(self.device)
            h = self.model(batched_g)
            root_embs = extract_root_logits(batched_g, h)
            all_embs.append(root_embs)

        return torch.cat(all_embs, dim=0)

    def score_edges(self, h, edges):
        """Score edges using dot product or MLP decoder."""
        if self.decoder is not None:
            return self.decoder(h, edges)
        return (h[edges[:, 0]] * h[edges[:, 1]]).sum(dim=-1)

    def _filter_collisions(self, neg_edges):
        """Replace negative edges that collide with real edges."""
        src, dst = neg_edges[:, 0], neg_edges[:, 1]
        N = self.num_nodes
        for _ in range(10):
            hashes = src.long() * N + dst.long()
            collision = torch.isin(hashes, self.edge_set) | (src == dst)
            if not collision.any():
                break
            n_bad = collision.sum().item()
            dst[collision] = torch.randint(0, N, (n_bad,))
        return neg_edges

    def _sample_random_negatives(self, n):
        """Sample random node pairs as guaranteed non-edges."""
        neg_edges = torch.stack([
            torch.randint(0, self.num_nodes, (n,)),
            torch.randint(0, self.num_nodes, (n,))
        ], dim=1)
        return self._filter_collisions(neg_edges).to(self.device)

    def _sample_hard_negatives(self, pos_edges):
        """Sample hard negatives via 2-hop random walks, guaranteed non-edges."""
        src = pos_edges[:, 0]
        walk_nodes, _ = dgl.sampling.random_walk(
            self.train_graph.cpu(), src.cpu(), metapath=[None, None])
        hard_dst = walk_nodes[:, 2]

        failed = hard_dst == -1
        if failed.any():
            hard_dst[failed] = torch.randint(0, self.num_nodes, (failed.sum(),))

        neg_edges = torch.stack([src.cpu(), hard_dst], dim=1)
        return self._filter_collisions(neg_edges).to(self.device)

    def train(self):
        params = list(self.model.parameters())
        if self.decoder is not None:
            params += list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.model_config['lr'])

        test_score = None
        n_train = self.train_pos_edges.shape[0]

        for e in range(self.train_config['epochs']):
            self.model.train()
            h = self._compute_all_embeddings()

            if self.neg_sampling == 'hard':
                train_neg_edges = self._sample_hard_negatives(self.train_pos_edges)
            else:
                train_neg_edges = self._sample_random_negatives(n_train)

            pos_scores = self.score_edges(h, self.train_pos_edges)
            neg_scores = self.score_edges(h, train_neg_edges)
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(pos_scores.shape[0]),
                torch.zeros(neg_scores.shape[0])
            ]).to(self.device)
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.model.eval()
                h = self._compute_all_embeddings()
                val_pos = torch.sigmoid(self.score_edges(h, self.val_pos_edges))
                val_neg = torch.sigmoid(self.score_edges(h, self.val_neg_edges))
            val_score = self.eval(val_pos, val_neg)

            if val_score[self.train_config['metric']] > self.best_score:
                self.best_score = val_score[self.train_config['metric']]
                self.patience_knt = 0
                with torch.no_grad():
                    test_pos = torch.sigmoid(self.score_edges(h, self.test_pos_edges))
                    test_neg = torch.sigmoid(self.score_edges(h, self.test_neg_edges))
                test_score = self.eval(test_pos, test_neg)
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, '
                      'test AUC {:.4f}, PRC {:.4f}'.format(
                          e, loss, val_score['AUROC'], val_score['AUPRC'],
                          test_score['AUROC'], test_score['AUPRC']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break

        return test_score
