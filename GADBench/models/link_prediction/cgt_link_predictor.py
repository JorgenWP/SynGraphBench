"""
MergedCompGraphLinkPredictor — paper-style link prediction on computation graphs.

For each edge (u, v), the two endpoints' computation-graph trees are
merged by a root-root edge and scored with a single GNN forward pass.
This preserves the joint-computation property of whole-graph LP (where
u and v share message-passing paths) while operating on the sampled
tree subgraphs that CGT synthesises.

Exposed as `CompGraphLinkPredictor` for backward compatibility with
existing imports.
"""

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.comp_graph import (
    MergedOriginalCompGraphDataset,
    compute_merged_tree_adj,
    compute_template_edges,
    dgl_to_adj_list,
    extract_edge_root_embeddings,
    make_merged_comp_graph_collate,
)
from models.link_prediction.link_predictor import BaseDetector, MLPDecoder

CG_LP_SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE']


class MergedCompGraphLinkPredictor(BaseDetector):
    """Link prediction via per-edge merged endpoint computation graphs."""

    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        self.device = train_config['device']

        model_config['output_emb'] = True
        if model_config['model'] == 'GraphSAGE':
            model_config.setdefault('agg', 'mean')

        model_name = model_config['model']
        if model_name not in CG_LP_SUPPORTED_MODELS:
            raise ValueError(
                f"'{model_name}' not supported for merged computation graph "
                f"link prediction. Supported: {CG_LP_SUPPORTED_MODELS}")

        import models.gnn as gnn_module
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

        self.step_num = train_config.get('step_num', 2)
        self.sample_num = train_config.get('sample_num', 5)
        self.noise_num = train_config.get('noise_num', 0)
        self.self_connection = train_config.get('self_connection', False)
        self.batch_size = train_config.get('batch_size', 256)

        num_layers = model_config.get('num_layers', 2)
        if num_layers != self.step_num:
            print(
                f"  WARNING: num_layers={num_layers} != step_num={self.step_num}. "
                f"GNN depth should equal merged-tree depth for a fair "
                f"comparison with full-graph LP.")

        # Tree sampling uses train_graph (no test/val edge leakage)
        self.train_adj_list = dgl_to_adj_list(data.train_graph)
        self.features = data.graph.ndata['feature'].cpu().numpy().astype(
            np.float32)

        merged_adj = compute_merged_tree_adj(
            self.step_num, self.sample_num + self.noise_num,
            self.self_connection)
        self.template_src, self.template_dst, self.num_merged_nodes = \
            compute_template_edges(merged_adj)
        self.per_tree_num_nodes = self.num_merged_nodes // 2

    def _score_edges(self, edges):
        """Score edges by building batched merged computation graphs."""
        if edges.shape[0] == 0:
            return torch.empty(0, device=self.device)

        ds = MergedOriginalCompGraphDataset(
            self.train_adj_list, self.features, edges,
            self.step_num, self.sample_num,
            self.noise_num, self.self_connection)
        collate = make_merged_comp_graph_collate(
            self.template_src, self.template_dst, self.num_merged_nodes)
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate, num_workers=0)

        scores = []
        for batched in loader:
            batched = batched.to(self.device)
            h = self.model(batched)
            h_u, h_v = extract_edge_root_embeddings(
                batched, h, self.per_tree_num_nodes)
            if self.decoder is not None:
                s = self.decoder.score_from_pair(h_u, h_v)
            else:
                s = (h_u * h_v).sum(dim=-1)
            scores.append(s)
        return torch.cat(scores, dim=0)

    def _filter_collisions(self, neg_edges):
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
        neg = torch.stack([
            torch.randint(0, self.num_nodes, (n,)),
            torch.randint(0, self.num_nodes, (n,)),
        ], dim=1)
        return self._filter_collisions(neg).to(self.device)

    def _sample_hard_negatives(self, pos_edges):
        src = pos_edges[:, 0]
        walk_nodes, _ = dgl.sampling.random_walk(
            self.train_graph.cpu(), src.cpu(), metapath=[None, None])
        hard_dst = walk_nodes[:, 2]
        failed = hard_dst == -1
        if failed.any():
            hard_dst[failed] = torch.randint(
                0, self.num_nodes, (int(failed.sum()),))
        neg = torch.stack([src.cpu(), hard_dst], dim=1)
        return self._filter_collisions(neg).to(self.device)

    def train(self):
        params = list(self.model.parameters())
        if self.decoder is not None:
            params += list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.model_config['lr'])

        test_score = None
        n_train = self.train_pos_edges.shape[0]

        for e in range(self.train_config['epochs']):
            self.model.train()

            if self.neg_sampling == 'hard':
                train_neg_edges = self._sample_hard_negatives(
                    self.train_pos_edges)
            else:
                train_neg_edges = self._sample_random_negatives(n_train)

            all_train_edges = torch.cat(
                [self.train_pos_edges, train_neg_edges], dim=0)
            scores = self._score_edges(all_train_edges)
            labels = torch.cat([
                torch.ones(n_train, device=self.device),
                torch.zeros(train_neg_edges.shape[0], device=self.device),
            ])
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_pos = torch.sigmoid(self._score_edges(self.val_pos_edges))
                val_neg = torch.sigmoid(self._score_edges(self.val_neg_edges))
            val_score = self.eval(val_pos, val_neg)

            if val_score[self.train_config['metric']] > self.best_score:
                self.best_score = val_score[self.train_config['metric']]
                self.patience_knt = 0
                with torch.no_grad():
                    test_pos = torch.sigmoid(
                        self._score_edges(self.test_pos_edges))
                    test_neg = torch.sigmoid(
                        self._score_edges(self.test_neg_edges))
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


# Back-compat alias for existing imports
CompGraphLinkPredictor = MergedCompGraphLinkPredictor
