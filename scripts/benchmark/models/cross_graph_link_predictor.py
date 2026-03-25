"""Cross-graph GNN link predictor: train+val on synthetic graph, test on original."""

import os, sys, torch
import torch.nn.functional as F
import dgl

_gadbench = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GADBench'))
if _gadbench not in sys.path:
    sys.path.insert(0, _gadbench)

from models.link_prediction.link_predictor import BaseDetector, MLPDecoder

CROSS_GRAPH_LP_SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE']


class CrossGraphLinkPredictor(BaseDetector):
    """
    Train entirely on synthetic graph edges, test on original graph edges.

    Train:      synthetic graph train edges  (GNN on synthetic train graph)
    Val:        synthetic graph val edges     (early stopping, GNN on synthetic train graph)
    Test:       original graph test edges     (final evaluation, GNN on original train graph)
    """

    def __init__(self, train_config, model_config, syn_data, orig_data):
        from models.gnn import GCN, GIN, GraphSAGE
        _gnn = {'GCN': GCN, 'GIN': GIN, 'GraphSAGE': GraphSAGE}

        device = train_config['device']
        self.train_config = train_config
        self.model_config = model_config
        self.neg_sampling = train_config.get('neg_sampling', 'random')

        # --- Synthetic graph (train + val) ---
        self.syn_graph = syn_data.graph.to(device)
        self.syn_train_graph = syn_data.train_graph.to(device)
        self.syn_num_nodes = syn_data.graph.num_nodes()
        self.syn_edge_set = syn_data.edge_set

        self.syn_train_pos_edges = syn_data.train_pos_edges.to(device)
        self.syn_val_pos_edges = syn_data.val_pos_edges.to(device)
        self.syn_val_neg_edges = syn_data.val_neg_edges.to(device)

        # --- Original graph (test only) ---
        self.orig_train_graph = orig_data.train_graph.to(device)
        self.orig_num_nodes = orig_data.graph.num_nodes()
        self.orig_edge_set = orig_data.edge_set

        self.test_pos_edges = orig_data.test_pos_edges.to(device)
        self.test_neg_edges = orig_data.test_neg_edges.to(device)

        # Build model
        model_config = {k: v for k, v in model_config.items()}
        model_config['in_feats'] = syn_data.graph.ndata['feature'].shape[1]
        model_config['output_emb'] = True
        if model_config['model'] == 'GraphSAGE':
            model_config.setdefault('agg', 'mean')

        self.model = _gnn[model_config['model']](**model_config).to(device)

        # Edge decoder
        h_feats = model_config.get('h_feats', 32)
        decoder = train_config.get('decoder', 'dot')
        if decoder == 'mlp':
            self.decoder = MLPDecoder(
                h_feats, model_config.get('drop_rate', 0)
            ).to(device)
        else:
            self.decoder = None

        self.best_score = -1
        self.patience_knt = 0

    def score_edges(self, h, edges):
        if self.decoder is not None:
            return self.decoder(h, edges)
        return (h[edges[:, 0]] * h[edges[:, 1]]).sum(dim=-1)

    def _filter_collisions(self, neg_edges, num_nodes, edge_set):
        src, dst = neg_edges[:, 0], neg_edges[:, 1]
        for _ in range(10):
            hashes = src.long() * num_nodes + dst.long()
            collision = torch.isin(hashes, edge_set) | (src == dst)
            if not collision.any():
                break
            n_bad = collision.sum().item()
            dst[collision] = torch.randint(0, num_nodes, (n_bad,))
        return neg_edges

    def _sample_random_negatives(self, n, num_nodes, edge_set):
        neg_edges = torch.stack([
            torch.randint(0, num_nodes, (n,)),
            torch.randint(0, num_nodes, (n,))
        ], dim=1)
        return self._filter_collisions(neg_edges, num_nodes, edge_set).to(
            self.train_config['device'])

    def _sample_hard_negatives(self, pos_edges, train_graph, num_nodes, edge_set):
        src = pos_edges[:, 0]
        walk_nodes, _ = dgl.sampling.random_walk(
            train_graph.cpu(), src.cpu(), metapath=[None, None])
        hard_dst = walk_nodes[:, 2]
        failed = hard_dst == -1
        if failed.any():
            hard_dst[failed] = torch.randint(0, num_nodes, (failed.sum(),))
        neg_edges = torch.stack([src.cpu(), hard_dst], dim=1)
        return self._filter_collisions(neg_edges, num_nodes, edge_set).to(
            self.train_config['device'])

    def eval(self, pos_scores, neg_scores):
        from sklearn.metrics import roc_auc_score, average_precision_score
        with torch.no_grad():
            labels = torch.cat([
                torch.ones(pos_scores.shape[0]),
                torch.zeros(neg_scores.shape[0])
            ]).numpy()
            probs = torch.cat([pos_scores, neg_scores]).cpu().numpy()
            score = {}
            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)
            k = int(labels.sum())
            score['RecK'] = sum(labels[probs.argsort()[-k:]]) / k
        return score

    def train(self):
        params = list(self.model.parameters())
        if self.decoder is not None:
            params += list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.model_config['lr'])

        metric = self.train_config['metric']
        patience = self.train_config['patience']
        device = self.train_config['device']

        test_score = None
        n_train = self.syn_train_pos_edges.shape[0]

        for e in range(self.train_config['epochs']):
            # Train on synthetic graph
            self.model.train()
            h = self.model(self.syn_train_graph)

            if self.neg_sampling == 'hard':
                train_neg = self._sample_hard_negatives(
                    self.syn_train_pos_edges, self.syn_train_graph,
                    self.syn_num_nodes, self.syn_edge_set)
            else:
                train_neg = self._sample_random_negatives(
                    n_train, self.syn_num_nodes, self.syn_edge_set)

            pos_scores = self.score_edges(h, self.syn_train_pos_edges)
            neg_scores = self.score_edges(h, train_neg)
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(pos_scores.shape[0]),
                torch.zeros(neg_scores.shape[0])
            ]).to(device)
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validate on synthetic val edges
            self.model.eval()
            with torch.no_grad():
                h_syn = self.model(self.syn_train_graph)
                val_pos = torch.sigmoid(self.score_edges(h_syn, self.syn_val_pos_edges))
                val_neg = torch.sigmoid(self.score_edges(h_syn, self.syn_val_neg_edges))
            val_score = self.eval(val_pos, val_neg)

            if val_score[metric] > self.best_score:
                self.best_score = val_score[metric]
                self.patience_knt = 0
                # Test on original graph
                with torch.no_grad():
                    h_orig = self.model(self.orig_train_graph)
                    test_pos = torch.sigmoid(self.score_edges(h_orig, self.test_pos_edges))
                    test_neg = torch.sigmoid(self.score_edges(h_orig, self.test_neg_edges))
                test_score = self.eval(test_pos, test_neg)
                print(f'  Epoch {e}, Loss {loss:.4f}, '
                      f'Val AUC {val_score["AUROC"]:.4f} PRC {val_score["AUPRC"]:.4f} | '
                      f'Test AUC {test_score["AUROC"]:.4f} PRC {test_score["AUPRC"]:.4f}')
            else:
                self.patience_knt += 1
                if self.patience_knt > patience:
                    break

        return test_score
