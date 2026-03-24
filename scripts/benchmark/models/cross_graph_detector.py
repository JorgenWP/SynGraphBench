"""Cross-graph GNN detector: train+val on synthetic graph, test on original."""

import os, sys, torch
import torch.nn.functional as F
import numpy as np
import dgl

_gadbench = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GADBench'))
if _gadbench not in sys.path:
    sys.path.insert(0, _gadbench)

from models.anomaly_detection.detector import BaseDetector

CROSS_GRAPH_SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE']


class CrossGraphGNNDetector(BaseDetector):
    """
    Train entirely on synthetic graph, test on original graph.

    Train:      synthetic graph train nodes
    Val:        synthetic graph val nodes  (early stopping)
    Test:       original graph test nodes  (final evaluation)
    """

    def __init__(self, train_config, model_config, syn_data, orig_data):
        from models.gnn import GCN, GIN, GraphSAGE
        _gnn = {'GCN': GCN, 'GIN': GIN, 'GraphSAGE': GraphSAGE}

        device = train_config['device']
        self.train_config = train_config
        self.model_config = model_config

        # --- Synthetic graph (train + val) ---
        # All GADBench original datasets have self-loops on every node.
        # Add them unconditionally to the synthetic graph to match this convention.
        syn_graph = syn_data.graph.to(device)
        syn_graph = dgl.add_self_loop(dgl.remove_self_loop(syn_graph))
        self.syn_graph      = syn_graph
        self.syn_train_mask = syn_graph.ndata['train_mask'].bool()
        self.syn_val_mask   = syn_graph.ndata['val_mask'].bool()
        self.syn_labels     = syn_graph.ndata['label']

        # Class imbalance weight from synthetic train labels
        syn_train_labels = self.syn_labels[self.syn_train_mask]
        pos = syn_train_labels.sum().item()
        self.weight = (len(syn_train_labels) - pos) / max(pos, 1)

        # --- Original graph (test only) ---
        orig_graph = orig_data.graph.to(device)
        self.orig_graph  = orig_graph
        self.test_mask   = orig_graph.ndata['test_mask'].bool()
        self.orig_labels = orig_graph.ndata['label']

        # Build model — feature dim must match between synthetic and original
        cfg = {k: v for k, v in model_config.items() if k != 'model'}
        cfg['in_feats'] = syn_graph.ndata['feature'].shape[1]
        self.model = _gnn[model_config['model']](**cfg).to(device)

        self.best_score   = -1
        self.patience_knt = 0

    def train(self):
        optimizer     = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        metric        = self.train_config['metric']
        patience      = self.train_config['patience']
        device        = self.train_config['device']
        weight_tensor = torch.tensor([1., self.weight], device=device)

        train_labels = self.syn_labels[self.syn_train_mask]
        val_labels   = self.syn_labels[self.syn_val_mask]
        test_labels  = self.orig_labels[self.test_mask]

        test_score = None
        for e in range(self.train_config['epochs']):
            # Train on synthetic graph
            self.model.train()
            logits = self.model(self.syn_graph)
            loss = F.cross_entropy(
                logits[self.syn_train_mask], train_labels, weight=weight_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validate on synthetic val nodes
            self.model.eval()
            with torch.no_grad():
                val_probs = self.model(self.syn_graph).softmax(1)[:, 1]
                val_score = self.eval(val_labels, val_probs[self.syn_val_mask])

            if val_score[metric] > self.best_score:
                with torch.no_grad():
                    test_probs = self.model(self.orig_graph).softmax(1)[:, 1]
                    test_score = self.eval(test_labels, test_probs[self.test_mask])
                self.best_score   = val_score[metric]
                self.patience_knt = 0
                print(f'  Epoch {e}, Loss {loss:.4f}, '
                      f'Val AUROC {val_score["AUROC"]:.4f} AUPRC {val_score["AUPRC"]:.4f} | '
                      f'Test AUROC {test_score["AUROC"]:.4f} AUPRC {test_score["AUPRC"]:.4f}')
            else:
                self.patience_knt += 1
                if self.patience_knt > patience:
                    break

        return test_score
