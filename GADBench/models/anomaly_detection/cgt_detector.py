"""
CompGraphDetector: Train GADBench GNN models on CGT computation graph trees.

Uses the same GNN architectures as GADBench (GCN, GIN, GraphSAGE) but feeds
them batched DGL computation graph trees instead of a single full graph.
Only root node predictions are used for classification.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from data.comp_graph import comp_graph_collate, extract_root_logits

# GNN models that support computation graph mode
CG_SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE']


class CompGraphDetector:
    """Train and evaluate GADBench GNN models on computation graph trees.

    Uses the same GNN architectures as GADBench but feeds them batched
    DGL computation graph trees instead of a single full graph. Only
    root node predictions are used for classification.
    """

    def __init__(self, train_config, model_config,
                 train_dataset, val_dataset, test_dataset):
        self.train_config = train_config
        self.model_config = model_config
        self.device = train_config['device']

        batch_size = train_config.get('batch_size', 256)
        loader_kw = dict(
            batch_size=batch_size,
            num_workers=0,
            collate_fn=comp_graph_collate,
            drop_last=False,
        )
        self.train_loader = DataLoader(
            train_dataset, shuffle=True, **loader_kw)
        self.val_loader = DataLoader(
            val_dataset, shuffle=False, **loader_kw)
        self.test_loader = DataLoader(
            test_dataset, shuffle=False, **loader_kw)

        # Class weight for imbalanced anomaly detection
        train_labels = np.asarray(train_dataset.get_labels())
        num_pos = train_labels.sum()
        num_neg = len(train_labels) - num_pos
        self.weight = num_neg / max(num_pos, 1)

        # Instantiate GADBench GNN
        import models.gnn as gnn_module
        model_name = model_config['model']
        gnn_cls = getattr(gnn_module, model_name, None)
        if gnn_cls is None or model_name not in CG_SUPPORTED_MODELS:
            raise ValueError(
                f"'{model_name}' not supported for computation graph mode. "
                f"Supported: {CG_SUPPORTED_MODELS}")
        self.model = gnn_cls(**model_config).to(self.device)

        self.best_score = -1
        self.patience_knt = 0

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.model_config['lr'])
        w = torch.tensor([1., self.weight], dtype=torch.float32,
                         device=self.device)

        best_test_score = None

        for epoch in range(self.train_config['epochs']):
            self.model.train()
            total_loss = 0
            for batched_g, labels in self.train_loader:
                batched_g = batched_g.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(batched_g)
                root_logits = extract_root_logits(batched_g, logits)

                loss = F.cross_entropy(root_logits, labels, weight=w)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_score = self._evaluate_loader(self.val_loader)

            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                best_test_score = self._evaluate_loader(self.test_loader)
                avg_loss = total_loss / max(len(self.train_loader), 1)
                print(
                    f'Epoch {epoch}, Loss {avg_loss:.4f}, '
                    f'Val AUC {val_score["AUROC"]:.4f}, '
                    f'PRC {val_score["AUPRC"]:.4f}, '
                    f'RecK {val_score["RecK"]:.4f}, '
                    f'test AUC {best_test_score["AUROC"]:.4f}, '
                    f'PRC {best_test_score["AUPRC"]:.4f}, '
                    f'RecK {best_test_score["RecK"]:.4f}')
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break

        return best_test_score

    def _evaluate_loader(self, loader):
        self.model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for batched_g, labels in loader:
                batched_g = batched_g.to(self.device)
                logits = self.model(batched_g)
                root_logits = extract_root_logits(batched_g, logits)
                probs = root_logits.softmax(1)[:, 1]
                all_probs.append(probs.cpu())
                all_labels.append(labels)

        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        score = {}
        score['AUROC'] = roc_auc_score(all_labels, all_probs)
        score['AUPRC'] = average_precision_score(all_labels, all_probs)
        k = int(all_labels.sum())
        score['RecK'] = (
            sum(all_labels[all_probs.argsort()[-k:]]) / max(sum(all_labels), 1)
        )
        return score
