"""
CompGraphDetector: Train GADBench GNN models on CGT computation graph trees.

Uses the same GNN architectures as GADBench (GCN, GIN, GraphSAGE) but feeds
them batched DGL computation graph trees instead of a single full graph.
Only root node predictions are used for classification.

Also provides XGBoost / XGBGraph tree-model detectors that consume the same
computation-graph datasets. XGBoost uses the root feature directly;
XGBGraph runs GIN_noparam over the batched tree and takes the root
embedding as the XGBoost input.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from data.comp_graph import make_comp_graph_collate, extract_root_logits

# GNN models that support computation graph mode
CG_SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE']


def _score(labels, probs):
    k = int(labels.sum())
    return {
        'AUROC': roc_auc_score(labels, probs),
        'AUPRC': average_precision_score(labels, probs),
        'RecK': sum(labels[probs.argsort()[-k:]]) / max(labels.sum(), 1),
    }


def _collect_root_features(dataset, device, gin=None, batch_size=256):
    """Materialize (X, y) numpy arrays of per-sample root features.

    gin=None: X is the root row of the per-sample feature tensor.
              For synthetic datasets, this is the root's cluster center;
              for original datasets, the raw root-node feature.
    gin set:  X is GIN_noparam(batched_tree)[root_positions], shape
              [N, feat_dim * (num_layers + 1)].
    """
    collate = make_comp_graph_collate(
        dataset.template_src, dataset.template_dst, dataset.num_tree_nodes)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0,
                        collate_fn=collate, drop_last=False, shuffle=False)
    xs, ys = [], []
    for batched_g, labels in loader:
        if gin is None:
            roots = extract_root_logits(batched_g, batched_g.ndata['feature'])
        else:
            batched_g = batched_g.to(device)
            with torch.no_grad():
                emb = gin(batched_g)
            roots = extract_root_logits(batched_g, emb).cpu()
        xs.append(roots)
        ys.append(labels)
    return torch.cat(xs).numpy(), torch.cat(ys).numpy()


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

        def _make_loader(dataset, shuffle):
            collate_fn = make_comp_graph_collate(
                dataset.template_src, dataset.template_dst,
                dataset.num_tree_nodes)
            return DataLoader(
                dataset, batch_size=batch_size, num_workers=0,
                collate_fn=collate_fn, drop_last=False, shuffle=shuffle)

        self.train_loader = _make_loader(train_dataset, shuffle=True)
        self.val_loader = _make_loader(val_dataset, shuffle=False)
        self.test_loader = _make_loader(test_dataset, shuffle=False)

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
        return _score(all_labels, all_probs)


class CGTXGBoostDetector:
    """XGBoost on CGT computation graphs. Feature = root row of each
    sample's tensor (cluster center for synthetic, raw feature for
    original)."""

    def __init__(self, train_config, model_config,
                 train_dataset, val_dataset, test_dataset):
        self.train_config = train_config
        self.model_config = model_config
        self.device = train_config['device']
        self.batch_size = train_config.get('batch_size', 256)
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = test_dataset

        labels = np.asarray(train_dataset.get_labels())
        num_pos = labels.sum()
        num_neg = len(labels) - num_pos
        self.weight = num_neg / max(num_pos, 1)

        import xgboost as xgb
        eval_metric = (roc_auc_score if train_config['metric'] == 'AUROC'
                       else average_precision_score)
        self.model = xgb.XGBClassifier(
            tree_method='hist', eval_metric=eval_metric, **model_config)

        self.gin = None
        self.best_score = -1

    def train(self):
        X_tr, y_tr = _collect_root_features(
            self.train_ds, self.device, self.gin, self.batch_size)
        X_va, y_va = _collect_root_features(
            self.val_ds, self.device, self.gin, self.batch_size)
        X_te, y_te = _collect_root_features(
            self.test_ds, self.device, self.gin, self.batch_size)

        print(f"  X_tr.shape={X_tr.shape}, X_va.shape={X_va.shape}, "
              f"X_te.shape={X_te.shape}")

        sw = np.where(y_tr == 0, 1.0, self.weight)
        self.model.fit(X_tr, y_tr, sample_weight=sw,
                       eval_set=[(X_va, y_va)])

        val_probs = self.model.predict_proba(X_va)[:, 1]
        test_probs = self.model.predict_proba(X_te)[:, 1]
        self.best_score = _score(y_va, val_probs)[self.train_config['metric']]
        return _score(y_te, test_probs)


class CGTXGBGraphDetector(CGTXGBoostDetector):
    """XGBoost on GIN_noparam-aggregated root embeddings of CGT
    computation graphs. Root embedding = input feat concat per-layer GIN
    aggregates, dim feat_dim * (num_layers + 1). num_layers must equal
    cg_depth (enforced by benchmark entry point)."""

    def __init__(self, train_config, model_config,
                 train_dataset, val_dataset, test_dataset):
        super().__init__(train_config, model_config,
                         train_dataset, val_dataset, test_dataset)
        from models.gnn import GIN_noparam
        self.gin = GIN_noparam(**model_config).to(self.device).eval()
