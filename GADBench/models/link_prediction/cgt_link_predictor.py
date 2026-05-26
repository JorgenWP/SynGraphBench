"""
MergedCompGraphLinkPredictor — paper-style link prediction on computation graphs.

For each edge (u, v), the two endpoints' computation-graph trees are
merged by a root-root edge and scored with a single GNN forward pass.
This preserves the joint-computation property of whole-graph LP (where
u and v share message-passing paths) while operating on the sampled
tree subgraphs that CGT synthesises.

Exposed as `CompGraphLinkPredictor` for backward compatibility with
existing imports.

Also provides tree-model link predictors that consume the same merged
comp graphs:
  - `CGTXGBoostLinkPredictor`: Hadamard of the two raw root features
    (positions 0 and T of the merged tree) → XGBoost. CGT-only (no
    full-graph LP baseline).
  - `CGTXGBGraphLinkPredictor`: CG counterpart of `XGBGraphLinkPredictor`.
    Adds `GIN_noparam` over the merged tree before the Hadamard step.
    Enforces `num_layers == step_num`.
"""

import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
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

        # Cache of pre-built merged-CG batches for fixed eval edge sets
        # (val_pos/val_neg/test_pos/test_neg). Keyed by id(edges) so the
        # immutable edge tensors created once per trial deduplicate the
        # ~414+ batches/epoch otherwise rebuilt for evaluation.
        self._eval_batch_cache = {}

    def _edge_loader(self, edges):
        ds = MergedOriginalCompGraphDataset(
            self.train_adj_list, self.features, edges,
            self.step_num, self.sample_num,
            self.noise_num, self.self_connection)
        collate = make_merged_comp_graph_collate(
            self.template_src, self.template_dst, self.num_merged_nodes)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate, num_workers=8)

    def _score_batch_on_device(self, batched):
        h = self.model(batched)
        h_u, h_v = extract_edge_root_embeddings(
            batched, h, self.per_tree_num_nodes)
        if self.decoder is not None:
            return self.decoder.score_from_pair(h_u, h_v)
        return (h_u * h_v).sum(dim=-1)

    def _score_batch(self, batched):
        return self._score_batch_on_device(batched.to(self.device))

    def _score_edges(self, edges):
        """Score edges by building batched merged computation graphs.

        For fixed edge sets (val_pos/val_neg/test_pos/test_neg) we build
        the batched merged-CG graphs once and cache them; subsequent
        calls only re-run the GNN forward. Train-time edges (resampled
        every epoch) are not cached — `id(edges)` differs per call.
        """
        if edges.shape[0] == 0:
            return torch.empty(0, device=self.device)

        cache_key = id(edges)
        cached = self._eval_batch_cache.get(cache_key)
        if cached is None:
            cached = [b.to(self.device) for b in self._edge_loader(edges)]
            self._eval_batch_cache[cache_key] = cached

        scores = [self._score_batch_on_device(b) for b in cached]
        return torch.cat(scores, dim=0)

    def _train_step(self, edges, labels, ts=None):
        """Per-batch forward+backward with gradient accumulation.

        Peak memory is O(one batch) because each batch's autograd graph
        is freed by its own backward. Accumulated gradient equals the
        gradient of mean-BCE over all edges: each batch contributes
        (1/N) * sum_{i in batch} BCE_i, summing to (1/N) * sum_i BCE_i.

        If `ts` (dict) is provided, accumulates per-batch timings:
          - ts['data_wait']: wall time GPU was idle waiting on DataLoader.
          - ts['gpu_compute']: wall time spent in score+BCE+backward.
          - ts['n_batches']: number of batches processed.
        """
        n_total = edges.shape[0]
        if n_total == 0:
            return 0.0

        use_cuda = (torch.cuda.is_available()
                    and 'cuda' in str(self.device))

        loss_sum = 0.0
        offset = 0

        if use_cuda:
            torch.cuda.synchronize()
        t_prev = time.time()
        for batched in self._edge_loader(edges):
            t_after_next = time.time()
            if ts is not None:
                ts['data_wait'] += t_after_next - t_prev

            s = self._score_batch(batched)
            n_b = s.shape[0]
            batch_labels = labels[offset:offset + n_b]
            batch_loss = F.binary_cross_entropy_with_logits(
                s, batch_labels, reduction='sum') / n_total
            batch_loss.backward()
            loss_sum += batch_loss.item()
            offset += n_b

            if use_cuda:
                torch.cuda.synchronize()
            t_after_gpu = time.time()
            if ts is not None:
                ts['gpu_compute'] += t_after_gpu - t_after_next
                ts['n_batches'] += 1
            t_prev = t_after_gpu
        return loss_sum

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

        use_cuda = (torch.cuda.is_available()
                    and 'cuda' in str(self.device))

        def sync():
            if use_cuda:
                torch.cuda.synchronize()

        metric = self.train_config['metric']
        test_score = None
        val_auprc_curve  = []
        test_auprc_curve = []
        n_train = self.train_pos_edges.shape[0]
        ts_total = {'neg': 0.0, 'data_wait': 0.0, 'gpu_compute': 0.0,
                    'opt': 0.0, 'val': 0.0, 'test': 0.0, 'n_batches': 0}
        n_epochs = 0
        trial_start = time.time()

        for e in range(self.train_config['epochs']):
            self.model.train()

            sync(); t0 = time.time()
            if self.neg_sampling == 'hard':
                train_neg_edges = self._sample_hard_negatives(
                    self.train_pos_edges)
            else:
                train_neg_edges = self._sample_random_negatives(n_train)
            sync(); t1 = time.time()
            t_neg = t1 - t0

            all_train_edges = torch.cat(
                [self.train_pos_edges, train_neg_edges], dim=0)
            labels = torch.cat([
                torch.ones(n_train, device=self.device),
                torch.zeros(train_neg_edges.shape[0], device=self.device),
            ])

            epoch_step_ts = {
                'data_wait': 0.0, 'gpu_compute': 0.0, 'n_batches': 0}
            optimizer.zero_grad()
            loss = self._train_step(all_train_edges, labels, epoch_step_ts)
            optimizer.step()
            sync(); t2 = time.time()
            t_opt = (t2 - t1) - epoch_step_ts['data_wait'] - epoch_step_ts['gpu_compute']

            self.model.eval()
            with torch.no_grad():
                val_pos = torch.sigmoid(self._score_edges(self.val_pos_edges))
                val_neg = torch.sigmoid(self._score_edges(self.val_neg_edges))
            val_score = self.eval(val_pos, val_neg)
            sync(); t3 = time.time()
            t_val = t3 - t2

            with torch.no_grad():
                test_pos = torch.sigmoid(
                    self._score_edges(self.test_pos_edges))
                test_neg = torch.sigmoid(
                    self._score_edges(self.test_neg_edges))
            epoch_test = self.eval(test_pos, test_neg)
            sync(); t4 = time.time()
            t_test = t4 - t3
            val_auprc_curve.append(val_score['AUPRC'])
            test_auprc_curve.append(epoch_test['AUPRC'])

            if val_score[metric] > self.best_score:
                self.best_score = val_score[metric]
                self.patience_knt = 0
                test_score = epoch_test
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, '
                      'test AUC {:.4f}, PRC {:.4f}'.format(
                          e, loss, val_score['AUROC'], val_score['AUPRC'],
                          test_score['AUROC'], test_score['AUPRC']))
            else:
                self.patience_knt += 1

            ts_total['neg'] += t_neg
            ts_total['data_wait'] += epoch_step_ts['data_wait']
            ts_total['gpu_compute'] += epoch_step_ts['gpu_compute']
            ts_total['opt'] += t_opt
            ts_total['val'] += t_val
            ts_total['test'] += t_test
            ts_total['n_batches'] += epoch_step_ts['n_batches']
            n_epochs += 1

            if self.patience_knt > self.train_config['patience']:
                break

        trial_wall = time.time() - trial_start
        sum_phases = (ts_total['neg'] + ts_total['data_wait']
                      + ts_total['gpu_compute'] + ts_total['opt']
                      + ts_total['val'] + ts_total['test'])
        other = trial_wall - sum_phases
        print(f"  [Trial summary] {n_epochs} epochs "
              f"({ts_total['n_batches']} batches) | totals: "
              f"neg={ts_total['neg']:.1f}s "
              f"data={ts_total['data_wait']:.1f}s "
              f"gpu={ts_total['gpu_compute']:.1f}s "
              f"opt={ts_total['opt']:.1f}s "
              f"val={ts_total['val']:.1f}s "
              f"test={ts_total['test']:.1f}s "
              f"other={other:.1f}s")

        if test_score is not None:
            test_score['val_auprc_curve']  = val_auprc_curve
            test_score['test_auprc_curve'] = test_auprc_curve
        return test_score


# Back-compat alias for existing imports
CompGraphLinkPredictor = MergedCompGraphLinkPredictor


class CGTXGBoostLinkPredictor(BaseDetector):
    """XGBoost LP on merged comp graphs. Per edge: Hadamard of
    (h_u, h_v) at positions 0 and T of the merged tree. Base class
    uses raw root features (no GIN); subclass adds GIN_noparam
    aggregation. CGT-only.
    """

    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        import xgboost as xgb

        self.device = train_config['device']
        self.step_num = train_config.get('step_num', 2)
        self.sample_num = train_config.get('sample_num', 5)
        self.noise_num = train_config.get('noise_num', 0)
        self.self_connection = train_config.get('self_connection', False)
        self.batch_size = train_config.get('batch_size', 256)

        self.train_adj_list = dgl_to_adj_list(data.train_graph)
        self.features = data.graph.ndata['feature'].cpu().numpy().astype(
            np.float32)

        merged_adj = compute_merged_tree_adj(
            self.step_num, self.sample_num + self.noise_num,
            self.self_connection)
        self.template_src, self.template_dst, self.num_merged_nodes = \
            compute_template_edges(merged_adj)
        self.per_tree_num_nodes = self.num_merged_nodes // 2

        eval_metric = (roc_auc_score if train_config['metric'] == 'AUROC'
                       else average_precision_score)
        cfg = {k: v for k, v in model_config.items() if k != 'model'}
        self.model = xgb.XGBClassifier(
            tree_method='hist', eval_metric=eval_metric, verbose=False, **cfg)

        self.gin = None

    def _edge_features(self, edges):
        if edges.shape[0] == 0:
            feat_dim = self.features.shape[1]
            if self.gin is not None:
                feat_dim *= self.model_config.get('num_layers', 2) + 1
            return np.zeros((0, feat_dim), dtype=np.float32)

        edges_cpu = edges.cpu() if torch.is_tensor(edges) else edges
        ds = MergedOriginalCompGraphDataset(
            self.train_adj_list, self.features, edges_cpu,
            self.step_num, self.sample_num,
            self.noise_num, self.self_connection)
        collate = make_merged_comp_graph_collate(
            self.template_src, self.template_dst, self.num_merged_nodes)
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate, num_workers=8)

        feats = []
        with torch.no_grad():
            for batched in loader:
                if self.gin is None:
                    emb = batched.ndata['feature']
                else:
                    batched = batched.to(self.device)
                    emb = self.gin(batched)
                h_u, h_v = extract_edge_root_embeddings(
                    batched, emb, self.per_tree_num_nodes)
                feats.append((h_u * h_v).cpu().numpy())
        return np.vstack(feats)

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
        n_train = self.train_pos_edges.shape[0]

        if self.neg_sampling == 'hard':
            train_neg = self._sample_hard_negatives(self.train_pos_edges)
        else:
            train_neg = self._sample_random_negatives(n_train)

        train_X = np.vstack([
            self._edge_features(self.train_pos_edges),
            self._edge_features(train_neg),
        ])
        train_y = np.concatenate([
            np.ones(n_train), np.zeros(train_neg.shape[0])])

        n_val_pos = self.val_pos_edges.shape[0]
        n_val_neg = self.val_neg_edges.shape[0]
        val_X = np.vstack([
            self._edge_features(self.val_pos_edges),
            self._edge_features(self.val_neg_edges),
        ])
        val_y = np.concatenate([np.ones(n_val_pos), np.zeros(n_val_neg)])

        self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)],
                       verbose=False)

        val_probs = self.model.predict_proba(val_X)[:, 1]
        val_pos_probs = torch.tensor(val_probs[:n_val_pos])
        val_neg_probs = torch.tensor(val_probs[n_val_pos:])
        val_score = self.eval(val_pos_probs, val_neg_probs)
        self.best_score = val_score[self.train_config['metric']]

        n_test_pos = self.test_pos_edges.shape[0]
        n_test_neg = self.test_neg_edges.shape[0]
        test_X = np.vstack([
            self._edge_features(self.test_pos_edges),
            self._edge_features(self.test_neg_edges),
        ])
        test_probs = self.model.predict_proba(test_X)[:, 1]
        test_pos_probs = torch.tensor(test_probs[:n_test_pos])
        test_neg_probs = torch.tensor(test_probs[n_test_pos:])
        return self.eval(test_pos_probs, test_neg_probs)


class CGTXGBGraphLinkPredictor(CGTXGBoostLinkPredictor):
    """XGBoost LP on GIN_noparam-aggregated root embeddings of merged
    comp graphs. Per-edge feature dim: feat_dim * (num_layers + 1).
    `num_layers` must equal `step_num` (enforced).
    """

    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        from models.gnn import GIN_noparam

        num_layers = model_config.get('num_layers', 2)
        if num_layers != self.step_num:
            raise ValueError(
                f"CGTXGBGraphLinkPredictor requires num_layers == step_num; "
                f"got num_layers={num_layers}, step_num={self.step_num}.")

        self.gin = GIN_noparam(**model_config).to(self.device).eval()
