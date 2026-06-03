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
    _build_csr_adj,
    build_merged_skeleton,
    compute_merged_tree_adj,
    compute_template_edges,
    dgl_to_adj_list,
    extract_edge_root_embeddings,
    make_merged_comp_graph_collate,
)
from models.link_prediction.link_predictor import BaseDetector, MLPDecoder

CG_LP_SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE']

# DataLoader workers for merged-CG tree sampling are auto-sized from the
# per-loader batch count (no CLI flag — scale-driven choice). The worker payload
# is now a small int64 node-ID array (feature gather + graph assembly happen on
# the main process / GPU), so workers only parallelise the light numpy tree
# sampler and their benefit saturates fast; tiny loaders pay more in per-epoch
# fork/teardown than they save.
_CG_LP_MIN_BATCHES_FOR_WORKERS = 16   # below this, sample in-process (0 workers)
_CG_LP_MAX_WORKERS = 4                # overlap saturates; cap to avoid oversubscription


def _resolve_num_workers(num_batches):
    """Pick DataLoader worker count from a loader's batch count.

    Few batches -> 0 (in-process; fork/teardown would dominate). Otherwise a
    small fixed count that overlaps CPU tree sampling with GPU compute. The
    count depends only on the workload (not on CPUs available), so the
    worker-RNG-dependent tree sampling stays reproducible across SLURM
    allocations rather than shifting with `--cpus-per-task`.
    """
    if num_batches < _CG_LP_MIN_BATCHES_FOR_WORKERS:
        return 0
    return _CG_LP_MAX_WORKERS


class _MergedCGBatchMixin:
    """Shared merged computation-graph machinery for the CGT link predictors.

    The DataLoader (workers) only samples merged-tree node IDs; the expensive,
    feature-dimension-dependent work — gathering node features and assembling
    the batched DGL graph — runs on the main process / GPU here, reusing:

      * a resident, zero-padded feature matrix on the device (gather by index),
      * a CSR adjacency built once per trial (not per epoch), and
      * a batched-graph skeleton cached per batch size (topology is fixed; only
        ``ndata['feature']`` changes per batch).

    This replaces the previous path that built a full ``DGLGraph`` with features
    inside each worker and serialised it across the process boundary every
    batch — the dominant cost in the old pipeline. DataLoader worker count is
    auto-sized per loader (see ``_resolve_num_workers``), and negative sampling
    runs on-device against a resident edge-set so it doesn't bottleneck the
    now-faster training loop.
    """

    def _setup_merged_cg(self, data, test_features):
        self.step_num = self.train_config.get('step_num', 2)
        self.sample_num = self.train_config.get('sample_num', 5)
        self.noise_num = self.train_config.get('noise_num', 0)
        self.self_connection = self.train_config.get('self_connection', False)
        self.batch_size = self.train_config.get('batch_size', 256)

        # Tree sampling uses train_graph (no test/val edge leakage).
        self.train_adj_list = dgl_to_adj_list(data.train_graph)
        self.features = data.graph.ndata['feature'].cpu().numpy().astype(
            np.float32)
        self.num_nodes_feat = self.features.shape[0]
        self.feat_dim = self.features.shape[1]

        # CSR adjacency built once per trial and reused across every epoch's
        # loader (was previously rebuilt inside each dataset construction).
        self._csr = _build_csr_adj(self.train_adj_list, self.num_nodes_feat)

        # Resident, zero-padded feature matrices on the device. Trees emit node
        # IDs in [0, N]; row N (empty_id) is the zero pad. Gather is by index.
        self.feat_gpu = self._padded_device_features(self.features)

        # TSTR test path: real features for test-edge scoring when self.features
        # carries synthetic/quantized train+val rows. See the class docstrings
        # and the anomaly-benchmark mirror for rationale. When None, the
        # detector falls back to self.features everywhere (original-cg path).
        if test_features is not None:
            tf = np.ascontiguousarray(test_features, dtype=np.float32)
            if tf.shape != self.features.shape:
                raise ValueError(
                    f"test_features shape {tf.shape} != "
                    f"features shape {self.features.shape}")
            self.test_features = tf
            self.test_feat_gpu = self._padded_device_features(tf)
        else:
            self.test_features = None
            self.test_feat_gpu = None

        merged_adj = compute_merged_tree_adj(
            self.step_num, self.sample_num + self.noise_num,
            self.self_connection)
        self.template_src, self.template_dst, self.num_merged_nodes = \
            compute_template_edges(merged_adj)
        self.per_tree_num_nodes = self.num_merged_nodes // 2

        # Resident sorted edge-hash set for on-device negative collision checks
        # (was a CPU torch.isin against a large edge set every epoch).
        self.edge_set_gpu = self.edge_set.to(self.device)

        # Batched-graph skeletons keyed by batch size (number of edges).
        self._skeleton_cache = {}
        # Node-ID caches for the fixed eval edge sets (val/test). Trees are
        # sampled once and frozen; features are gathered fresh each epoch.
        self._eval_batch_cache = {}
        self._test_eval_batch_cache = {}

    def _padded_device_features(self, feats_np):
        t = torch.from_numpy(np.ascontiguousarray(feats_np, dtype=np.float32))
        pad = torch.zeros(1, t.shape[1], dtype=t.dtype)
        return torch.cat([t, pad], dim=0).to(self.device)

    def _edge_loader(self, edges):
        """DataLoader yielding flat int64 node-ID tensors per batch."""
        ds = MergedOriginalCompGraphDataset(
            None, self.num_nodes_feat, edges,
            self.step_num, self.sample_num,
            self.noise_num, self.self_connection, csr=self._csr)
        collate = make_merged_comp_graph_collate()
        num_batches = (len(ds) + self.batch_size - 1) // self.batch_size
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate, num_workers=_resolve_num_workers(num_batches))

    def _get_skeleton(self, m):
        g = self._skeleton_cache.get(m)
        if g is None:
            g = build_merged_skeleton(
                self.template_src, self.template_dst,
                self.num_merged_nodes, m).to(self.device)
            self._skeleton_cache[m] = g
        return g

    def _assemble(self, node_ids, feat_gpu):
        """Build a batched merged-CG graph from device node IDs + feature gather."""
        m = node_ids.shape[0] // self.num_merged_nodes
        g = self._get_skeleton(m)
        g.ndata['feature'] = feat_gpu[node_ids]
        return g

    # --- Negative sampling (on-device; shared by GNN and tree-model LP) ------
    # Training negatives are resampled every epoch, so this ran hot once the
    # data pipeline was fixed. Drawing and collision-filtering on the device
    # (against the resident `edge_set_gpu`) avoids the per-epoch CPU work and
    # host->device transfer. Returns edges already on `self.device`.

    def _filter_collisions(self, neg_edges):
        src, dst = neg_edges[:, 0], neg_edges[:, 1]
        N = self.num_nodes
        for _ in range(10):
            hashes = src.long() * N + dst.long()
            collision = torch.isin(hashes, self.edge_set_gpu) | (src == dst)
            if not collision.any():
                break
            n_bad = int(collision.sum().item())
            dst[collision] = torch.randint(
                0, N, (n_bad,), device=neg_edges.device)
        return neg_edges

    def _sample_random_negatives(self, n):
        neg = torch.stack([
            torch.randint(0, self.num_nodes, (n,), device=self.device),
            torch.randint(0, self.num_nodes, (n,), device=self.device),
        ], dim=1)
        return self._filter_collisions(neg)

    def _sample_hard_negatives(self, pos_edges):
        # 2-hop walk needs the CPU graph; the rest stays on-device.
        src = pos_edges[:, 0]
        walk_nodes, _ = dgl.sampling.random_walk(
            self.train_graph.cpu(), src.cpu(), metapath=[None, None])
        hard_dst = walk_nodes[:, 2].to(self.device)
        failed = hard_dst == -1
        if failed.any():
            hard_dst = hard_dst.clone()
            hard_dst[failed] = torch.randint(
                0, self.num_nodes, (int(failed.sum().item()),),
                device=self.device)
        neg = torch.stack([src.to(self.device), hard_dst], dim=1)
        return self._filter_collisions(neg)


class MergedCompGraphLinkPredictor(_MergedCGBatchMixin, BaseDetector):
    """Link prediction via per-edge merged endpoint computation graphs."""

    def __init__(self, train_config, model_config, data, test_features=None):
        super().__init__(train_config, model_config, data)
        self.device = train_config['device']

        # Static merged-CG state: CSR adjacency, resident device feature
        # matrices, templates, and skeleton/eval caches (see the mixin).
        self._setup_merged_cg(data, test_features)

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

        num_layers = model_config.get('num_layers', 2)
        if num_layers != self.step_num:
            print(
                f"  WARNING: num_layers={num_layers} != step_num={self.step_num}. "
                f"GNN depth should equal merged-tree depth for a fair "
                f"comparison with full-graph LP.")

    def _score_batch_on_device(self, batched):
        h = self.model(batched)
        h_u, h_v = extract_edge_root_embeddings(
            batched, h, self.per_tree_num_nodes)
        if self.decoder is not None:
            return self.decoder.score_from_pair(h_u, h_v)
        return (h_u * h_v).sum(dim=-1)

    def _score_node_ids(self, node_ids, feat_gpu):
        return self._score_batch_on_device(self._assemble(node_ids, feat_gpu))

    def _score_cached(self, edges, cache, feat_gpu):
        """Score a fixed edge set, caching its (frozen) merged-tree node IDs.

        For val/test edge sets the trees are sampled once and cached as device
        node-ID tensors; each call only re-gathers features from `feat_gpu` and
        re-runs the GNN forward. Train-time edges are not cached here — they are
        resampled every epoch and streamed directly through `_train_step`.
        """
        if edges.shape[0] == 0:
            return torch.empty(0, device=self.device)

        cache_key = id(edges)
        cached = cache.get(cache_key)
        if cached is None:
            cached = [nid.to(self.device) for nid in self._edge_loader(edges)]
            cache[cache_key] = cached

        scores = [self._score_node_ids(nid, feat_gpu) for nid in cached]
        return torch.cat(scores, dim=0)

    def _score_edges(self, edges):
        """Score edges against self.features (train/val path)."""
        return self._score_cached(edges, self._eval_batch_cache, self.feat_gpu)

    def _score_test_edges(self, edges):
        """Score edges against real (TSTR) features for the test path.

        Mirrors _score_edges but gathers node features from self.test_feat_gpu
        so test-edge computation trees never read the synthetic/quantized rows
        planted at train+val mask positions of self.features. The node IDs are
        identical to the train/val path (sampling is feature-independent); only
        the gathered feature values differ, so a separate cache keeps the two
        feature sources cleanly partitioned.
        """
        return self._score_cached(
            edges, self._test_eval_batch_cache, self.test_feat_gpu)

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
        for node_ids_cpu in self._edge_loader(edges):
            t_after_next = time.time()
            if ts is not None:
                ts['data_wait'] += t_after_next - t_prev

            node_ids = node_ids_cpu.to(self.device, non_blocking=True)
            s = self._score_node_ids(node_ids, self.feat_gpu)
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

            test_score_fn = (self._score_test_edges
                             if self.test_features is not None
                             else self._score_edges)
            with torch.no_grad():
                test_pos = torch.sigmoid(
                    test_score_fn(self.test_pos_edges))
                test_neg = torch.sigmoid(
                    test_score_fn(self.test_neg_edges))
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


class CGTXGBoostLinkPredictor(_MergedCGBatchMixin, BaseDetector):
    """XGBoost LP on merged comp graphs. Per edge: Hadamard of
    (h_u, h_v) at positions 0 and T of the merged tree. Base class
    uses raw root features (no GIN); subclass adds GIN_noparam
    aggregation. CGT-only.
    """

    def __init__(self, train_config, model_config, data, test_features=None):
        super().__init__(train_config, model_config, data)
        import xgboost as xgb

        self.device = train_config['device']

        # Static merged-CG state: CSR adjacency, resident device feature
        # matrices, templates, and skeleton caches (see the mixin).
        self._setup_merged_cg(data, test_features)

        eval_metric = (roc_auc_score if train_config['metric'] == 'AUROC'
                       else average_precision_score)
        cfg = {k: v for k, v in model_config.items() if k != 'model'}
        self.model = xgb.XGBClassifier(
            tree_method='hist', eval_metric=eval_metric, verbose=False, **cfg)

        self.gin = None

    def _edge_features(self, edges, feat_gpu=None):
        fg = self.feat_gpu if feat_gpu is None else feat_gpu
        if edges.shape[0] == 0:
            feat_dim = self.feat_dim
            if self.gin is not None:
                feat_dim *= self.model_config.get('num_layers', 2) + 1
            return np.zeros((0, feat_dim), dtype=np.float32)

        feats = []
        with torch.no_grad():
            for node_ids_cpu in self._edge_loader(edges):
                node_ids = node_ids_cpu.to(self.device)
                g = self._assemble(node_ids, fg)
                # Base XGBoost reads the raw root features; XGBGraph first runs
                # GIN_noparam message passing over the merged tree.
                emb = g.ndata['feature'] if self.gin is None else self.gin(g)
                h_u, h_v = extract_edge_root_embeddings(
                    g, emb, self.per_tree_num_nodes)
                feats.append((h_u * h_v).cpu().numpy())
        return np.vstack(feats)

    def _test_edge_features(self, edges):
        """Materialize edge features using real (TSTR) node features."""
        return self._edge_features(edges, feat_gpu=self.test_feat_gpu)

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
        test_feat_fn = (self._test_edge_features
                        if self.test_features is not None
                        else self._edge_features)
        test_X = np.vstack([
            test_feat_fn(self.test_pos_edges),
            test_feat_fn(self.test_neg_edges),
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

    def __init__(self, train_config, model_config, data, test_features=None):
        super().__init__(train_config, model_config, data,
                         test_features=test_features)
        from models.gnn import GIN_noparam

        num_layers = model_config.get('num_layers', 2)
        if num_layers != self.step_num:
            raise ValueError(
                f"CGTXGBGraphLinkPredictor requires num_layers == step_num; "
                f"got num_layers={num_layers}, step_num={self.step_num}.")

        self.gin = GIN_noparam(**model_config).to(self.device).eval()
