import argparse
import math
import os
import numpy as np
import torch
import dgl
from sklearn.preprocessing import normalize

from data.comp_graph import (
    OriginalCompGraphDataset,
    SyntheticCompGraphDataset,
    dgl_to_adj_list,
)
from models.anomaly_detection.cgt_detector import CG_SUPPORTED_MODELS

SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE', 'XGBGraph']
LP_SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE']


def parse_link_args():
    """Parse arguments for link prediction benchmark."""
    parser = argparse.ArgumentParser(
        description='Benchmark GNNs on link prediction: original vs synthetic')

    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--datasets', type=str, required=True,
                            help='Comma-separated dataset names')
    data_group.add_argument('--models', type=str, default=','.join(LP_SUPPORTED_MODELS),
                            help='Comma-separated model names')
    data_group.add_argument('--data_dir', type=str, default=None,
                            help='Path to datasets root (default: datasets/)')
    data_group.add_argument('--synthetic_dir', type=str, default=None,
                            help='Path to synthetic datasets (default: datasets/synthetic)')
    data_group.add_argument('--output_dir', type=str, default=None,
                            help='Directory to save results (default: results/evaluate)')
    data_group.add_argument('--synthetic_type', type=str, default='comp-graph',
                            choices=['graph', 'comp-graph'],
                            help='Synthetic data format: "graph" or "comp-graph"')
    data_group.add_argument('--generator', type=str, default=None,
                            help='Generative model subfolder (e.g. "cgt", "bigg")')
    data_group.add_argument('--synthetic_name', type=str, default=None,
                            help='Exact filename stem for a specific variant')
    data_group.add_argument('--task', type=str, default='hidden_links',
                            choices=['hidden_labels', 'hidden_links', 'structure'],
                            help='Task subfolder under <dataset>/ '
                                 '(hidden_labels, hidden_links, or structure). '
                                 'Resolved path: '
                                 '<synthetic_dir>/<generator>/<dataset>/<task>/<stem>[.pt].')

    lp_group = parser.add_argument_group('Link prediction')
    lp_group.add_argument('--val_ratio', type=float, default=0.05,
                          help='Fraction of edges for validation')
    lp_group.add_argument('--test_ratio', type=float, default=0.1,
                          help='Fraction of edges for test')
    lp_group.add_argument('--neg_sampling', type=str, default='random',
                          choices=['random', 'hard'],
                          help='Negative sampling strategy')
    lp_group.add_argument('--decoder', type=str, default='dot',
                          choices=['dot', 'mlp'],
                          help='Edge decoder: dot product or MLP')

    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--trials', type=int, default=1)
    train_group.add_argument('--epochs', type=int, default=200)
    train_group.add_argument('--patience', type=int, default=50)
    train_group.add_argument('--batch_size', type=int, default=256,
                             help='Batch size (CGT comp-graph mode only)')

    model_group = parser.add_argument_group('Model architecture')
    model_group.add_argument('--lr', type=float, default=0.01)
    model_group.add_argument('--drop_rate', type=float, default=0.0)
    model_group.add_argument('--h_feats', type=int, default=32)
    model_group.add_argument('--num_layers', type=int, default=2)

    return parser.parse_args()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark GNNs on original vs synthetic graph data')

    # --- Data ---
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--datasets', type=str, required=True,
                            help='Comma-separated dataset names (e.g., "reddit,weibo,amazon")')
    data_group.add_argument('--models', type=str, default=','.join(SUPPORTED_MODELS),
                            help='Comma-separated model names')
    data_group.add_argument('--data_dir', type=str, default=None,
                            help='Path to original datasets (default: datasets/original)')
    data_group.add_argument('--synthetic_dir', type=str, default=None,
                            help='Path to synthetic datasets (default: datasets/synthetic)')
    data_group.add_argument('--output_dir', type=str, default=None,
                            help='Directory to save results (default: results/evaluate)')
    data_group.add_argument('--synthetic_type', type=str, default='comp-graph',
                            choices=['graph', 'comp-graph'],
                            help='Format of the synthetic data, determines evaluation mode: '
                                 '"graph" = full DGL graph, evaluated with whole-graph GNNs; '
                                 '"comp-graph" = computation graph sequences (.pt), evaluated '
                                 'with computation-graph GNNs.')
    data_group.add_argument('--generator', type=str, default=None,
                            help='Name of the generative model — used as the subfolder '
                                 'under --synthetic_dir (e.g. "bigg", "cgt"). Required. '
                                 'Resolved path: '
                                 '<synthetic_dir>/<generator>/<synthetic_name or dataset>[.pt].')
    data_group.add_argument('--synthetic_name', type=str, default=None,
                            help='Filename stem of the synthetic variant to evaluate. '
                                 'Required when a dataset has multiple synthetic variants. '
                                 'Resolved path: <synthetic_dir>/<generator>/<dataset>/<task>/<stem>[.pt]. '
                                 'Example: "--generator bigg --task hidden_labels --datasets tolokers --synthetic_name blksize_1024_b_1_lr_0.001_epochs_50" '
                                 'resolves to synthetic/bigg/tolokers/hidden_labels/blksize_1024_b_1_lr_0.001_epochs_50.')
    data_group.add_argument('--task', type=str, default='hidden_labels',
                            choices=['hidden_labels', 'hidden_links', 'structure'],
                            help='Task subfolder under <dataset>/ '
                                 '(hidden_labels, hidden_links, or structure). '
                                 'Resolved path: '
                                 '<synthetic_dir>/<generator>/<dataset>/<task>/<stem>[.pt].')
    data_group.add_argument('--semi_supervised', type=int, default=0,
                            help='Use semi-supervised split (0 or 1)')
    data_group.add_argument('--trial_id', type=int, default=0,
                            help='Trial ID for mask split (must match CGT training)')

    # --- Training ---
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--trials', type=int, default=1,
                             help='Number of evaluation trials per model/dataset')
    train_group.add_argument('--epochs', type=int, default=200,
                             help='Max training epochs')
    train_group.add_argument('--patience', type=int, default=50,
                             help='Early stopping patience')
    train_group.add_argument('--batch_size', type=int, default=256,
                             help='Batch size for computation graph mode (CGT); '
                                  'not used by whole-graph GNNs')

    # --- Model architecture ---
    # These must match between whole-graph and computation-graph GNNs for a fair comparison.
    # num_layers controls the receptive field depth and must equal cg_depth used during
    # CGT training (stored in the .pt file). lr, drop_rate, and h_feats must also be
    # identical across both training modes.
    model_group = parser.add_argument_group('Model architecture')
    model_group.add_argument('--lr', type=float, default=0.01,
                             help='Learning rate for Adam optimizer (applies to both '
                                  'whole-graph and computation-graph GNNs)')
    model_group.add_argument('--drop_rate', type=float, default=0.0,
                             help='Dropout rate (applies to both whole-graph and '
                                  'computation-graph GNNs)')
    model_group.add_argument('--h_feats', type=int, default=32,
                             help='Hidden feature dimension. Overridden to 16 for '
                                  'tsocial regardless of this value.')
    model_group.add_argument('--num_layers', type=int, default=2,
                             help='Number of GNN layers / message-passing hops. '
                                  'For a fair comparison with computation-graph GNNs '
                                  'this must equal the cg_depth used during CGT training '
                                  '(default: 2).')

    return parser.parse_args()


def apply_normalization(features, stats):
    """Apply a previously computed normalization to a feature tensor.

    Parameters
    ----------
    features : torch.Tensor
        Node feature matrix of shape (N, D).
    stats : dict
        Stats dict saved by BiGG pipeline (contains 'method' and params).
        If stats contains ``'binary_idx'``, those columns are left untouched.
    """
    binary_idx = stats.get('binary_idx', [])
    if binary_idx:
        all_idx = list(range(features.shape[1]))
        cont_idx = sorted(set(all_idx) - set(binary_idx))
        cont_features = features[:, cont_idx]
    else:
        cont_features = features

    method = stats['method']

    if method == 'zscore':
        cont_features = (cont_features - stats['mean']) / stats['std']
    elif method == 'minmax':
        cont_features = (cont_features - stats['min']) / stats['denom']
    elif method == 'row':
        norms = cont_features.norm(p=2, dim=1, keepdim=True)
        norms[norms == 0] = 1.0
        cont_features = cont_features / norms
    elif method == 'quantile':
        sorted_values = stats['sorted_values']  # (N_train, D_cont)
        N_train = sorted_values.shape[0]
        D = cont_features.shape[1]
        eps = 1e-6
        transformed = torch.empty_like(cont_features)
        for d in range(D):
            col = cont_features[:, d]
            sv = sorted_values[:, d]
            ranks = torch.searchsorted(sv, col).clamp(0, N_train - 1).float()
            uniform = (ranks + 0.5) / N_train
            uniform = uniform.clamp(eps, 1.0 - eps)
            transformed[:, d] = math.sqrt(2) * torch.erfinv(2 * uniform - 1)
        cont_features = transformed

    if binary_idx:
        features = features.clone()
        features[:, cont_idx] = cont_features
    else:
        features = cont_features

    return features


def invert_normalization(features, stats):
    """Invert a previously computed normalization to recover original-space features.

    Only lossless methods (zscore, minmax, quantile) are invertible.
    Raises ``ValueError`` for ``'row'`` normalization (lossy).

    Parameters
    ----------
    features : torch.Tensor
        Normalised feature matrix of shape (N, D).
    stats : dict
        Stats dict saved by BiGG pipeline.
        If stats contains ``'binary_idx'``, those columns are left untouched.
    """
    binary_idx = stats.get('binary_idx', [])
    if binary_idx:
        all_idx = list(range(features.shape[1]))
        cont_idx = sorted(set(all_idx) - set(binary_idx))
        cont_features = features[:, cont_idx]
    else:
        cont_features = features

    method = stats['method']

    if method == 'zscore':
        cont_features = cont_features * stats['std'] + stats['mean']
    elif method == 'minmax':
        cont_features = cont_features * stats['denom'] + stats['min']
    elif method == 'quantile':
        sorted_values = stats['sorted_values']  # (N_train, D_cont)
        N_train = sorted_values.shape[0]
        D = cont_features.shape[1]
        inverted = torch.empty_like(cont_features)
        for d in range(D):
            col = cont_features[:, d]
            sv = sorted_values[:, d]
            uniform = 0.5 * (1.0 + torch.erf(col / math.sqrt(2)))
            indices = (uniform * (N_train - 1)).clamp(0, N_train - 1)
            lo = indices.long().clamp(0, N_train - 2)
            hi = lo + 1
            frac = indices - lo.float()
            inverted[:, d] = sv[lo] * (1 - frac) + sv[hi] * frac
        cont_features = inverted
    elif method == 'row':
        raise ValueError(
            "Row (L2) normalization is lossy and cannot be inverted — "
            "original magnitudes are not stored."
        )

    if binary_idx:
        features = features.clone()
        features[:, cont_idx] = cont_features
    else:
        features = cont_features

    return features


def load_bigg_synthetic_graph(path):
    """Load a BiGG-generated DGL graph from *path*.

    *path* may be either:
    - A file path: loaded directly as a single DGL graph.
    - A directory (subsampled run): all ``subgraph_*`` files are loaded and
      combined into one block-diagonal graph via ``dgl.batch()``.

    Also returns the norm_stats dict (or None) located alongside the graph.
    """
    if os.path.isdir(path):
        subgraph_files = sorted(
            f for f in os.listdir(path) if f.startswith('subgraph_')
        )
        if not subgraph_files:
            raise FileNotFoundError(f'No subgraph_* files found in {path}')
        graphs = []
        for fname in subgraph_files:
            gs, _ = dgl.load_graphs(os.path.join(path, fname))
            graphs.append(gs[0])
        combined = dgl.batch(graphs)
        stats_path = os.path.join(path, 'norm_stats.pt')
        norm_stats = torch.load(stats_path, weights_only=False) if os.path.exists(stats_path) else None
        print(f'  Loaded {len(graphs)} subgraphs → combined: '
              f'{combined.num_nodes()} nodes, {combined.num_edges()} edges')
        return combined, norm_stats
    else:
        graphs, _ = dgl.load_graphs(path)
        stats_path = path + '_norm_stats.pt'
        norm_stats = torch.load(stats_path, weights_only=False) if os.path.exists(stats_path) else None
        return graphs[0], norm_stats


def load_cgt_synthetic_data(syn_path):
    """Load CGT synthetic data from a .pt file."""
    try:
        return torch.load(syn_path, weights_only=False)
    except TypeError:
        return torch.load(syn_path)


def build_synthetic_dgl_graph(original_graph, synthetic_data,
                              trial_id=0, semi_supervised=False):
    """
    Build a DGL graph with synthetic node features for train/val nodes.

    Keeps the original graph structure (edges) and test node features/labels.
    Replaces train/val node features with CGT-generated features derived from
    cluster centers corresponding to the generated computation graph root nodes.

    CGT L2-normalizes features before clustering, so synthetic features are in
    L2-normalized space. If original features are unnormalized there may be a
    distribution shift between synthetic train/val and original test features.
    """
    mask_col = trial_id + (10 if semi_supervised else 0)
    train_mask = original_graph.ndata['train_masks'][:, mask_col].bool()
    val_mask = original_graph.ndata['val_masks'][:, mask_col].bool()

    train_node_ids = torch.where(train_mask)[0]
    val_node_ids = torch.where(val_mask)[0]

    cluster_centers = synthetic_data['cluster_centers']
    gen_train_seqs = synthetic_data['gen_train_ids']
    gen_val_seqs = synthetic_data['gen_val_ids']

    # Use explicit node IDs when available (more robust than mask recovery)
    if 'ids' in synthetic_data:
        saved_ids = synthetic_data['ids']
        train_node_ids = torch.tensor(saved_ids['train'], dtype=torch.long)
        val_node_ids = torch.tensor(saved_ids['val'], dtype=torch.long)

    # Root of each computation graph tree = position 0
    syn_train_feats = cluster_centers[gen_train_seqs[:, 0]].float()
    syn_val_feats = cluster_centers[gen_val_seqs[:, 0]].float()

    orig_feat_dim = original_graph.ndata['feature'].shape[1]
    syn_feat_dim = syn_train_feats.shape[1]
    if orig_feat_dim != syn_feat_dim:
        raise ValueError(
            f"Feature dimension mismatch: original={orig_feat_dim}, "
            f"synthetic={syn_feat_dim}")
    if len(train_node_ids) != len(syn_train_feats):
        raise ValueError(
            f"Train node count mismatch: {len(train_node_ids)} vs "
            f"{len(syn_train_feats)} synthetic sequences")
    if len(val_node_ids) != len(syn_val_feats):
        raise ValueError(
            f"Val node count mismatch: {len(val_node_ids)} vs "
            f"{len(syn_val_feats)} synthetic sequences")

    new_features = original_graph.ndata['feature'].clone().float()
    new_features[train_node_ids] = syn_train_feats
    new_features[val_node_ids] = syn_val_feats

    src, dst = original_graph.edges()
    syn_graph = dgl.graph((src, dst), num_nodes=original_graph.num_nodes())
    syn_graph.ndata['feature'] = new_features
    syn_graph.ndata['label'] = original_graph.ndata['label'].clone()
    syn_graph.ndata['train_masks'] = original_graph.ndata['train_masks'].clone()
    syn_graph.ndata['val_masks'] = original_graph.ndata['val_masks'].clone()
    syn_graph.ndata['test_masks'] = original_graph.ndata['test_masks'].clone()

    print(f"  Synthetic graph: {syn_graph.num_nodes()} nodes, "
          f"{syn_graph.num_edges()} edges | "
          f"replaced {len(train_node_ids)} train + {len(val_node_ids)} val features")

    return syn_graph


def print_comparison(all_results, datasets, models):
    """Print formatted comparison of original vs synthetic results."""
    sources = sorted(set(r['source'] for r in all_results))

    print("\n" + "=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)

    for dataset in datasets:
        print(f"\n  Dataset: {dataset}")
        header = (f"  {'Model':<14} {'Source':<24} "
                  f"{'AUROC':>15} {'AUPRC':>15} {'RecK':>15}")
        print(header)
        print(f"  {'-' * 86}")

        for model in models:
            for source in sources:
                matches = [r for r in all_results
                           if r['source'] == source
                           and r['dataset'] == dataset
                           and r['model'] == model]
                if matches:
                    r = matches[0]
                    auroc = f"{r['AUROC_mean']:.4f}\u00b1{r['AUROC_std']:.4f}"
                    auprc = f"{r['AUPRC_mean']:.4f}\u00b1{r['AUPRC_std']:.4f}"
                    reck = f"{r['RecK_mean']:.4f}\u00b1{r['RecK_std']:.4f}"
                    print(f"  {model:<14} {source:<24} "
                          f"{auroc:>15} {auprc:>15} {reck:>15}")
            print()


def _extract_cg_params(syn_data):
    """Extract computation graph tree parameters from CGT .pt data."""
    step_num = syn_data.get('cg_depth', syn_data.get('subgraph_step_num'))
    sample_num = syn_data.get('cg_fanout', syn_data.get('subgraph_sample_num'))
    noise_num = syn_data.get('noise_num', 0)
    self_conn = syn_data.get('self_connection', False)
    return step_num, sample_num, noise_num, self_conn


def _assert_pt_alignment(syn_data, expected_trial_id, expected_semi_supervised,
                         expected_test_ids, source_label):
    """Verify a CGT .pt file matches the benchmark's split configuration.

    Hard-fails on metadata mismatch or structural test-id disagreement.
    Warns + falls back to the structural check when the .pt predates
    provenance metadata (no 'trial_id' / 'semi_supervised' keys).
    """
    saved_trial = syn_data.get('trial_id')
    saved_semi = syn_data.get('semi_supervised')

    if saved_trial is None or saved_semi is None:
        print(f"  WARNING [{source_label}]: .pt lacks provenance metadata "
              f"(trial_id/semi_supervised). Falling back to structural check. "
              f"Re-run CGT training to embed metadata.")
    else:
        assert saved_trial == expected_trial_id, (
            f"[{source_label}] trial_id mismatch: .pt was trained with "
            f"trial_id={saved_trial}, benchmark expects {expected_trial_id}.")
        assert bool(saved_semi) == bool(expected_semi_supervised), (
            f"[{source_label}] semi_supervised mismatch: .pt="
            f"{bool(saved_semi)}, benchmark={bool(expected_semi_supervised)}.")

    saved_test = set(int(x) for x in syn_data['ids']['test'])
    got_test = set(int(x) for x in expected_test_ids)
    assert saved_test == got_test, (
        f"[{source_label}] test split desync: .pt's ids['test'] "
        f"(|S|={len(saved_test)}) != mask-derived test_ids "
        f"(|S|={len(got_test)}). Likely trial_id or semi_supervised mismatch, "
        f"or a renamed/moved .pt.")


def _assert_link_pt_alignment(syn_data, expected_trial_id, expected_test_edges,
                              source_label):
    """Verify a CGT .pt file was trained under the same hidden_links split.

    Link-prediction provenance: the .pt must carry `hidden_test_edges`
    equal to the downstream `LinkDataset.split(trial_id).test_pos_edges`.
    Any mismatch means CGT saw edges that the GNN is about to test on
    (or vice versa), which breaks the fairness invariant.
    """
    saved_trial = syn_data.get('trial_id')
    saved_task = syn_data.get('task')

    if saved_task != 'hidden_links':
        print(f"  WARNING [{source_label}]: .pt task={saved_task!r} != "
              f"'hidden_links'. CGT may not have withheld test edges.")

    if saved_trial is None:
        print(f"  WARNING [{source_label}]: .pt lacks trial_id metadata. "
              f"Re-train CGT to embed it.")
    elif saved_trial != expected_trial_id:
        raise AssertionError(
            f"[{source_label}] trial_id mismatch: .pt was trained with "
            f"trial_id={saved_trial}, benchmark expects {expected_trial_id}.")

    saved_edges = syn_data.get('hidden_test_edges')
    if saved_edges is None:
        print(f"  WARNING [{source_label}]: .pt lacks hidden_test_edges; "
              f"cannot verify test-edge alignment.")
        return

    if torch.is_tensor(saved_edges):
        saved_set = {(int(r[0]), int(r[1])) for r in saved_edges.cpu()}
    else:
        saved_set = {(int(s), int(d)) for s, d in saved_edges}
    exp = expected_test_edges.cpu() if torch.is_tensor(expected_test_edges) \
        else expected_test_edges
    expected_set = {(int(r[0]), int(r[1])) for r in exp}

    if saved_set != expected_set:
        raise AssertionError(
            f"[{source_label}] hidden_test_edges mismatch: "
            f"saved |S|={len(saved_set)}, expected |S|={len(expected_set)}. "
            f"CGT's trial-{expected_trial_id} edge split differs from the "
            f"downstream split. Check val_ratio/test_ratio and that both "
            f"sides derive seed as 3407 + trial_id*10.")


def resolve_cgt_trial_paths(syn_path, num_trials):
    """Check for per-trial .pt files: {stem}_t{t}.pt for t in 0..num_trials-1.

    Returns list of paths if all num_trials files exist, else None.
    """
    base, ext = os.path.splitext(syn_path)
    paths = [f"{base}_t{t}{ext}" for t in range(num_trials)]
    found = [p for p in paths if os.path.exists(p)]

    if len(found) == num_trials:
        return paths
    elif len(found) > 0:
        print(f"  WARNING: found {len(found)}/{num_trials} per-trial .pt files. "
              f"Need all {num_trials}. Falling back to single-file mode.")
    return None


def make_cgt_rebuild_fn(original_graph, trial_paths,
                        trial_id_offset=0, semi_supervised=False):
    """Return a rebuild_datasets_fn(t) that loads per-trial .pt files.

    For trial t: loads trial_paths[t], builds synthetic train/val from it,
    and builds the test set from the original graph using mask column
    (trial_id_offset + t) so the test split varies across trials.
    """
    # Precompute graph data shared across trials
    adj_list = dgl_to_adj_list(original_graph)
    features = original_graph.ndata['feature'].cpu().numpy().astype(np.float32)
    features = normalize(features, axis=1, norm='l2')
    labels = original_graph.ndata['label'].cpu().numpy().astype(np.int64)

    def rebuild(t):
        syn_data = load_cgt_synthetic_data(trial_paths[t])
        step_num, sample_num, noise_num, self_conn = _extract_cg_params(syn_data)
        total_sample = sample_num + noise_num

        # Test set from original graph with matching mask column
        mask_col = (trial_id_offset + t) + (10 if semi_supervised else 0)
        test_ids = original_graph.ndata['test_masks'][:, mask_col].bool().nonzero(
            as_tuple=True)[0].numpy()

        _assert_pt_alignment(
            syn_data,
            expected_trial_id=trial_id_offset + t,
            expected_semi_supervised=semi_supervised,
            expected_test_ids=test_ids,
            source_label=f'synthetic-cgt[t={t}]',
        )

        test_ds = OriginalCompGraphDataset(
            adj_list, features, labels, test_ids,
            step_num, sample_num, noise_num, self_conn)

        cluster_centers = syn_data['cluster_centers']
        if not isinstance(cluster_centers, torch.Tensor):
            cluster_centers = torch.tensor(cluster_centers)

        syn_train = SyntheticCompGraphDataset(
            syn_data['gen_train_ids'], syn_data['train_labels'],
            cluster_centers, step_num, total_sample, self_conn)
        syn_val = SyntheticCompGraphDataset(
            syn_data['gen_val_ids'], syn_data['val_labels'],
            cluster_centers, step_num, total_sample, self_conn)

        print(f"  Trial {t}: syn_train={len(syn_train)}, syn_val={len(syn_val)}, "
              f"test={len(test_ds)}")

        return syn_train, syn_val, test_ds

    return rebuild


def build_cgt_datasets(original_graph, syn_data):
    """Build CGT computation graph datasets for synthetic training.

    Features are L2-normalized to match the space CGT uses internally
    (CGT normalizes features before clustering, so cluster centers are in
    L2-normalized space). Normalizing here ensures the test set's feature
    distribution matches the synthetic train/val features.

    Returns:
        syn_train: SyntheticCompGraphDataset for train nodes
        syn_val: SyntheticCompGraphDataset for val nodes
        test_ds: OriginalCompGraphDataset for test nodes (L2-normalized features)
    """
    step_num, sample_num, noise_num, self_conn = _extract_cg_params(syn_data)
    total_sample = sample_num + noise_num

    # Get node IDs from saved splits
    ids = syn_data['ids']
    test_ids = ids['test']

    # Build test set from original graph (L2-normalized features to match
    # the cluster center space used for synthetic train/val)
    adj_list = dgl_to_adj_list(original_graph)
    features = original_graph.ndata['feature'].cpu().numpy().astype(np.float32)
    features = normalize(features, axis=1, norm='l2')
    labels = original_graph.ndata['label'].cpu().numpy().astype(np.int64)

    test_ds = OriginalCompGraphDataset(
        adj_list, features, labels, test_ids,
        step_num, sample_num, noise_num, self_conn)

    # Build synthetic train/val from CGT-generated cluster center features
    cluster_centers = syn_data['cluster_centers']
    if not isinstance(cluster_centers, torch.Tensor):
        cluster_centers = torch.tensor(cluster_centers)

    syn_train = SyntheticCompGraphDataset(
        syn_data['gen_train_ids'], syn_data['train_labels'],
        cluster_centers, step_num, total_sample, self_conn)
    syn_val = SyntheticCompGraphDataset(
        syn_data['gen_val_ids'], syn_data['val_labels'],
        cluster_centers, step_num, total_sample, self_conn)

    print(f"  CG datasets: syn_train={len(syn_train)}, syn_val={len(syn_val)}, "
          f"test={len(test_ds)} | "
          f"tree_nodes={test_ds.num_tree_nodes} "
          f"(step={step_num}, sample={sample_num}, noise={noise_num})")

    return syn_train, syn_val, test_ds


def build_original_cg_datasets(original_graph, syn_data,
                               trial_id=0, semi_supervised=False):
    """Build computation graph datasets from original data for all splits.

    Uses the same tree structure (step_num, sample_num, etc.) from the CGT
    .pt file, with L2-normalized features for train/val/test. Normalization
    matches the feature space CGT operates in, making the original-CG baseline
    directly comparable to the synthetic-CGT condition.

    Node IDs are derived from the graph's pre-stored mask columns so that
    different trial_ids produce different train/val/test splits, matching the
    split-varying behaviour of the whole-graph evaluation path.

    Returns:
        train_ds, val_ds, test_ds: OriginalCompGraphDataset for each split
    """
    step_num, sample_num, noise_num, self_conn = _extract_cg_params(syn_data)

    mask_col = trial_id + (10 if semi_supervised else 0)
    train_ids = original_graph.ndata['train_masks'][:, mask_col].bool().nonzero(as_tuple=True)[0].numpy()
    val_ids = original_graph.ndata['val_masks'][:, mask_col].bool().nonzero(as_tuple=True)[0].numpy()
    test_ids = original_graph.ndata['test_masks'][:, mask_col].bool().nonzero(as_tuple=True)[0].numpy()

    adj_list = dgl_to_adj_list(original_graph)
    features = original_graph.ndata['feature'].cpu().numpy().astype(np.float32)
    features = normalize(features, axis=1, norm='l2')
    labels = original_graph.ndata['label'].cpu().numpy().astype(np.int64)

    train_ds = OriginalCompGraphDataset(
        adj_list, features, labels, train_ids,
        step_num, sample_num, noise_num, self_conn)
    val_ds = OriginalCompGraphDataset(
        adj_list, features, labels, val_ids,
        step_num, sample_num, noise_num, self_conn)
    test_ds = OriginalCompGraphDataset(
        adj_list, features, labels, test_ids,
        step_num, sample_num, noise_num, self_conn)

    print(f"  Original CG datasets: train={len(train_ds)}, val={len(val_ds)}, "
          f"test={len(test_ds)} | "
          f"tree_nodes={test_ds.num_tree_nodes} "
          f"(step={step_num}, sample={sample_num}, noise={noise_num})")

    return train_ds, val_ds, test_ds
