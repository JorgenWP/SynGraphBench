import argparse
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
        header = (f"  {'Model':<14} {'Source':<16} "
                  f"{'AUROC':>18} {'AUPRC':>18} {'RecK':>18}")
        print(header)
        print(f"  {'-' * 82}")

        for model in models:
            for source in sources:
                matches = [r for r in all_results
                           if r['source'] == source
                           and r['dataset'] == dataset
                           and r['model'] == model]
                if matches:
                    r = matches[0]
                    print(f"  {model:<14} {source:<16} "
                          f"{r['AUROC_mean']:.4f}\u00b1{r['AUROC_std']:.4f}   "
                          f"{r['AUPRC_mean']:.4f}\u00b1{r['AUPRC_std']:.4f}   "
                          f"{r['RecK_mean']:.4f}\u00b1{r['RecK_std']:.4f}")
            print()


def _extract_cg_params(syn_data):
    """Extract computation graph tree parameters from CGT .pt data."""
    step_num = syn_data.get('cg_depth', syn_data.get('subgraph_step_num'))
    sample_num = syn_data.get('cg_fanout', syn_data.get('subgraph_sample_num'))
    noise_num = syn_data.get('noise_num', 0)
    self_conn = syn_data.get('self_connection', False)
    return step_num, sample_num, noise_num, self_conn


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
          f"tree_nodes={test_ds.tree_adj.shape[0]} "
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
          f"tree_nodes={test_ds.tree_adj.shape[0]} "
          f"(step={step_num}, sample={sample_num}, noise={noise_num})")

    return train_ds, val_ds, test_ds
