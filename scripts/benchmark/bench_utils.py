import argparse
import os
import numpy as np
import torch
import dgl

from data.comp_graph import (
    OriginalCompGraphDataset,
    SyntheticCompGraphDataset,
    dgl_to_adj_list,
)
from models.anomaly_detection.cgt_detector import CG_SUPPORTED_MODELS

SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE', 'XGBGraph']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark GNNs on original vs synthetic graph data')
    parser.add_argument('--datasets', type=str, required=True,
                        help='Comma-separated dataset names (e.g., "reddit,weibo,amazon")')
    parser.add_argument('--models', type=str, default=','.join(SUPPORTED_MODELS),
                        help='Comma-separated model names')
    parser.add_argument('--trials', type=int, default=1,
                        help='Number of evaluation trials per model/dataset')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to original datasets (default: datasets/original)')
    parser.add_argument('--synthetic_dir', type=str, default=None,
                        help='Path to synthetic datasets (default: datasets/synthetic)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: results/evaluate)')
    parser.add_argument('--synthetic_type', type=str, default='auto',
                        choices=['auto', 'graph', 'cgt'],
                        help='Type of synthetic data: '
                             '"graph" = full DGL graph (BiGG, etc.), '
                             '"cgt" = CGT computation graph sequences (.pt), '
                             '"auto" = detect from file extension')
    parser.add_argument('--semi_supervised', type=int, default=0,
                        help='Use semi-supervised split (0 or 1)')
    parser.add_argument('--trial_id', type=int, default=0,
                        help='Trial ID for mask split (must match CGT training)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Max training epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for computation graph mode (CGT)')
    parser.add_argument('--synthetic_model', type=str, default=None,
                        help='Generator model prefix for synthetic filenames '
                             '(e.g. "cgt" looks for cgt_<dataset>.pt, '
                             '"bigg" looks for bigg_<dataset>). '
                             'If not set, falls back to <dataset>.pt / <dataset>.')
    return parser.parse_args()


def find_synthetic_path(synthetic_dir, dataset_name, synthetic_type,
                        synthetic_model=None):
    """Locate the synthetic data file and resolve its type.

    Naming convention:
        With --synthetic_model <model>:
            CGT:   <synthetic_dir>/<model>_<dataset>.pt
            Graph: <synthetic_dir>/<model>_<dataset>
        Without (legacy fallback):
            CGT:   <synthetic_dir>/<dataset>.pt
            Graph: <synthetic_dir>/<dataset>

    Returns (path, resolved_type) or (None, None) if not found.
    """
    if synthetic_model:
        prefix = f'{synthetic_model}_{dataset_name}'
    else:
        prefix = dataset_name

    cgt_path = os.path.join(synthetic_dir, f'{prefix}.pt')
    graph_path = os.path.join(synthetic_dir, prefix)

    if synthetic_type == 'cgt':
        return (cgt_path, 'cgt') if os.path.exists(cgt_path) else (None, None)
    if synthetic_type == 'graph':
        return (graph_path, 'graph') if os.path.exists(graph_path) else (None, None)

    # auto-detect: prefer full graph file (no extension), fall back to .pt
    if os.path.isfile(graph_path) and not graph_path.endswith('.pt'):
        return graph_path, 'graph'
    if os.path.exists(cgt_path):
        return cgt_path, 'cgt'
    return None, None


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
    step_num = syn_data['subgraph_step_num']
    sample_num = syn_data['subgraph_sample_num']
    noise_num = syn_data.get('noise_num', 0)
    self_conn = syn_data.get('self_connection', False)
    return step_num, sample_num, noise_num, self_conn


def build_cgt_datasets(original_graph, syn_data):
    """Build CGT computation graph datasets for synthetic training.

    Returns:
        syn_train: SyntheticCompGraphDataset for train nodes
        syn_val: SyntheticCompGraphDataset for val nodes
        test_ds: OriginalCompGraphDataset for test nodes (original features)
    """
    step_num, sample_num, noise_num, self_conn = _extract_cg_params(syn_data)
    total_sample = sample_num + noise_num

    # Get node IDs from saved splits
    ids = syn_data['ids']
    test_ids = ids['test']

    # Build test set from original graph (real features + structure)
    adj_list = dgl_to_adj_list(original_graph)
    features = original_graph.ndata['feature'].cpu().numpy().astype(np.float32)
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


def build_original_cg_datasets(original_graph, syn_data):
    """Build computation graph datasets from original data for all splits.

    Uses the same tree structure (step_num, sample_num, etc.) from the CGT
    .pt file, but with original features for train/val/test.

    Returns:
        train_ds, val_ds, test_ds: OriginalCompGraphDataset for each split
    """
    step_num, sample_num, noise_num, self_conn = _extract_cg_params(syn_data)

    ids = syn_data['ids']
    train_ids = ids['train']
    val_ids = ids['val']
    test_ids = ids['test']

    adj_list = dgl_to_adj_list(original_graph)
    features = original_graph.ndata['feature'].cpu().numpy().astype(np.float32)
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
