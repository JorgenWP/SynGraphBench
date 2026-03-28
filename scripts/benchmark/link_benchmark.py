"""
Benchmark GNN models on link prediction: original vs synthetic graph data.

Supports two types of synthetic data:
  - "graph": Full DGL graphs (e.g. from BiGG), loaded like original data.
  - "comp-graph": CGT computation graph sequences (.pt). GADBench GNNs
                  generate node embeddings via batched computation graph
                  trees, then score edges with dot product or MLP decoder.

Usage:
    python scripts/benchmark/link_benchmark.py --datasets reddit --trials 1
    python scripts/benchmark/link_benchmark.py --datasets reddit --synthetic_type comp-graph
    python scripts/benchmark/link_benchmark.py --datasets reddit --synthetic_type graph
"""

import os
import sys
import time
import random
import warnings
import numpy as np
import torch
import pandas as pd

warnings.filterwarnings("ignore")

# Resolve project paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
_gadbench_dir = os.path.join(_project_root, 'GADBench')

if _gadbench_dir not in sys.path:
    sys.path.insert(0, _gadbench_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from link_utils import LinkDataset, save_results
from models.link_prediction.link_predictor import BaseGNNLinkPredictor, XGBGraphLinkPredictor
from models.link_prediction.cgt_link_predictor import CompGraphLinkPredictor
from bench_utils import (
    parse_link_args, load_cgt_synthetic_data,
    print_comparison,
)
from models.cross_graph_link_predictor import (
    CrossGraphLinkPredictor, CrossGraphXGBGraphLinkPredictor,
    CROSS_GRAPH_LP_SUPPORTED_MODELS,
)

SEED_LIST = list(range(3407, 10000, 10))

SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE', 'XGBGraph']


def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def evaluate_link_models(dataset_name, models, data_dir, data_source,
                         trials, val_ratio, test_ratio, neg_sampling,
                         decoder, epochs, patience, lr, drop_rate,
                         h_feats, num_layers, synthetic_dir=None):
    """Train and evaluate link prediction models on full graphs."""
    results = []

    for model_name in models:
        if model_name not in SUPPORTED_MODELS:
            print(f"  WARNING: '{model_name}' not supported for link prediction. Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  {data_source.upper()} | {dataset_name} | {model_name}")
        print(f"{'='*60}")

        auc_list, pre_list, rec_list = [], [], []
        time_cost = 0

        prefix = (synthetic_dir + '/') if data_source != 'original' else (data_dir + '/original/')
        data = LinkDataset(dataset_name, prefix=prefix)

        for t in range(trials):
            torch.cuda.empty_cache()
            seed = SEED_LIST[t]
            set_seed(seed)

            print(f"  Trial {t}, seed={seed}")
            data.split(val_ratio, test_ratio, t, neg_sampling)

            train_config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'epochs': epochs,
                'patience': patience,
                'metric': 'AUPRC',
                'neg_sampling': neg_sampling,
                'decoder': decoder,
                'seed': seed,
            }
            model_config = {
                'model': model_name,
                'lr': lr,
                'drop_rate': drop_rate,
                'h_feats': h_feats,
                'num_layers': num_layers,
            }

            if model_name == 'XGBGraph':
                detector = XGBGraphLinkPredictor(train_config, model_config, data)
            else:
                detector = BaseGNNLinkPredictor(train_config, model_config, data)
            st = time.time()
            test_score = detector.train()
            ed = time.time()
            time_cost += ed - st

            auc_list.append(test_score['AUROC'])
            pre_list.append(test_score['AUPRC'])
            rec_list.append(test_score['RecK'])

            print(f"  -> AUROC={test_score['AUROC']:.4f}, "
                  f"AUPRC={test_score['AUPRC']:.4f}, "
                  f"RecK={test_score['RecK']:.4f}")

            del detector

        del data

        if auc_list:
            results.append({
                'source': data_source,
                'dataset': dataset_name,
                'model': model_name,
                'AUROC_mean': np.mean(auc_list),
                'AUROC_std': np.std(auc_list),
                'AUPRC_mean': np.mean(pre_list),
                'AUPRC_std': np.std(pre_list),
                'RecK_mean': np.mean(rec_list),
                'RecK_std': np.std(rec_list),
                'time_per_trial': time_cost / len(auc_list),
            })

    return results


def _run_cg_link_trials(dataset_name, cg_models, data, source_label,
                        trials, val_ratio, test_ratio, neg_sampling, decoder,
                        epochs, patience, batch_size, lr, drop_rate,
                        h_feats, num_layers, step_num, sample_num):
    """Run CompGraphLinkPredictor trials for a set of models.

    Each trial re-splits the edges (via data.split) so that different
    train/val/test edge sets are used, then builds computation graph
    trees and trains the link predictor.
    """
    results = []

    for model_name in cg_models:
        print(f"\n{'='*60}")
        print(f"  {source_label.upper()} | {dataset_name} | {model_name}")
        print(f"{'='*60}")

        auc_list, pre_list, rec_list = [], [], []
        time_cost = 0

        for t in range(trials):
            torch.cuda.empty_cache()
            seed = SEED_LIST[t]
            set_seed(seed)

            print(f"  Trial {t}, seed={seed}")
            data.split(val_ratio, test_ratio, t, neg_sampling)

            train_config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'epochs': epochs,
                'patience': patience,
                'metric': 'AUPRC',
                'neg_sampling': neg_sampling,
                'decoder': decoder,
                'seed': seed,
                'step_num': step_num,
                'sample_num': sample_num,
                'batch_size': batch_size,
            }
            model_config = {
                'model': model_name,
                'lr': lr,
                'drop_rate': drop_rate,
                'h_feats': h_feats,
                'num_layers': num_layers,
            }

            detector = CompGraphLinkPredictor(train_config, model_config, data)
            st = time.time()
            test_score = detector.train()
            ed = time.time()
            time_cost += ed - st

            auc_list.append(test_score['AUROC'])
            pre_list.append(test_score['AUPRC'])
            rec_list.append(test_score['RecK'])

            print(f"  -> AUROC={test_score['AUROC']:.4f}, "
                  f"AUPRC={test_score['AUPRC']:.4f}, "
                  f"RecK={test_score['RecK']:.4f}")

            del detector

        if auc_list:
            results.append({
                'source': source_label,
                'dataset': dataset_name,
                'model': model_name,
                'AUROC_mean': np.mean(auc_list),
                'AUROC_std': np.std(auc_list),
                'AUPRC_mean': np.mean(pre_list),
                'AUPRC_std': np.std(pre_list),
                'RecK_mean': np.mean(rec_list),
                'RecK_std': np.std(rec_list),
                'time_per_trial': time_cost / len(auc_list),
            })

    return results


def evaluate_link_models_cgt(dataset_name, models, data_dir,
                             trials, val_ratio, test_ratio,
                             neg_sampling, decoder, epochs, patience,
                             syn_path, batch_size, lr, drop_rate,
                             h_feats, num_layers):
    """Evaluate link prediction using CGT computation graph trees.

    Runs two comparisons:
      1. Original-CG: node embeddings via computation graph trees built
         from the original graph's train edges (baseline for CG format).
      2. Synthetic-CGT: node embeddings via computation graph trees built
         from a synthetic graph's train edges, tested on original graph's
         test edges. The synthetic graph is reconstructed from CGT cluster
         centers (root node features replaced with synthetic features).
    """
    results = []

    data = LinkDataset(dataset_name, prefix=data_dir + '/original/')
    syn_data = load_cgt_synthetic_data(syn_path)

    cg_models = [m for m in models if m in SUPPORTED_MODELS]
    skipped = [m for m in models if m not in SUPPORTED_MODELS]
    if skipped:
        print(f"  NOTE: {skipped} not supported for CG link prediction. Skipping.")

    # Extract CGT computation graph parameters
    step_num = syn_data.get('cg_depth', syn_data.get('subgraph_step_num'))
    sample_num = syn_data.get('cg_fanout', syn_data.get('subgraph_sample_num'))

    # --- Original data as computation graphs (CG baseline) ---
    results.extend(_run_cg_link_trials(
        dataset_name, cg_models, data, 'original-cg',
        trials, val_ratio, test_ratio, neg_sampling, decoder,
        epochs, patience, batch_size, lr, drop_rate,
        h_feats, num_layers, step_num, sample_num))

    # --- CGT synthetic computation graphs ---
    # Build a synthetic graph by replacing train/val node features with
    # CGT-generated cluster center features, then run CG link prediction.
    # Test edges still come from the original graph's edge split.
    from bench_utils import build_synthetic_dgl_graph
    syn_graph = build_synthetic_dgl_graph(data.graph, syn_data)
    syn_link_data = LinkDataset.__new__(LinkDataset)
    syn_link_data.name = dataset_name + '_synthetic_cgt'
    syn_link_data.graph = syn_graph.long()

    results.extend(_run_cg_link_trials(
        dataset_name, cg_models, syn_link_data, 'synthetic-cgt',
        trials, val_ratio, test_ratio, neg_sampling, decoder,
        epochs, patience, batch_size, lr, drop_rate,
        h_feats, num_layers, step_num, sample_num))

    del data, syn_link_data
    return results


def evaluate_link_models_cross_graph(dataset_name, models, data_dir, dataset_dir, syn_file_name,
                                     trials, val_ratio, test_ratio, neg_sampling,
                                     decoder, epochs, patience, lr, drop_rate,
                                     h_feats, num_layers):
    """Train GNNs on synthetic graph edges, validate on synthetic val, test on original test edges."""
    results = []

    for model_name in models:
        if model_name not in CROSS_GRAPH_LP_SUPPORTED_MODELS:
            print(f"  NOTE: '{model_name}' skipped in cross-graph link prediction mode.")
            continue

        print(f"\n{'='*60}")
        print(f"  SYNTHETIC-GRAPH (CROSS) | {dataset_name} | {model_name}")
        print(f"{'='*60}")

        auc_list, pre_list, rec_list = [], [], []
        time_cost = 0

        for t in range(trials):
            torch.cuda.empty_cache()
            seed = SEED_LIST[t]
            set_seed(seed)

            # Synthetic graph: train + val edges
            syn_data = LinkDataset(syn_file_name, prefix=dataset_dir + '/')
            syn_data.split(val_ratio, test_ratio, t, neg_sampling)

            # Original graph: test edges
            orig_data = LinkDataset(dataset_name, prefix=data_dir + '/original/')
            orig_data.split(val_ratio, test_ratio, t, neg_sampling)

            train_config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'epochs': epochs,
                'patience': patience,
                'metric': 'AUPRC',
                'neg_sampling': neg_sampling,
                'decoder': decoder,
                'seed': seed,
            }
            model_config = {
                'model': model_name,
                'lr': lr,
                'drop_rate': drop_rate,
                'h_feats': h_feats,
                'num_layers': num_layers,
            }

            print(f"  Trial {t}, seed={seed}")

            if syn_data.val_pos_edges.shape[0] == 0:
                print(f"  Skipping trial {t}: synthetic graph too sparse (0 val edges)")
                del syn_data, orig_data
                continue

            if model_name == 'XGBGraph':
                detector = CrossGraphXGBGraphLinkPredictor(train_config, model_config, syn_data, orig_data)
            else:
                detector = CrossGraphLinkPredictor(train_config, model_config, syn_data, orig_data)

            st = time.time()
            test_score = detector.train()
            ed = time.time()
            time_cost += ed - st

            auc_list.append(test_score['AUROC'])
            pre_list.append(test_score['AUPRC'])
            rec_list.append(test_score['RecK'])
            print(f"  -> AUROC={test_score['AUROC']:.4f}, "
                  f"AUPRC={test_score['AUPRC']:.4f}, "
                  f"RecK={test_score['RecK']:.4f}")
            del detector, syn_data, orig_data

        if auc_list:
            results.append({
                'source': 'synthetic-graph',
                'dataset': dataset_name,
                'model': model_name,
                'AUROC_mean': np.mean(auc_list),
                'AUROC_std': np.std(auc_list),
                'AUPRC_mean': np.mean(pre_list),
                'AUPRC_std': np.std(pre_list),
                'RecK_mean': np.mean(rec_list),
                'RecK_std': np.std(rec_list),
                'time_per_trial': time_cost / len(auc_list),
            })

    return results


def main():
    args = parse_link_args()

    # Resolve default paths relative to project root
    if args.data_dir is None:
        args.data_dir = os.path.join(_project_root, 'datasets')
    if args.synthetic_dir is None:
        args.synthetic_dir = os.path.join(_project_root, 'datasets', 'synthetic')
    if args.output_dir is None:
        args.output_dir = os.path.join(_project_root, 'results', 'evaluate')

    args.data_dir = os.path.abspath(args.data_dir)
    args.synthetic_dir = os.path.abspath(args.synthetic_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    datasets = [d.strip() for d in args.datasets.split(',')]
    models = [m.strip() for m in args.models.split(',')]

    os.makedirs(args.output_dir, exist_ok=True)

    print("Arguments:")
    print("\nData:")
    print(f"  Datasets:       {datasets}")
    print(f"  Models:         {models}")
    print(f"  Synthetic type: {args.synthetic_type}")
    print(f"  Generator:      {args.generator}")
    print(f"  Synthetic name: {args.synthetic_name if args.synthetic_name else '(use dataset name)'}")
    print(f"  Task:           {args.task}")
    print(f"  Data dir:       {args.data_dir}")
    print(f"  Synthetic dir:  {args.synthetic_dir}")
    print(f"  Output dir:     {args.output_dir}")

    print("\nLink Prediction:")
    print(f"  Trials:         {args.trials}")
    print(f"  Val ratio:      {args.val_ratio}")
    print(f"  Test ratio:     {args.test_ratio}")
    print(f"  Neg sampling:   {args.neg_sampling}")
    print(f"  Decoder:        {args.decoder}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Patience:       {args.patience}")
    print(f"  Batch size:     {args.batch_size}  (CGT only)")

    print("\nModel Architecture:")
    print(f"  LR:             {args.lr}")
    print(f"  Drop rate:      {args.drop_rate}")
    print(f"  Hidden feats:   {args.h_feats}")
    print(f"  Num layers:     {args.num_layers}")

    all_results = []

    # --- Phase 1: Evaluate on original data ---
    print("\n" + "#" * 80)
    print("# PHASE 1: LINK PREDICTION ON ORIGINAL DATA")
    print("#" * 80)

    for dataset_name in datasets:
        results = evaluate_link_models(
            dataset_name, models, args.data_dir,
            'original', args.trials,
            args.val_ratio, args.test_ratio,
            args.neg_sampling, args.decoder,
            args.epochs, args.patience,
            args.lr, args.drop_rate, args.h_feats, args.num_layers)
        all_results.extend(results)

    # --- Phase 2: Evaluate on synthetic data ---
    print("\n" + "#" * 80)
    print("# PHASE 2: LINK PREDICTION ON SYNTHETIC DATA")
    print("#" * 80)

    for dataset_name in datasets:
        # Resolve path: synthetic_dir/<generator>/<dataset>/<task>/<stem>[.pt]
        gen_dir = os.path.join(args.synthetic_dir, args.generator)
        dataset_dir = os.path.join(gen_dir, dataset_name)
        task_dir = os.path.join(dataset_dir, args.task)
        stem = args.synthetic_name
        if args.synthetic_type == 'comp-graph':
            syn_path = os.path.join(task_dir, f'{stem}.pt')
        else:
            syn_path = os.path.join(task_dir, stem)

        if not os.path.exists(syn_path):
            print(f"\n  Skipping {dataset_name}: {syn_path} not found")
            continue

        print(f"\n  Found {args.synthetic_type} synthetic data: {syn_path}")

        if args.synthetic_type == 'comp-graph':
            results = evaluate_link_models_cgt(
                dataset_name, models, args.data_dir,
                args.trials, args.val_ratio, args.test_ratio,
                args.neg_sampling, args.decoder,
                args.epochs, args.patience, syn_path,
                args.batch_size, args.lr, args.drop_rate,
                args.h_feats, args.num_layers)
        else:
            # Full graph (BiGG, etc.): train+val on synthetic, test on original.
            results = evaluate_link_models_cross_graph(
                dataset_name, models, args.data_dir, task_dir, stem,
                args.trials, args.val_ratio, args.test_ratio,
                args.neg_sampling, args.decoder,
                args.epochs, args.patience,
                args.lr, args.drop_rate, args.h_feats, args.num_layers)
        all_results.extend(results)

    # --- Save and display results ---
    if all_results:
        results_df = pd.DataFrame(all_results)

        xlsx_path = os.path.join(args.output_dir, 'link_prediction_results.xlsx')
        results_df.to_excel(xlsx_path, index=False)

        csv_path = os.path.join(args.output_dir, 'link_prediction_results.csv')
        results_df.to_csv(csv_path, index=False)

        print(f"\nResults saved to:\n  {xlsx_path}\n  {csv_path}")
        print_comparison(all_results, datasets, models)
    else:
        print("\nNo results to save.")


if __name__ == '__main__':
    main()
