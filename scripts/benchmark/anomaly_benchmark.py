"""
Benchmark GNN models on original vs synthetic graph data.

Supports two types of synthetic data:
  - "graph": Full DGL graphs (e.g. from BiGG), loaded like original data.
  - "cgt":   CGT computation graph sequences (.pt). GADBench GNNs are
             extended to operate on batched computation graph trees,
             analogous to CGT's own GNN pipeline but using GADBench's
             GCN/GIN/GraphSAGE architectures.

Usage:
    # Auto-detect synthetic data type:
    python scripts/benchmark/anomaly_benchmark.py --datasets reddit --trials 1

    # Explicitly specify type:
    python scripts/benchmark/anomaly_benchmark.py --datasets reddit --synthetic_type cgt
    python scripts/benchmark/anomaly_benchmark.py --datasets reddit --synthetic_type graph
"""

import os
import sys
import time
import random
import warnings
import numpy as np
import torch
import pandas as pd

import dgl

warnings.filterwarnings("ignore")

# Resolve project paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
_gadbench_dir = os.path.join(_project_root, 'GADBench')

if _gadbench_dir not in sys.path:
    sys.path.insert(0, _gadbench_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from utils import Dataset as GADBenchDataset, model_detector_dict
from bench_utils import (
    parse_args,
    load_cgt_synthetic_data, build_cgt_datasets,
    build_original_cg_datasets, print_comparison,
)
from models.anomaly_detection.cgt_detector import CompGraphDetector, CG_SUPPORTED_MODELS

SEED_LIST = list(range(3407, 10000, 10))

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def evaluate_models(dataset_name, models, data_dir,
                    trials, semi_supervised, trial_id,
                    epochs, patience, lr, drop_rate, h_feats, num_layers):
    """Train and evaluate all specified models on the original dataset."""
    results = []

    for model_name in models:
        if model_name not in model_detector_dict:
            print(f"  WARNING: '{model_name}' not in GADBench. Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  ORIGINAL | {dataset_name} | {model_name}")
        print(f"{'='*60}")

        auc_list, pre_list, rec_list = [], [], []
        time_cost = 0

        for t in range(trials):
            torch.cuda.empty_cache()
            seed = SEED_LIST[t]
            set_seed(seed)

            train_config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'epochs': epochs,
                'patience': patience,
                'metric': 'AUPRC',
                'inductive': False,
                'seed': seed,
            }

            data = GADBenchDataset(dataset_name, prefix=data_dir + '/')

            # Apply train/val/test split masks — advance by t so each trial
            # uses a different pre-stored mask column (trial_id is the starting offset)
            data.split(semi_supervised, trial_id + t)

            model_config = {
                'model': model_name,
                'lr': lr,
                'drop_rate': drop_rate,
                'h_feats': h_feats,
                'num_layers': num_layers,
            }
            if dataset_name == 'tsocial':
                model_config['h_feats'] = 16

            print(f"  Trial {t}, seed={seed}")
            detector = model_detector_dict[model_name](
                train_config, model_config, data)

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
                'source': 'original',
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


def _run_cg_trials(dataset_name, cg_models, train_ds, val_ds, test_ds,
                   feat_dim, source_label, trials, epochs, patience,
                   batch_size, lr, drop_rate, h_feats, num_layers,
                   rebuild_datasets_fn=None):
    """Run CompGraphDetector trials for a set of models on given CG datasets.

    Args:
        rebuild_datasets_fn: optional callable(t) -> (train_ds, val_ds, test_ds).
            When provided, datasets are rebuilt each trial so that different
            train/val/test splits are used. When None, the passed-in datasets
            are reused across all trials (only the seed varies).
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

            if rebuild_datasets_fn is not None:
                train_ds, val_ds, test_ds = rebuild_datasets_fn(t)

            train_config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'epochs': epochs,
                'patience': patience,
                'metric': 'AUPRC',
                'batch_size': batch_size,
                'seed': seed,
            }

            model_config = {
                'model': model_name,
                'lr': lr,
                'drop_rate': drop_rate,
                'h_feats': h_feats,
                'num_layers': num_layers,
                'in_feats': feat_dim,
            }
            if dataset_name == 'tsocial':
                model_config['h_feats'] = 16

            print(f"  Trial {t}, seed={seed}")
            detector = CompGraphDetector(
                train_config, model_config, train_ds, val_ds, test_ds)

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


def evaluate_models_cgt(dataset_name, models, data_dir,
                        trials, epochs, patience, syn_path,
                        batch_size, lr, drop_rate, h_feats, num_layers,
                        trial_id=0, semi_supervised=False):
    """Evaluate GNN models on CGT computation graphs.

    Runs two comparisons:
      1. Original-CG: train/val/test all built as computation graphs
         from the original graph (baseline for the CG format). Each trial
         uses a different pre-stored mask split (trial_id + t).
      2. Synthetic-CGT: train/val from CGT-generated cluster-center
         sequences, test from original graph computation graphs. The
         synthetic split is fixed in the .pt file, so trials here only
         vary the random seed (initialization variance).
    """
    results = []

    # Load data once per dataset
    data = GADBenchDataset(dataset_name, prefix=data_dir + '/')
    syn_data = load_cgt_synthetic_data(syn_path)

    feat_dim = data.graph.ndata['feature'].shape[1]

    cg_models = [m for m in models if m in CG_SUPPORTED_MODELS]
    skipped = [m for m in models if m not in CG_SUPPORTED_MODELS]
    if skipped:
        print(f"  NOTE: {skipped} not supported in computation graph mode. "
              f"Skipping.")

    # --- Original data as computation graphs (CG baseline) ---
    # Rebuild datasets each trial so that different mask splits are used,
    # matching the same split-varying behaviour as the whole-graph path.
    orig_train0, orig_val0, orig_test0 = build_original_cg_datasets(
        data.graph, syn_data, trial_id=trial_id, semi_supervised=semi_supervised)
    results.extend(_run_cg_trials(
        dataset_name, cg_models, orig_train0, orig_val0, orig_test0,
        feat_dim, 'original-cg', trials, epochs, patience,
        batch_size, lr, drop_rate, h_feats, num_layers,
        rebuild_datasets_fn=lambda t: build_original_cg_datasets(
            data.graph, syn_data,
            trial_id=trial_id + t, semi_supervised=semi_supervised)))

    # --- CGT synthetic computation graphs ---
    # The synthetic train/val split is baked into the .pt file, so it cannot
    # vary across trials; only the random seed changes.
    syn_train, syn_val, test_ds = build_cgt_datasets(data.graph, syn_data)
    results.extend(_run_cg_trials(
        dataset_name, cg_models, syn_train, syn_val, test_ds,
        feat_dim, 'synthetic-cgt', trials, epochs, patience,
        batch_size, lr, drop_rate, h_feats, num_layers))

    del data
    return results


def evaluate_models_cross_graph(dataset_name, models, data_dir, dataset_dir, syn_file_name,
                                trials, semi_supervised, trial_id,
                                epochs, patience, lr, drop_rate, h_feats, num_layers):
    """Train GNNs on synthetic graph, validate on synthetic val, test on original test nodes."""
    from models.cross_graph_detector import CrossGraphGNNDetector, CROSS_GRAPH_SUPPORTED_MODELS
    results = []

    for model_name in models:
        if model_name not in CROSS_GRAPH_SUPPORTED_MODELS:
            print(f"  NOTE: '{model_name}' skipped in cross-graph mode.")
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

            # Synthetic graph: train + val from synthetic masks
            syn_data = GADBenchDataset(syn_file_name, prefix=dataset_dir + '/')
            syn_data.split(semi_supervised, trial_id + t)

            # Original graph: test nodes only
            orig_data = GADBenchDataset(dataset_name, prefix=data_dir + '/')
            orig_data.split(semi_supervised, trial_id + t)

            train_config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'epochs': epochs, 'patience': patience, 'metric': 'AUPRC', 'seed': seed,
            }
            model_config = {
                'model': model_name, 'lr': lr, 'drop_rate': drop_rate,
                'h_feats': 16 if dataset_name == 'tsocial' else h_feats,
                'num_layers': num_layers,
            }

            print(f"  Trial {t}, seed={seed}")
            detector = CrossGraphGNNDetector(train_config, model_config, syn_data, orig_data)

            st = time.time()
            test_score = detector.train()
            ed = time.time()
            time_cost += ed - st

            auc_list.append(test_score['AUROC'])
            pre_list.append(test_score['AUPRC'])
            rec_list.append(test_score['RecK'])
            print(f"  -> AUROC={test_score['AUROC']:.4f}, AUPRC={test_score['AUPRC']:.4f}, RecK={test_score['RecK']:.4f}")
            del detector, syn_data, orig_data

        if auc_list:
            results.append({
                'source': 'synthetic-graph', 'dataset': dataset_name, 'model': model_name,
                'AUROC_mean': np.mean(auc_list), 'AUROC_std': np.std(auc_list),
                'AUPRC_mean': np.mean(pre_list), 'AUPRC_std': np.std(pre_list),
                'RecK_mean':  np.mean(rec_list),  'RecK_std':  np.std(rec_list),
                'time_per_trial': time_cost / len(auc_list),
            })
    return results


def main():
    args = parse_args()

    # Resolve default paths relative to project root
    if args.data_dir is None:
        args.data_dir = os.path.join(_project_root, 'datasets', 'original')
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
    print(f"  Trial ID:       {args.trial_id}")

    print("\nTraining:")
    print(f"  Trials:         {args.trials}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Patience:       {args.patience}")
    print(f"  Batch size:     {args.batch_size}  (CGT only)")

    print("\nModel Architecture:")
    print(f"  LR:             {args.lr}")
    print(f"  Drop rate:      {args.drop_rate}")
    print(f"  Hidden feats:   {args.h_feats}")
    print(f"  Num layers:     {args.num_layers}  (must equal cg_depth in CGT .pt)")

    all_results = []

    # --- Phase 1: Evaluate on original data ---
    print("\n" + "#" * 80)
    print("# PHASE 1: EVALUATING ON ORIGINAL DATA")
    print("#" * 80)

    for dataset_name in datasets:
        results = evaluate_models(
            dataset_name, models, args.data_dir,
            args.trials, args.semi_supervised, args.trial_id,
            args.epochs, args.patience,
            args.lr, args.drop_rate, args.h_feats, args.num_layers)
        all_results.extend(results)

    # --- Phase 2: Evaluate on synthetic data ---
    print("\n" + "#" * 80)
    print("# PHASE 2: EVALUATING ON SYNTHETIC DATA")
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
            # CGT: use computation graph trees with GADBench GNNs
            results = evaluate_models_cgt(
                dataset_name, models, args.data_dir,
                args.trials, args.epochs, args.patience, syn_path,
                args.batch_size, args.lr, args.drop_rate,
                args.h_feats, args.num_layers,
                trial_id=args.trial_id,
                semi_supervised=bool(args.semi_supervised))
        else:
            # Full graph (BiGG, etc.): train+val on synthetic, test on original.
            results = evaluate_models_cross_graph(
                dataset_name, models, args.data_dir, task_dir, stem,
                args.trials, args.semi_supervised, args.trial_id,
                args.epochs, args.patience,
                args.lr, args.drop_rate, args.h_feats, args.num_layers)
        all_results.extend(results)

    # --- Save and display results ---
    if all_results:
        results_df = pd.DataFrame(all_results)

        xlsx_path = os.path.join(args.output_dir, 'evaluation_results.xlsx')
        results_df.to_excel(xlsx_path, index=False)

        csv_path = os.path.join(args.output_dir, 'evaluation_results.csv')
        results_df.to_csv(csv_path, index=False)

        print(f"\nResults saved to:\n  {xlsx_path}\n  {csv_path}")
        print_comparison(all_results, datasets, models)
    else:
        print("\nNo results to save.")


if __name__ == '__main__':
    main()
