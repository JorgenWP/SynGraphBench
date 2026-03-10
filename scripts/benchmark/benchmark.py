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
    python scripts/benchmark/benchmark.py --datasets reddit --trials 1

    # Explicitly specify type:
    python scripts/benchmark/benchmark.py --datasets reddit --synthetic_type cgt
    python scripts/benchmark/benchmark.py --datasets reddit --synthetic_type graph
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

from utils import Dataset as GADBenchDataset, model_detector_dict
from bench_utils import (
    parse_args, find_synthetic_path,
    load_cgt_synthetic_data, build_cgt_datasets,
    print_comparison,
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
                    data_source, trials, semi_supervised, trial_id,
                    epochs, patience,
                    synthetic_dir=None, synthetic_type=None):
    """Train and evaluate all specified models on a single dataset.

    Args:
        data_source: 'original' or 'synthetic-graph'
        synthetic_dir: path to synthetic data (only for synthetic sources)
        synthetic_type: resolved type ('graph' only), used when data_source != 'original'
    """
    results = []

    for model_name in models:
        if model_name not in model_detector_dict:
            print(f"  WARNING: '{model_name}' not in GADBench. Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  {data_source.upper()} | {dataset_name} | {model_name}")
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

            if synthetic_type == 'graph':
                # Full synthetic graph (BiGG, etc.) — load directly
                data = GADBenchDataset(dataset_name, prefix=synthetic_dir + '/')
            else:
                # Original data
                data = GADBenchDataset(dataset_name, prefix=data_dir + '/')

            # Apply train/val/test split masks
            data.split(semi_supervised, trial_id)

            model_config = {'model': model_name, 'lr': 0.01, 'drop_rate': 0}
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


def evaluate_models_cgt(dataset_name, models, data_dir,
                        trials, epochs, patience, syn_path,
                        batch_size=256):
    """Evaluate GNN models trained on CGT synthetic computation graphs.

    Trains GADBench GNNs (GCN, GIN, GraphSAGE) on batched synthetic
    computation graph trees, then tests on computation graphs built
    from the original test nodes.
    """
    results = []

    # Load data once per dataset
    data = GADBenchDataset(dataset_name, prefix=data_dir + '/')
    syn_data = load_cgt_synthetic_data(syn_path)
    syn_train, syn_val, test_ds = build_cgt_datasets(data.graph, syn_data)

    feat_dim = data.graph.ndata['feature'].shape[1]

    cg_models = [m for m in models if m in CG_SUPPORTED_MODELS]
    skipped = [m for m in models if m not in CG_SUPPORTED_MODELS]
    if skipped:
        print(f"  NOTE: {skipped} not supported in computation graph mode. "
              f"Skipping.")

    for model_name in cg_models:
        print(f"\n{'='*60}")
        print(f"  SYNTHETIC-CGT | {dataset_name} | {model_name}")
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
                'batch_size': batch_size,
                'seed': seed,
            }

            model_config = {
                'model': model_name,
                'lr': 0.01,
                'drop_rate': 0,
                'in_feats': feat_dim,
            }
            if dataset_name == 'tsocial':
                model_config['h_feats'] = 16

            print(f"  Trial {t}, seed={seed}")
            detector = CompGraphDetector(
                train_config, model_config, syn_train, syn_val, test_ds)

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
                'source': 'synthetic-cgt',
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

    del data
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

    print("GNN Benchmark: Original vs Synthetic Data")
    print(f"  Datasets:       {datasets}")
    print(f"  Models:         {models}")
    print(f"  Trials:         {args.trials}")
    print(f"  Synthetic type: {args.synthetic_type}")
    print(f"  Data dir:       {args.data_dir}")
    print(f"  Synthetic dir:  {args.synthetic_dir}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Trial ID:       {args.trial_id}")

    all_results = []

    # --- Phase 1: Evaluate on original data ---
    print("\n" + "#" * 80)
    print("# PHASE 1: EVALUATING ON ORIGINAL DATA")
    print("#" * 80)

    for dataset_name in datasets:
        results = evaluate_models(
            dataset_name, models, args.data_dir,
            'original', args.trials, args.semi_supervised, args.trial_id,
            args.epochs, args.patience)
        all_results.extend(results)

    # --- Phase 2: Evaluate on synthetic data ---
    print("\n" + "#" * 80)
    print("# PHASE 2: EVALUATING ON SYNTHETIC DATA")
    print("#" * 80)

    for dataset_name in datasets:
        syn_path, resolved_type = find_synthetic_path(
            args.synthetic_dir, dataset_name, args.synthetic_type)
        if syn_path is None:
            print(f"\n  Skipping {dataset_name}: no synthetic data found in "
                  f"{args.synthetic_dir}")
            continue

        print(f"\n  Found {resolved_type} synthetic data: {syn_path}")

        if resolved_type == 'cgt':
            # CGT: use computation graph trees with GADBench GNNs
            results = evaluate_models_cgt(
                dataset_name, models, args.data_dir,
                args.trials, args.epochs, args.patience,
                syn_path, batch_size=args.batch_size)
        else:
            # Full graph (BiGG, etc.): use standard full-graph GNNs
            results = evaluate_models(
                dataset_name, models, args.data_dir,
                f'synthetic-{resolved_type}', args.trials,
                args.semi_supervised, args.trial_id,
                args.epochs, args.patience,
                synthetic_dir=args.synthetic_dir,
                synthetic_type=resolved_type)
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
