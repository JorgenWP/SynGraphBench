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
from models.link_prediction.link_predictor import (
    BaseGNNLinkPredictor, XGBoostLinkPredictor, XGBGraphLinkPredictor)
from models.link_prediction.cgt_link_predictor import (
    CompGraphLinkPredictor, CGTXGBoostLinkPredictor, CGTXGBGraphLinkPredictor)
from bench_utils import (
    parse_link_args, load_cgt_synthetic_data,
    print_comparison, resolve_cgt_trial_paths,
    _assert_link_pt_alignment, build_synthetic_dgl_graph,
    format_duration,
)
from models.cross_graph_link_predictor import (
    CrossGraphLinkPredictor, CrossGraphXGBoostLinkPredictor,
    CrossGraphXGBGraphLinkPredictor, CROSS_GRAPH_LP_SUPPORTED_MODELS,
)

SEED_LIST = list(range(3407, 10000, 10))

SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE', 'XGBoost', 'XGBGraph']
CGT_LP_SUPPORTED_MODELS = SUPPORTED_MODELS

CGT_LP_TREE_MODELS = {
    'XGBoost': CGTXGBoostLinkPredictor,
    'XGBGraph': CGTXGBGraphLinkPredictor,
}


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
                         h_feats, num_layers, synthetic_dir=None,
                         curve_records=None):
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
        val_curves, test_curves = [], []
        time_list = []

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

            if model_name == 'XGBoost':
                detector = XGBoostLinkPredictor(train_config, model_config, data)
            elif model_name == 'XGBGraph':
                detector = XGBGraphLinkPredictor(train_config, model_config, data)
            else:
                detector = BaseGNNLinkPredictor(train_config, model_config, data)
            st = time.time()
            test_score = detector.train()
            ed = time.time()
            dt = ed - st
            time_list.append(dt)

            auc_list.append(test_score['AUROC'])
            pre_list.append(test_score['AUPRC'])
            rec_list.append(test_score['RecK'])

            if 'val_auprc_curve' in test_score:
                val_curves.append(test_score['val_auprc_curve'])
                test_curves.append(test_score['test_auprc_curve'])

            print(f"  -> AUROC={test_score['AUROC']:.4f}, "
                  f"AUPRC={test_score['AUPRC']:.4f}, "
                  f"RecK={test_score['RecK']:.4f}  [{dt:.1f}s]")

            del detector

        del data

        if time_list:
            total = sum(time_list)
            print(f"  [{model_name} on {dataset_name}] {len(time_list)} trials — "
                  f"total {format_duration(total)}, "
                  f"mean {format_duration(total / len(time_list))} "
                  f"± {format_duration(float(np.std(time_list)))}")

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
                'time_per_trial': sum(time_list) / len(time_list),
            })

        if val_curves and curve_records is not None:
            max_len = max(len(c) for c in val_curves)
            def pad(curve):
                return curve + [curve[-1]] * (max_len - len(curve))
            val_arr  = np.array([pad(c) for c in val_curves])
            test_arr = np.array([pad(c) for c in test_curves])
            for epoch in range(max_len):
                curve_records.append({
                    'source': data_source,
                    'dataset': dataset_name, 'model': model_name, 'epoch': epoch,
                    'val_auprc_mean':  float(val_arr[:, epoch].mean()),
                    'val_auprc_std':   float(val_arr[:, epoch].std()),
                    'test_auprc_mean': float(test_arr[:, epoch].mean()),
                    'test_auprc_std':  float(test_arr[:, epoch].std()),
                })

    return results


def _run_cg_link_trials(dataset_name, cg_models, base_data, source_label,
                        trials, val_ratio, test_ratio, neg_sampling, decoder,
                        epochs, patience, batch_size, lr, drop_rate,
                        h_feats, num_layers, step_num, sample_num,
                        syn_graph_factory=None, curve_records=None):
    """Run MergedCompGraphLinkPredictor trials across models and trials.

    For each trial:
      - Split base_data's edges (same split logic as full-graph LP).
      - If syn_graph_factory is provided (synthetic-cgt mode), build a
        per-trial synthetic graph from that trial's CGT .pt file and wrap
        it in a LinkDataset. The factory also runs alignment checks.
      - Train and evaluate the merged-CG link predictor.
    """
    results = []

    for model_name in cg_models:
        print(f"\n{'='*60}")
        print(f"  {source_label.upper()} | {dataset_name} | {model_name}")
        print(f"{'='*60}")

        auc_list, pre_list, rec_list = [], [], []
        val_curves, test_curves = [], []
        time_list = []

        for t in range(trials):
            torch.cuda.empty_cache()
            seed = SEED_LIST[t]
            set_seed(seed)

            print(f"  Trial {t}, seed={seed}")
            base_data.split(val_ratio, test_ratio, t, neg_sampling)

            if syn_graph_factory is not None:
                syn_graph = syn_graph_factory(t, base_data.test_pos_edges)
                trial_data = LinkDataset.__new__(LinkDataset)
                trial_data.name = dataset_name + f'_synthetic_cgt_t{t}'
                trial_data.graph = syn_graph.long()
                trial_data.split(val_ratio, test_ratio, t, neg_sampling)
            else:
                trial_data = base_data

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

            detector_cls = CGT_LP_TREE_MODELS.get(
                model_name, CompGraphLinkPredictor)
            detector = detector_cls(
                train_config, model_config, trial_data)
            st = time.time()
            test_score = detector.train()
            ed = time.time()
            dt = ed - st
            time_list.append(dt)

            auc_list.append(test_score['AUROC'])
            pre_list.append(test_score['AUPRC'])
            rec_list.append(test_score['RecK'])

            if 'val_auprc_curve' in test_score:
                val_curves.append(test_score['val_auprc_curve'])
                test_curves.append(test_score['test_auprc_curve'])

            print(f"  -> AUROC={test_score['AUROC']:.4f}, "
                  f"AUPRC={test_score['AUPRC']:.4f}, "
                  f"RecK={test_score['RecK']:.4f}  [{dt:.1f}s]")

            del detector
            if syn_graph_factory is not None:
                del trial_data

        if time_list:
            total = sum(time_list)
            print(f"  [{model_name} on {dataset_name}] {len(time_list)} trials — "
                  f"total {format_duration(total)}, "
                  f"mean {format_duration(total / len(time_list))} "
                  f"± {format_duration(float(np.std(time_list)))}")

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
                'time_per_trial': sum(time_list) / len(time_list),
            })

        if val_curves and curve_records is not None:
            max_len = max(len(c) for c in val_curves)
            def pad(curve):
                return curve + [curve[-1]] * (max_len - len(curve))
            val_arr  = np.array([pad(c) for c in val_curves])
            test_arr = np.array([pad(c) for c in test_curves])
            for epoch in range(max_len):
                curve_records.append({
                    'source': source_label,
                    'dataset': dataset_name, 'model': model_name, 'epoch': epoch,
                    'val_auprc_mean':  float(val_arr[:, epoch].mean()),
                    'val_auprc_std':   float(val_arr[:, epoch].std()),
                    'test_auprc_mean': float(test_arr[:, epoch].mean()),
                    'test_auprc_std':  float(test_arr[:, epoch].std()),
                })

    return results


def evaluate_link_models_cgt(dataset_name, models, data_dir,
                             trials, val_ratio, test_ratio,
                             neg_sampling, decoder, epochs, patience,
                             trial_paths, batch_size, lr, drop_rate,
                             h_feats, num_layers, eval_mode='both',
                             curve_records=None):
    """Evaluate link prediction on CGT merged computation graphs.

    Runs two comparisons:
      1. Original-CG: merged endpoint trees built from the original graph
         (with the trial's test edges withheld), baseline for the CG
         format.
      2. Synthetic-CGT: merged endpoint trees built from a hybrid graph
         whose train/val root features are replaced by CGT cluster-center
         features (via build_synthetic_dgl_graph). Each trial loads its
         own .pt with alignment-checked test edges.

    Features are L2-normalized once here so both downstream paths feed
    unit-norm features to MergedCompGraphLinkPredictor — matching CGT's
    training space (CGT/task/utils/utils.py L2-normalizes before clustering)
    and the L2-normalized cluster centers (CGT/generator/cluster.py). This
    mirrors the AD pipeline (build_cgt_datasets / build_original_cg_datasets)
    and removes the scale clash where train/val cluster centers met raw
    test/other features inside a single merged comp graph.
    """
    results = []

    data = LinkDataset(dataset_name, prefix=data_dir + '/original/')

    # L2-normalize features in place. data.train_graph (built later by
    # data.split() via dgl.remove_edges) inherits ndata, so a single
    # normalization here covers both original-CG and synthetic-CGT paths.
    feats = data.graph.ndata['feature'].float()
    norms = feats.norm(p=2, dim=1, keepdim=True).clamp(min=1.0)
    data.graph.ndata['feature'] = feats / norms
    print(f"  L2-normalized features for {dataset_name}: "
          f"mean row-norm={data.graph.ndata['feature'].norm(dim=1).mean():.4f}")

    # CG params come from the first trial's .pt (all trials share these)
    first_syn = load_cgt_synthetic_data(trial_paths[0])
    step_num = first_syn.get('cg_depth', first_syn.get('subgraph_step_num'))
    sample_num = first_syn.get('cg_fanout', first_syn.get('subgraph_sample_num'))

    cg_models = [m for m in models if m in CGT_LP_SUPPORTED_MODELS]
    skipped = [m for m in models if m not in CGT_LP_SUPPORTED_MODELS]
    if skipped:
        print(f"  NOTE: {skipped} not supported for CG link prediction. Skipping.")

    if 'XGBGraph' in cg_models and num_layers != step_num:
        raise ValueError(
            f"XGBGraph on CGT link prediction requires num_layers == step_num; "
            f"got num_layers={num_layers}, step_num={step_num} "
            f"(from {trial_paths[0]}).")

    # --- Original-CG baseline: same graph for every trial ---
    if eval_mode in ('original_cg', 'both'):
        results.extend(_run_cg_link_trials(
            dataset_name, cg_models, data, 'original-cg',
            trials, val_ratio, test_ratio, neg_sampling, decoder,
            epochs, patience, batch_size, lr, drop_rate,
            h_feats, num_layers, step_num, sample_num,
            curve_records=curve_records))

    # --- Synthetic-CGT: per-trial hybrid graph built from per-trial .pt ---
    if eval_mode in ('synthetic_cgt', 'both'):
        def make_syn_graph(t, expected_test_edges):
            syn_data = load_cgt_synthetic_data(trial_paths[t])
            _assert_link_pt_alignment(
                syn_data, expected_trial_id=t,
                expected_test_edges=expected_test_edges,
                source_label=f'synthetic-cgt[t={t}]')
            return build_synthetic_dgl_graph(
                data.graph, syn_data, trial_id=t)

        results.extend(_run_cg_link_trials(
            dataset_name, cg_models, data, 'synthetic-cgt',
            trials, val_ratio, test_ratio, neg_sampling, decoder,
            epochs, patience, batch_size, lr, drop_rate,
            h_feats, num_layers, step_num, sample_num,
            syn_graph_factory=make_syn_graph,
            curve_records=curve_records))

    del data
    return results


def evaluate_link_models_cross_graph(dataset_name, models, data_dir, dataset_dir, syn_file_name,
                                     trials, val_ratio, test_ratio, neg_sampling,
                                     decoder, epochs, patience, lr, drop_rate,
                                     h_feats, num_layers, norm_stats_path=None,
                                     curve_records=None):
    """Train GNNs on synthetic graph edges, validate on synthetic val, test on original test edges."""

    # Load normalization stats if available
    norm_stats = None
    if norm_stats_path and os.path.exists(norm_stats_path):
        norm_stats = torch.load(norm_stats_path, weights_only=False)
        print(f"  Loaded normalization stats ({norm_stats['method']}) from {norm_stats_path}")

    results = []

    for model_name in models:
        if model_name not in CROSS_GRAPH_LP_SUPPORTED_MODELS:
            print(f"  NOTE: '{model_name}' skipped in cross-graph link prediction mode.")
            continue

        print(f"\n{'='*60}")
        print(f"  SYNTHETIC-GRAPH (CROSS) | {dataset_name} | {model_name}")
        print(f"{'='*60}")

        auc_list, pre_list, rec_list = [], [], []
        val_curves, test_curves = [], []
        time_list = []

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

            # Apply same normalization as was used during synthetic generation
            if norm_stats is not None:
                from bench_utils import apply_normalization
                orig_data.graph.ndata['feature'] = apply_normalization(
                    orig_data.graph.ndata['feature'], norm_stats)
                orig_data.train_graph.ndata['feature'] = apply_normalization(
                    orig_data.train_graph.ndata['feature'], norm_stats)

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

            if model_name == 'XGBoost':
                detector = CrossGraphXGBoostLinkPredictor(train_config, model_config, syn_data, orig_data)
            elif model_name == 'XGBGraph':
                detector = CrossGraphXGBGraphLinkPredictor(train_config, model_config, syn_data, orig_data)
            else:
                detector = CrossGraphLinkPredictor(train_config, model_config, syn_data, orig_data)

            st = time.time()
            test_score = detector.train()
            ed = time.time()
            dt = ed - st
            time_list.append(dt)

            auc_list.append(test_score['AUROC'])
            pre_list.append(test_score['AUPRC'])
            rec_list.append(test_score['RecK'])

            if 'val_auprc_curve' in test_score:
                val_curves.append(test_score['val_auprc_curve'])
                test_curves.append(test_score['test_auprc_curve'])

            print(f"  -> AUROC={test_score['AUROC']:.4f}, "
                  f"AUPRC={test_score['AUPRC']:.4f}, "
                  f"RecK={test_score['RecK']:.4f}  [{dt:.1f}s]")
            del detector, syn_data, orig_data

        if time_list:
            total = sum(time_list)
            print(f"  [{model_name} on {dataset_name}] {len(time_list)} trials — "
                  f"total {format_duration(total)}, "
                  f"mean {format_duration(total / len(time_list))} "
                  f"± {format_duration(float(np.std(time_list)))}")

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
                'time_per_trial': sum(time_list) / len(time_list) if time_list else 0,
            })

        if val_curves and curve_records is not None:
            max_len = max(len(c) for c in val_curves)
            def pad(curve):
                return curve + [curve[-1]] * (max_len - len(curve))
            val_arr  = np.array([pad(c) for c in val_curves])
            test_arr = np.array([pad(c) for c in test_curves])
            for epoch in range(max_len):
                curve_records.append({
                    'source': 'synthetic-graph',
                    'dataset': dataset_name, 'model': model_name, 'epoch': epoch,
                    'val_auprc_mean':  float(val_arr[:, epoch].mean()),
                    'val_auprc_std':   float(val_arr[:, epoch].std()),
                    'test_auprc_mean': float(test_arr[:, epoch].mean()),
                    'test_auprc_std':  float(test_arr[:, epoch].std()),
                })

    return results


def main():
    args = parse_link_args()
    script_start = time.time()

    # Resolve default paths relative to project root
    if args.data_dir is None:
        args.data_dir = os.path.join(_project_root, 'datasets')
    if args.synthetic_dir is None:
        args.synthetic_dir = os.path.join(_project_root, 'datasets', 'synthetic')
    if args.output_dir is None:
        # Auto-derive: results/evaluate/{generator}/{dataset}/{task}/{synthetic_name}
        # Mirrors anomaly_benchmark.py and datasets/synthetic/{generator}/
        # {dataset}/{task}/{variant}/ so results from different tasks on the
        # same variant stem don't share a directory.
        # For multi-dataset runs, the dataset level is omitted.
        datasets_tmp = [d.strip() for d in args.datasets.split(',')]
        stem = args.synthetic_name or 'original'
        if len(datasets_tmp) == 1:
            args.output_dir = os.path.join(
                _project_root, 'results', 'evaluate',
                args.generator, datasets_tmp[0], args.task, stem)
        else:
            args.output_dir = os.path.join(
                _project_root, 'results', 'evaluate',
                args.generator, args.task, stem)

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
    all_curve_records = []

    # --- Phase 1: Evaluate on original data ---
    if args.skip_phase1:
        print("\n" + "#" * 80)
        print("# PHASE 1: SKIPPED (--skip_phase1)")
        print("#" * 80)
    else:
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
                args.lr, args.drop_rate, args.h_feats, args.num_layers,
                curve_records=all_curve_records)
            all_results.extend(results)

    # --- Phase 2: Evaluate on synthetic data ---
    print("\n" + "#" * 80)
    print("# PHASE 2: LINK PREDICTION ON SYNTHETIC DATA")
    print("#" * 80)

    for dataset_name in datasets:
        # Resolve path: synthetic_dir/<generator>/<dataset>/<task>/<stem>[.pt]
        # If --graph_path is set, override path resolution entirely (used for
        # task-agnostic artifacts like datasets/kanon/<dataset>/<stem>.dgl).
        if args.graph_path is not None:
            if args.synthetic_type != 'graph':
                raise ValueError('--graph_path is only supported with '
                                 '--synthetic_type graph.')
            syn_path = args.graph_path
            task_dir = os.path.dirname(os.path.abspath(syn_path))
            stem = os.path.splitext(os.path.basename(syn_path))[0]
        else:
            gen_dir = os.path.join(args.synthetic_dir, args.generator)
            dataset_dir = os.path.join(gen_dir, dataset_name)
            task_dir = os.path.join(dataset_dir, args.task)
            stem = args.synthetic_name
            if args.synthetic_type == 'comp-graph':
                variant_dir = os.path.join(task_dir, stem)
                canonical_path = os.path.join(variant_dir, f'{stem}.pt')
                trial_paths = resolve_cgt_trial_paths(canonical_path, args.trials)
                if trial_paths is None:
                    if os.path.exists(canonical_path):
                        print(f"  Falling back to single-file CGT .pt for all "
                              f"{args.trials} trials: {canonical_path}")
                        trial_paths = [canonical_path] * args.trials
                    else:
                        print(f"\n  Skipping {dataset_name}: no per-trial .pt at "
                              f"{variant_dir}/{stem}_t*.pt and no canonical "
                              f"{canonical_path}")
                        continue
                syn_path = trial_paths[0]
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
                args.epochs, args.patience, trial_paths,
                args.batch_size, args.lr, args.drop_rate,
                args.h_feats, args.num_layers,
                eval_mode=args.eval_mode,
                curve_records=all_curve_records)
        else:
            # Full graph (BiGG, etc.): train+val on synthetic, test on original.
            norm_stats_path = os.path.join(task_dir, stem + '_norm_stats.pt')
            results = evaluate_link_models_cross_graph(
                dataset_name, models, args.data_dir, task_dir, stem,
                args.trials, args.val_ratio, args.test_ratio,
                args.neg_sampling, args.decoder,
                args.epochs, args.patience,
                args.lr, args.drop_rate, args.h_feats, args.num_layers,
                norm_stats_path=norm_stats_path,
                curve_records=all_curve_records)
        all_results.extend(results)

    # --- Save and display results ---
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Suffix results filename when running a non-default subset so parallel
        # SLURM array tasks (one per eval_mode) don't clobber each other's files.
        if args.skip_phase1 or args.eval_mode != 'both':
            phase_tag = 'phase2only' if args.skip_phase1 else 'allphases'
            results_stem = f'link_prediction_results__{phase_tag}_{args.eval_mode}'
        else:
            results_stem = 'link_prediction_results'

        xlsx_path = os.path.join(args.output_dir, f'{results_stem}.xlsx')
        results_df.to_excel(xlsx_path, index=False)

        csv_path = os.path.join(args.output_dir, f'{results_stem}.csv')
        results_df.to_csv(csv_path, index=False)

        print(f"\nResults saved to:\n  {xlsx_path}\n  {csv_path}")
        print_comparison(all_results, datasets, models)
    else:
        print("\nNo results to save.")

    if all_curve_records:
        # Suffix curves CSV the same way as results, so parallel SLURM array
        # tasks (one per eval_mode) don't clobber each other's curve files.
        if args.skip_phase1 or args.eval_mode != 'both':
            phase_tag = 'phase2only' if args.skip_phase1 else 'allphases'
            curves_stem = f'link_divergence_curves__{phase_tag}_{args.eval_mode}'
        else:
            curves_stem = 'link_divergence_curves'
        curves_df = pd.DataFrame(all_curve_records)
        curves_path = os.path.join(args.output_dir, f'{curves_stem}.csv')
        curves_df.to_csv(curves_path, index=False)
        print(f"  Divergence curves saved to: {curves_path}")

    print(f"\n[Total] {format_duration(time.time() - script_start)}")


if __name__ == '__main__':
    main()
