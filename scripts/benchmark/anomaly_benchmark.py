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
    load_bigg_synthetic_graph, load_bigg_real_subsampled_graph,
    apply_normalization,
    apply_inferred_trial_id,
    discover_bigg_split_bundle,
    resolve_cgt_trial_paths, make_cgt_rebuild_fn,
    _assert_pt_alignment, format_duration,
)
from models.anomaly_detection.cgt_detector import (
    CompGraphDetector, CG_SUPPORTED_MODELS,
    CGTXGBoostDetector, CGTXGBGraphDetector)

CGT_TREE_MODELS = {
    'XGBoost': CGTXGBoostDetector,
    'XGBGraph': CGTXGBGraphDetector,
}

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


def _restrict_test_mask(data, ratio, seed, portion):
    """Replace ``data.graph.ndata['test_mask']`` with an anomaly-stratified subset.

    Used by the BO tuning framework so selection (``portion='tune'``) and
    final reporting (``portion='heldout'``) operate on disjoint test nodes.
    The stratification on the binary anomaly label keeps the positive rate
    stable across the two halves on imbalanced datasets like weibo.
    """
    if ratio is None:
        return
    if not (0.0 < ratio < 1.0):
        raise ValueError(
            f"--tune_test_ratio must be in (0, 1) when set, got {ratio}")
    from sklearn.model_selection import train_test_split as _tts
    mask = data.graph.ndata['test_mask'].bool()
    test_idx = mask.nonzero(as_tuple=True)[0].cpu().numpy()
    if test_idx.size == 0:
        return
    labels = data.graph.ndata['label'][test_idx].cpu().numpy()
    pos = int((labels > 0).sum())
    neg = int((labels == 0).sum())
    if pos < 2 or neg < 2:
        # Stratification needs >=2 of each class; fall back to unstratified.
        tune_idx, heldout_idx = _tts(
            test_idx, train_size=ratio, random_state=seed, shuffle=True)
    else:
        tune_idx, heldout_idx = _tts(
            test_idx, train_size=ratio, random_state=seed,
            stratify=labels, shuffle=True)
    keep = tune_idx if portion == 'tune' else heldout_idx
    new_mask = torch.zeros_like(data.graph.ndata['test_mask'])
    new_mask[keep] = 1
    data.graph.ndata['test_mask'] = new_mask.bool()


def evaluate_models(dataset_name, models, data_dir,
                    trials, semi_supervised, trial_id,
                    epochs, patience, lr, drop_rate, h_feats, num_layers,
                    curve_records=None, split_ids=None, seeds_per_split=1,
                    per_trial_records=None,
                    tune_test_ratio=None, tune_test_seed=0,
                    tune_portion='tune'):
    """Train and evaluate all specified models on the original dataset.

    ``split_ids`` overrides the per-trial GADBench split. When ``None``, the
    legacy ``trial_id + t`` rotation is used (single-variant runs). When a
    list is passed, trial ``t`` uses ``split_ids[t]`` — this lets BiGG
    split-bundle runs rotate the baseline through the same splits the
    bundle's per-variant synthetic graphs were trained on.

    ``seeds_per_split`` only applies in bundle mode (``split_ids`` provided).
    Each split is repeated ``seeds_per_split`` times with the *same* seeds
    (``SEED_LIST[:seeds_per_split]``) so trial-to-trial variance does not
    conflate seed and split. Total runs = ``len(split_ids) * seeds_per_split``.
    In single-variant mode, ``seeds_per_split`` is ignored and the legacy
    one-seed-per-trial rotation is used.
    """
    results = []
    if split_ids is not None and len(split_ids) != trials:
        raise ValueError(
            f"split_ids has {len(split_ids)} entries but trials={trials}")

    if split_ids is not None:
        runs = [(split_ids[s], SEED_LIST[k])
                for s in range(trials)
                for k in range(seeds_per_split)]
    else:
        runs = [(trial_id + t, SEED_LIST[t]) for t in range(trials)]

    for model_name in models:
        if model_name not in model_detector_dict:
            print(f"  WARNING: '{model_name}' not in GADBench. Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  ORIGINAL | {dataset_name} | {model_name}")
        print(f"{'='*60}")

        auc_list, pre_list, rec_list = [], [], []
        val_curves, test_curves = [], []
        time_list = []

        for t, (split_col, seed) in enumerate(runs):
            torch.cuda.empty_cache()
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
            data.split(semi_supervised, split_col)
            _restrict_test_mask(data, tune_test_ratio, tune_test_seed, tune_portion)

            model_config = {
                'model': model_name,
                'lr': lr,
                'drop_rate': drop_rate,
                'h_feats': h_feats,
                'num_layers': num_layers,
            }
            if dataset_name == 'tsocial':
                model_config['h_feats'] = 16

            print(f"  Trial {t}, split={split_col}, seed={seed}")
            detector = model_detector_dict[model_name](
                train_config, model_config, data)

            st = time.time()
            test_score = detector.train()
            ed = time.time()
            dt = ed - st
            time_list.append(dt)

            auc_list.append(test_score['AUROC'])
            pre_list.append(test_score['AUPRC'])
            rec_list.append(test_score['RecK'])

            if per_trial_records is not None:
                per_trial_records.append({
                    'source': 'original',
                    'dataset': dataset_name,
                    'model': model_name,
                    'split_id': int(split_col),
                    'seed': int(seed),
                    'AUROC': float(test_score['AUROC']),
                    'AUPRC': float(test_score['AUPRC']),
                    'RecK':  float(test_score['RecK']),
                    'time_sec': float(dt),
                })

            if 'val_auprc_curve' in test_score:
                val_curves.append(test_score['val_auprc_curve'])
                test_curves.append(test_score['test_auprc_curve'])

            print(f"  -> AUROC={test_score['AUROC']:.4f}, "
                  f"AUPRC={test_score['AUPRC']:.4f}, "
                  f"RecK={test_score['RecK']:.4f}  [{dt:.1f}s]")

            del detector

        if time_list:
            total = sum(time_list)
            print(f"  [{model_name}] {len(time_list)} trials — "
                  f"total {format_duration(total)}, "
                  f"mean {format_duration(total / len(time_list))} "
                  f"± {format_duration(float(np.std(time_list)))}")

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
                    'source': 'original',
                    'dataset': dataset_name, 'model': model_name, 'epoch': epoch,
                    'val_auprc_mean':  float(val_arr[:, epoch].mean()),
                    'val_auprc_std':   float(val_arr[:, epoch].std()),
                    'test_auprc_mean': float(test_arr[:, epoch].mean()),
                    'test_auprc_std':  float(test_arr[:, epoch].std()),
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
        time_list = []

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
            detector_cls = CGT_TREE_MODELS.get(model_name, CompGraphDetector)
            detector = detector_cls(
                train_config, model_config, train_ds, val_ds, test_ds)

            st = time.time()
            test_score = detector.train()
            ed = time.time()
            dt = ed - st
            time_list.append(dt)

            auc_list.append(test_score['AUROC'])
            pre_list.append(test_score['AUPRC'])
            rec_list.append(test_score['RecK'])

            print(f"  -> AUROC={test_score['AUROC']:.4f}, "
                  f"AUPRC={test_score['AUPRC']:.4f}, "
                  f"RecK={test_score['RecK']:.4f}  [{dt:.1f}s]")

            del detector

        if time_list:
            total = sum(time_list)
            print(f"  [{model_name}] {len(time_list)} trials — "
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
    # Prefer per-trial files (_t0.pt, _t1.pt, …); fall back to single file.
    trial_paths_probe = resolve_cgt_trial_paths(syn_path, trials)
    if trial_paths_probe is not None:
        syn_data = load_cgt_synthetic_data(trial_paths_probe[0])
    elif os.path.exists(syn_path):
        syn_data = load_cgt_synthetic_data(syn_path)
    else:
        raise FileNotFoundError(
            f"Synthetic data not found: {syn_path}\n"
            f"Also could not find per-trial files (_t0..t{trials-1}.pt).")

    feat_dim = data.graph.ndata['feature'].shape[1]

    supported = set(CG_SUPPORTED_MODELS) | set(CGT_TREE_MODELS)
    cg_models = [m for m in models if m in supported]
    skipped = [m for m in models if m not in supported]
    if skipped:
        print(f"  NOTE: {skipped} not supported in computation graph mode. "
              f"Skipping.")

    if 'XGBGraph' in cg_models:
        cg_depth_pt = int(
            syn_data.get('cg_depth', syn_data.get('subgraph_step_num')))
        if cg_depth_pt != num_layers:
            raise ValueError(
                f"XGBGraph on CGT requires num_layers == cg_depth; "
                f"got num_layers={num_layers}, cg_depth={cg_depth_pt} "
                f"(in {syn_path}).")

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
    # When per-trial .pt files exist (one per split), load a different file
    # each trial so the synthetic split varies — matching the original-CG
    # baseline. Otherwise fall back to single-file mode (seed-only variation).
    trial_paths = resolve_cgt_trial_paths(syn_path, trials)

    if trial_paths is not None:
        print(f"  Found {len(trial_paths)} per-trial .pt files — varying splits across trials.")
        syn_data_0 = load_cgt_synthetic_data(trial_paths[0])
        syn_train0, syn_val0, test_ds0 = build_cgt_datasets(data.graph, syn_data_0)
        rebuild_fn = make_cgt_rebuild_fn(
            data.graph, trial_paths,
            trial_id_offset=trial_id, semi_supervised=semi_supervised)
        results.extend(_run_cg_trials(
            dataset_name, cg_models, syn_train0, syn_val0, test_ds0,
            feat_dim, 'synthetic-cgt', trials, epochs, patience,
            batch_size, lr, drop_rate, h_feats, num_layers,
            rebuild_datasets_fn=rebuild_fn))
    else:
        # Single-file mode: verify the .pt was trained under the same
        # trial_id / semi_supervised the benchmark is running with.
        mask_col = trial_id + (10 if semi_supervised else 0)
        single_test_ids = data.graph.ndata['test_masks'][:, mask_col].bool().nonzero(
            as_tuple=True)[0].numpy()
        _assert_pt_alignment(
            syn_data,
            expected_trial_id=trial_id,
            expected_semi_supervised=semi_supervised,
            expected_test_ids=single_test_ids,
            source_label='synthetic-cgt[single-file]',
        )
        syn_train, syn_val, test_ds = build_cgt_datasets(data.graph, syn_data)
        results.extend(_run_cg_trials(
            dataset_name, cg_models, syn_train, syn_val, test_ds,
            feat_dim, 'synthetic-cgt', trials, epochs, patience,
            batch_size, lr, drop_rate, h_feats, num_layers))

    del data
    return results


def evaluate_models_cross_graph(dataset_name, models, data_dir, dataset_dir, syn_file_names,
                                trials, semi_supervised, split_ids,
                                epochs, patience, lr, drop_rate, h_feats, num_layers,
                                norm_stats_paths=None, curve_records=None,
                                source_label='synthetic-graph',
                                seeds_per_split=1,
                                per_trial_records=None,
                                tune_test_ratio=None, tune_test_seed=0,
                                tune_portion='tune'):
    """Train GNNs on a train-source graph, validate on its val mask, test on original test nodes.

    *source_label* identifies the training-source in printed banners and result
    rows — e.g. ``'synthetic-graph'`` (BiGG-generated subgraphs) or
    ``'real-subsampled-graph'`` (the real forest-fire subsamples used to train
    the generative model). The rest of the pipeline is identical.

    *syn_file_names*, *norm_stats_paths*, and *split_ids* are per-split lists
    of length ``trials``. In single-variant mode all entries are identical
    (same synthetic graph reused across trials, only the split column varies).
    In split-bundle mode each entry corresponds to one BiGG variant +
    matching GADBench split column.

    ``seeds_per_split`` repeats each (synthetic graph, split) combination with
    the same ``SEED_LIST[:seeds_per_split]`` seeds so seed and split variance
    are not conflated. Total runs = ``trials * seeds_per_split``.

    Per-epoch val/test AUPRC curves are appended to curve_records (if provided),
    averaged across all runs, for val/test divergence analysis.
    """
    from models.cross_graph_detector import (CrossGraphGNNDetector,
                                                CrossGraphXGBGraphDetector,
                                                CrossGraphXGBDetector,
                                                CROSS_GRAPH_SUPPORTED_MODELS)
    if len(syn_file_names) != trials or len(split_ids) != trials:
        raise ValueError(
            f"per-trial lists length mismatch: trials={trials}, "
            f"syn_file_names={len(syn_file_names)}, "
            f"split_ids={len(split_ids)}")
    if norm_stats_paths is None:
        norm_stats_paths = [None] * trials
    elif len(norm_stats_paths) != trials:
        raise ValueError(
            f"norm_stats_paths length {len(norm_stats_paths)} != trials={trials}")

    # Cache parsed norm_stats by path so we don't reload the same file each trial
    norm_stats_cache = {}
    def _load_norm_stats(path):
        if not path or not os.path.exists(path):
            return None
        if path not in norm_stats_cache:
            norm_stats_cache[path] = torch.load(path, weights_only=False)
            print(f"  Loaded normalization stats "
                  f"({norm_stats_cache[path]['method']}) from {path}")
        return norm_stats_cache[path]

    results = []

    for model_name in models:
        if model_name not in CROSS_GRAPH_SUPPORTED_MODELS:
            print(f"  NOTE: '{model_name}' skipped in cross-graph mode.")
            continue

        print(f"\n{'='*60}")
        print(f"  {source_label.upper()} (CROSS) | {dataset_name} | {model_name}")
        print(f"{'='*60}")

        auc_list, pre_list, rec_list = [], [], []
        val_curves, test_curves = [], []
        time_list = []

        runs = [(s, k) for s in range(trials) for k in range(seeds_per_split)]

        for t, (s, k) in enumerate(runs):
            torch.cuda.empty_cache()
            seed = SEED_LIST[k]
            set_seed(seed)

            split_col = split_ids[s]
            syn_file_name = syn_file_names[s]
            norm_stats = _load_norm_stats(norm_stats_paths[s])

            # Synthetic graph: train + val from synthetic masks
            syn_data = GADBenchDataset(syn_file_name, prefix=dataset_dir + '/')
            syn_data.split(semi_supervised, split_col)

            # Original graph: test nodes only
            orig_data = GADBenchDataset(dataset_name, prefix=data_dir + '/')
            orig_data.split(semi_supervised, split_col)
            _restrict_test_mask(orig_data, tune_test_ratio, tune_test_seed, tune_portion)

            # Apply same normalization as was used during synthetic generation
            if norm_stats is not None:
                from bench_utils import apply_normalization
                orig_data.graph.ndata['feature'] = apply_normalization(
                    orig_data.graph.ndata['feature'], norm_stats)

            train_config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'epochs': epochs, 'patience': patience, 'metric': 'AUPRC', 'seed': seed,
            }
            model_config = {
                'model': model_name, 'lr': lr, 'drop_rate': drop_rate,
                'h_feats': 16 if dataset_name == 'tsocial' else h_feats,
                'num_layers': num_layers,
            }

            print(f"  Trial {t}, split={split_col}, seed={seed}")
            if model_name == 'XGBGraph':
                detector = CrossGraphXGBGraphDetector(train_config, model_config, syn_data, orig_data)
            elif model_name == 'XGBoost':
                detector = CrossGraphXGBDetector(train_config, model_config, syn_data, orig_data)
            else:
                detector = CrossGraphGNNDetector(train_config, model_config, syn_data, orig_data)

            st = time.time()
            test_score = detector.train()
            ed = time.time()
            dt = ed - st
            time_list.append(dt)

            auc_list.append(test_score['AUROC'])
            pre_list.append(test_score['AUPRC'])
            rec_list.append(test_score['RecK'])

            if per_trial_records is not None:
                per_trial_records.append({
                    'source': source_label,
                    'dataset': dataset_name,
                    'model': model_name,
                    'split_id': int(split_col),
                    'seed': int(seed),
                    'AUROC': float(test_score['AUROC']),
                    'AUPRC': float(test_score['AUPRC']),
                    'RecK':  float(test_score['RecK']),
                    'time_sec': float(dt),
                })

            print(f"  -> AUROC={test_score['AUROC']:.4f}, AUPRC={test_score['AUPRC']:.4f}, "
                  f"RecK={test_score['RecK']:.4f}  [{dt:.1f}s]")

            if 'val_auprc_curve' in test_score:
                val_curves.append(test_score['val_auprc_curve'])
                test_curves.append(test_score['test_auprc_curve'])

            del detector, syn_data, orig_data

        if time_list:
            total = sum(time_list)
            print(f"  [{model_name}] {len(time_list)} trials — "
                  f"total {format_duration(total)}, "
                  f"mean {format_duration(total / len(time_list))} "
                  f"± {format_duration(float(np.std(time_list)))}")

        if auc_list:
            results.append({
                'source': source_label, 'dataset': dataset_name, 'model': model_name,
                'AUROC_mean': np.mean(auc_list), 'AUROC_std': np.std(auc_list),
                'AUPRC_mean': np.mean(pre_list), 'AUPRC_std': np.std(pre_list),
                'RecK_mean':  np.mean(rec_list),  'RecK_std':  np.std(rec_list),
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


def _build_bigg_variant_caches(variant_syn_path, cache_dir, stem,
                               cdf_invert, data_dir, dataset_name):
    """Build (or reuse) combined-graph + real-subsampled caches for one BiGG variant.

    Caches live at ``cache_dir/{stem}_combined_v2[_cdf_invert]`` and
    ``cache_dir/{stem}_real_sub_combined_v3[_cdf_invert]`` so single-variant
    runs and split-bundle runs can share or isolate caches naturally.

    Returns dict with keys: ``eval_stem``, ``norm_stats_path``,
    ``real_sub_stem``, ``real_sub_norm_path`` (the latter two are ``None``
    when no real subsamples are available).
    """
    cdf_tag = '' if cdf_invert == 'linear' else f'_{cdf_invert}'
    out = {'eval_stem': stem, 'norm_stats_path': None,
           'real_sub_stem': None, 'real_sub_norm_path': None}

    if os.path.isdir(variant_syn_path):
        combined_stem = stem + '_combined_v2' + cdf_tag
        combined_path = os.path.join(cache_dir, combined_stem)
        if not os.path.exists(combined_path):
            print(f'  Combining subgraphs → {combined_path}')
            combined_graph, norm_stats = load_bigg_synthetic_graph(
                variant_syn_path, cdf_mode=cdf_invert)
            dgl.save_graphs(combined_path, [combined_graph])
            if norm_stats is not None:
                torch.save(norm_stats, combined_path + '_norm_stats.pt')
        else:
            print(f'  Using cached combined graph: {combined_path}')
        out['eval_stem'] = combined_stem
        out['norm_stats_path'] = combined_path + '_norm_stats.pt'

        real_sub_stem = stem + '_real_sub_combined_v3' + cdf_tag
        real_sub_path = os.path.join(cache_dir, real_sub_stem)
        real_sub_norm_path = real_sub_path + '_norm_stats.pt'
        if os.path.exists(real_sub_path):
            print(f'  Using cached real-subsampled combined graph: {real_sub_path}')
            out['real_sub_stem'] = real_sub_stem
            out['real_sub_norm_path'] = (real_sub_norm_path
                                         if os.path.exists(real_sub_norm_path) else None)
        else:
            orig_for_masks = GADBenchDataset(
                dataset_name, prefix=data_dir + '/').graph
            real_combined, real_norm = load_bigg_real_subsampled_graph(
                variant_syn_path, orig_for_masks, cdf_mode=cdf_invert)
            if real_combined is not None:
                print(f'  Combining real subsamples → {real_sub_path}')
                dgl.save_graphs(real_sub_path, [real_combined])
                if real_norm is not None:
                    torch.save(real_norm, real_sub_norm_path)
                out['real_sub_stem'] = real_sub_stem
                out['real_sub_norm_path'] = (real_sub_norm_path
                                             if os.path.exists(real_sub_norm_path) else None)
            else:
                print(f'  No training_subsamples/ found in {variant_syn_path}; '
                      f'skipping real-subsampled source.')
    else:
        inverted_stem = stem + '_inverted_v2' + cdf_tag
        inverted_path = os.path.join(cache_dir, inverted_stem)
        if not os.path.exists(inverted_path):
            print(f'  Inverting synthetic features → {inverted_path}')
            syn_graph, norm_stats = load_bigg_synthetic_graph(
                variant_syn_path, cdf_mode=cdf_invert)
            dgl.save_graphs(inverted_path, [syn_graph])
            if norm_stats is not None:
                torch.save(norm_stats, inverted_path + '_norm_stats.pt')
        else:
            print(f'  Using cached inverted synthetic graph: {inverted_path}')
        out['eval_stem'] = inverted_stem
        out['norm_stats_path'] = inverted_path + '_norm_stats.pt'

    return out


def main():
    t_start = time.time()
    args = parse_args()
    apply_inferred_trial_id(args)

    # Resolve default paths relative to project root
    if args.data_dir is None:
        args.data_dir = os.path.join(_project_root, 'datasets', 'original')
    if args.synthetic_dir is None:
        args.synthetic_dir = os.path.join(_project_root, 'datasets', 'synthetic')
    if args.output_dir is None:
        # Auto-derive: results/evaluate/{generator}/{dataset}/{synthetic_name}
        # For multi-dataset runs, dataset level is omitted.
        datasets_tmp = [d.strip() for d in args.datasets.split(',')]
        stem = args.synthetic_name or 'original'
        if len(datasets_tmp) == 1:
            args.output_dir = os.path.join(
                _project_root, 'results', 'evaluate',
                args.generator, datasets_tmp[0], stem)
        else:
            args.output_dir = os.path.join(
                _project_root, 'results', 'evaluate',
                args.generator, stem)

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
    print(f"  CDF invert:     {args.cdf_invert}")

    print("\nTraining:")
    print(f"  Trials:         {args.trials}")
    print(f"  Seeds/split:    {args.seeds_per_split}  (BiGG bundle mode only)")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Patience:       {args.patience}")
    print(f"  Batch size:     {args.batch_size}  (CGT only)")

    print("\nModel Architecture:")
    print(f"  LR:             {args.lr}")
    print(f"  Drop rate:      {args.drop_rate}")
    print(f"  Hidden feats:   {args.h_feats}")
    print(f"  Num layers:     {args.num_layers}  (must equal cg_depth in CGT .pt)")

    all_results = []
    all_curve_records = []
    all_per_trial_records = [] if args.dump_per_trial else None
    if args.tune_test_ratio is not None:
        print(f"  Tune-mask:      ratio={args.tune_test_ratio} "
              f"seed={args.tune_test_seed} portion={args.tune_portion}")

    # --- Pre-resolve per-dataset state (syn path, bundle vs single variant) ---
    # Bundle detection happens up front so Phase 1 can rotate the baseline
    # through the bundle's split ids and so we error out on misconfiguration
    # before any training starts.
    per_dataset_state = {}
    for dataset_name in datasets:
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
            if stem is None:
                syn_path = None
            elif args.synthetic_type == 'comp-graph':
                variant_dir = os.path.join(task_dir, stem)
                syn_path = os.path.join(variant_dir, f'{stem}.pt')
            else:
                syn_path = os.path.join(task_dir, stem)

        bundle = None
        if args.synthetic_type == 'graph' and syn_path:
            bundle = discover_bigg_split_bundle(syn_path)
            if bundle is not None:
                ids = [sid for sid, _ in bundle]
                if args.trial_id != 0:
                    raise ValueError(
                        f"--trial_id={args.trial_id} cannot be used with the "
                        f"BiGG split bundle '{stem}'. Bundle mode iterates "
                        f"through splits {ids} automatically.")
                if args.trials != 1 and args.trials != len(bundle):
                    raise ValueError(
                        f"--trials={args.trials} but bundle '{stem}' contains "
                        f"{len(bundle)} per-split variants (splits {ids}). "
                        f"Pass --trials {len(bundle)} or omit it (defaults to "
                        f"bundle size).")
                if args.trials != len(bundle):
                    print(f"  Bundle '{stem}' overrides --trials to {len(bundle)} "
                          f"(was {args.trials}).")
                    args.trials = len(bundle)
                print(f"  BiGG split bundle '{stem}': {len(bundle)} variants — "
                      f"splits {ids}")

        per_dataset_state[dataset_name] = {
            'syn_path': syn_path,
            'task_dir': task_dir,
            'stem': stem,
            'bundle': bundle,
        }

    # --- Phase 1: Evaluate on original data ---
    if args.skip_original:
        print("\n" + "#" * 80)
        print("# PHASE 1 SKIPPED (per --skip_original)")
        print("#" * 80)
    else:
        print("\n" + "#" * 80)
        print("# PHASE 1: EVALUATING ON ORIGINAL DATA")
        print("#" * 80)
        t_phase1 = time.time()

        baseline_override = None
        if args.baseline_split_ids:
            baseline_override = [int(x) for x in args.baseline_split_ids.split(',')]
            args.trials = len(baseline_override)
            print(f"  Baseline-only mode: split_ids={baseline_override} "
                  f"seeds_per_split={args.seeds_per_split} trials={args.trials}")

        for dataset_name in datasets:
            bundle = per_dataset_state[dataset_name]['bundle']
            if baseline_override is not None:
                split_ids_for_phase1 = baseline_override
                seeds_for_phase1 = args.seeds_per_split
            elif bundle is not None:
                split_ids_for_phase1 = [sid for sid, _ in bundle]
                seeds_for_phase1 = args.seeds_per_split
            else:
                split_ids_for_phase1 = None
                seeds_for_phase1 = 1
            results = evaluate_models(
                dataset_name, models, args.data_dir,
                args.trials, args.semi_supervised, args.trial_id,
                args.epochs, args.patience,
                args.lr, args.drop_rate, args.h_feats, args.num_layers,
                curve_records=all_curve_records,
                split_ids=split_ids_for_phase1,
                seeds_per_split=seeds_for_phase1,
                per_trial_records=all_per_trial_records,
                tune_test_ratio=args.tune_test_ratio,
                tune_test_seed=args.tune_test_seed,
                tune_portion=args.tune_portion)
            all_results.extend(results)

        phase1_elapsed = time.time() - t_phase1
        print(f"\n[Phase 1: original-data] {format_duration(phase1_elapsed)}")

    # --- Phase 2: Evaluate on synthetic data ---
    print("\n" + "#" * 80)
    print("# PHASE 2: EVALUATING ON SYNTHETIC DATA")
    print("#" * 80)
    t_phase2 = time.time()

    for dataset_name in datasets:
        state = per_dataset_state[dataset_name]
        syn_path = state['syn_path']
        task_dir = state['task_dir']
        stem = state['stem']
        bundle = state['bundle']

        # Prefer per-trial files; fall back to single file.
        if args.synthetic_type == 'comp-graph':
            trial_paths = resolve_cgt_trial_paths(syn_path, args.trials)
            if trial_paths is not None:
                print(f"\n  Found {len(trial_paths)} per-trial .pt files for {dataset_name}")
            elif syn_path and os.path.exists(syn_path):
                print(f"\n  Found {args.synthetic_type} synthetic data: {syn_path}")
            else:
                print(f"\n  Skipping {dataset_name}: {syn_path} not found")
                continue
        elif not syn_path or not os.path.exists(syn_path):
            print(f"\n  Skipping {dataset_name}: {syn_path} not found")
            continue
        else:
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
            all_results.extend(results)
            continue

        # --- BiGG full-graph path ---
        # Build per-trial config. In single-variant mode every trial reuses
        # the same synthetic graph; only the split column rotates. In
        # split-bundle mode each trial loads its own variant + split. Caches
        # for bundle variants live inside the bundle dir so re-bundling
        # doesn't invalidate them.
        #
        # Synthetic features are inverted back to raw space at cache-build
        # time (when a lossless normalization was used during training), so
        # all benchmark rows live in the dataset's native feature space and
        # the original-vs-synthetic delta isn't conflated with normalization.
        # The ``_v2`` suffix invalidates caches that pre-date that change.
        # ``cdf_invert`` is appended to the cache stem when not the default
        # ``linear`` so switching modes doesn't reuse stale caches.
        if bundle is not None:
            cache_parent = syn_path
            variant_paths = [(sid, os.path.join(syn_path, sub), sub)
                             for sid, sub in bundle]
        else:
            cache_parent = task_dir
            variant_paths = [(args.trial_id + t, syn_path, stem)
                             for t in range(args.trials)]

        syn_file_names, norm_stats_paths = [], []
        real_sub_file_names, real_sub_norm_paths = [], []
        split_ids = []
        real_sub_available = True
        for sid, variant_path, variant_stem in variant_paths:
            cache = _build_bigg_variant_caches(
                variant_path, cache_parent, variant_stem,
                args.cdf_invert, args.data_dir, dataset_name)
            syn_file_names.append(cache['eval_stem'])
            norm_stats_paths.append(cache['norm_stats_path'])
            split_ids.append(sid)
            if cache['real_sub_stem'] is None:
                real_sub_available = False
            else:
                real_sub_file_names.append(cache['real_sub_stem'])
                real_sub_norm_paths.append(cache['real_sub_norm_path'])

        seeds_for_phase2 = args.seeds_per_split if bundle is not None else 1
        results = evaluate_models_cross_graph(
            dataset_name, models, args.data_dir, cache_parent,
            syn_file_names,
            args.trials, args.semi_supervised, split_ids,
            args.epochs, args.patience,
            args.lr, args.drop_rate, args.h_feats, args.num_layers,
            norm_stats_paths=norm_stats_paths,
            curve_records=all_curve_records,
            seeds_per_split=seeds_for_phase2,
            per_trial_records=all_per_trial_records,
            tune_test_ratio=args.tune_test_ratio,
            tune_test_seed=args.tune_test_seed,
            tune_portion=args.tune_portion)
        all_results.extend(results)

        # Train on the REAL forest-fire subsamples, test on full original.
        # Shares subsampling + cross-graph testing with the synthetic path,
        # so the delta between the two isolates generative-model fidelity
        # from the subsampling effect.
        if real_sub_available and real_sub_file_names:
            results = evaluate_models_cross_graph(
                dataset_name, models, args.data_dir, cache_parent,
                real_sub_file_names,
                args.trials, args.semi_supervised, split_ids,
                args.epochs, args.patience,
                args.lr, args.drop_rate, args.h_feats, args.num_layers,
                norm_stats_paths=real_sub_norm_paths,
                curve_records=all_curve_records,
                source_label='real-subsampled-graph',
                seeds_per_split=seeds_for_phase2,
                per_trial_records=all_per_trial_records,
                tune_test_ratio=args.tune_test_ratio,
                tune_test_seed=args.tune_test_seed,
                tune_portion=args.tune_portion)
            all_results.extend(results)

    phase2_elapsed = time.time() - t_phase2
    print(f"\n[Phase 2: synthetic-data] {format_duration(phase2_elapsed)}")

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

    if all_curve_records:
        curves_df = pd.DataFrame(all_curve_records)
        curves_path = os.path.join(args.output_dir, 'divergence_curves.csv')
        curves_df.to_csv(curves_path, index=False)
        print(f"  Divergence curves saved to: {curves_path}")

    if all_per_trial_records is not None and all_per_trial_records:
        per_trial_df = pd.DataFrame(all_per_trial_records)
        per_trial_path = os.path.join(args.output_dir, 'per_trial_results.csv')
        per_trial_df.to_csv(per_trial_path, index=False)
        print(f"  Per-trial results saved to: {per_trial_path}")

    print(f"\n[Total] {format_duration(time.time() - t_start)}")


if __name__ == '__main__':
    main()
