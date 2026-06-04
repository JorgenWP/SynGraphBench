"""Final-report stage: re-evaluate the best HPs on the held-out 50% of test nodes.

Runs after a BO study is finished. Loads the best params from the study DB,
reuses the trial's bundle if it still exists (resume hit), re-runs the
benchmark with ``--tune_portion heldout`` so we report on nodes that BO
never selected against. Real-data baseline is rotated through the same
splits for the gap-ratio table.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict

import optuna
import pandas as pd
import yaml

from .bigg_invoke import (
    expected_output_dir, PROJECT_ROOT, build_save_name,
    bo_bigg_cache_root, resolve_benchmark_python,
    load_n_subgraphs_per_split,
)
from .search_space import WARM_START_DEFAULTS


BENCHMARK_SCRIPT = os.path.join(PROJECT_ROOT, 'scripts', 'benchmark',
                                'anomaly_benchmark.py')


def _load_config(dataset: str) -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'configs', f'{dataset}.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_bundle(bundle_dir: str, dataset: str,
                   save_names_by_split: Dict[int, str],
                   cache_root: str):
    os.makedirs(bundle_dir, exist_ok=True)
    for sid, save_name in save_names_by_split.items():
        target = expected_output_dir(dataset, save_name, cache_root=cache_root)
        if not os.path.isdir(target):
            raise FileNotFoundError(
                f'best-config output missing for split {sid}: {target}')
        link = os.path.join(bundle_dir, save_name)
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)
        os.symlink(target, link)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['tolokers', 'questions', 'weibo', 'reddit', 'yelp', 'amazon', 'tfinance', 'elliptic'])
    parser.add_argument('--mode', required=True, choices=['shared', 'per_split'])
    parser.add_argument('--split_id', type=int, default=None)
    parser.add_argument('--study_version', default='v1')
    parser.add_argument('--warm_start', action='store_true',
                        help='Evaluate the warm-start (trial-0) config on '
                             'held-out instead of the BO-selected best.')
    args = parser.parse_args()

    cfg = _load_config(args.dataset)
    fixed_hp = cfg['fixed_hp']
    n_subgraphs_by_split = load_n_subgraphs_per_split(
        args.dataset, fixed_hp['subsampling_config'],
        fixed_hp['n_splits'], fixed_hp.get('min_subgraph_nodes', 0))
    seeds_per_split = cfg.get('seeds_per_split', 3)
    tune_test_ratio = cfg.get('tune_test_ratio', 0.5)
    tune_test_seed = cfg.get('tune_test_seed', 20260523)
    benchmark_env = cfg['benchmark']
    models = cfg.get('models', ['GCN', 'GIN', 'GraphSAGE', 'XGBGraph', 'XGBoost'])

    base_root = os.path.join(PROJECT_ROOT, 'experiments', 'bo_tuning',
                             args.dataset, args.mode)
    if args.mode == 'per_split':
        if args.split_id is None:
            parser.error('--split_id required in per_split mode')
        base_root = os.path.join(base_root, f'split{args.split_id}')
    if args.warm_start:
        # Trial 0's enqueued defaults; never the BO-selected best, so it was
        # only ever benchmarked on the tune portion. Re-evaluate it on held-out
        # to get a true held-out improvement baseline. No study DB needed.
        params = WARM_START_DEFAULTS[args.dataset]
        print(f'[final-report] warm-start config: {params}')
    else:
        study_db = os.path.join(base_root, 'study.db')
        study_name = f'bo_bigg_{args.dataset}_{args.mode}_{args.study_version}'
        if args.mode == 'per_split':
            study_name += f'_split{args.split_id}'
        storage = optuna.storages.RDBStorage(
            f'sqlite:///{study_db}',
            engine_kwargs={'connect_args': {'timeout': 30}})
        study = optuna.load_study(study_name=study_name, storage=storage)
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise RuntimeError(f'no completed trials in {study_name}')
        best = max(completed, key=lambda t: t.value)
        print(f'[final-report] best trial {best.number}: '
              f'objective={best.value:.4f} params={best.params}')
        params = best.params

    split_ids = (list(range(fixed_hp['n_splits'])) if args.mode == 'shared'
                 else [args.split_id])
    save_names = {sid: build_save_name(fixed_hp, params, sid,
                                       n_subgraphs_by_split[sid])
                  for sid in split_ids}

    # BO trials wrote to the BO-scoped cache (see bo_bigg_cache_root); the
    # canonical datasets/synthetic/bigg/<dataset>/hidden_labels/ path is not
    # populated unless someone re-trained outside the BO pipeline. Look up
    # bundles in the cache and fail fast if any are missing.
    cache_root = bo_bigg_cache_root(args.dataset)
    for sid, sn in save_names.items():
        target = expected_output_dir(args.dataset, sn, cache_root=cache_root)
        if not os.path.isdir(target):
            raise FileNotFoundError(
                f'best-config split {sid} not in BO cache: {target}')

    final_dir = os.path.join(
        base_root, 'final_report_warmstart' if args.warm_start else 'final_report')
    bundle_dir = os.path.join(final_dir, 'bundle')
    os.makedirs(final_dir, exist_ok=True)
    _ensure_bundle(bundle_dir, args.dataset, save_names, cache_root)

    out_dir = os.path.join(final_dir, 'benchmark_heldout')
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(final_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        resolve_benchmark_python(benchmark_env), '-u', BENCHMARK_SCRIPT,
        '--datasets', args.dataset,
        '--models', ','.join(models),
        '--generator', 'bigg',
        '--synthetic_type', 'graph',
        '--task', 'hidden_labels',
        '--graph_path', bundle_dir,
        '--seeds_per_split', str(seeds_per_split),
        '--epochs', str(benchmark_env['epochs']),
        '--patience', str(benchmark_env['patience']),
        '--lr', str(benchmark_env['lr']),
        '--drop_rate', str(benchmark_env['drop_rate']),
        '--h_feats', str(benchmark_env['h_feats']),
        '--num_layers', str(benchmark_env['num_layers']),
        '--dump_per_trial',
        '--tune_test_ratio', str(tune_test_ratio),
        '--tune_test_seed', str(tune_test_seed),
        '--tune_portion', 'heldout',
        '--output_dir', out_dir,
    ]
    print(f'[final-report] benchmark cmd: {" ".join(cmd)}')
    with open(os.path.join(log_dir, 'benchmark.out'), 'w') as outf, \
         open(os.path.join(log_dir, 'benchmark.err'), 'w') as errf:
        rc = subprocess.run(cmd, stdout=outf, stderr=errf,
                            cwd=PROJECT_ROOT, check=False).returncode
    if rc != 0:
        raise RuntimeError(f'held-out benchmark failed (code {rc}); see {log_dir}')

    src = os.path.join(out_dir, 'per_trial_results.csv')
    dst = os.path.join(final_dir, 'held_out_warmstart_report.csv'
                       if args.warm_start else 'held_out_final_report.csv')
    shutil.copy(src, dst)
    print(f'[final-report] wrote {dst}')


if __name__ == '__main__':
    main()
