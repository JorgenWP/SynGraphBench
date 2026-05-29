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
    expected_output_dir, PROJECT_ROOT, build_save_name, train_one_split,
    resolve_benchmark_python,
)


BENCHMARK_SCRIPT = os.path.join(PROJECT_ROOT, 'scripts', 'benchmark',
                                'anomaly_benchmark.py')


def _load_config(dataset: str) -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'configs', f'{dataset}.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_bundle(bundle_dir: str, dataset: str, save_names_by_split: Dict[int, str]):
    os.makedirs(bundle_dir, exist_ok=True)
    for sid, save_name in save_names_by_split.items():
        target = expected_output_dir(dataset, save_name)
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
    parser.add_argument('--retrain_missing', action='store_true',
                        help='If a best-config save_dir is missing, retrain it. '
                             'Default behaviour fails fast on missing dirs so '
                             'we never silently re-run trials on cached splits.')
    args = parser.parse_args()

    cfg = _load_config(args.dataset)
    fixed_hp = cfg['fixed_hp']
    n_subgraphs = fixed_hp['n_subgraphs']
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

    split_ids = (list(range(fixed_hp['n_splits'])) if args.mode == 'shared'
                 else [args.split_id])
    save_names = {sid: build_save_name(fixed_hp, best.params, sid, n_subgraphs)
                  for sid in split_ids}

    # Retrain any missing splits (the canonical save_dir might have been
    # purged or never finished). Without --retrain_missing this fails fast.
    for sid, sn in save_names.items():
        target = expected_output_dir(args.dataset, sn)
        if not os.path.isdir(target):
            if not args.retrain_missing:
                raise FileNotFoundError(
                    f'best-config split {sid} not on disk: {target}\n'
                    f'pass --retrain_missing to rebuild it.')
            print(f'[final-report] retraining missing split {sid}')
            log_dir = os.path.join(base_root, 'final_report', 'logs',
                                   f'retrain_split{sid}')
            res = train_one_split(fixed_hp, best.params, sid, n_subgraphs,
                                  log_dir=log_dir,
                                  stdout_fname='train.out',
                                  stderr_fname='train.err')
            if res['status'] != 'completed':
                raise RuntimeError(f'retrain failed for split {sid}: {res["failure"]}')

    final_dir = os.path.join(base_root, 'final_report')
    bundle_dir = os.path.join(final_dir, 'bundle')
    os.makedirs(final_dir, exist_ok=True)
    _ensure_bundle(bundle_dir, args.dataset, save_names)

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
    dst = os.path.join(final_dir, 'held_out_final_report.csv')
    shutil.copy(src, dst)
    print(f'[final-report] wrote {dst}')


if __name__ == '__main__':
    main()
