"""BO coordinator. Opens (or resumes) an Optuna study and runs trials.

Topology A from the plan: one long-running process, walltime budget set via
``--max_wall_seconds`` so a SLURM walltime kill happens cleanly. Trials run
sequentially; restart the coordinator (same study DB) to pick up where it
left off.

Modes:
  shared    — one study per (dataset). Each trial trains all 5 splits.
  per_split — one study per (dataset, split). Each trial trains 1 split.

Use ``--max_trials`` to cap trials in one process (e.g. ``--max_trials 1``
for a topology-C array worker that just pulls one trial and exits).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import optuna
import yaml

from .bigg_invoke import PROJECT_ROOT
from .search_space import (
    suggest_params, WARM_START_DEFAULTS, DEFAULT_BOUNDS,
)
from .cleanup_stale import cleanup_stale_running
from .run_trial import run_trial


SAMPLER_SEED = 20260523
N_STARTUP_TRIALS = 8
STALE_AFTER_SEC_DEFAULT = 6 * 3600


def _load_config(dataset: str) -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'configs', f'{dataset}.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def _experiment_root(dataset: str, mode: str, split_id=None) -> str:
    base = os.path.join(PROJECT_ROOT, 'experiments', 'bo_tuning',
                        dataset, mode)
    if mode == 'per_split':
        return os.path.join(base, f'split{split_id}')
    return base


def _study_name(dataset: str, mode: str, study_version: str, split_id=None) -> str:
    base = f'bo_bigg_{dataset}_{mode}_{study_version}'
    if mode == 'per_split':
        return f'{base}_split{split_id}'
    return base


def _open_study(dataset, mode, study_version, bounds, defaults, split_id=None,
                stale_after_sec=STALE_AFTER_SEC_DEFAULT):
    root = _experiment_root(dataset, mode, split_id=split_id)
    os.makedirs(root, exist_ok=True)
    db_path = os.path.join(root, 'study.db')
    storage = optuna.storages.RDBStorage(
        f'sqlite:///{db_path}',
        engine_kwargs={'connect_args': {'timeout': 30}})
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=N_STARTUP_TRIALS, multivariate=True, group=True,
        seed=SAMPLER_SEED + (split_id or 0))
    study = optuna.create_study(
        study_name=_study_name(dataset, mode, study_version, split_id),
        storage=storage, direction='maximize', sampler=sampler,
        load_if_exists=True)

    cleanup_stale_running(study, stale_after_sec=stale_after_sec)

    # Warm-start once: if no trial has used the warm-start defaults yet,
    # enqueue them so trial 0 reproduces a known-good config.
    completed_params = [t.params for t in study.trials
                        if t.state in (optuna.trial.TrialState.COMPLETE,
                                       optuna.trial.TrialState.RUNNING,
                                       optuna.trial.TrialState.WAITING)]
    if not completed_params and defaults is not None:
        # Clamp defaults to current bounds so suggest_float doesn't reject.
        clamped = _clamp_to_bounds(defaults, bounds)
        study.enqueue_trial(clamped, skip_if_exists=True)
    return study, root, db_path


def _clamp_to_bounds(params, bounds):
    out = {}
    for k, v in params.items():
        lo, hi = bounds.get(k, DEFAULT_BOUNDS[k])
        out[k] = float(min(max(v, lo), hi))
    return out


def _append_jsonl(root, payload):
    path = os.path.join(root, 'trial_log.jsonl')
    with open(path, 'a') as f:
        f.write(json.dumps(payload, default=str) + '\n')


def _run_one_trial(study, *, dataset, fixed_hp, n_subgraphs, models,
                   seeds_per_split, tune_test_ratio, tune_test_seed,
                   tune_portion, mode, benchmark_env, root, study_split_id=None,
                   bounds=None):
    """Ask, run, tell. Returns the completed trial's number."""
    trial = study.ask()
    params = suggest_params(trial, bounds=bounds)
    trial_dir = os.path.join(root, 'trials', f'trial_{trial.number:04d}')
    os.makedirs(trial_dir, exist_ok=True)

    sample_source = (
        'enqueued' if trial.number < len(study.get_trials(deepcopy=False)) and
        (trial.params == params and trial.number == 0) else
        ('random' if trial.number < N_STARTUP_TRIALS else 'TPE'))

    split_ids = (list(range(fixed_hp['n_splits']))
                 if mode == 'shared' else [study_split_id])

    metrics = run_trial(
        dataset=dataset, fixed_hp=fixed_hp, params=params,
        split_ids=split_ids, trial_dir=trial_dir,
        n_subgraphs=n_subgraphs, models=models,
        seeds_per_split=seeds_per_split,
        tune_test_ratio=tune_test_ratio,
        tune_test_seed=tune_test_seed,
        tune_portion=tune_portion,
        mode=mode, benchmark_env=benchmark_env)

    metrics['trial_number'] = trial.number
    metrics['sample_source'] = sample_source
    metrics['fixed_hp'] = fixed_hp

    # Store the full metric blob as a user attr (Optuna will JSON it).
    for k, v in metrics.items():
        try:
            trial.set_user_attr(k, v)
        except Exception:
            trial.set_user_attr(k, json.loads(json.dumps(v, default=str)))

    _append_jsonl(root, metrics)

    if metrics.get('status') == 'COMPLETED' and 'objective' in metrics:
        study.tell(trial, metrics['objective'])
    else:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
    return trial.number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['tolokers', 'questions', 'weibo'])
    parser.add_argument('--mode', required=True, choices=['shared', 'per_split'])
    parser.add_argument('--split_id', type=int, default=None,
                        help='Required in per_split mode (0..n_splits-1).')
    parser.add_argument('--n_trials', type=int, default=40,
                        help='Target total trials in the study (caps at this).')
    parser.add_argument('--max_trials', type=int, default=None,
                        help='Cap on trials done in THIS process (e.g. 1 for '
                             'array workers). None = no per-process cap.')
    parser.add_argument('--max_wall_seconds', type=int, default=None,
                        help='Stop accepting new trials past this wall-clock.')
    parser.add_argument('--stale_after_sec', type=int,
                        default=STALE_AFTER_SEC_DEFAULT)
    parser.add_argument('--study_version', default='v1',
                        help='Suffix on the study name. Bump to fork a new study.')
    args = parser.parse_args()

    if args.mode == 'per_split' and args.split_id is None:
        parser.error('--split_id is required in per_split mode')

    cfg = _load_config(args.dataset)
    fixed_hp = cfg['fixed_hp']
    bounds = cfg.get('search_space', {})
    benchmark_env = cfg['benchmark']
    n_subgraphs = fixed_hp['n_subgraphs']
    seeds_per_split = cfg.get('seeds_per_split', 3)
    tune_test_ratio = cfg.get('tune_test_ratio', 0.5)
    tune_test_seed = cfg.get('tune_test_seed', 20260523)
    tune_portion = cfg.get('tune_portion', 'tune')
    models = cfg.get('models', ['GCN', 'GIN', 'GraphSAGE', 'XGBGraph', 'XGBoost'])

    defaults = WARM_START_DEFAULTS.get(args.dataset)

    study, root, db_path = _open_study(
        args.dataset, args.mode, args.study_version, bounds, defaults,
        split_id=args.split_id, stale_after_sec=args.stale_after_sec)

    print(f'[bo-coordinator] study={study.study_name} db={db_path}')
    print(f'[bo-coordinator] mode={args.mode} '
          f'completed={len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])} '
          f'failed={len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])} '
          f'target={args.n_trials}')

    t_start = time.time()
    done_this_proc = 0
    while True:
        completed = sum(1 for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE)
        if completed >= args.n_trials:
            print(f'[bo-coordinator] target n_trials={args.n_trials} reached.')
            break
        if args.max_trials is not None and done_this_proc >= args.max_trials:
            print(f'[bo-coordinator] max_trials={args.max_trials} reached for this process.')
            break
        if (args.max_wall_seconds is not None
                and (time.time() - t_start) >= args.max_wall_seconds):
            print(f'[bo-coordinator] wall budget exhausted '
                  f'({args.max_wall_seconds}s).')
            break

        n = _run_one_trial(
            study, dataset=args.dataset, fixed_hp=fixed_hp,
            n_subgraphs=n_subgraphs, models=models,
            seeds_per_split=seeds_per_split,
            tune_test_ratio=tune_test_ratio,
            tune_test_seed=tune_test_seed,
            tune_portion=tune_portion,
            mode=args.mode, benchmark_env=benchmark_env,
            root=root, study_split_id=args.split_id, bounds=bounds)
        done_this_proc += 1
        print(f'[bo-coordinator] trial {n} done; '
              f'best_value={study.best_value if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else None}')

    print('[bo-coordinator] exiting.')


if __name__ == '__main__':
    main()
