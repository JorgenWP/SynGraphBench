"""Rebuild a flat summary CSV and best_params.json from an Optuna study DB.

The summary is regenerable at any point — useful when a study runs across
many SLURM jobs and you want a quick scan of progress without opening the
SQLite directly.
"""

from __future__ import annotations

import argparse
import json
import os

import optuna
import pandas as pd


SUMMARY_COLUMNS = [
    'trial_number', 'status', 'objective', 'cvar', 'n_collapsed_splits',
    'lr', 'kl_weight', 'lw_cont', 'lw_label',
    'wall_sec', 'sample_source', 'base_rate',
]


def _flatten(trial) -> dict:
    row = {
        'trial_number': trial.number,
        'status': trial.state.name,
        'objective': trial.value,
    }
    # Pull all standard fields from user_attrs (set in coordinator).
    for k in ('cvar', 'n_collapsed_splits', 'wall_sec', 'sample_source', 'base_rate'):
        row[k] = trial.user_attrs.get(k)
    for k in ('lr', 'kl_weight', 'lw_cont', 'lw_label'):
        row[k] = trial.params.get(k)

    sg = trial.user_attrs.get('split_gap') or {}
    msap = trial.user_attrs.get('mean_syn_auprc_per_split') or {}
    # Expand split-keyed dicts as their own columns.
    for s_str, v in (sg.items() if isinstance(sg, dict) else []):
        row[f'split_gap_{s_str}'] = v
    for s_str, v in (msap.items() if isinstance(msap, dict) else []):
        row[f'mean_syn_auprc_split_{s_str}'] = v
    return row


def aggregate(study_db, study_name, out_dir):
    storage = optuna.storages.RDBStorage(
        f'sqlite:///{study_db}',
        engine_kwargs={'connect_args': {'timeout': 30}})
    study = optuna.load_study(study_name=study_name, storage=storage)
    rows = [_flatten(t) for t in study.trials]
    if not rows:
        print(f'no trials in {study_name}')
        return
    df = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, 'summary.csv')
    df.to_csv(summary_path, index=False)
    print(f'wrote {summary_path}  (n_trials={len(df)})')

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = max(completed, key=lambda t: t.value)
        best_path = os.path.join(out_dir, 'best_params.json')
        with open(best_path, 'w') as f:
            json.dump({
                'trial_number': best.number,
                'value': best.value,
                'params': best.params,
                'user_attrs': {k: v for k, v in best.user_attrs.items()
                               if k in ('cvar', 'n_collapsed_splits',
                                         'split_gap',
                                         'mean_syn_auprc_per_split')},
            }, f, indent=2)
        print(f'wrote {best_path}  (trial {best.number}, obj={best.value:.4f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_db', required=True)
    parser.add_argument('--study_name', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    aggregate(args.study_db, args.study_name, args.out_dir)
