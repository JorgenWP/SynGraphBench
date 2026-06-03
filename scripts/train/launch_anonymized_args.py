"""Emit the train_bigg_subsample.sh invocation for one anonymized BiGG cell.

Used by scripts/train/train_bigg_anonymized.slurm. For a given (dataset,
task, split_id, cluster_size) tuple it:

1. Loads experiments/bo_tuning/configs/<dataset>.yaml's fixed_hp block.
2. Reads experiments/results/BO_Tuning/<dataset>/per_split/rollup/best_params.csv
   and picks the row with split_id == split.
3. Resolves the unique cluster cache leaf at
   cache/clustering/<dataset>/<task>/[t<split>/]k<K>_c*.
4. Reuses experiments.bo_tuning.bigg_invoke._positional_args to build
   positions 1..38 (byte-identical to every BO trial's invocation), then
   appends positions 39..43 (cache_root, cluster_size, cluster_num,
   trial_id, task).
5. Emits bash-eval-friendly KEY=value lines on stdout that the SLURM
   wrapper sources.

Stdout protocol (each line independent, single-quoted values):
    SAVE_ROOT='...'
    CLUSTER_CACHE_DIR='...'
    BO_TRIAL_NUMBER='...'
    BO_OBJECTIVE='...'
    BO_LR='...'
    BO_KL_WEIGHT='...'
    BO_LW_CONT='...'
    BO_LW_LABEL='...'
    TRAIN_ARGS=(arg1 arg2 ... arg43)
"""

from __future__ import annotations

import argparse
import glob
import os
import shlex
import sys
from pathlib import Path

import pandas as pd
import yaml


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.bo_tuning.bigg_invoke import (  # noqa: E402
    _positional_args, load_n_subgraphs_per_split,
)


K_LEVELS = [20, 50, 100, 500, 1000]


def _load_fixed_hp(dataset: str) -> dict:
    cfg = _PROJECT_ROOT / 'experiments' / 'bo_tuning' / 'configs' / f'{dataset}.yaml'
    if not cfg.exists():
        raise FileNotFoundError(f'BO config not found: {cfg}')
    with open(cfg) as f:
        return yaml.safe_load(f)['fixed_hp']


def _load_best_params_row(dataset: str, split_id: int) -> dict:
    csv = (_PROJECT_ROOT / 'experiments' / 'results' / 'BO_Tuning' / dataset
           / 'per_split' / 'rollup' / 'best_params.csv')
    if not csv.exists():
        raise FileNotFoundError(f'best_params.csv not found: {csv}')
    df = pd.read_csv(csv)
    sub = df[df['split_id'] == split_id]
    if sub.empty:
        raise ValueError(f'No row for split_id={split_id} in {csv}.')
    if len(sub) > 1:
        raise ValueError(f'Duplicate rows for split_id={split_id} in {csv}.')
    return sub.iloc[0].to_dict()


def _resolve_cache_leaf(dataset: str, task: str, split_id: int,
                        cluster_size: int) -> tuple[str, int]:
    """Return (absolute_cache_dir, cluster_num) for the unique k{K}_c* leaf."""
    root = _PROJECT_ROOT / 'cache' / 'clustering' / dataset / task
    if task == 'hidden_links':
        parent = root  # trial-invariant
    else:
        parent = root / f't{split_id}'
    pattern = str(parent / f'k{cluster_size}_c*')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f'No clustering cache leaf matched {pattern}. '
            f'Run scripts/cluster/precompute_clusters.py for '
            f'(dataset={dataset}, task={task}, '
            f'{"" if task == "hidden_links" else f"trial={split_id}, "}'
            f'k={cluster_size}).')
    if len(matches) > 1:
        raise ValueError(
            f'Ambiguous clustering cache leaves at {pattern}: {matches}')
    leaf = matches[0]
    if not os.path.isfile(os.path.join(leaf, 'DONE')):
        raise FileNotFoundError(
            f'Clustering cache present but DONE missing: {leaf}. '
            f'Producer may have crashed; rerun precompute_clusters.py.')
    cluster_num = int(os.path.basename(leaf).split('_c')[1])
    return leaf, cluster_num


def _emit_bash(payload: dict, train_args: list[str]) -> None:
    for key in ('SAVE_ROOT', 'CLUSTER_CACHE_DIR',
                'BO_TRIAL_NUMBER', 'BO_OBJECTIVE',
                'BO_LR', 'BO_KL_WEIGHT', 'BO_LW_CONT', 'BO_LW_LABEL'):
        print(f"{key}={shlex.quote(str(payload[key]))}")
    quoted = ' '.join(shlex.quote(a) for a in train_args)
    print(f'TRAIN_ARGS=({quoted})')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--task', required=True, choices=['hidden_labels', 'hidden_links'])
    p.add_argument('--split', type=int, required=True)
    p.add_argument('--K', type=int, required=True, choices=K_LEVELS)
    args = p.parse_args()

    fixed = _load_fixed_hp(args.dataset)
    bp = _load_best_params_row(args.dataset, args.split)
    params = {
        'lr':        float(bp['lr']),
        'kl_weight': float(bp['kl_weight']),
        'lw_cont':   float(bp['lw_cont']),
        'lw_label':  float(bp['lw_label']),
    }

    # Surface stale subsample pickles early — the post-filter K matters for
    # downstream consistency even though we don't predict save_name here.
    load_n_subgraphs_per_split(
        args.dataset, fixed['subsampling_config'],
        fixed['n_splits'], fixed.get('min_subgraph_nodes', 0))

    cache_dir, cluster_num = _resolve_cache_leaf(
        args.dataset, args.task, args.split, args.K)

    # Positions 1..38 — reuse the BO coordinator's exact mapping.
    train_args = _positional_args(fixed, params, args.split)

    # Positions 39..43 — cache + task.
    cache_root_rel = str(_PROJECT_ROOT / 'cache' / 'clustering')
    train_args += [
        cache_root_rel,             # 39 cache_root
        str(args.K),                # 40 cluster_size
        str(cluster_num),           # 41 cluster_num
        str(args.split),            # 42 trial_id (== split_id for hidden_labels)
        args.task,                  # 43 task
    ]

    save_root = str(_PROJECT_ROOT / 'datasets' / 'synthetic' / 'bigg'
                    / 'anonymized' / args.dataset / args.task
                    / f'{args.K}_anonymity')

    payload = {
        'SAVE_ROOT':         save_root,
        'CLUSTER_CACHE_DIR': cache_dir,
        'BO_TRIAL_NUMBER':   bp['trial_number'],
        'BO_OBJECTIVE':      bp['objective'],
        'BO_LR':             params['lr'],
        'BO_KL_WEIGHT':      params['kl_weight'],
        'BO_LW_CONT':        params['lw_cont'],
        'BO_LW_LABEL':       params['lw_label'],
    }
    _emit_bash(payload, train_args)


if __name__ == '__main__':
    main()
