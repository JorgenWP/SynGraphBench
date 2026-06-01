"""Print one best-trial save_name per split, computed from best_params.json.

Throwaway helper used to drive a targeted bigg_cache rsync from the cluster.
Run from the repo root:

    python -m experiments.bo_tuning._print_best_save_names --dataset weibo \
        > best_save_names.txt
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

from .bigg_invoke import build_save_name, load_n_subgraphs_per_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--n_splits', type=int, default=5)
    args = p.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), 'configs',
                            f'{args.dataset}.yaml')
    cfg = yaml.safe_load(open(cfg_path))
    fixed = cfg['fixed_hp']
    n_subgraphs_by_split = load_n_subgraphs_per_split(
        args.dataset, fixed['subsampling_config'],
        args.n_splits, fixed.get('min_subgraph_nodes', 0))

    for s in range(args.n_splits):
        bp_path = (f'experiments/bo_tuning/{args.dataset}/per_split/'
                   f'split{s}/best_params.json')
        bp = json.load(open(bp_path))
        print(build_save_name(fixed, bp['params'], s, n_subgraphs_by_split[s]))


if __name__ == '__main__':
    main()
