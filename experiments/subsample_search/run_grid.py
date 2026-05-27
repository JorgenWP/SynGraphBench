"""Run the full §3 sampler grid for one or more datasets.

Per (dataset × config × split × model) emits one utility row.
Per (dataset × config × split) emits one structure row.
Resumable: skips configs whose utility rows are already present in utility.csv.

Usage:
    python experiments/subsample_search/run_grid.py --datasets tolokers
    python experiments/subsample_search/run_grid.py --datasets tolokers,questions,weibo
    python experiments/subsample_search/run_grid.py --datasets tolokers --methods forest_fire,metis
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from splitsource import load_split_source
from sampling import sample_with_method
from masking import build_combined_graph
from benchmark import run_models
from structure_stats import compute_subgraph_stats
from grid import enumerate_configs, enumerate_planned_configs


MODELS = ['GCN', 'GIN', 'GraphSAGE', 'XGBGraph', 'XGBoost']
DEFAULT_SPLITS = [0, 1, 2, 3, 4]


def split_seed(split_id: int) -> int:
    """Deterministic seed per split; splits 0/1/2 land on the historic
    seeds 3407/3417/3427 so existing CSV rows remain reproducible."""
    return 3407 + 10 * split_id


def _assert_size_params_fit(dataset: str, split_id: int, n_lcc: int, configs: list) -> None:
    """Warn (don't fail) when K * target_size exceeds the train+val LCC.

    For M=1 the sampler will exhaust its seed pool and `num_subgraphs_actual`
    will fall below K. For M>1 or M=∞ overlap is expected. Either way the run
    proceeds — only the realized K may differ from the planned K."""
    for cfg in configs:
        ts = cfg['target_size']
        if ts is None:
            continue
        K = cfg['K']
        if K <= 0:
            continue
        budget = K * ts
        if budget <= n_lcc:
            continue
        M = cfg.get('multiplicity_cap')
        if M == 1:
            print(f'  [warn] {cfg["params_tag"]} M=1 budget {K}*{ts}={budget} > '
                  f'LCC {n_lcc} — expect num_subgraphs_actual ≤ {n_lcc // ts}',
                  flush=True)
        else:
            print(f'  [warn] {cfg["params_tag"]} K*ts={budget} > LCC {n_lcc} '
                  f'— overlap expected under M={M}', flush=True)


# ---------------------------------------------------------------------------
# Append-on-row CSV writers (so an interrupt doesn't lose results)
# ---------------------------------------------------------------------------

def _append_row(path: str, row: dict) -> None:
    df = pd.DataFrame([row])
    write_header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=write_header, index=False)


def _completed_utility_set(utility_csv: str) -> set:
    """Set of (dataset, params_tag, split_id, model) tuples already done."""
    if not os.path.exists(utility_csv):
        return set()
    df = pd.read_csv(utility_csv)
    return set(zip(df['dataset'], df['params_tag'], df['split_id'], df['model']))


# ---------------------------------------------------------------------------
# One-trial inner loop — sample + assemble + run all models
# ---------------------------------------------------------------------------

def _persist_partitions(
    parts, dataset: str, cfg: dict, split_id: int, seed: int, persist_root: str,
) -> None:
    """Pickle the partition list for `(dataset, params_tag, split_id)` so BiGG
    (or any downstream consumer) can load it later. Idempotent — skip if file
    already exists."""
    out_dir = os.path.join(persist_root, dataset, cfg['params_tag'])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'split{split_id}.pkl')
    if os.path.exists(out_path):
        return
    payload = {
        'partitions': parts,
        'meta': {
            'dataset': dataset,
            'method': cfg['method'],
            'params_tag': cfg['params_tag'],
            'target_size': cfg['target_size'],
            'K_request': cfg['K'],
            'multiplicity_cap': cfg['multiplicity_cap'],
            'split_id': split_id,
            'seed': seed,
            'method_kwargs': cfg.get('method_kwargs', {}),
        },
    }
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f'  [persist] {out_path} ({len(parts)} partitions)')


def _run_one_trial(
    dataset: str, src, cfg: dict, split_id: int, seed: int, device: str,
    utility_csv: str, structure_csv: str,
    completed: set, epochs: int, patience: int,
    persist_root: Optional[str] = None,
) -> None:
    needed = [m for m in MODELS
              if (dataset, cfg['params_tag'], split_id, m) not in completed]
    if not needed:
        return

    # ---- Sample ----
    node_data = torch.cat([src.features, src.labels.unsqueeze(1).float()], dim=1)
    parts = sample_with_method(
        method=cfg['method'],
        g_nx=src.g_nx,
        node_data=node_data,
        K=cfg['K'] if cfg['K'] > 0 else 1,                     # bfs_partition: K is auto
        seed=seed,
        multiplicity_cap=cfg['multiplicity_cap'],
        **cfg['method_kwargs'],
    )

    if not parts:
        print(f'  [skip] {cfg["params_tag"]} produced 0 partitions')
        return

    # ---- Persist partitions for downstream BiGG reuse ----
    if persist_root is not None:
        _persist_partitions(parts, dataset, cfg, split_id, seed, persist_root)

    # ---- Structure stats (one row per config × split) ----
    stats = compute_subgraph_stats(parts, src)
    struct_row = {
        'dataset': dataset,
        'method': cfg['method'],
        'params_tag': cfg['params_tag'],
        'target_size': cfg['target_size'],
        'K_request': cfg['K'],
        'multiplicity_cap': cfg['multiplicity_cap'] if cfg['multiplicity_cap'] is not None else -1,
        'split_id': split_id,
        'seed': seed,
        **stats,
    }
    _append_row(structure_csv, struct_row)

    # ---- Combined graph ----
    combined = build_combined_graph(parts, src).to(device)
    orig = src.orig_dgl.to(device)

    # ---- Per-model runs ----
    rows = run_models(combined, orig, needed, seed, device,
                      epochs=epochs, patience=patience)
    for r in rows:
        r.update({
            'dataset': dataset,
            'method': cfg['method'],
            'params_tag': cfg['params_tag'],
            'target_size': cfg['target_size'],
            'K_request': cfg['K'],
            'multiplicity_cap': cfg['multiplicity_cap'] if cfg['multiplicity_cap'] is not None else -1,
            'split_id': split_id,
            'num_subgraphs_actual': stats['num_subgraphs'],
        })
        _append_row(utility_csv, r)
        completed.add((dataset, cfg['params_tag'], split_id, r['model']))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='tolokers',
                    help='Comma-separated dataset names.')
    ap.add_argument('--methods', default=None,
                    help='Comma-separated subset of method names; default = all.')
    ap.add_argument('--exclude_methods', default=None,
                    help='Comma-separated method names to skip after --methods/--params_tags filtering.')
    ap.add_argument('--params_tags', default=None,
                    help='Comma-separated explicit params_tags; if provided, overrides --methods.')
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--splits', type=int, nargs='+', default=DEFAULT_SPLITS,
                    help='Train/val/test split IDs (DGL multi-mask columns); '
                         f'default {DEFAULT_SPLITS}.')
    ap.add_argument('--persist_dir', default=None,
                    help='Where to save the produced subsamples for BiGG reuse. '
                         'Defaults to <project_root>/datasets/bigg_subsamples.')
    ap.add_argument('--no_persist', action='store_true',
                    help='Skip writing subsamples to disk.')
    ap.add_argument('--plan_csv', default=None,
                    help='Path to a plan.csv produced by plan_grid.py. When set, '
                         'each cell\'s K is taken from the plan and non-ok cells '
                         'are dropped. Default: artifacts/grid/plan.csv if it '
                         'exists. Pass an empty string to force legacy DEFAULT_K.')
    args = ap.parse_args()

    project_root = os.path.abspath(os.path.join(_HERE, '..', '..'))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'datasets', 'original')
    if args.out_dir is None:
        args.out_dir = os.path.join(_HERE, 'artifacts', 'grid')
    os.makedirs(args.out_dir, exist_ok=True)

    utility_csv = os.path.join(args.out_dir, 'utility.csv')
    structure_csv = os.path.join(args.out_dir, 'structure.csv')

    # Plan CSV default: same dir as utility/structure. Empty string disables planning.
    if args.plan_csv is None:
        args.plan_csv = os.path.join(args.out_dir, 'plan.csv')
    use_plan = bool(args.plan_csv) and os.path.exists(args.plan_csv)

    persist_root = None
    if not args.no_persist:
        persist_root = (args.persist_dir
                        or os.path.join(project_root, 'datasets', 'bigg_subsamples'))
        os.makedirs(persist_root, exist_ok=True)

    trials = [(s, split_seed(s)) for s in args.splits]

    datasets = [d.strip() for d in args.datasets.split(',')]
    method_filter = ([m.strip() for m in args.methods.split(',')]
                     if args.methods else None)
    tag_filter = ([t.strip() for t in args.params_tags.split(',')]
                  if args.params_tags else None)
    exclude_filter = ({m.strip() for m in args.exclude_methods.split(',')}
                      if args.exclude_methods else None)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'== grid run | datasets={datasets} | splits={args.splits} | device={device} ==')
    print(f'   utility CSV: {utility_csv}')
    print(f'   structure CSV: {structure_csv}')
    print(f'   plan CSV:    {args.plan_csv if use_plan else "(disabled — using DEFAULT_K)"}')
    if persist_root is not None:
        print(f'   persist dir:  {persist_root}')
    else:
        print(f'   persist dir:  (disabled)')

    completed = _completed_utility_set(utility_csv)
    print(f'   resume: {len(completed)} (dataset, tag, split, model) rows already done')

    t_start = time.time()
    n_done = 0
    for dataset in datasets:
        if use_plan:
            configs = enumerate_planned_configs(dataset, args.plan_csv)
        else:
            configs = enumerate_configs(dataset)
        if tag_filter is not None:
            configs = [c for c in configs if c['params_tag'] in tag_filter]
        elif method_filter is not None:
            configs = [c for c in configs if c['method'] in method_filter]
        if exclude_filter is not None:
            configs = [c for c in configs if c['method'] not in exclude_filter]

        print(f'\n=== {dataset}: {len(configs)} configs × {len(trials)} trials × '
              f'{len(MODELS)} models ===')

        # Cache LCC source per split (expensive: nx connected_components)
        sources = {}
        for split_id, seed in trials:
            for cfg in configs:
                key = (dataset, cfg['params_tag'], split_id)
                if all((dataset, cfg['params_tag'], split_id, m) in completed for m in MODELS):
                    continue
                if split_id not in sources:
                    print(f'\n[load] {dataset} split {split_id}')
                    sources[split_id] = load_split_source(dataset, split_id, args.data_dir)
                    _assert_size_params_fit(dataset, split_id,
                                            sources[split_id].g_nx.number_of_nodes(),
                                            configs)
                src = sources[split_id]
                print(f'\n[{dataset} | {cfg["params_tag"]} | split {split_id} | seed {seed}]')
                try:
                    _run_one_trial(dataset, src, cfg, split_id, seed, device,
                                   utility_csv, structure_csv,
                                   completed, args.epochs, args.patience,
                                   persist_root=persist_root)
                    n_done += 1
                except Exception as e:
                    print(f'  ERROR: {e}')
                    traceback.print_exc(limit=4)
                    continue

    elapsed = time.time() - t_start
    print(f'\n== grid done | {n_done} new (config, trial) blocks in {elapsed:.0f}s ==')


if __name__ == '__main__':
    main()
