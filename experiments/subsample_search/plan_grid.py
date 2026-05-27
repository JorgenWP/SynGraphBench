"""Plan the per-cell K for one dataset using the cost model + sampler dry-runs.

Reads `experiments/bigg_capacity/results.csv` (via cost_model.py), fits the BiGG
cost model, enumerates this dataset's grid configs, runs the sampler alone on
split 0 to measure expected `(avg_m, p95_m)` per cell, and writes a plan row
per cell to `artifacts/grid/plan.csv`.

A cell's `K_final` is:
  * cfg['K']  for natural-K methods (metis, bfs_partition);
  * predicted-K from the cost model otherwise.

Cells are dropped (status != 'ok') if predicted K is below `--K_floor` or if
predicted peak VRAM at p95_m exceeds `--oom_factor * --gpu_vram_gb`.

Usage:
    python -m experiments.subsample_search.plan_grid --dataset questions
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cost_model import fit_cost_model, CostModel  # noqa: E402
from dryrun import measure_sample_shape, _NATURAL_K_METHODS  # noqa: E402
from grid import enumerate_configs  # noqa: E402
from splitsource import load_split_source  # noqa: E402


_DEFAULT_PLAN = os.path.join(_HERE, 'artifacts', 'grid', 'plan.csv')

_PLAN_COLS = [
    'dataset', 'params_tag', 'method', 'target_size', 'multiplicity_cap',
    'K_config',                # K as stated in cfg (DEFAULT_K or natural K)
    'avg_n', 'avg_m', 'max_n', 'max_m', 'p95_m', 'n_succeeded',
    'predicted_per_sg_s', 'predicted_peak_vram_gb', 'predicted_K',
    'K_final',                 # the K the grid run will actually use
    'status',                  # 'ok' | 'drop_low_K' | 'drop_vram_risk' | 'drop_dryrun_empty'
    'm_ceiling', 'vram_budget_gb',
]


def _lcc_K_cap(cfg: Dict[str, Any], n_lcc: int) -> Optional[int]:
    """Upper bound on K for M=1 iterative cells: disjoint subgraphs of size
    target_size can't exceed `n_lcc // target_size`. Returns None when no cap
    applies (M != 1, no target_size, or natural-K methods)."""
    if cfg['method'] in _NATURAL_K_METHODS:
        return None
    if cfg.get('multiplicity_cap') != 1:
        return None
    ts = cfg.get('target_size')
    if ts is None:
        return None
    return max(0, int(n_lcc // ts))


def _plan_one_cell(
    dataset: str,
    cfg: Dict[str, Any],
    split_id: int,
    seed: int,
    source,
    model: CostModel,
    K_floor: int,
    oom_factor: float,
    gpu_vram_gb: float,
    target_full_epochs: int,
    budget_s: float,
    conf_sigma: float,
    cache_path: str,
) -> Dict[str, Any]:
    shape = measure_sample_shape(
        dataset=dataset, cfg=cfg, split_id=split_id, source=source,
        seed=seed, cache_path=cache_path,
    )

    m_ceiling = model.m_ceiling.get(dataset, float('inf'))
    vram_budget_gb = oom_factor * gpu_vram_gb

    predicted_per_sg = model.predict_per_sg_total(
        dataset=dataset, m=shape['avg_m'],
        target_full_epochs=target_full_epochs, conf_sigma=conf_sigma,
    )
    predicted_K = model.predict_K(
        dataset=dataset, m=shape['avg_m'], budget_s=budget_s,
        target_full_epochs=target_full_epochs, conf_sigma=conf_sigma,
    )
    # VRAM is per-subgraph and the OOM tail is driven by p95_m, not avg_m.
    predicted_vram = model.predict_peak_vram_gb(
        dataset=dataset, m=shape['p95_m'], conf_sigma=conf_sigma,
    )

    natural_K = cfg['method'] in _NATURAL_K_METHODS
    n_lcc = source.g_nx.number_of_nodes()
    lcc_K_cap = _lcc_K_cap(cfg, n_lcc)        # None when no cap applies
    K_after_cap = predicted_K
    if K_after_cap is not None and lcc_K_cap is not None:
        K_after_cap = min(K_after_cap, lcc_K_cap)

    # If the cost model can't predict VRAM (dataset not in capacity sweep), be
    # conservative and drop rather than admit an unbounded cell.
    vram_unknown = predicted_vram is None
    if shape['n_succeeded'] == 0:
        status = 'drop_dryrun_empty'
        K_final = 0
    elif vram_unknown or predicted_vram > vram_budget_gb:
        status = 'drop_vram_risk'
        K_final = 0
    elif natural_K:
        status = 'ok'
        # metis carries its K in cfg ({5, 15}); bfs_partition leaves cfg['K']=0
        # as an "auto-determined at sample time" placeholder, so fall back to
        # the empirical K observed by the dryrun.
        K_final = int(cfg['K']) if int(cfg.get('K') or 0) > 0 else int(shape['n_succeeded'])
    elif K_after_cap is None or K_after_cap < K_floor:
        status = 'drop_low_K'
        K_final = 0
    else:
        status = 'ok'
        K_final = int(K_after_cap)

    return {
        'dataset': dataset,
        'params_tag': cfg['params_tag'],
        'method': cfg['method'],
        'target_size': cfg.get('target_size'),
        'multiplicity_cap': cfg.get('multiplicity_cap'),
        'K_config': int(cfg.get('K', 0) or 0),
        'avg_n': shape['avg_n'],
        'avg_m': shape['avg_m'],
        'max_n': shape['max_n'],
        'max_m': shape['max_m'],
        'p95_m': shape['p95_m'],
        'n_succeeded': int(shape['n_succeeded']),
        'predicted_per_sg_s': predicted_per_sg,
        'predicted_peak_vram_gb': predicted_vram if predicted_vram is not None else '',
        'predicted_K': predicted_K if predicted_K is not None else '',
        'K_final': int(K_final),
        'status': status,
        'm_ceiling': m_ceiling,
        'vram_budget_gb': vram_budget_gb,
    }


def _merge_into_plan(out_path: str, dataset: str, rows: List[Dict[str, Any]]) -> None:
    """Replace any existing rows for `dataset`; append `rows` to plan.csv."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        kept = existing[existing['dataset'] != dataset]
    else:
        kept = pd.DataFrame(columns=_PLAN_COLS)
    new = pd.DataFrame(rows, columns=_PLAN_COLS)
    combined = pd.concat([kept, new], ignore_index=True)
    combined.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split_id', type=int, default=0,
                    help='Split to dry-run on (n,m distributions are split-stable).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--output_csv', default=_DEFAULT_PLAN)
    ap.add_argument('--K_floor', type=int, default=5,
                    help='Drop iterative cells with predicted K below this.')
    ap.add_argument('--gpu_vram_gb', type=float, default=80.0,
                    help='Target GPU memory budget (default: 80 GB, A100/H100).')
    ap.add_argument('--oom_factor', type=float, default=0.85,
                    help='VRAM headroom: drop cells with predicted peak VRAM '
                         '> oom_factor * gpu_vram_gb.')
    ap.add_argument('--budget_s', type=float, default=3600.0)
    ap.add_argument('--target_full_epochs', type=int, default=300)
    ap.add_argument('--conf_sigma', type=float, default=0.5)
    ap.add_argument('--exclude_methods', default=None,
                    help='Comma-separated method names to skip (e.g. metis).')
    args = ap.parse_args()

    if args.data_dir is None:
        project_root = os.path.abspath(os.path.join(_HERE, '..', '..'))
        args.data_dir = os.path.join(project_root, 'datasets', 'original')

    print(f'[plan_grid] dataset={args.dataset} split_id={args.split_id} '
          f'K_floor={args.K_floor} gpu_vram={args.gpu_vram_gb}GB '
          f'oom_factor={args.oom_factor} '
          f'budget={args.budget_s}s epochs={args.target_full_epochs}')

    model = fit_cost_model()
    print(f'[plan_grid] cost model: '
          f'slope_train={model.slope_train:.3f}  slope_gen={model.slope_gen:.3f}  '
          f'm_ceiling[{args.dataset}]={model.m_ceiling.get(args.dataset, float("inf")):,.0f}')

    source = load_split_source(args.dataset, args.split_id, args.data_dir)
    configs = enumerate_configs(args.dataset)
    if args.exclude_methods:
        excl = {m.strip() for m in args.exclude_methods.split(',')}
        configs = [c for c in configs if c['method'] not in excl]
        print(f'[plan_grid] excluded methods: {sorted(excl)}')
    print(f'[plan_grid] enumerated {len(configs)} configs')

    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        row = _plan_one_cell(
            dataset=args.dataset, cfg=cfg, split_id=args.split_id, seed=args.seed,
            source=source, model=model,
            K_floor=args.K_floor, oom_factor=args.oom_factor,
            gpu_vram_gb=args.gpu_vram_gb,
            target_full_epochs=args.target_full_epochs, budget_s=args.budget_s,
            conf_sigma=args.conf_sigma,
            cache_path=os.path.join(_HERE, 'artifacts', 'grid', 'dryrun_stats.csv'),
        )
        rows.append(row)
        vram_str = (f'{row["predicted_peak_vram_gb"]:>5.1f}GB'
                    if isinstance(row['predicted_peak_vram_gb'], (int, float))
                    else f'{"":>7}')
        print(f'  {row["params_tag"]:<40} '
              f'avg_m={row["avg_m"]:>8.0f}  p95_m={row["p95_m"]:>8.0f}  '
              f'pred_vram={vram_str}  '
              f'pred_K={str(row["predicted_K"]):>5}  K_final={row["K_final"]:>3}  '
              f'{row["status"]}')

    _merge_into_plan(args.output_csv, args.dataset, rows)

    n_ok = sum(1 for r in rows if r['status'] == 'ok')
    print(f'\n[plan_grid] wrote {len(rows)} rows ({n_ok} ok) to {args.output_csv}')


if __name__ == '__main__':
    main()
