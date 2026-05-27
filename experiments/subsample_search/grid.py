"""Grid configuration enumeration.

A *config* fully specifies one sampler run. Configs are named by a stable
`params_tag` so that downstream CSVs / notebooks can group rows easily.

`target_size` is handled in two modes:

* **Frozen** (tolokers, weibo) — single value per dataset, kept exactly as it
  was when BO ran on those datasets so existing `params_tag` strings (and the
  pickled partitions / util.csv / BO trial state keyed by them) remain valid.
* **Swept** (questions, amazon, reddit, yelp, tfinance, tsocial) — list of
  values per dataset. The `(target_size, K)` tradeoff under the 1 h / 80 GB
  compute budget is explored at each size; `plan_grid.py` drops sizes that
  exceed the VRAM gate.

Frozen-mode `params_tag` is byte-identical to the pre-sweep scheme; sweep-mode
tags additionally embed `ts{size}` for size-dependent methods (forest_fire,
rwr, uniform, bfs_partition, bfs_snowball-with-size). METIS and depth-only
bfs_snowball are size-independent and are emitted once per dataset."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# Datasets whose `target_size` is locked because downstream BO state depends
# on their existing `params_tag` scheme. Do NOT add to this list lightly:
# changing a frozen dataset's value would invalidate pickled partitions,
# dryrun cache rows, utility.csv rows, and BO trial checkpoints for that
# dataset.
PER_DATASET_TARGET_SIZE_FROZEN: Dict[str, int] = {
    'tolokers': 500,
}

# Datasets where `target_size` is searched. Cells whose predicted peak VRAM
# exceeds the budget are dropped by `plan_grid.py`'s VRAM gate; no need to
# pre-filter the candidate set per dataset.
PER_DATASET_TARGET_SIZES_SWEEP: Dict[str, List[int]] = {
    'questions': [500, 1500, 2500],
    'reddit':    [500, 1500, 2500],
    'amazon':    [500, 1500, 2500],
    'weibo':     [500, 1500, 2500],
    'yelp':      [500, 1500, 2500],
    'tfinance':  [500, 1500, 2500],
    'tsocial':   [500, 1500, 2500],
}

DEFAULT_K = 15  # nominal K for iterative methods; overridden per cell by the
                # cost model in `plan_grid.py` to fit the 1-hour budget.


def _target_sizes_for(dataset: str) -> List[int]:
    """Return the list of `target_size` values to enumerate for `dataset`.

    Frozen datasets return a single-element list (so the loop in
    `enumerate_configs` is uniform between modes); swept datasets return their
    sweep set."""
    if dataset in PER_DATASET_TARGET_SIZE_FROZEN:
        return [PER_DATASET_TARGET_SIZE_FROZEN[dataset]]
    return PER_DATASET_TARGET_SIZES_SWEEP[dataset]


def enumerate_configs(dataset: str) -> List[Dict[str, Any]]:
    """All grid configs for `dataset`.

    In frozen mode (~17 configs) the output matches the pre-sweep scheme
    byte-for-byte. In sweep mode the size-dependent methods are emitted once
    per `target_size`, giving roughly `n_sizes × 13 + 6` configs (METIS and
    depth-only bfs_snowball are size-independent and emitted once)."""
    sizes = _target_sizes_for(dataset)
    sweep_mode = len(sizes) > 1
    K = DEFAULT_K
    out: List[Dict[str, Any]] = []

    for size in sizes:
        # In frozen mode (single size), preserve the original params_tag exactly
        # so existing artifacts remain addressable.
        ts_tag = f'ts{size}_' if sweep_mode else ''

        # ---- Forest fire: 3 burn probs × 3 multiplicity caps ----
        for burn in [0.3, 0.5, 0.7]:
            for M in [1, 2, None]:                # None = ∞
                tag_M = 'inf' if M is None else str(M)
                out.append({
                    'method': 'forest_fire',
                    'params_tag': f'ff_{ts_tag}b{burn}_M{tag_M}',
                    'K': K,
                    'target_size': size,
                    'multiplicity_cap': M,
                    'method_kwargs': {'burn_prob': burn, 'target_size': size},
                })

        # ---- RWR: 2 restart probs × 2 multiplicity caps ----
        for restart in [0.10, 0.15]:
            for M in [1, 2]:
                out.append({
                    'method': 'rwr',
                    'params_tag': f'rwr_{ts_tag}r{restart}_M{M}',
                    'K': K,
                    'target_size': size,
                    'multiplicity_cap': M,
                    'method_kwargs': {'restart_prob': restart, 'target_size': size},
                })

        # ---- Uniform random + induced + LCC (M = ∞) ----
        out.append({
            'method': 'uniform',
            'params_tag': f'uniform_{ts_tag}K{K}',
            'K': K,
            'target_size': size,
            'multiplicity_cap': None,
            'method_kwargs': {'target_size': size},
        })

        # ---- BFS partition (the previous default; disjoint, M = 1) ----
        out.append({
            'method': 'bfs_partition',
            'params_tag': f'bfs_part_{ts_tag}mn32',
            'K': 0,                                   # auto-determined
            'target_size': size,
            'multiplicity_cap': 1,
            'method_kwargs': {'target_size': size, 'max_neighbors': 32},
        })

        # ---- BFS snowball with explicit size — 6 cells (mn × M) ----
        # params_tag already encodes ts{size}; the size loop just expands it.
        for mn in [8, 16, 32]:
            for M in [1, None]:
                tag_M = 'inf' if M is None else str(M)
                out.append({
                    'method': 'bfs_snowball',
                    'params_tag': f'bfs_snow_ts{size}_mn{mn}_dinf_M{tag_M}',
                    'K': K,
                    'target_size': size,
                    'multiplicity_cap': M,
                    'method_kwargs': {'target_size': size, 'max_neighbors': mn},
                })

        # ---- BFS snowball combined size + depth=4 safety cap (mn × M=1) ----
        for mn in [8, 16]:
            out.append({
                'method': 'bfs_snowball',
                'params_tag': f'bfs_snow_ts{size}_mn{mn}_d4_M1',
                'K': K,
                'target_size': size,
                'multiplicity_cap': 1,
                'method_kwargs': {'target_size': size, 'max_neighbors': mn, 'max_depth': 4},
            })

    # ---- METIS: 2 partition counts (M = 1 by construction) ----
    # Size-independent: METIS auto-balances.
    for parts in [5, 15]:
        out.append({
            'method': 'metis',
            'params_tag': f'metis_K{parts}',
            'K': parts,
            'target_size': None,
            'multiplicity_cap': 1,
            'method_kwargs': {},
        })

    # ---- BFS snowball depth-only: 4 cells (mn × max_depth, M = 1) ----
    # Size-independent: target_size=None, partitions bounded by depth/fanout.
    for d in [2, 3]:
        for mn in [8, 16]:
            out.append({
                'method': 'bfs_snowball',
                'params_tag': f'bfs_snow_tsinf_mn{mn}_d{d}_M1',
                'K': K,
                'target_size': None,
                'multiplicity_cap': 1,
                'method_kwargs': {'target_size': None, 'max_neighbors': mn, 'max_depth': d},
            })

    return out


def config_id(dataset: str, params_tag: str, split_id: int) -> str:
    """Stable identifier used for checkpoint dedup."""
    return f'{dataset}__{params_tag}__split{split_id}'


def enumerate_planned_configs(dataset: str, plan_csv: str) -> List[Dict[str, Any]]:
    """Like `enumerate_configs(dataset)` but with per-cell K from `plan_csv`.

    Behavior:
      * Loads `plan_csv`, filters to rows for `dataset` with `status == 'ok'`.
      * Joins onto `enumerate_configs(dataset)` by `params_tag`.
      * Overrides `cfg['K']` with the plan's `K_final` for matched cells.
      * Drops cells absent from the plan or with status != 'ok'.
      * If the plan file is missing or has zero rows for `dataset`, falls back
        to the unmodified `enumerate_configs(dataset)` with `DEFAULT_K` and
        prints a loud warning (used for the tolokers/weibo legacy path)."""
    import os
    import pandas as pd

    cfgs = enumerate_configs(dataset)

    if not os.path.exists(plan_csv):
        print(f'  [grid] WARNING: plan_csv {plan_csv!r} not found — '
              f'falling back to DEFAULT_K={DEFAULT_K} for {dataset}', flush=True)
        return cfgs

    plan = pd.read_csv(plan_csv)
    plan = plan[(plan['dataset'] == dataset) & (plan['status'] == 'ok')]
    if len(plan) == 0:
        print(f'  [grid] WARNING: no ok plan rows for {dataset!r} in {plan_csv!r} — '
              f'falling back to DEFAULT_K={DEFAULT_K}', flush=True)
        return cfgs

    K_by_tag: Dict[str, int] = {
        str(r['params_tag']): int(r['K_final']) for _, r in plan.iterrows()
    }
    planned: List[Dict[str, Any]] = []
    for cfg in cfgs:
        tag = cfg['params_tag']
        if tag not in K_by_tag:
            continue
        new_cfg = dict(cfg)
        new_cfg['K'] = K_by_tag[tag]
        planned.append(new_cfg)
    print(f'  [grid] {dataset}: {len(planned)}/{len(cfgs)} configs survived planning '
          f'(plan_csv={plan_csv})', flush=True)
    return planned
