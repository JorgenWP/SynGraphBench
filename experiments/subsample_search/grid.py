"""Grid configuration enumeration.

A *config* fully specifies one sampler run. Configs are named by a stable
`params_tag` so that downstream CSVs / notebooks can group rows easily.

Per-dataset overrides apply only to the per-subgraph `target_size` (and the
analogous `target_size` for `bfs_partition`). Everything else is dataset-
independent so cross-dataset comparisons are clean."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# Per-dataset target_size for forest_fire / rwr / bfs_snowball / uniform / bfs_partition.
# Chosen so that K * target_size sits at ~75–90% of the train+val LCC: enough that
# K disjoint subgraphs (M=1) can fit inside the source without exhausting the seed
# pool. Values that push K * target_size above the LCC over-saturate the M=1 case
# and inflate multiplicity counters under M=∞; `run_grid.py` asserts the bound
# at runtime against every loaded split's actual LCC.
PER_DATASET_TARGET_SIZE: Dict[str, int] = {
    'tolokers':  500,    # K*ts=7500, LCC≈8675   →  87%
    'questions': 1500,   # K*ts=22500, LCC≈30300 →  74%
    'weibo':     300,    # K*ts=4500, LCC≈5035   →  89%
}

DEFAULT_K = 15  # number of subgraphs for iterative methods


def enumerate_configs(dataset: str) -> List[Dict[str, Any]]:
    """All grid configs for `dataset`. ~17 configs per dataset."""
    size = PER_DATASET_TARGET_SIZE[dataset]
    K = DEFAULT_K
    out: List[Dict[str, Any]] = []

    # ---- Forest fire: 3 burn probs × 3 multiplicity caps ----
    for burn in [0.3, 0.5, 0.7]:
        for M in [1, 2, None]:                # None = ∞
            tag_M = 'inf' if M is None else str(M)
            out.append({
                'method': 'forest_fire',
                'params_tag': f'ff_b{burn}_M{tag_M}',
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
                'params_tag': f'rwr_r{restart}_M{M}',
                'K': K,
                'target_size': size,
                'multiplicity_cap': M,
                'method_kwargs': {'restart_prob': restart, 'target_size': size},
            })

    # ---- METIS: 2 partition counts (M = 1 by construction) ----
    for parts in [5, 15]:
        out.append({
            'method': 'metis',
            'params_tag': f'metis_K{parts}',
            'K': parts,
            'target_size': None,        # METIS auto-balances
            'multiplicity_cap': 1,
            'method_kwargs': {},
        })

    # ---- Uniform random + induced + LCC (M = ∞) ----
    out.append({
        'method': 'uniform',
        'params_tag': f'uniform_K{K}',
        'K': K,
        'target_size': size,
        'multiplicity_cap': None,
        'method_kwargs': {'target_size': size},
    })

    # ---- BFS partition (the previous default; disjoint, M = 1) ----
    out.append({
        'method': 'bfs_partition',
        'params_tag': f'bfs_part_mn32',
        'K': 0,                                   # auto-determined
        'target_size': size,
        'multiplicity_cap': 1,
        'method_kwargs': {'target_size': size, 'max_neighbors': 32},
    })

    # ---- BFS snowball (overlapping, parametric over size + depth + fanout + M) ----
    # 6 size-bound (mn ∈ {8,16,32} × M ∈ {1, ∞})
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

    # 4 depth-only (target_size=None, max_depth ∈ {2,3} × mn ∈ {8,16}, M = 1)
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

    # 2 combined (target_size + max_depth=4 safety cap, mn ∈ {8,16}, M = 1)
    for mn in [8, 16]:
        out.append({
            'method': 'bfs_snowball',
            'params_tag': f'bfs_snow_ts{size}_mn{mn}_d4_M1',
            'K': K,
            'target_size': size,
            'multiplicity_cap': 1,
            'method_kwargs': {'target_size': size, 'max_neighbors': mn, 'max_depth': 4},
        })

    return out


def config_id(dataset: str, params_tag: str, split_id: int) -> str:
    """Stable identifier used for checkpoint dedup."""
    return f'{dataset}__{params_tag}__split{split_id}'
