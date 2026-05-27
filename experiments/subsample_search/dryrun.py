"""Sampler-only "dry run" helper for the planning step.

For each (dataset, method, params) cell we want to know expected per-subgraph
`(n, m)` *before* training. This module runs the sampler alone (no BiGG), caches
the resulting summary stats to `artifacts/grid/dryrun_stats.csv`, and returns
them as a dict. Idempotent on the cache key
`(dataset, params_tag, split_id, K_probe, seed)`."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from sampling import sample_with_method
from splitsource import SplitSource

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CACHE = os.path.join(_HERE, 'artifacts', 'grid', 'dryrun_stats.csv')

_CACHE_COLS = [
    'dataset', 'params_tag', 'split_id', 'K_probe', 'seed',
    'n_succeeded', 'avg_n', 'avg_m', 'max_n', 'max_m', 'p95_m',
]

_NATURAL_K_METHODS = {'metis', 'bfs_partition'}


def _load_cache(cache_path: str) -> pd.DataFrame:
    if not os.path.exists(cache_path):
        return pd.DataFrame(columns=_CACHE_COLS)
    return pd.read_csv(cache_path)


def _cache_hit(cache: pd.DataFrame, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if cache.empty:
        return None
    mask = (
        (cache['dataset'] == key['dataset'])
        & (cache['params_tag'] == key['params_tag'])
        & (cache['split_id'] == key['split_id'])
        & (cache['K_probe'] == key['K_probe'])
        & (cache['seed'] == key['seed'])
    )
    rows = cache[mask]
    if len(rows) == 0:
        return None
    return rows.iloc[-1].to_dict()


def _append_cache(cache_path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df = pd.DataFrame([{c: row[c] for c in _CACHE_COLS}])
    header = not os.path.exists(cache_path)
    df.to_csv(cache_path, mode='a', header=header, index=False)


def _probe_K_for(cfg: Dict[str, Any], default_probe: int = 5) -> int:
    """Natural-K methods (metis, bfs_partition) sample at cfg['K']; iterative
    methods sample at a small fixed probe size."""
    if cfg['method'] in _NATURAL_K_METHODS:
        return max(int(cfg.get('K', default_probe)), 1)
    return default_probe


def measure_sample_shape(
    dataset: str,
    cfg: Dict[str, Any],
    split_id: int,
    source: SplitSource,
    seed: int = 0,
    K_probe: Optional[int] = None,
    cache_path: str = _DEFAULT_CACHE,
) -> Dict[str, Any]:
    """Run the sampler in `cfg` once, return per-subgraph shape stats.

    Cache schema (one row per (dataset, params_tag, split_id, K_probe, seed)):
        n_succeeded, avg_n, avg_m, max_n, max_m, p95_m

    `K_probe=None` picks 5 for iterative methods and cfg['K'] for metis /
    bfs_partition."""
    if K_probe is None:
        K_probe = _probe_K_for(cfg)
    key = {
        'dataset': dataset,
        'params_tag': cfg['params_tag'],
        'split_id': split_id,
        'K_probe': int(K_probe),
        'seed': int(seed),
    }
    cache = _load_cache(cache_path)
    hit = _cache_hit(cache, key)
    if hit is not None:
        return hit

    node_data = torch.cat(
        [source.features, source.labels.unsqueeze(1).float()], dim=1
    )

    parts = sample_with_method(
        method=cfg['method'],
        g_nx=source.g_nx,
        node_data=node_data,
        K=int(K_probe),
        seed=int(seed),
        multiplicity_cap=cfg.get('multiplicity_cap'),
        **cfg.get('method_kwargs', {}),
    )

    ns = np.array([p[0].number_of_nodes() for p in parts], dtype=float)
    ms = np.array([p[0].number_of_edges() for p in parts], dtype=float)
    row = {
        **key,
        'n_succeeded': int(len(parts)),
        'avg_n': float(ns.mean()) if len(ns) else 0.0,
        'avg_m': float(ms.mean()) if len(ms) else 0.0,
        'max_n': float(ns.max()) if len(ns) else 0.0,
        'max_m': float(ms.max()) if len(ms) else 0.0,
        'p95_m': float(np.percentile(ms, 95)) if len(ms) else 0.0,
    }
    _append_cache(cache_path, row)
    return row
