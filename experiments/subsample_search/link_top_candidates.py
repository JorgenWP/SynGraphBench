"""Top-N anomaly-detection candidates per dataset (notebook §10 ranking).

Replicates the per-(dataset, params_tag) ranking used in
`analysis.ipynb` §10 — `subsampled - full_graph` AUPRC averaged across
models, sorted descending, head(N) per dataset — and exposes it both
as an importable helper and a CLI dump.

The link-prediction subsample sweep consumes this to decide which
existing partition pickles to re-evaluate under LP.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd


_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_UTIL = os.path.join(_HERE, 'artifacts', 'grid', 'utility.csv')
_DEFAULT_BASE = os.path.join(_HERE, 'artifacts', 'baselines.csv')


def load_top_candidates(
    dataset: str,
    top_n: int = 10,
    util_csv: str = _DEFAULT_UTIL,
    base_csv: str = _DEFAULT_BASE,
) -> List[dict]:
    """Top-N (method, params_tag) cells for `dataset` by mean AUPRC_delta.

    AUPRC_delta = subsample_mean(AUPRC) - full_graph_baseline(AUPRC), averaged
    first within (model, params_tag) over splits, then across models. Matches
    the §10 ranking in `analysis.ipynb`.

    Returns: [{'method', 'params_tag', 'AUPRC_delta'}, ...] sorted desc.
    """
    util = pd.read_csv(util_csv)
    base = pd.read_csv(base_csv)

    util = util[util['dataset'] == dataset]
    base = base[base['dataset'] == dataset]
    if util.empty:
        raise ValueError(f'No utility rows for dataset={dataset} in {util_csv}')
    if base.empty:
        raise ValueError(f'No baseline rows for dataset={dataset} in {base_csv}')

    util_avg = (util.groupby(['method', 'params_tag', 'model'])
                    ['AUPRC'].mean().reset_index())
    base_avg = (base.groupby(['model'])['AUPRC']
                    .mean().reset_index()
                    .rename(columns={'AUPRC': 'AUPRC_base'}))

    merged = util_avg.merge(base_avg, on='model')
    merged['AUPRC_delta'] = merged['AUPRC'] - merged['AUPRC_base']

    config_means = (merged.groupby(['method', 'params_tag'])
                          ['AUPRC_delta'].mean().reset_index()
                          .sort_values('AUPRC_delta', ascending=False)
                          .head(top_n))

    return config_means.to_dict('records')


def _format_table(rows: List[dict]) -> str:
    df = pd.DataFrame(rows)
    if df.empty:
        return '(no rows)'
    df['AUPRC_delta'] = df['AUPRC_delta'].round(4)
    return df.to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--top_n', type=int, default=10)
    ap.add_argument('--util_csv', default=_DEFAULT_UTIL)
    ap.add_argument('--base_csv', default=_DEFAULT_BASE)
    args = ap.parse_args()

    rows = load_top_candidates(args.dataset, args.top_n,
                               args.util_csv, args.base_csv)
    print(_format_table(rows))


if __name__ == '__main__':
    main()
