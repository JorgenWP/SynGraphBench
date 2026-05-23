"""Composite BO objective: CVaR over per-split utility gap, plus collapse penalty.

Per trial we get the per-(split, classifier, seed) AUPRC from the benchmark's
``per_trial_results.csv``. The objective averages over downstream-classifier
seeds, takes the gap synthetic/real per (model, split), averages over models
to get one number per split, then aggregates across splits with CVaR_60%
(mean of the 3 worst splits when there are 5) so a single collapsed split
dominates the score. A separate collapse penalty fires when the mean
synthetic AUPRC on a split falls below the base rate, which is a heavier
failure mode than gap noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLLAPSE_PENALTY = 0.2          # subtracted per collapsed split
CVAR_FRACTION = 0.6             # 0.6 → average of worst 60% of splits


def _cvar_lower(values, fraction=CVAR_FRACTION):
    """Average of the lowest ``fraction`` of ``values`` (>=1 sample guaranteed)."""
    arr = np.sort(np.asarray(values, dtype=float))
    k = max(1, int(round(len(arr) * fraction)))
    return float(arr[:k].mean())


def compute_objective(per_trial_df, base_rate, mode='shared',
                      synthetic_source='synthetic-graph',
                      original_source='original'):
    """Score one BO trial from a per-trial benchmark CSV.

    Returns a dict with the scalar ``objective`` plus the intermediate
    quantities used to compute it (saved as metadata for the thesis).

    ``mode='shared'`` uses CVaR over all splits; ``mode='per_split'`` expects
    rows for a single split and degenerates the CVaR to the plain mean.
    """
    df = per_trial_df.copy()
    syn = df[df['source'] == synthetic_source]
    orig = df[df['source'] == original_source]
    if syn.empty or orig.empty:
        raise ValueError(
            f"per_trial_results.csv missing rows for "
            f"source in ({synthetic_source!r}, {original_source!r}); "
            f"have sources {sorted(df['source'].unique())}")

    # Mean over the 3 downstream-classifier seeds per (split, model).
    syn_mean = (syn.groupby(['split_id', 'model'])['AUPRC']
                .mean().rename('auprc_synth').reset_index())
    orig_mean = (orig.groupby(['split_id', 'model'])['AUPRC']
                 .mean().rename('auprc_real').reset_index())
    joined = pd.merge(syn_mean, orig_mean, on=['split_id', 'model'], how='inner')
    if joined.empty:
        raise ValueError(
            "no overlapping (split_id, model) pairs between synthetic and "
            "original sources; check the benchmark ran both phases.")
    # gap[m, s] = syn / real ; guard against zero baselines.
    joined['gap'] = joined['auprc_synth'] / joined['auprc_real'].clip(lower=1e-6)

    # split_gap[s] = mean over models
    split_gap = (joined.groupby('split_id')['gap']
                 .mean().sort_index())
    mean_syn = (joined.groupby('split_id')['auprc_synth']
                .mean().sort_index())

    n_collapsed = int((mean_syn < base_rate).sum())

    if mode == 'shared':
        cvar = _cvar_lower(split_gap.values, fraction=CVAR_FRACTION)
    elif mode == 'per_split':
        # one split → CVaR = mean = the single split's gap
        cvar = float(split_gap.mean())
    else:
        raise ValueError(f"mode must be 'shared' or 'per_split', got {mode!r}")

    objective = cvar - COLLAPSE_PENALTY * n_collapsed

    # Nested dict layouts for metadata.
    gap_pms = {int(s): {} for s in split_gap.index}
    for _, row in joined.iterrows():
        gap_pms[int(row['split_id'])][str(row['model'])] = float(row['gap'])

    return {
        'objective': float(objective),
        'cvar': float(cvar),
        'n_collapsed_splits': n_collapsed,
        'base_rate': float(base_rate),
        'split_gap': {int(s): float(v) for s, v in split_gap.items()},
        'mean_syn_auprc_per_split': {int(s): float(v) for s, v in mean_syn.items()},
        'gap_per_model_per_split': gap_pms,
    }


def base_rate_from_test_mask(dataset_name, data_dir, ratio, seed, portion,
                             semi_supervised=0, split_id=0):
    """Compute the anomaly base rate on the (tune|heldout) portion of test nodes.

    Imports done lazily so this module stays cheap to import.
    """
    import sys, os, torch
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gadbench_dir = os.path.join(project_root, 'GADBench')
    scripts_dir = os.path.join(project_root, 'scripts', 'benchmark')
    for p in (gadbench_dir, scripts_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    from utils import Dataset as GADBenchDataset
    from anomaly_benchmark import _restrict_test_mask

    data = GADBenchDataset(dataset_name, prefix=data_dir + '/')
    data.split(semi_supervised, split_id)
    _restrict_test_mask(data, ratio, seed, portion)
    mask = data.graph.ndata['test_mask'].bool()
    labels = data.graph.ndata['label']
    n = int(mask.sum())
    if n == 0:
        return 0.0
    return float(((labels[mask] > 0).sum().item()) / n)
