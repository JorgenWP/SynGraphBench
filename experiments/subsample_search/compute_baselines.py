"""Full-graph baseline: train + test each model on the original graph,
3 splits per dataset. Output goes to `artifacts/baselines.csv`.

Hyperparams are pinned to match `run_grid.py`: epochs=200, patience=50, lr=0.01,
drop_rate=0.0, h_feats=32, num_layers=2.

Privacy baseline (``--anonymity_k``): replace all node features with their
k-anonymity cluster centroids (from cache/clustering), then restore the *real*
features on the test-mask nodes. Measures the utility cost of the anonymization
step alone (no generative model). Output goes to `artifacts/baselines_anonymized.csv`
with a `k_anonymity` column; the legacy path/behavior is untouched when the flag
is absent."""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
_GADBENCH = os.path.join(_PROJECT_ROOT, 'GADBench')
if _GADBENCH not in sys.path:
    sys.path.insert(0, _GADBENCH)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils import Dataset as GADBenchDataset, model_detector_dict  # noqa: E402

from anonymize_utils import resolve_anonymized_features  # noqa: E402


MODELS = ['GCN', 'GIN', 'GraphSAGE', 'XGBGraph', 'XGBoost']
DATASETS = ['tolokers', 'questions', 'weibo', 'reddit']
SEEDS = [3407, 3417, 3427, 3437, 3447]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(DATASETS),
                    help='Comma-separated dataset names. Default: all.')
    ap.add_argument('--anonymity_k', type=int, nargs='+', default=None,
                    help='k-anonymity levels for the privacy baseline. When set, '
                         'node features are replaced by their cluster centroids '
                         '(test nodes keep real features) and results go to '
                         'baselines_anonymized.csv. Absent = legacy original baseline.')
    ap.add_argument('--cache_root', default='cache/clustering',
                    help='Clustering cache root (relative paths anchor to project root).')
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(',')]
    anonymize = args.anonymity_k is not None
    k_levels = args.anonymity_k if anonymize else [None]

    out_name = 'baselines_anonymized.csv' if anonymize else 'baselines.csv'
    out_path = os.path.join(_HERE, 'artifacts', out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = os.path.join(_PROJECT_ROOT, 'datasets', 'original')

    def _key(r):
        base = (r['dataset'], r['model'], r['split_id'])
        return base + (r['k_anonymity'],) if anonymize else base

    rows = []
    if os.path.exists(out_path):
        rows = pd.read_csv(out_path).to_dict('records')
        done = {_key(r) for r in rows}
        print(f'Resuming from {len(rows)} existing rows')
    else:
        done = set()

    for dataset in datasets:
        for k in k_levels:
            for split_id, seed in enumerate(SEEDS):
                for model_name in MODELS:
                    key = ((dataset, model_name, split_id, k) if anonymize
                           else (dataset, model_name, split_id))
                    if key in done:
                        continue
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    _set_seed(seed)
                    data = GADBenchDataset(dataset, prefix=data_dir + '/')
                    data.split(False, split_id)
                    if anonymize:
                        feats = data.graph.ndata['feature']
                        anon = resolve_anonymized_features(
                            args.cache_root, dataset, 'hidden_labels',
                            split_id, k, feats)
                        test_mask = data.graph.ndata['test_mask'].bool()
                        anon[test_mask] = feats[test_mask]   # real features on test nodes
                        data.graph.ndata['feature'] = anon
                    tc = {'device': device, 'epochs': 200, 'patience': 50,
                          'metric': 'AUPRC', 'inductive': False, 'seed': seed}
                    mc = {'model': model_name, 'lr': 0.01, 'drop_rate': 0.0,
                          'h_feats': 32, 'num_layers': 2}
                    if dataset == 'tsocial':
                        mc['h_feats'] = 16

                    detector = model_detector_dict[model_name](tc, mc, data)
                    st = time.time()
                    score = detector.train()
                    elapsed = time.time() - st
                    row = {
                        'dataset': dataset,
                        'model': model_name,
                        'split_id': split_id,
                        'seed': seed,
                        'AUROC': float(score['AUROC']),
                        'AUPRC': float(score['AUPRC']),
                        'RecK': float(score['RecK']),
                        'time_sec': elapsed,
                    }
                    if anonymize:
                        row['k_anonymity'] = k
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(out_path, index=False)
                    ktag = f'/k{k}' if anonymize else ''
                    print(f'  {dataset}/{model_name}/split{split_id}{ktag}: '
                          f'AUROC={row["AUROC"]:.4f} AUPRC={row["AUPRC"]:.4f} '
                          f'RecK={row["RecK"]:.4f}  ({elapsed:.1f}s)')
                    done.add(key)
                    del detector

    print(f'\nSaved {len(rows)} rows to {out_path}')


if __name__ == '__main__':
    main()
