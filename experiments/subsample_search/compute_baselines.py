"""Full-graph baseline: train + test each model on the original graph,
3 splits per dataset. Output goes to `artifacts/baselines.csv`.

Hyperparams are pinned to match `run_grid.py`: epochs=200, patience=50, lr=0.01,
drop_rate=0.0, h_feats=32, num_layers=2."""
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

from utils import Dataset as GADBenchDataset, model_detector_dict  # noqa: E402


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
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(',')]

    out_path = os.path.join(_HERE, 'artifacts', 'baselines.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = os.path.join(_PROJECT_ROOT, 'datasets', 'original')

    rows = []
    if os.path.exists(out_path):
        rows = pd.read_csv(out_path).to_dict('records')
        done = {(r['dataset'], r['model'], r['split_id']) for r in rows}
        print(f'Resuming from {len(rows)} existing rows')
    else:
        done = set()

    for dataset in datasets:
        for split_id, seed in enumerate(SEEDS):
            for model_name in MODELS:
                if (dataset, model_name, split_id) in done:
                    continue
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                _set_seed(seed)
                data = GADBenchDataset(dataset, prefix=data_dir + '/')
                data.split(False, split_id)
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
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_path, index=False)
                print(f'  {dataset}/{model_name}/split{split_id}: '
                      f'AUROC={row["AUROC"]:.4f} AUPRC={row["AUPRC"]:.4f} '
                      f'RecK={row["RecK"]:.4f}  ({elapsed:.1f}s)')
                del detector

    print(f'\nSaved {len(rows)} rows to {out_path}')


if __name__ == '__main__':
    main()
