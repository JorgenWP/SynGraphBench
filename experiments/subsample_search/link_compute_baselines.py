"""Full-graph LP baselines: train+val+test all on the original graph.

Output: `artifacts/baselines_link.csv`. Rows are tagged with `decoder` +
`neg_sampling` so one CSV holds all sweep variants. Resumable on
(dataset, model, split_id, decoder, neg_sampling).

Hyperparams match `experiments/subsample_search/link_run_grid.py` so the
cross-graph deltas are read against a comparable full-graph reference.
"""
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

from link_utils import LinkDataset  # noqa: E402
from models.link_prediction.link_predictor import (  # noqa: E402
    BaseGNNLinkPredictor, XGBGraphLinkPredictor, XGBoostLinkPredictor,
)


MODELS = ['GCN', 'GIN', 'GraphSAGE', 'XGBoost', 'XGBGraph']
DEFAULT_DATASETS = ['tolokers', 'questions', 'weibo', 'reddit',
                    'amazon', 'tfinance', 'yelp']
DEFAULT_SPLITS = [0, 1, 2, 3, 4]


def _split_seed(split_id: int) -> int:
    return 3407 + 10 * split_id


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(DEFAULT_DATASETS))
    ap.add_argument('--neg_sampling', choices=['random', 'hard'], required=True)
    ap.add_argument('--decoder', choices=['dot', 'mlp'], required=True)
    ap.add_argument('--splits', type=int, nargs='+', default=DEFAULT_SPLITS)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--out_path', default=None)
    args = ap.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(_PROJECT_ROOT, 'datasets', 'original')
    if args.out_path is None:
        args.out_path = os.path.join(_HERE, 'artifacts', 'baselines_link.csv')
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(args.out_path):
        existing = pd.read_csv(args.out_path)
        done = set(zip(existing['dataset'], existing['model'],
                       existing['split_id'], existing['decoder'],
                       existing['neg_sampling']))
        print(f'Resuming from {len(existing)} rows ({len(done)} unique combos)')
    else:
        existing = pd.DataFrame()
        done = set()

    print(f'== link baselines | decoder={args.decoder} | '
          f'neg={args.neg_sampling} | datasets={datasets} | '
          f'splits={args.splits} | device={device} ==')

    new_rows = []
    for dataset in datasets:
        for split_id in args.splits:
            seed = _split_seed(split_id)
            _set_seed(seed)
            data = LinkDataset(dataset, prefix=args.data_dir + '/')
            data.split(val_ratio=args.val_ratio,
                       test_ratio=args.test_ratio,
                       trial_id=split_id,
                       neg_sampling=args.neg_sampling)

            for model_name in MODELS:
                key = (dataset, model_name, split_id,
                       args.decoder, args.neg_sampling)
                if key in done:
                    continue

                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                _set_seed(seed)

                train_config = {
                    'device': device,
                    'epochs': args.epochs,
                    'patience': args.patience,
                    'metric': 'AUPRC',
                    'neg_sampling': args.neg_sampling,
                    'decoder': args.decoder,
                    'seed': seed,
                }
                model_config = {
                    'model': model_name,
                    'lr': 0.01,
                    'drop_rate': 0.0,
                    'h_feats': 32,
                    'num_layers': 2,
                }

                if model_name == 'XGBGraph':
                    det = XGBGraphLinkPredictor(train_config, model_config, data)
                elif model_name == 'XGBoost':
                    det = XGBoostLinkPredictor(train_config, model_config, data)
                else:
                    det = BaseGNNLinkPredictor(train_config, model_config, data)

                st = time.time()
                score = det.train()
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
                    'decoder': args.decoder,
                    'neg_sampling': args.neg_sampling,
                }
                new_rows.append(row)
                pd.concat([existing, pd.DataFrame(new_rows)],
                          ignore_index=True).to_csv(args.out_path, index=False)
                print(f'  {dataset}/{model_name}/split{split_id}: '
                      f'AUROC={row["AUROC"]:.4f} AUPRC={row["AUPRC"]:.4f} '
                      f'RecK={row["RecK"]:.4f}  ({elapsed:.1f}s)')
                done.add(key)
                del det

            del data

    total_rows = len(existing) + len(new_rows)
    print(f'\nSaved {total_rows} total rows ({len(new_rows)} new) '
          f'to {args.out_path}')


if __name__ == '__main__':
    main()
