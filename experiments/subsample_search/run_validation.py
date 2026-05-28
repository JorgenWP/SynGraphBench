"""Validation deliverable for the subsample-search experiment.

Forest-fire subsampling on tolokers, 3 paired (split, seed) trials × 5 models.
Goal: confirm that the leak-safe first-occurrence masking + LCC-restricted
forest fire produce numbers that strictly exceed the existing
real-subsampled-graph row in
results/evaluate/bigg/tolokers/.../evaluation_results.csv.

Usage:
    cd <project root>
    conda activate bigg
    python experiments/subsample_search/run_validation.py
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from splitsource import load_split_source
from sampling import sample_partitions
from masking import build_combined_graph
from benchmark import run_models


MODELS = ['GCN', 'GIN', 'GraphSAGE', 'XGBGraph', 'XGBoost']
SEEDS = [3407, 3417, 3427]                                # one per split
TRIALS = list(zip(range(len(SEEDS)), SEEDS))              # [(split_id, seed), ...]


def _print_summary(df: pd.DataFrame) -> None:
    summary = (df.groupby('model')[['AUROC', 'AUPRC', 'RecK']]
                 .agg(['mean', 'std']).round(4))
    print('\n=== Per-model summary across 3 paired trials ===')
    print(summary)


def _print_comparison(df: pd.DataFrame, baseline_csv: str) -> None:
    if not os.path.exists(baseline_csv):
        print(f'\n[skip comparison] baseline CSV not found: {baseline_csv}')
        return
    base = pd.read_csv(baseline_csv)
    base = base[base['source'] == 'real-subsampled-graph']
    if base.empty:
        print(f'\n[skip comparison] no real-subsampled-graph rows in {baseline_csv}')
        return

    new_means = (df.groupby('model')[['AUROC', 'AUPRC', 'RecK']].mean())
    print('\n=== AUROC delta vs existing real-subsampled-graph row ===')
    print(f'{"model":<12} {"existing":>10} {"new":>10} {"delta":>10}')
    for m in MODELS:
        existing_rows = base[base['model'] == m]
        if existing_rows.empty or m not in new_means.index:
            continue
        ex = float(existing_rows['AUROC_mean'].iloc[0])
        nw = float(new_means.loc[m, 'AUROC'])
        print(f'{m:<12} {ex:>10.4f} {nw:>10.4f} {nw - ex:>+10.4f}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='tolokers')
    ap.add_argument('--num_subgraphs', type=int, default=15)
    ap.add_argument('--target_size', type=int, default=500)
    ap.add_argument('--burn_prob', type=float, default=0.7)
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--baseline_csv', default=None,
                    help='Existing pipeline evaluation_results.csv to compare against.')
    args = ap.parse_args()

    project_root = os.path.abspath(os.path.join(_HERE, '..', '..'))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'datasets', 'original')
    if args.out_dir is None:
        args.out_dir = os.path.join(_HERE, 'artifacts', args.dataset, 'validation')
    if args.baseline_csv is None:
        args.baseline_csv = os.path.join(
            project_root, 'results', 'evaluate', 'bigg', args.dataset,
            'blksize_-1_b_1_lr_0.0003_epochs_300_noise_0.3_ss_0.3_norm_cdf_bfs_'
            'lw_0.01_0.1_det_masked_binfeat_vae16_kl0.0_sub15_size500_p0.7',
            'evaluation_results.csv',
        )
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'== {args.dataset} | forest_fire | K={args.num_subgraphs} '
          f'size={args.target_size} burn={args.burn_prob} | device={device} ==')

    rows = []
    for split_id, seed in TRIALS:
        print(f'\n[split {split_id} | seed {seed}] loading split source...')
        src = load_split_source(args.dataset, split_id, args.data_dir)
        print(f'  trainval LCC: N={src.orig_ids.numel()} '
              f'(train={int(src.is_train.sum())}, val={int(src.is_val.sum())})')

        print(f'[split {split_id}] forest fire...')
        node_data = torch.cat([src.features, src.labels.unsqueeze(1).float()], dim=1)
        partitions = sample_partitions(
            src.g_nx, node_data,
            K=args.num_subgraphs,
            target_size=args.target_size,
            burn_prob=args.burn_prob,
            seed=seed,
        )

        combined = build_combined_graph(partitions, src)
        combined = combined.to(device)
        orig = src.orig_dgl.to(device)

        print(f'[split {split_id}] running {len(MODELS)} models...')
        trial_rows = run_models(
            combined, orig, MODELS, seed, device,
            epochs=args.epochs, patience=args.patience,
        )
        for r in trial_rows:
            r.update({
                'dataset': args.dataset,
                'split_id': split_id,
                'K': args.num_subgraphs,
                'target_size': args.target_size,
                'burn_prob': args.burn_prob,
            })
            rows.append(r)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, 'utility.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv} ({len(df)} rows)')

    _print_summary(df)
    _print_comparison(df, args.baseline_csv)


if __name__ == '__main__':
    main()
