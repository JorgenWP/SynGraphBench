"""Assemble thesis-ready BO results under ``experiments/results/BO_Tuning/``.

Reads per-split outputs from ``experiments/bo_tuning/{dataset}/per_split/split{N}/``
(the canonical Optuna/process directory) and writes copies + cross-split
rollups to ``experiments/results/BO_Tuning/{dataset}/per_split/``.

Per-split artifacts copied: ``summary.csv``, ``best_params.json``,
``held_out_final_report.csv``, ``held_out_aggregated.csv``,
``divergence_curves.csv``, ``divergence_curves.png``.

Rollup artifacts derived from the per-split copies:
``rollup/best_params.csv`` (one row per split),
``rollup/trial_summary.csv`` (concat with ``split_id`` column),
``rollup/held_out.csv``,
``rollup/held_out_aggregated.csv`` (mean/std across splits),
``rollup/figures/objective_trajectory.png``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


PER_SPLIT_COPIES = [
    ('summary.csv', 'summary.csv'),
    ('best_params.json', 'best_params.json'),
    ('final_report/held_out_final_report.csv', 'held_out_final_report.csv'),
    ('final_report/benchmark_heldout/evaluation_results.csv', 'held_out_aggregated.csv'),
    ('final_report/benchmark_heldout/divergence_curves.csv', 'divergence_curves.csv'),
    ('final_report/divergence_curves.png', 'divergence_curves.png'),
]


def _copy_split(src_split: Path, dst_split: Path) -> None:
    dst_split.mkdir(parents=True, exist_ok=True)
    for src_rel, dst_name in PER_SPLIT_COPIES:
        src_file = src_split / src_rel
        if not src_file.exists():
            print(f'  missing {src_rel} under {src_split.name}, skipping')
            continue
        shutil.copy2(src_file, dst_split / dst_name)


def _best_params_row(bp_path: Path, split_id: int) -> dict | None:
    if not bp_path.exists():
        return None
    bp = json.loads(bp_path.read_text())
    row = {
        'split_id': split_id,
        'trial_number': bp.get('trial_number'),
        'objective': bp.get('value'),
    }
    row.update(bp.get('params', {}))
    ua = bp.get('user_attrs', {})
    row['cvar'] = ua.get('cvar')
    row['n_collapsed_splits'] = ua.get('n_collapsed_splits')
    return row


def _build_rollup(dst_root: Path, splits: list[int]) -> None:
    rollup = dst_root / 'rollup'
    figures = rollup / 'figures'
    figures.mkdir(parents=True, exist_ok=True)

    # best_params.csv — one row per split
    rows = [r for s in splits
            if (r := _best_params_row(dst_root / f'split{s}' / 'best_params.json', s))]
    if rows:
        pd.DataFrame(rows).to_csv(rollup / 'best_params.csv', index=False)

    # trial_summary.csv — concat summaries with split_id column
    summary_frames = []
    for s in splits:
        p = dst_root / f'split{s}' / 'summary.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.insert(0, 'split_id', s)
        summary_frames.append(df)
    trial_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    if not trial_summary.empty:
        trial_summary.to_csv(rollup / 'trial_summary.csv', index=False)

    # held_out.csv — concat per-split held-out reports
    held_frames = []
    for s in splits:
        p = dst_root / f'split{s}' / 'held_out_final_report.csv'
        if not p.exists():
            continue
        held_frames.append(pd.read_csv(p))
    held_out = pd.concat(held_frames, ignore_index=True) if held_frames else pd.DataFrame()
    if not held_out.empty:
        held_out.to_csv(rollup / 'held_out.csv', index=False)

        # held_out_aggregated.csv — mean/std across (splits × seeds) per (source, model)
        metrics = ['AUROC', 'AUPRC', 'RecK']
        agg = (held_out.groupby(['source', 'dataset', 'model'])[metrics]
               .agg(['mean', 'std'])
               .reset_index())
        agg.columns = ['_'.join(c).rstrip('_') for c in agg.columns]
        agg.to_csv(rollup / 'held_out_aggregated.csv', index=False)

    # objective_trajectory.png — best-so-far per split vs trial number
    if not trial_summary.empty and 'objective' in trial_summary.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for s in sorted(trial_summary['split_id'].unique()):
            d = trial_summary[(trial_summary['split_id'] == s)
                              & (trial_summary['status'] == 'COMPLETE')].copy()
            if d.empty:
                continue
            d = d.sort_values('trial_number')
            d['best_so_far'] = d['objective'].cummax()
            ax.plot(d['trial_number'], d['best_so_far'], label=f'split {s}', marker='o', markersize=3)
        ax.set_xlabel('trial')
        ax.set_ylabel('best objective so far')
        ax.set_title('BO objective trajectory per split')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures / 'objective_trajectory.png', dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--mode', default='per_split', choices=['per_split'],
                        help='Only per_split is supported today.')
    parser.add_argument('--repo_root', default=None,
                        help='Project root (defaults to ../../ relative to this file).')
    args = parser.parse_args()

    repo = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[2]
    src_root = repo / 'experiments' / 'bo_tuning' / args.dataset / 'per_split'
    dst_root = repo / 'experiments' / 'results' / 'BO_Tuning' / args.dataset / 'per_split'

    if not src_root.is_dir():
        raise SystemExit(f'no source directory at {src_root}')

    splits = sorted(int(d.name.removeprefix('split'))
                    for d in src_root.iterdir()
                    if d.is_dir() and d.name.startswith('split')
                    and d.name.removeprefix('split').isdigit())
    if not splits:
        raise SystemExit(f'no split{{N}} subdirs found under {src_root}')

    print(f'splits present: {splits}')
    for s in splits:
        print(f'copying split{s} ...')
        _copy_split(src_root / f'split{s}', dst_root / f'split{s}')

    print('building rollup ...')
    _build_rollup(dst_root, splits)
    print(f'wrote results to {dst_root}')


if __name__ == '__main__':
    main()
