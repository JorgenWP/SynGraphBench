"""Inspect min_subgraph_nodes thresholds against the actual partition pickles.

For each BO YAML in ``experiments/bo_tuning/configs/``, loads each split's
partition pickle, applies the YAML's ``min_subgraph_nodes`` filter, and
prints the dropped sizes plus the 3 smallest *kept* partitions per split.
Shows whether the threshold sits in a clean gap or slices a continuum.

Usage:
    python experiments/bo_tuning/_check_subsample_thresholds.py
"""
import glob
import os
import pickle

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'bo_tuning', 'configs')
SUBSAMPLE_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'bigg_subsamples')


def main():
    for cfg_path in sorted(glob.glob(os.path.join(CONFIG_DIR, '*.yaml'))):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        fh = cfg['fixed_hp']
        dataset = fh['dataset']
        n_splits = fh['n_splits']
        subcfg = fh['subsampling_config']
        threshold = fh.get('min_subgraph_nodes', 0)

        print(f'=== {dataset} | subsampling={subcfg} | min_subgraph_nodes={threshold} ===')

        for s in range(n_splits):
            pkl = os.path.join(SUBSAMPLE_DIR, dataset, subcfg, f'split{s}.pkl')
            if not os.path.exists(pkl):
                print(f'  split{s}: pickle missing at {pkl}')
                continue
            with open(pkl, 'rb') as f:
                payload = pickle.load(f)
            sizes = sorted(sg.number_of_nodes() for sg, _, _ in payload['partitions'])
            dropped = [n for n in sizes if n < threshold]
            kept = [n for n in sizes if n >= threshold]
            smallest_kept = kept[:3]
            gap = (smallest_kept[0] - dropped[-1]) if dropped and kept else None
            gap_str = f'gap={gap}' if gap is not None else 'gap=n/a'
            print(f'  split{s}: K_raw={len(sizes):>3} K_kept={len(kept):>3} '
                  f'dropped={dropped} smallest_kept={smallest_kept} {gap_str}')
        print()


if __name__ == '__main__':
    main()
