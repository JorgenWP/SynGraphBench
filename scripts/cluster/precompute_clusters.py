"""Precompute and cache k-anonymity clustering per (dataset, task, trial, k).

Writes one cache entry per `(dataset, task, [trial,] cluster_size, cluster_num)`
under `cache/clustering/`. Each entry contains:
  - cluster_ids.pt     (N,) LongTensor, node -> cluster id
  - raw_centers.pt     (K, D) FloatTensor, mean of pre-L2-norm features per cluster
                       (BiGG's input; BiGG applies its own CDF/quantile normalization)
  - l2_centers.pt      (K, D) FloatTensor, L2-normed in original space
                       (CGT's input; matches CGT's current convention)
  - meta.json          dataset/task/trial/k/etc. + member-count stats
  - DONE               touchfile, written last; resume = skip entries with DONE

For `hidden_labels` the fit-set is the per-trial `train+val` derived from
`graph.ndata['{train,val}_masks'][:, trial]`, so clustering is per-trial.

For `hidden_links` the fit-set is *all nodes* (since
`split_node_ids_for_hidden_links` gives `train=80%`, `val=20%`, `test=[]`,
union = all) and the result is trial-invariant — one fit per
`(dataset, cluster_size, cluster_num)` suffices, so the cache key drops the
trial segment.

Run from project root, e.g.:
    python scripts/cluster/precompute_clusters.py \
        --datasets tolokers \
        --cluster-sizes 1 50 \
        --cluster-nums 512 597 \
        --tasks hidden_labels hidden_links \
        --trials 0 1 2
"""

import argparse
import json
import os
import os.path as osp
import random
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import normalize

# Make `CGT/` importable when running from project root.
_THIS_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, osp.join(_PROJECT_ROOT, 'CGT'))

from generator.cluster import cluster_fit_and_assign  # noqa: E402
from task.utils.utils import load_dgl_raw_features  # noqa: E402


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', '-C', _PROJECT_ROOT, 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _cache_dir(cache_root, dataset, task, trial, cluster_size, cluster_num):
    """Cache layout: hidden_links drops the trial segment (trial-invariant)."""
    leaf = f"k{cluster_size}_c{cluster_num}"
    if task == 'hidden_links':
        return osp.join(cache_root, dataset, task, leaf)
    return osp.join(cache_root, dataset, task, f"t{trial}", leaf)


def _resolve_fit_ids(task, trial, train_masks, val_masks, num_nodes):
    """Return the node ids used to fit k-means for (task, trial).

    hidden_labels: per-trial train+val from DGL masks.
    hidden_links:  all nodes (trial-invariant).
    """
    if task == 'hidden_links':
        return np.arange(num_nodes, dtype=np.int64).tolist()
    if train_masks is None or val_masks is None:
        raise ValueError(
            f"task=hidden_labels requires train_masks/val_masks on the graph; "
            f"dataset has none."
        )
    train_ids = np.where(train_masks[:, trial])[0]
    val_ids = np.where(val_masks[:, trial])[0]
    return np.concatenate([train_ids, val_ids]).tolist()


def _save_atomic(cache_dir, cluster_ids, raw_centers, l2_centers, meta):
    """Write the four files then touch DONE last so resume is safe."""
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(torch.from_numpy(cluster_ids).long(),
               osp.join(cache_dir, 'cluster_ids.pt'))
    torch.save(torch.from_numpy(raw_centers).float(),
               osp.join(cache_dir, 'raw_centers.pt'))
    torch.save(torch.from_numpy(l2_centers).float(),
               osp.join(cache_dir, 'l2_centers.pt'))
    with open(osp.join(cache_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    # DONE last — its presence is the resume marker.
    open(osp.join(cache_dir, 'DONE'), 'w').close()


def _compute_raw_centers(feat_raw, cluster_ids, k):
    """Mean of truly-raw (pre-L2-norm) features per cluster.

    Empty clusters get a zero row (kept consistent with CGT's empty_id
    convention, though that row is not appended here).
    """
    d = feat_raw.shape[1]
    raw_centers = np.zeros((k, d), dtype=np.float32)
    for c in range(k):
        mask = cluster_ids == c
        if mask.any():
            raw_centers[c] = feat_raw[mask].mean(axis=0)
    return raw_centers


def _process_one(args_ns, feat_raw, feat_l2, fit_ids, dataset, task,
                 trial, cluster_size, cluster_num, cache_root, seed, force):
    cache_dir = _cache_dir(cache_root, dataset, task, trial,
                           cluster_size, cluster_num)
    done_path = osp.join(cache_dir, 'DONE')
    trial_str = '-' if task == 'hidden_links' else f"t{trial}"
    tag = f"[{dataset} {task} {trial_str} k{cluster_size} c{cluster_num}]"

    if osp.isfile(done_path) and not force:
        print(f"{tag} skipping (cached, DONE exists)")
        return 'skipped'

    fit_set_size = len(fit_ids)
    if cluster_size * cluster_num > fit_set_size:
        print(f"{tag} SKIP — infeasible: cluster_size*cluster_num "
              f"({cluster_size * cluster_num}) > |fit_set| ({fit_set_size})")
        return 'infeasible'

    print(f"{tag} starting | fit_set={fit_set_size} | feat_dim={feat_raw.shape[1]}")
    t0 = perf_counter()

    _set_seed(seed)
    args_ns.cluster_size = cluster_size
    args_ns.cluster_num = cluster_num
    # Force cluster_sample_num >= fit_set so the fit uses every fit-set node.
    args_ns.cluster_sample_num = fit_set_size

    cluster_ids_np, centers_orig, stats = cluster_fit_and_assign(
        args_ns, feat_l2, fit_ids=fit_ids)

    # CGT-side centers: L2-normalize the originals (CGT's current convention,
    # CGT/generator/cluster.py L2-norm step). Treat zero rows safely.
    norms = np.linalg.norm(centers_orig, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    l2_centers = (centers_orig / norms).astype(np.float32)

    # BiGG-side centers: average pre-L2-norm features per cluster.
    raw_centers = _compute_raw_centers(feat_raw, cluster_ids_np,
                                       centers_orig.shape[0])

    meta = {
        'dataset': dataset,
        'task': task,
        'trial': None if task == 'hidden_links' else int(trial),
        'cluster_size': int(cluster_size),
        'cluster_num': int(cluster_num),
        'cluster_sample_num': int(args_ns.cluster_sample_num),
        'fit_set_size': int(fit_set_size),
        'num_nodes': int(feat_raw.shape[0]),
        'feat_dim': int(feat_raw.shape[1]),
        'seed': int(seed),
        'git_commit': _git_commit(),
        'cluster_stats': stats,
    }

    _save_atomic(cache_dir, cluster_ids_np, raw_centers, l2_centers, meta)
    elapsed = perf_counter() - t0
    print(f"{tag} wrote cache to {cache_dir} in {elapsed:.2f}s")
    return 'ok'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='GADBench dataset names (e.g. tolokers questions elliptic).')
    parser.add_argument('--cluster-sizes', nargs='+', type=int, required=True,
                        help='k-anonymity floors; paired index-wise with --cluster-nums.')
    parser.add_argument('--cluster-nums', nargs='+', type=int, required=True,
                        help='Number of clusters; paired index-wise with --cluster-sizes.')
    parser.add_argument('--tasks', nargs='+',
                        default=['hidden_labels', 'hidden_links'],
                        choices=['hidden_labels', 'hidden_links'])
    parser.add_argument('--trials', nargs='+', type=int,
                        default=list(range(10)),
                        help='Trial ids; only used for hidden_labels.')
    parser.add_argument('--cache-root', default='cache/clustering')
    parser.add_argument('--data-dir', default='datasets/original')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true',
                        help='Re-fit and overwrite even if DONE exists.')
    cli = parser.parse_args()

    if len(cli.cluster_sizes) != len(cli.cluster_nums):
        parser.error("--cluster-sizes and --cluster-nums must have the same length")

    pairs = list(zip(cli.cluster_sizes, cli.cluster_nums))
    cache_root = osp.abspath(cli.cache_root) if osp.isabs(cli.cache_root) \
        else osp.join(_PROJECT_ROOT, cli.cache_root)

    summary = {'ok': 0, 'skipped': 0, 'infeasible': 0, 'failed': 0}

    for dataset in cli.datasets:
        print(f"\n=== Dataset: {dataset} ===")
        args_ns = SimpleNamespace(
            data_dir=cli.data_dir,
            dataset=dataset,
            dp_feature=False,
        )
        t_load = perf_counter()
        feat_raw, train_masks, val_masks, _test_masks = load_dgl_raw_features(args_ns)
        feat_l2 = normalize(feat_raw, axis=1, norm='l2').astype(np.float32)
        print(f"loaded raw features in {perf_counter() - t_load:.2f}s: "
              f"N={feat_raw.shape[0]}, D={feat_raw.shape[1]}")

        for task in cli.tasks:
            if task == 'hidden_links':
                trial_iter = [None]  # single trial-agnostic pass
            else:
                trial_iter = cli.trials

            for trial in trial_iter:
                try:
                    fit_ids = _resolve_fit_ids(task, trial, train_masks,
                                               val_masks, feat_raw.shape[0])
                except ValueError as e:
                    print(f"[{dataset} {task}] SKIP — {e}")
                    summary['failed'] += 1
                    continue

                for cluster_size, cluster_num in pairs:
                    try:
                        result = _process_one(
                            args_ns, feat_raw, feat_l2, fit_ids,
                            dataset, task, trial,
                            cluster_size, cluster_num,
                            cache_root, cli.seed, cli.force,
                        )
                        summary[result] = summary.get(result, 0) + 1
                    except Exception as e:
                        trial_str = '-' if task == 'hidden_links' else f"t{trial}"
                        print(f"[{dataset} {task} {trial_str} "
                              f"k{cluster_size} c{cluster_num}] FAILED: {e!r}")
                        summary['failed'] += 1

    print(f"\n=== Done. Summary: {summary} ===")


if __name__ == '__main__':
    main()
