"""Shared k-anonymity feature-anonymization helper for the privacy baselines.

The privacy baseline trains a GNN on the real graph topology but with node
features replaced by their k-anonymity *cluster centroids* (raw feature space),
isolating the cost of the anonymization step from the generative model. This
module resolves the precomputed clustering cache and returns the per-node
centroid feature matrix.

Pure stdlib + torch so it imports cleanly into any benchmark env. Fail-loud on
a missing/ambiguous cache leaf or a meta.json mismatch — never refit clusters
on the fly (see methodology guard-rail #3 in the Clustering-cache skill).
"""
from __future__ import annotations

import glob
import json
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))

# raw_centers.pt holds per-cluster means in *raw* feature space — the right
# tensor for GADBench detectors, which read graph.ndata['feature'] without
# normalization. l2_centers.pt is CGT-only.
_CENTERS_FILE = 'raw_centers.pt'


def _resolve_leaf(cache_root, dataset, task, trial, k):
    """Return the unique cache leaf dir for (dataset, task, [trial], k).

    cluster_num (the ``_c<N>`` suffix) varies per (dataset, k), so the leaf is
    matched by glob ``k{k}_c*`` and asserted unique.
    """
    if not os.path.isabs(cache_root):
        cache_root = os.path.join(_PROJECT_ROOT, cache_root)

    if task == 'hidden_links':            # trial-invariant — no t<trial> segment
        base = os.path.join(cache_root, dataset, task)
    elif task == 'hidden_labels':
        if trial is None:
            raise ValueError("hidden_labels requires a trial (split_id)")
        base = os.path.join(cache_root, dataset, task, f't{trial}')
    else:
        raise ValueError(f"unsupported task {task!r}")

    matches = sorted(glob.glob(os.path.join(base, f'k{k}_c*')))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No clustering cache leaf 'k{k}_c*' under {base}. "
            f"Run scripts/cluster/precompute_clusters.py for "
            f"dataset={dataset} task={task} cluster_size={k}.")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous cache leaves for k{k} under {base}: {matches}")
    return matches[0]


def resolve_anonymized_features(cache_root, dataset, task, trial, k,
                                original_feats):
    """Return a (N, D) tensor with every row replaced by its cluster centroid.

    ``trial`` is the split_id for ``hidden_labels`` (per-trial cache) and is
    ignored for ``hidden_links`` (trial-invariant). Validates the cache against
    ``original_feats`` (feat_dim, N) and the requested k (cluster_size).
    """
    leaf = _resolve_leaf(cache_root, dataset, task, trial, k)

    if not os.path.isfile(os.path.join(leaf, 'DONE')):
        raise FileNotFoundError(
            f"Cache leaf {leaf} has no DONE marker — incomplete. "
            f"Re-run scripts/cluster/precompute_clusters.py.")

    with open(os.path.join(leaf, 'meta.json')) as f:
        meta = json.load(f)

    n, d = original_feats.shape
    if meta['feat_dim'] != d:
        raise ValueError(
            f"feat_dim mismatch: cache {meta['feat_dim']} vs graph {d} ({leaf})")
    if meta['cluster_size'] != k:
        raise ValueError(
            f"cluster_size mismatch: cache {meta['cluster_size']} vs requested "
            f"{k} ({leaf})")
    if meta.get('num_nodes', n) != n:
        raise ValueError(
            f"num_nodes mismatch: cache {meta['num_nodes']} vs graph {n} ({leaf})")

    cluster_ids = torch.load(os.path.join(leaf, 'cluster_ids.pt')).long()  # (N,)
    raw_centers = torch.load(os.path.join(leaf, _CENTERS_FILE)).float()    # (K,D)
    if cluster_ids.shape[0] != n:
        raise ValueError(
            f"cluster_ids length {cluster_ids.shape[0]} != graph N {n} ({leaf})")

    return raw_centers[cluster_ids].to(original_feats.dtype)               # (N,D)
