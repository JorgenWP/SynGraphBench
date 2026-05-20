"""Build a feature k-anonymized DGL graph artifact.

Loads an original DGL graph, groups its nodes into size-constrained k-means
clusters (size_min = k, n_clusters = N // k), and replaces each node's
feature vector with the **un-normalized** mean of its cluster. Topology,
labels, masks, and edge data are preserved verbatim.

Output:
    ../datasets/kanon/<dataset>/<dataset>_k{k}_{cluster_norm}_s{seed}.dgl
    ../datasets/kanon/<dataset>/<dataset>_k{k}_{cluster_norm}_s{seed}.meta.pt

The artifact can be used (a) as a benchmark input via the GADBench harness
(privacy-baseline utility curve) or (b) as a training input to the BiGG
pipeline so the synthetic output inherits k-anonymity.
"""

import argparse
import os
from time import perf_counter

import dgl
import numpy as np
import torch
from k_means_constrained import KMeansConstrained
from sklearn.preprocessing import normalize as sk_normalize


def _parse_args():
    p = argparse.ArgumentParser(description='k-anonymize a DGL graph by clustering features.')
    p.add_argument('-dataset', type=str, required=True,
                   help='Dataset name (file at <data_dir>/original/<dataset>).')
    p.add_argument('-k', type=int, required=True,
                   help='Anonymity level (size_min for clusters). n_clusters = N // k.')
    p.add_argument('-cluster_norm', type=str, default='zscore',
                   choices=['none', 'zscore', 'l2'],
                   help='Feature normalization used only for the clustering distance. '
                        'Centroid values are always the un-normalized mean.')
    p.add_argument('-seed', type=int, default=0)
    p.add_argument('-trial_id', type=int, default=0,
                   help='Which GADBench trial split to use. Defines the '
                        'train+val anonymization scope; test nodes for that '
                        'trial keep their original features.')
    p.add_argument('-max_iter', type=int, default=100,
                   help='KMeansConstrained max_iter cap.')
    p.add_argument('-data_dir', type=str, default='../datasets',
                   help='Datasets root. Reads <data_dir>/original/<dataset>, '
                        'writes <data_dir>/kanon/<dataset>/.')
    return p.parse_args()


def _make_cluster_feats(feats_raw, mode):
    if mode == 'none':
        return feats_raw, None
    if mode == 'zscore':
        mean = feats_raw.mean(axis=0)
        std = feats_raw.std(axis=0)
        std_safe = std.copy()
        std_safe[std_safe == 0] = 1.0
        return (feats_raw - mean) / std_safe, {'method': 'zscore', 'mean': mean, 'std': std}
    if mode == 'l2':
        return sk_normalize(feats_raw, axis=1, norm='l2'), {'method': 'l2'}
    raise ValueError(f'unknown cluster_norm: {mode}')


def main():
    args = _parse_args()

    in_path = os.path.join(args.data_dir, 'original', args.dataset)
    out_dir = os.path.join(args.data_dir, 'kanon', args.dataset)
    stem = f'{args.dataset}_k{args.k}_{args.cluster_norm}_s{args.seed}_t{args.trial_id}'
    out_graph_path = os.path.join(out_dir, f'{stem}.dgl')
    out_meta_path = os.path.join(out_dir, f'{stem}.meta.pt')

    print(f'[kanon] loading {in_path}')
    graphs, _ = dgl.load_graphs(in_path)
    graph = graphs[0]
    feats_raw = graph.ndata['feature'].numpy().astype(np.float32)
    N, D = feats_raw.shape

    # Trial-specific train+val mask defines the anonymization scope.
    # Test nodes for this trial keep their original features so that
    # downstream evaluation measures the utility cost of training on
    # quantized features against a real-world test distribution.
    train_mask = graph.ndata['train_masks'][:, args.trial_id].bool().numpy()
    val_mask = graph.ndata['val_masks'][:, args.trial_id].bool().numpy()
    anon_mask = train_mask | val_mask
    n_anon = int(anon_mask.sum())
    n_test_like = N - n_anon

    n_clusters = n_anon // args.k
    if n_clusters < 1:
        raise ValueError(
            f'k={args.k} too large for n_anon={n_anon} (n_clusters would be 0). '
            f'Reduce -k or check trial mask coverage.'
        )

    print(f'[kanon] N={N}, D={D}, k={args.k}, trial={args.trial_id}, '
          f'anonymized={n_anon}, untouched={n_test_like}, '
          f'n_clusters={n_clusters}, norm={args.cluster_norm}')

    feats_anon_raw = feats_raw[anon_mask]
    feats_cluster, norm_meta = _make_cluster_feats(feats_anon_raw, args.cluster_norm)

    print(f'[kanon] running KMeansConstrained (max_iter={args.max_iter})...')
    t0 = perf_counter()
    clf = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=args.k,
        init='k-means++',
        n_init=1,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    clf.fit(feats_cluster)
    cluster_ids = clf.labels_.astype(np.int64)
    t_cluster = perf_counter() - t0
    print(f'[kanon] clustering done in {t_cluster:.2f}s')

    # Centroids = mean of ORIGINAL features within each cluster, over the
    # anonymized (train+val) subset only.
    centroids = np.zeros((n_clusters, D), dtype=np.float32)
    sizes = np.bincount(cluster_ids, minlength=n_clusters)
    for c in range(n_clusters):
        centroids[c] = feats_anon_raw[cluster_ids == c].mean(axis=0)

    # Replace only the anonymized rows; leave the rest untouched.
    new_feats = feats_raw.copy()
    new_feats[anon_mask] = centroids[cluster_ids]
    graph.ndata['feature'] = torch.from_numpy(new_feats)

    os.makedirs(out_dir, exist_ok=True)
    dgl.save_graphs(out_graph_path, [graph])
    print(f'[kanon] wrote {out_graph_path}')

    meta = {
        'dataset': args.dataset,
        'k': args.k,
        'n_clusters': int(n_clusters),
        'cluster_norm': args.cluster_norm,
        'seed': args.seed,
        'trial_id': args.trial_id,
        'n_anonymized': n_anon,
        'n_untouched': n_test_like,
        'max_iter': args.max_iter,
        'cluster_sizes': sizes.tolist(),
        'cluster_size_min': int(sizes.min()),
        'cluster_size_max': int(sizes.max()),
        'norm_meta': norm_meta,
        'cluster_runtime_sec': t_cluster,
    }
    torch.save(meta, out_meta_path)
    print(f'[kanon] wrote {out_meta_path}')


if __name__ == '__main__':
    main()
