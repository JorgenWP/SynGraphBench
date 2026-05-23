import numpy as np
import os
import random
import torch

from time import perf_counter
from sklearn.decomposition import PCA
from k_means_constrained import KMeansConstrained

# To run differential private k-means, you need to download an open-source library from: https://github.com/google/differential-privacy/tree/main/learning/clustering
#from .clustering import clustering_algorithm
#from .clustering import clustering_params

# Above this node count, KMeansConstrained.predict (min-cost flow over N*K edges)
# becomes infeasible on RAM/time; fall back to argmin + greedy repair instead.
# Empirical — validate on Elliptic (~204K nodes) and lower if needed.
_CONSTRAINED_MAX_N = 500_000


def kmeans(feats, cluster_num, cluster_size, cluster_sample_num):
    """
    k-means clustering. Returns the fitted PCA and KMeansConstrained objects so
    callers can re-run the constrained assignment over a larger node set.
    """
    if cluster_sample_num < feats.shape[0]:
        px_ids = random.sample(list(range(feats.shape[0])), cluster_sample_num)
        x = feats[px_ids]
    else:
        x = feats

    pca = PCA(n_components=min(feats.shape[1], 128))
    x_pca = pca.fit_transform(x)

    clf = KMeansConstrained(n_clusters=cluster_num, size_min=cluster_size, init='random', n_init=1, max_iter=8)
    clf.fit(x_pca)
    centers = pca.inverse_transform(clf.cluster_centers_)

    return centers, pca, clf


def DP_kmeans(feats, cluster_num, cluster_sample_num, epsilon=10, delta=1e-6):
    """
    Differential private k-means clustering
    Args:
        feats: feature vectors
        cluster_num: number of clusters
        cluster_sample_num: number of samples for clustering
        epsilon: privacy budget
        delta: privacy budget
    Return:
        centers: cluster centers
        cluster_num: number of clusters
    """
    if cluster_sample_num < feats.shape[0]:
        px_ids = random.sample(list(range(feats.shape[0])), cluster_sample_num)
        x = feats[px_ids]
    else:
        x = feats

    pca = PCA(n_components=128)
    x_pca = pca.fit_transform(x)
    x_pca_total = pca.transform(feats)

    data = clustering_params.Data(x_pca, radius=1.0)
    privacy_param = clustering_params.DifferentialPrivacyParam(epsilon=epsilon, delta=delta)
    clustering_result: clustering_algorithm.ClusteringResult = (clustering_algorithm.private_lsh_clustering(cluster_num, data, privacy_param))

    centers = pca.inverse_transform(clustering_result.centers)
    return centers, centers.shape[0]

def _repair_min_size(cluster_ids, feats, centers, size_min):
    """
    Greedy repair: while any cluster has fewer than `size_min` members, move the
    cheapest node out of an over-filled cluster into the most-deficient one.
    "Cheapest" = smallest squared-distance increase. Donors must have surplus
    (size > size_min) so the move never creates a new violation.

    Args:
        cluster_ids: (N,) int array — modified in place and returned.
        feats: (N, d) array in the same space as `centers`.
        centers: (K, d) cluster centers.
        size_min: required minimum cluster size.
    Returns:
        (cluster_ids, moves) — moves is the number of reassignments performed.
    """
    n_clusters = centers.shape[0]
    n = feats.shape[0]
    if size_min * n_clusters > n:
        raise ValueError(
            f"size_min ({size_min}) * n_clusters ({n_clusters}) > n_samples ({n}); "
            "cannot satisfy minimum cluster size."
        )

    sizes = np.bincount(cluster_ids, minlength=n_clusters)
    if sizes.min() >= size_min:
        return cluster_ids, 0

    # Squared distance from each node to its currently-assigned center.
    current_d2 = ((feats - centers[cluster_ids]) ** 2).sum(axis=1)
    moves = 0
    while sizes.min() < size_min:
        u = int(np.argmin(sizes))
        d_to_u = ((feats - centers[u]) ** 2).sum(axis=1)
        cost_inc = d_to_u - current_d2
        eligible = (sizes[cluster_ids] > size_min) & (cluster_ids != u)
        if not eligible.any():
            raise RuntimeError(
                "Greedy repair stalled: no donor cluster has surplus members. "
                f"sizes.min()={sizes.min()}, target size_min={size_min}."
            )
        masked_cost = np.where(eligible, cost_inc, np.inf)
        n_idx = int(np.argmin(masked_cost))
        c_old = int(cluster_ids[n_idx])
        cluster_ids[n_idx] = u
        sizes[c_old] -= 1
        sizes[u] += 1
        current_d2[n_idx] = d_to_u[n_idx]
        moves += 1
    return cluster_ids, moves


def cluster_feats(args, feats, fit_ids=None):
    """
    Cluster feature vectors. Final assignment is k-anonymity-preserving:
    constrained min-cost flow over all nodes for N <= _CONSTRAINED_MAX_N,
    or argmin + greedy repair otherwise (and on the DP path, which has no
    fitted KMeansConstrained object).

    Input:
        feats: original feature matrix (N, d).
        fit_ids: optional node id subset used to fit k-means (e.g. train+val);
            assignment still covers all nodes.
    Return:
        cluster_ids: (N+1,) LongTensor of cluster ids (with trailing empty_id).
        cluster_centers: (K+1, d) FloatTensor of centers (with trailing zero row).
    """
    start_time = perf_counter()
    fit_feats = feats if fit_ids is None else feats[fit_ids]
    fit_method = 'DP' if args.dp_feature else 'kmeans_constrained'
    print(f"[Clustering] fitting k-means on {len(fit_feats)}/{feats.shape[0]} nodes, "
          f"feat_dim={feats.shape[1]}, target k={args.cluster_num}, fit_method={fit_method}")
    if args.dp_feature:
        cluster_centers, cluster_num = DP_kmeans(fit_feats, args.cluster_num, args.cluster_sample_num)
        args.cluster_num = cluster_num
        pca = None
        clf = None
    else:
        cluster_centers, pca, clf = kmeans(fit_feats, args.cluster_num, args.cluster_size, args.cluster_sample_num)

    use_constrained = (clf is not None) and (feats.shape[0] <= _CONSTRAINED_MAX_N)
    if use_constrained:
        print(f"[Clustering] assigning {feats.shape[0]} nodes via constrained MCF "
              f"(size_min={args.cluster_size})")
        cluster_ids = clf.predict(pca.transform(feats), size_min=args.cluster_size).astype(np.int64)
        repair_moves = 0
    else:
        print(f"[Clustering] assigning {feats.shape[0]} nodes via argmin + greedy repair "
              f"(size_min={args.cluster_size}, N>{_CONSTRAINED_MAX_N} or DP)")
        batch_size = 1000
        cluster_ids = np.zeros(feats.shape[0], dtype=np.int64)
        for batch in range(feats.shape[0] // batch_size + 1):
            if batch < feats.shape[0] // batch_size:
                idx = list(range(batch * batch_size, (batch + 1) * batch_size))
            else:
                idx = list(range(batch * batch_size, feats.shape[0]))
            if not idx:
                continue
            cluster_ids[idx] = ((feats[idx, None, :] - cluster_centers[None, :, :]) ** 2).sum(-1).argmin(1)
        cluster_ids, repair_moves = _repair_min_size(cluster_ids, feats, cluster_centers, args.cluster_size)

    sizes = np.bincount(cluster_ids, minlength=cluster_centers.shape[0])
    empty = int((sizes == 0).sum())
    nonzero = sizes[sizes > 0]
    min_nonzero = int(nonzero.min()) if nonzero.size > 0 else 0
    print(f"[Clustering] produced {cluster_centers.shape[0]} clusters (empty={empty}); "
          f"member counts: min_nonzero={min_nonzero}, max={sizes.max()}, "
          f"mean={sizes.mean():.1f}, median={int(np.median(sizes))}, std={sizes.std():.1f}; "
          f"repair_moves={repair_moves}")

    # L2-normalize cluster centers so the synthetic feature pool lives
    # exactly on the unit sphere — matching CGT's L2-normalized training
    # input space and the eval framework's L2-normalized real test features.
    # Done after assignment + size repair so cluster memberships are
    # unaffected; only the saved/returned centers change.
    norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    cluster_centers = cluster_centers / norms
    print("[Clustering] L2-normalized cluster centers")

    # Append empty_id (zero row — produces no message-passing signal for
    # missing neighbours, intentionally outside the unit-sphere invariant)
    cluster_ids = torch.LongTensor(np.append(cluster_ids, args.cluster_num))
    cluster_centers = torch.FloatTensor(np.concatenate((cluster_centers, np.zeros((1, cluster_centers.shape[1]))), axis=0))

    print("Clustering time: {:.3f}".format(perf_counter() - start_time))

    return cluster_ids, cluster_centers

