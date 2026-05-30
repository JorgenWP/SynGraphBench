import json
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

# Above this fit-set size, KMeansConstrained.predict (min-cost flow over
# |fit set|*K edges) becomes infeasible on RAM/time; fall back to argmin +
# greedy repair instead. The constraint runs over the fit set (train+val), so
# this bounds the fit set, not all N.
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


def cluster_fit_and_assign(args, feats, fit_ids=None):
    """
    Fit k-means and assign every node to a cluster. Shared core used by both
    CGT's runtime path (`cluster_feats`) and the offline precompute cache
    (`scripts/cluster/precompute_clusters.py`).

    Returns centers in the original feature space (PCA-inverse-transformed)
    *before* any L2 normalization, and cluster_ids without the trailing
    empty_id row. Callers that need CGT's runtime conventions (unit-sphere
    centers + empty_id row) should use `cluster_feats` instead.

    Input:
        feats: original feature matrix (N, d).
        fit_ids: optional node id subset used to fit k-means AND to enforce the
            min cluster size (e.g. train+val). Assignment still covers all N
            nodes, but the k-anonymity floor is guaranteed only on the fit set;
            holdout nodes (outside fit_ids) are assigned by unconstrained
            nearest-center and never dilute the floor. None => fit/constrain on
            all nodes.
    Return:
        cluster_ids: (N,) int64 ndarray.
        cluster_centers: (K, d) float ndarray in original feature space
            (pre-L2-norm).
        stats: dict with member-count stats (incl. fit_min, the guaranteed
            train+val floor) and repair_moves for downstream logging / meta.json.
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

    # The k-anonymity (min cluster size) constraint is enforced ONLY on the fit
    # set (train+val for hidden_labels) — that is the population the generative
    # model is trained on, so the floor must hold there. Holdout nodes (e.g. the
    # anomaly test set) are assigned afterward by unconstrained nearest-center;
    # they appear in the partition only as neighbour / computation-graph context
    # and must not dilute the floor.
    n = feats.shape[0]
    fit_arr = (np.arange(n, dtype=np.int64) if fit_ids is None
               else np.asarray(fit_ids, dtype=np.int64))
    non_fit_mask = np.ones(n, dtype=bool)
    non_fit_mask[fit_arr] = False
    n_holdout = int(non_fit_mask.sum())

    # Size the constrained/argmin decision on the set the MCF actually runs over
    # (the fit set), not all N.
    use_constrained = (clf is not None) and (fit_feats.shape[0] <= _CONSTRAINED_MAX_N)
    cluster_ids = np.empty(n, dtype=np.int64)
    if use_constrained:
        print(f"[Clustering] assigning {fit_feats.shape[0]} fit nodes via constrained MCF "
              f"(size_min={args.cluster_size}); {n_holdout} holdout nodes via nearest-center")
        cluster_ids[fit_arr] = clf.predict(
            pca.transform(fit_feats), size_min=args.cluster_size).astype(np.int64)
        if n_holdout:
            # Nearest-center in PCA space, batched over nodes so the
            # (batch, K, d) distance tensor never blows up RAM (a single
            # (n_holdout, K, d) broadcast is hundreds of GB on large graphs).
            held_pca = pca.transform(feats[non_fit_mask])
            centers_pca = clf.cluster_centers_
            held_ids = np.empty(held_pca.shape[0], dtype=np.int64)
            batch_size = 1000
            for s in range(0, held_pca.shape[0], batch_size):
                chunk = held_pca[s:s + batch_size]
                d2 = ((chunk[:, None, :] - centers_pca[None, :, :]) ** 2).sum(-1)
                held_ids[s:s + batch_size] = d2.argmin(1)
            cluster_ids[non_fit_mask] = held_ids
        repair_moves = 0
    else:
        print(f"[Clustering] assigning {n} nodes via argmin + greedy repair "
              f"(size_min={args.cluster_size} on {fit_feats.shape[0]} fit nodes; "
              f"N>{_CONSTRAINED_MAX_N} or DP)")
        batch_size = 1000
        for batch in range(n // batch_size + 1):
            if batch < n // batch_size:
                idx = list(range(batch * batch_size, (batch + 1) * batch_size))
            else:
                idx = list(range(batch * batch_size, n))
            if not idx:
                continue
            cluster_ids[idx] = ((feats[idx, None, :] - cluster_centers[None, :, :]) ** 2).sum(-1).argmin(1)
        # Repair the floor on the fit set only; holdout nodes keep their argmin.
        fit_sub, repair_moves = _repair_min_size(
            cluster_ids[fit_arr].copy(), fit_feats, cluster_centers, args.cluster_size)
        cluster_ids[fit_arr] = fit_sub

    k = cluster_centers.shape[0]
    sizes = np.bincount(cluster_ids, minlength=k)
    empty = int((sizes == 0).sum())
    nonzero = sizes[sizes > 0]
    min_nonzero = int(nonzero.min()) if nonzero.size > 0 else 0
    # The k-anonymity floor as actually guaranteed: min per-cluster count over
    # the fit set (the population trained on). Fail loud if it slips below the
    # target — both the constrained and the greedy-repair paths guarantee it.
    fit_sizes = np.bincount(cluster_ids[fit_arr], minlength=k)
    fit_min = int(fit_sizes.min())
    if fit_min < args.cluster_size:
        raise RuntimeError(
            f"[Clustering] k-anonymity violated on fit set: min fit-cluster size "
            f"{fit_min} < cluster_size {args.cluster_size}.")
    print(f"[Clustering] produced {k} clusters (empty={empty}); "
          f"all-N member counts: min_nonzero={min_nonzero}, max={sizes.max()}, "
          f"mean={sizes.mean():.1f}, median={int(np.median(sizes))}, std={sizes.std():.1f}; "
          f"fit-set floor min={fit_min} (>= cluster_size={args.cluster_size}); "
          f"holdout={n_holdout}; repair_moves={repair_moves}")

    elapsed = perf_counter() - start_time
    print("Clustering time: {:.3f}".format(elapsed))

    stats = {
        'k': int(k),
        'empty': empty,
        'min_nonzero': min_nonzero,
        'max': int(sizes.max()),
        'mean': float(sizes.mean()),
        'median': int(np.median(sizes)),
        'std': float(sizes.std()),
        'fit_min': fit_min,
        'holdout_assigned': n_holdout,
        'repair_moves': int(repair_moves),
        'elapsed_seconds': float(elapsed),
        'fit_method': fit_method,
        'fit_set_size': int(len(fit_feats)),
        'total_nodes': int(n),
    }
    return cluster_ids, cluster_centers, stats


def cluster_feats(args, feats, fit_ids=None):
    """
    Cluster feature vectors. The k-anonymity (min cluster size) constraint is
    enforced on the fit set: constrained min-cost flow when |fit set| <=
    _CONSTRAINED_MAX_N, else argmin + greedy repair (also the DP path, which has
    no fitted KMeansConstrained object). Holdout nodes outside fit_ids are
    assigned afterward by unconstrained nearest-center.

    Input:
        feats: original feature matrix (N, d).
        fit_ids: optional node id subset used to fit k-means and enforce the min
            cluster size (e.g. train+val); assignment still covers all N nodes.
            None => all nodes.
    Return:
        cluster_ids: (N+1,) LongTensor of cluster ids (with trailing empty_id).
        cluster_centers: (K+1, d) FloatTensor of centers (with trailing zero row).
    """
    cluster_ids, cluster_centers, _ = cluster_fit_and_assign(args, feats, fit_ids=fit_ids)

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

    return cluster_ids, cluster_centers


def _cache_dir(cache_root, dataset, task, trial, cluster_size, cluster_num):
    """Resolve a clustering-cache entry dir.

    Mirrors scripts/cluster/precompute_clusters.py:_cache_dir — hidden_links is
    trial-invariant, so its layout drops the t<trial> segment.
    """
    leaf = f"k{cluster_size}_c{cluster_num}"
    if task == "hidden_links":
        return os.path.join(cache_root, dataset, task, leaf)
    return os.path.join(cache_root, dataset, task, f"t{trial}", leaf)


def load_cached_clusters(args, feats):
    """Load a precomputed k-anonymity clustering from the shared cache.

    Returns exactly the contract cluster_feats produces, so the rest of the CGT
    pipeline (GPT training, generation, QuantizedDataset) is unaffected:
        cluster_ids:     (N+1,) LongTensor  (trailing empty_id = args.cluster_num).
        cluster_centers: (K+1, d) FloatTensor (L2-normed centers + trailing zero row).

    Fails loud if the cache entry is missing (no DONE) or its meta.json disagrees
    with this run — never falls back to on-the-fly clustering, so CGT and the BiGG
    run it is compared against always read the identical partition.
    """
    cache_root = args.cache_root
    if not os.path.isabs(cache_root):
        # Anchor the relative default to the project root (CGT/generator/ -> ../..),
        # so the cache resolves regardless of the launch cwd.
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        cache_root = os.path.join(project_root, cache_root)

    cd = _cache_dir(cache_root, args.dataset, args.task,
                    args.trial_id, args.cluster_size, args.cluster_num)

    if not os.path.isfile(os.path.join(cd, "DONE")):
        raise FileNotFoundError(
            f"[CGT] cluster cache missing (no DONE) at: {cd}\n"
            f"Materialize it first (or the config may be infeasible: "
            f"cluster_size*cluster_num must be <= fit_set size):\n"
            f"  python scripts/cluster/precompute_clusters.py "
            f"--datasets {args.dataset} --cluster-sizes {args.cluster_size} "
            f"--cluster-nums {args.cluster_num} --tasks {args.task}"
            + ("" if args.task == "hidden_links" else f" --trials {args.trial_id}"))

    with open(os.path.join(cd, "meta.json")) as f:
        meta = json.load(f)
    if meta["feat_dim"] != feats.shape[1]:
        raise ValueError(
            f"[CGT] cache feat_dim={meta['feat_dim']} != "
            f"feats.shape[1]={feats.shape[1]} at {cd}")
    if meta["cluster_size"] != args.cluster_size or meta["cluster_num"] != args.cluster_num:
        raise ValueError(
            f"[CGT] cache size/num {(meta['cluster_size'], meta['cluster_num'])} "
            f"!= args {(args.cluster_size, args.cluster_num)} at {cd}")

    ids_np = torch.load(os.path.join(cd, "cluster_ids.pt")).numpy()
    centers_np = torch.load(os.path.join(cd, "l2_centers.pt")).numpy()
    if centers_np.shape[0] != args.cluster_num:
        raise ValueError(
            f"[CGT] cache l2_centers has {centers_np.shape[0]} rows, "
            f"expected cluster_num={args.cluster_num} at {cd}")
    if ids_np.shape[0] != feats.shape[0]:
        raise ValueError(
            f"[CGT] cache cluster_ids has {ids_np.shape[0]} entries, "
            f"expected N={feats.shape[0]} at {cd}")

    # Re-add the empty_id padding cluster_feats appends (cluster.py L2-norm block
    # above): trailing empty_id on the ids, trailing zero row on the centers.
    cluster_ids = torch.LongTensor(np.append(ids_np, args.cluster_num))
    cluster_centers = torch.FloatTensor(
        np.concatenate([centers_np, np.zeros((1, centers_np.shape[1]))], axis=0))
    print(f"[CGT] loaded cluster cache: {cd}")
    return cluster_ids, cluster_centers

