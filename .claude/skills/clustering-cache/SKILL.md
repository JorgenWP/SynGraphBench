---
name: clustering-cache
description: The shared k-anonymity clustering cache produced by scripts/cluster/. Cache layout, file-to-consumer mapping, and adaptation snippets for wiring CGT, BiGG, and the benchmark framework to read precomputed clusters instead of fitting their own.
---

# SynGraphBench — Clustering Cache

A precomputed, on-disk artifact that holds the k-anonymity clustering for each `(dataset, task, trial, cluster_size, cluster_num)`. It exists so CGT and BiGG can be compared fairly: both consume the same `cluster_ids` (identical partition), each picks the per-cluster feature representation that fits its own normalization pipeline. The producer is `scripts/cluster/precompute_clusters.py` (see the `execution-flow` skill for how to run it); this skill is about the *consumer* side — adapting training and benchmark scripts to load the cache.

## Why the cache exists

- **Methodological**: CGT and BiGG must operate under the same k-anonymity constraint. Today CGT clusters internally (`cluster_feats` in `CGT/generator/cluster.py`) and BiGG does no clustering at all. Externalizing the partition makes the constraint an explicit, shared input.
- **Practical**: clustering is deterministic given (dataset, fit-set, k_size, k_num, seed), so fitting once and caching avoids redundant work across the CGT and BiGG sweeps.
- **Auditability**: every cache entry records the inputs (`meta.json`) so reviewers can confirm two compared runs read identical partitions.

## Cache layout

Asymmetric across tasks because `hidden_links` clustering is trial-invariant (see "Why trial-invariant" below):

```
cache/clustering/<dataset>/
├── hidden_labels/
│   └── t<trial>/
│       └── k<cluster_size>_c<cluster_num>/
│           ├── cluster_ids.pt
│           ├── raw_centers.pt
│           ├── l2_centers.pt
│           ├── meta.json
│           └── DONE
└── hidden_links/                       # NOTE: no t<trial> segment
    └── k<cluster_size>_c<cluster_num>/
        └── (same five files)
```

`DONE` is touched last by the producer; its presence is the resume / consumer-readiness marker. Consumers MUST fail loudly if `DONE` is missing — never silently fall back to fitting on the fly.

### Why trial-invariant for hidden_links

`split_node_ids_for_hidden_links` (`CGT/task/utils/utils.py:202-218`) returns `train = 80%`, `val = 20%`, `test = []` per trial — the *union* covers all nodes regardless of trial. Features come straight from `graph.ndata['feature']` and are not modified by the per-trial edge split. A feature-only K-means on `train + val` therefore produces identical output across trials, so one fit per `(dataset, cluster_size, cluster_num)` suffices. `hidden_labels` doesn't get this shortcut — `split_ids_from_dgl` reads `graph.ndata['train_masks'][:, trial_id]`, so train+val genuinely differs per trial.

## Cache path resolver

The same helper drops into every consumer:

```python
def cache_dir(root, dataset, task, trial, cluster_size, cluster_num):
    leaf = f"k{cluster_size}_c{cluster_num}"
    if task == "hidden_links":          # trial-invariant; no t<trial> segment
        return f"{root}/{dataset}/{task}/{leaf}"
    return f"{root}/{dataset}/{task}/t{trial}/{leaf}"
```

Default `root = cache/clustering/` (relative to project root). Validate `os.path.isfile(f"{cache_dir}/DONE")` before reading; raise with a clear "missing — run scripts/cluster/precompute_clusters.py for ..." message otherwise.

## File-to-consumer mapping

| File | Shape / dtype | Consumer | Purpose |
|---|---|---|---|
| `cluster_ids.pt` | `(N,)` long | **CGT and BiGG** | The shared partition — this is the fairness lock |
| `l2_centers.pt` | `(K, D)` float | **CGT only** | L2-normed cluster centers in original feature space; drop-in for the centers `cluster_feats` would have returned |
| `raw_centers.pt` | `(K, D)` float | **BiGG only** | Per-cluster mean of *pre-L2-norm* features over the **fit set (train+val) members only** (matches the population `l2_centers` represent); fed into BiGG's CDF/quantile normalization in place of each node's own feature |
| `meta.json` | JSON | both | Provenance + sanity check — verify `cluster_size`, `cluster_num`, `feat_dim`, `fit_set_size`, `seed` match consumer expectations |

`cluster_ids.pt` covers **all N nodes**, but the k-anonymity floor is enforced only on the **fit set (train+val)** — the population the generative model trains on. K-means is fit on train+val, the min-size-constrained assignment runs on train+val, and the remaining (e.g. test) nodes are assigned afterward by **unconstrained nearest-center** purely as neighbour / computation-graph context (they never dilute the floor). So `min cluster size ≥ cluster_size` is guaranteed on `cluster_ids[train+val]`, not on the full-N counts — see `meta.json`'s `cluster_stats.fit_min`. The trailing empty_id row that `cluster_feats` appends (`cluster_ids[-1] = cluster_num`, `cluster_centers[-1] = 0`) is NOT in the cache — consumers that need it must append it themselves after loading.

## Consumer adaptation patterns

### CGT — load the cache instead of clustering in-pipeline

**Status: implemented for CGT *training*.** The loader is `load_cached_clusters(args, feats)` in `CGT/generator/cluster.py` (sits next to `cluster_feats`, with a consumer-side copy of the `_cache_dir` resolver). It is wired into the `train_and_generate` path at `CGT/generator/gpt/gpt.py:147`:

```python
# CGT/generator/gpt/gpt.py — train_and_generate
cluster_ids, cluster_centers = load_cached_clusters(args, feats)
```

`load_cached_clusters` resolves the cache dir (anchoring a relative `--cache_root` to the project root via `__file__`), **fails loud** if `DONE` is missing (with a message naming the `precompute_clusters.py` command), validates `meta.json` (`feat_dim`, `cluster_size`, `cluster_num`) and tensor shapes, then re-adds the empty_id padding so it returns the exact `(N+1,)` / `(K+1, D)` contract `cluster_feats` produced. Everything downstream (`gpt.train`/`generate`, `QuantizedDataset` in `dataset.py:127`) is unchanged.

The selecting flag is `--cache_root` (added in `CGT/args.py`, default `cache/clustering`); `--cluster_size`, `--cluster_num`, `--trial_id`, `--task`, `--dataset` already exist. `train_cgt.sh` cds to the project root, so the relative default resolves without any shell/SLURM change.

**Not yet migrated:** the eval/benchmark `run()` path at `gpt.py:108` (the `fit_ids=None` call) still uses `cluster_feats`. Both functions remain importable; migrating `run()` and the benchmark/BiGG consumers is future work.

### BiGG — quantize input features before normalization

BiGG never clustered before. The natural insertion point is the feature-preprocessing path in `bigg/bigg/extension/preprocessing.py` (where `zscore` / `quantile` / `cdf` / `row` / `minmax` are applied at lines 262-349). Swap each node's feature for its cluster's `raw_centers` row *before* normalization runs.

```python
# in BiGG's preprocessing path, before the chosen normalize_features() call:
cd = cache_dir(cache_root, dataset, task, trial, cluster_size, cluster_num)
assert os.path.isfile(f"{cd}/DONE"), f"missing cluster cache at {cd}"
cluster_ids = torch.load(f"{cd}/cluster_ids.pt").numpy()
raw_centers = torch.load(f"{cd}/raw_centers.pt").numpy()
feats_quantized = raw_centers[cluster_ids]        # (N, D); only K distinct rows
# hand feats_quantized to the existing normalize step instead of raw features
norm_features, norm_stats = normalize_features(feats_quantized, method=cfg.normalize)
print(f"[BiGG] loaded cluster cache: {cd}")
```

After CDF/quantile normalization → BiGG training → inversion via `_invert_to_raw_space` (`scripts/benchmark/bench_utils.py:427-503`), the generated graph's per-node features will be near-cluster-center but not exactly so (BiGG's decoder is continuous — `torch.sigmoid(raw_cont)` for CDF mode at `customized_models.py:462-474`).

**Strict vs soft k-anonymity at generation time:**
- *Soft (default)*: accept the continuous output. K-anonymity is enforced at input only.
- *Strict*: after `_invert_to_raw_space`, snap each generated node's features to the nearest `raw_centers` row before saving the synthetic graph. Document the choice in the run's metadata.

### Benchmark — load cache instead of reading centers from synthetic .pt

**Status: implemented for both tasks** (anomaly `anomaly_benchmark.py`, link `link_benchmark.py`). The benchmark runs in the **GADBench** conda env and must NOT import `CGT/generator/cluster.py` (it imports `k_means_constrained`/`sklearn`). So the loader is duplicated into `scripts/benchmark/bench_utils.py` — `apply_cluster_cache(syn_data, original_graph, cache_root, dataset, task, cluster_num, cluster_size, role)` plus `_cache_dir` and `parse_cluster_kc` — using only `json`/`os`/`re`/`numpy`/`torch`.

The benchmark builds cluster features from `syn_data['cluster_ids']` / `syn_data['cluster_centers']` in five paths; `apply_cluster_cache` mutates `syn_data` in place at each so they read the cache instead of the `.pt` payload, keyed by `(dataset, task, trial, cluster_size, cluster_num)`:

- **`role="quantized"`** — the cache is authoritative. Used by `build_original_cg_quantized_datasets` (AD) and `build_quantized_dgl_graph` (LP). Pins the quantization-only baseline to the same partition CGT/BiGG are compared against.
- **`role="synthetic"`** — the `.pt`'s `gen_*_ids` index the codebook the GPT trained on, so the cache codebook MUST equal the `.pt`'s (training already read the cache). `apply_cluster_cache` **asserts** `np.allclose(centers)` / `np.array_equal(ids)` and fails loud on mismatch (stale `.pt`). Used by `build_cgt_datasets`, `make_cgt_rebuild_fn` (AD), `build_synthetic_dgl_graph` (LP).

Key details: the cache stores `(N,)` ids and `(K,D)` `l2_centers`; the loader re-adds the trailing empty_id padding to `(N+1,)` / `(K+1,D)` (`build_synthetic_dgl_graph` reads `empty_id = cluster_centers.shape[0]-1`). Cache trial = `syn_data['trial_id']` for `hidden_labels` (fail loud if absent); `None` for `hidden_links`. `cluster_num`/`cluster_size` are parsed from the variant stem via `parse_cluster_kc` — **note the inversion**: stem `_k<cluster_num>_c<cluster_size>` vs cache leaf `k<cluster_size>_c<cluster_num>`.

**`original_cg` (full real-feature baseline) is intentionally NOT cache-substituted** — it is the richer-feature reference anchoring the ablation ladder (`original-cg → original-cg-quantized` = quantization cost; `original-cg-quantized → synthetic-cgt` = generation cost). The matched k-anonymity baseline is `original_cg_quantized`, which reads the cache. `eval_mode=original_cg`-only runs never touch the cache.

## CLI conventions for adapted scripts

When a train or benchmark script grows the ability to read the cache, the flags should be:

| Flag | Required | Notes |
|---|---|---|
| `--cluster-size` | yes | k-anonymity floor; together with `--cluster-num` selects the cache leaf |
| `--cluster-num` | yes | Number of clusters; pairs with `--cluster-size` |
| `--cache-root` | no | Default `cache/clustering`. Override only for testing alternate caches. |
| `--trial-id` / `--trial_id` | already exists | Used to resolve `t<trial>/` for `hidden_labels`; ignored for `hidden_links` |

CGT uses underscore-style flags, so the names landed as `--cache_root`, `--cluster_size`, `--cluster_num`, `--trial_id` (all already existed except `--cache_root`).

Don't add `--cluster-sample-num` or other K-means hyperparams to the consumer side — those are baked into the cache. Mismatch between consumer expectations and `meta.json` is a fail-loud condition. Note for CGT specifically: the producer fits on the full fit-set (forces `cluster_sample_num = fit_set_size`), so once training reads the cache, `--cluster_sample_num` no longer affects the partition (it still appears in the synthetic variant name).

## Methodology guard-rails

When adapting scripts, enforce these or the comparison silently breaks:

1. **Same cache, both models.** A CGT run and the BiGG run it's compared against must point at the same `(dataset, task, trial, cluster_size, cluster_num)` cache entry. Surface the resolved cache directory in every run's log (`[CGT] loaded cluster cache: ...`, `[BiGG] loaded cluster cache: ...`) so reviewers can confirm by grep.
2. **Same cache, matched baseline.** The matched k-anonymity baseline is `original_cg_quantized` (it reads `l2_centers`/`cluster_ids` from the cache), NOT a re-featured `original_cg`. `original_cg` deliberately stays in the full real-feature space as the richer-feature reference: `original-cg → original-cg-quantized` measures quantization cost, `original-cg-quantized → synthetic-cgt` measures generation cost. Keep both, and ensure the quantized baseline points at the same cache entry the synthetic run was trained against.
3. **Fail-loud on missing `DONE`.** Never compute clustering on the fly as a fallback when the cache is missing — re-running `scripts/cluster/precompute_clusters.py` is the only path to materialize an entry, and it should be a deliberate, logged act.
4. **No per-run reclustering of the same key.** If two scripts in the same comparison both decide to "just refit because the cache is stale," they will diverge (different random init, different repair tie-breaks). Refit once via the producer, write the cache, then read.

## Producer-side reference

For how to build the cache (CLI flags, SLURM template, resume semantics), see the "Clustering Precompute" subsection in the `execution-flow` skill. For the on-disk layout in the broader project tree, see `project-structure`. For the upstream CGT clustering details (PCA, KMeansConstrained, L2-norm step), see `generative-models`.
