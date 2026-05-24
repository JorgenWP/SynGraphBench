---
name: evaluation-framework
description: GADBench downstream evaluation framework — anomaly detection, link prediction extension, key design decisions, and important files.
---

# SynGraphBench — Evaluation Framework (GADBench)

GADBench serves as the **downstream evaluation framework** for both generative paradigms. It supports two distinct tasks.

* **Environment:** `GADBench` Conda environment (Python 3.10, PyTorch 1.13.1, DGL).

---

## Anomaly Detection (native)

The original GADBench capability. Trains GNN-based classifiers to identify anomalous nodes. Supports 25+ models (GCN, GIN, BWGNN, etc.) across fully-supervised, semi-supervised, and inductive settings. This is the primary benchmark task for comparing real vs. synthetic data utility.

* `GADBench/benchmark.py`: Native anomaly detection benchmark.
* `GADBench/random_search.py`: Hyperparameter tuning.
* `scripts/benchmark/anomaly_benchmark.py`: Project-level anomaly detection benchmark comparing original vs. synthetic data. Supports multi-trial CGT evaluation: when per-trial `.pt` files exist (`{stem}_t0.pt` through `{stem}_t9.pt`), each trial loads a different file with its own train/val split, matching the split-varying behaviour of the original-data baseline.
* `GADBench/models/anomaly_detection/cgt_detector.py`: `CompGraphDetector` (GCN/GIN/GraphSAGE on batched computation-graph trees, root-only prediction) plus tree-model detectors `CGTXGBoostDetector` (XGBoost on root row) and `CGTXGBGraphDetector` (XGBoost on `GIN_noparam`-aggregated root embedding). Tree-model dispatch lives in `scripts/benchmark/anomaly_benchmark.py::CGT_TREE_MODELS`; `XGBGraph` asserts `num_layers == cg_depth` against the .pt. Hyperparam grids reused from `GADBench/utils.py` `param_space['XGBoost' | 'XGBGraph']`.
* `scripts/benchmark/models/cross_graph_detector.py`: Cross-graph detectors (GNN + XGBGraph) — train on synthetic, test on original.
* **Hyperparameters:** `CGT/args.py` or `GADBench/benchmark.py`.
* **Label leakage prevention:** BiGG's `--mask_test_labels` flag excludes test node labels (split 0) from label loss during training. Without this, BiGG can memorize test labels, artificially inflating anomaly detection scores when synthetic-trained GNNs are tested on original test nodes.
* **BiGG trial_id auto-inference:** For `--synthetic_type graph` (BiGG) runs, both `anomaly_benchmark.py` and `link_benchmark.py` parse `_loadsub_..._split{N}_n` out of `--synthetic_name` and set `--trial_id=N` automatically (see `bench_utils.apply_inferred_trial_id`). The wrapper scripts `run_anomaly_benchmark.sh` / `run_link_benchmark.sh` therefore expose no `trial_id` positional. Reason: BiGG's training subsample for split N excludes only split N's test nodes, so a mismatched benchmark split silently leaks training data. An explicit `--trial_id` that disagrees with the parsed value raises `ValueError`. CGT runs are untouched — auto-inference no-ops when `synthetic_type != 'graph'`.
* **BiGG split-bundle mode (anomaly only):** If `--synthetic_name` resolves to a directory whose immediate sub-directories are per-split BiGG variants (each contains `subgraph_*` files and has `split{N}` in its name), `anomaly_benchmark.py` switches to bundle mode via `bench_utils.discover_bigg_split_bundle`. `--trials` is overridden to the number of variants; Phase 1 rotates the original-data baseline through the bundle's split ids; each Phase 2 trial loads its own variant + matching synthetic-split column. The real-subsampled-graph source is built per variant the same way. Per-variant caches (`{variant}_combined_v2`, `{variant}_real_sub_combined_v3`, `_norm_stats.pt`) are written inside the bundle dir so re-bundling doesn't invalidate them, and so single-variant runs against the same variant outside the bundle don't share caches. `--trial_id != 0` and any `--trials` mismatch raise `ValueError`. In bundle mode, `--seeds_per_split` (default 3) repeats each split with the same `SEED_LIST[:seeds_per_split]` seeds — total runs = `#splits × seeds_per_split`. Reusing the same seeds across every split decouples seed and split variance (the legacy one-seed-per-trial rotation conflated them). Use this when each split has independently-tuned hyper-parameters (so variant stems differ beyond the split tag) — for fixed-hparam splits you can still loop the single-variant call. Link prediction is **not** wired up yet.
* **Per-trial AUPRC dump + tune-mask split (anomaly only):** Four flags on `scripts/benchmark/anomaly_benchmark.py` support the BO tuning framework:
  * `--dump_per_trial` — writes `per_trial_results.csv` next to `evaluation_results.csv`, one row per (source, dataset, model, split_id, seed) with raw AUROC/AUPRC/RecK/time_sec. Aggregated mean/std rows are still written as before.
  * `--tune_test_ratio FLOAT` — if set in (0,1), restricts the per-split test mask to an anomaly-stratified subsample of that fraction.
  * `--tune_test_seed INT` — seed for the stratified split (fixed across all trials of a study so BO and final report use disjoint nodes).
  * `--tune_portion {tune,heldout}` — selects which half. BO selection uses `tune`; the final report on best HPs uses `heldout`.
  Implemented via `_restrict_test_mask` (calls `sklearn.model_selection.train_test_split(..., stratify=label)` and overwrites `data.graph.ndata['test_mask']` in-place) at the two `data.split(...)` sites in `evaluate_models` (`:135`) and `evaluate_models_cross_graph` (`:494`). `GADBench/utils.py` is **not** modified; the override is local to the benchmark to keep upstream intact. Stratification falls back to unstratified when a class has <2 samples in the test set.
* **Phase 1 caching (BO):** Two flags let the BO coordinator skip the deterministic original-data baseline on every trial. `--skip_original` gates Phase 1 entirely. `--baseline_split_ids "0,1,2,3,4"` makes Phase 1 iterate the given split ids when no bundle is supplied (used by the coordinator's one-shot baseline build at study startup); auto-sets `--trials` to `len(split_ids)`. The coordinator caches the resulting baseline CSV under `experiments/bo_tuning/{dataset}/baselines/{study_version}.csv` and merges it back in before scoring each trial. Without these flags the benchmark behaves exactly as before.

---

## Link Prediction (extension)

An extension added to this project that reuses existing GNN architectures for edge existence prediction. Rather than predicting node labels, the GNNs produce node embeddings (via an `output_emb=True` flag), which are then scored pairwise by a lightweight edge decoder.

**Design principle:** Minimal new code — existing GNN models are reused unchanged by toggling `output_emb=True`, and only a thin `BaseGNNLinkPredictor` wrapper is added on top.

* `GADBench/link_benchmark.py`: Link prediction benchmark (epochs, patience hyperparameters here).
* `GADBench/link_utils.py`: `LinkDataset` — edge splitting, negative sampling, model registry.
* `GADBench/models/link_prediction/link_predictor.py`: `BaseGNNLinkPredictor` and `XGBGraphLinkPredictor` — edge decoder and training loop. `MLPDecoder` exposes `forward(h, edges)` for full-graph LP and `score_from_pair(h_u, h_v)` for merged-CG LP.
* `GADBench/models/link_prediction/cgt_link_predictor.py`: `MergedCompGraphLinkPredictor` — paper-style merged-endpoints link prediction. For each edge (u,v), trees rooted at u and v are merged via a root-root edge into a single graph, a GNN forward pass produces h_u and h_v from the two root positions, and the decoder scores the pair. Preserves the joint-computation property of full-graph LP within a small merged subgraph. Exported as `CompGraphLinkPredictor` for back-compat; supports GCN/GIN/GraphSAGE and warns when `num_layers != step_num` (receptive field should match merged-tree depth). Also defines tree-model LP detectors: `CGTXGBoostLinkPredictor` (raw root features → Hadamard → XGBoost; CGT-only, no full-graph LP baseline) and `CGTXGBGraphLinkPredictor(CGTXGBoostLinkPredictor)` which adds `GIN_noparam` over the merged tree and enforces `num_layers == step_num`. Both dispatched via `CGT_LP_TREE_MODELS` in `scripts/benchmark/link_benchmark.py`; `CGT_LP_SUPPORTED_MODELS = SUPPORTED_MODELS + ['XGBoost']` extends the model filter on the CGT path. Hyperparam grids reused from `GADBench/utils.py` `param_space['XGBoost' | 'XGBGraph']`.
* `scripts/benchmark/models/cross_graph_link_predictor.py`: Cross-graph link predictors (GNN + XGBGraph) — train on synthetic edges, test on original edges.

### Key Design Details

**Edge splitting with connectivity preservation:** Edges are split into train/val/test sets, but a minimum spanning tree (via NetworkX) is computed first. Spanning-tree edges are never moved to val/test splits — this guarantees the training graph remains connected, which is critical for GNN message passing.

**Negative sampling:**
* `random` — uniform sampling with collision detection (vectorized hashing, 10-attempt retry per sample).
* `hard` — 2-hop random walks to generate structurally plausible negatives (harder for the model to distinguish).
* Val/test negatives are fixed at dataset creation for reproducibility. Training negatives are resampled each epoch to prevent overfitting to specific negatives.

**Edge decoders:**
* `dot` — simple dot product `(h[u] * h[v]).sum(dim=-1)`. No extra parameters; relies entirely on the GNN embedding quality.
* `mlp` — learnable scoring on Hadamard product: `Linear(h) → ReLU → Dropout → Linear(1)`. Adds capacity at the cost of extra parameters.

**Metrics:** AUROC, AUPRC, and Recall@K (where K = number of positive test edges).

**Merged computation-graph data utilities** (`GADBench/data/comp_graph.py`):
* `compute_merged_tree_adj(step_num, sample_num)`: two disjoint trees of size T joined by a bidirectional root-root edge → `[2T, 2T]` adjacency. Root_u at index 0, root_v at index T.
* `MergedOriginalCompGraphDataset(adj_list, features, edges, ...)`: lazily samples a tree for each endpoint of every edge and concatenates their features `[2T, D]`. Tree sampling uses `adj_list`, which the predictor builds from `data.train_graph` (no test/val edge leakage).
* `make_merged_comp_graph_collate`, `extract_edge_root_embeddings`: batch offsets + (h_u, h_v) extraction from the two root positions.

**CGT edge masking (`hidden_links`)**: for `task=hidden_links`, `CGT/train.py` calls `load_dgl_graph_with_hidden_links(trial_id, val_ratio, test_ratio)` in `CGT/task/utils/utils.py`, which mirrors `LinkDataset.split(trial_id)` byte-for-byte (seed `3407 + trial_id*10`, MST-protected split, first `int(E*test_ratio)` candidates → test). Test edges are stripped from adjacency; val edges remain (analogous to `mask_test_labels`). The withheld `hidden_test_edges` plus `trial_id` and `task` are saved in the `.pt` and asserted downstream via `_assert_link_pt_alignment` in `scripts/benchmark/bench_utils.py`.

**Per-trial CGT .pt resolution**: `scripts/benchmark/link_benchmark.py` now resolves `{stem}/{stem}_t{t}.pt` for `t in 0..trials-1` (via `resolve_cgt_trial_paths`). If all files exist, each trial loads its own .pt and builds its own synthetic graph. Missing files trigger single-file fallback with a warning. The original-cg baseline does not depend on .pt contents and reuses the same graph across trials.

---

## Feature Normalization in Cross-Graph Evaluation

When BiGG trains with feature normalization (e.g., `-normalize zscore`), the synthetic graph's features are in normalized space. The benchmarks must apply the same transform to the original graph's features before testing, otherwise the model trains on one feature distribution and tests on another.

* `bigg/bigg/extension/preprocessing.py`: `normalize_features()` returns `(features, stats)` — stats dict contains the method and parameters (e.g., mean/std for zscore, sorted_values for quantile/cdf). `apply_normalization()` applies saved stats to new data. `invert_normalization()` reverses lossless normalizations (zscore, minmax, quantile, cdf; errors on row).
* Normalization methods: `zscore` (zero mean, unit variance), `minmax` ([0,1] scaling), `row` (L2 row norm, lossy), `quantile` (rank-based inverse normal transform — any distribution to N(0,1)), `cdf` (empirical CDF — any distribution to Uniform[0,1]; couples to a sigmoid+BCE continuous head in the model).
* `bigg/bigg/extension/pipeline.py`: Saves `_norm_stats.pt` alongside the synthetic graph. When `--binary_feat` is enabled, also saves `_binary_idx.pt`.
* `scripts/benchmark/bench_utils.py`: `apply_normalization()` duplicated here for benchmark imports. When stats contain `binary_idx`, only non-binary columns are normalized.
* Both anomaly and link prediction benchmarks: load stats if the `_norm_stats.pt` file exists, apply to original graph features before cross-graph evaluation. If no stats file exists (non-normalized runs), features are used as-is.

This only affects the BiGG full-graph path. CGT handles its own L2 normalization internally (`build_cgt_datasets` in `bench_utils.py`).
