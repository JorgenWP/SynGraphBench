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

---

## Link Prediction (extension)

An extension added to this project that reuses existing GNN architectures for edge existence prediction. Rather than predicting node labels, the GNNs produce node embeddings (via an `output_emb=True` flag), which are then scored pairwise by a lightweight edge decoder.

**Design principle:** Minimal new code — existing GNN models are reused unchanged by toggling `output_emb=True`, and only a thin `BaseGNNLinkPredictor` wrapper is added on top.

* `GADBench/link_benchmark.py`: Link prediction benchmark (epochs, patience hyperparameters here).
* `GADBench/link_utils.py`: `LinkDataset` — edge splitting, negative sampling, model registry.
* `GADBench/models/link_prediction/link_predictor.py`: `BaseGNNLinkPredictor`, `XGBoostLinkPredictor` (raw node features → Hadamard → XGBoost), and `XGBGraphLinkPredictor` (`GIN_noparam` → Hadamard → XGBoost). Sister classes inheriting from `BaseDetector`. `MLPDecoder` exposes `forward(h, edges)` for full-graph LP and `score_from_pair(h_u, h_v)` for merged-CG LP.
* `GADBench/models/link_prediction/cgt_link_predictor.py`: `MergedCompGraphLinkPredictor` — paper-style merged-endpoints link prediction. For each edge (u,v), trees rooted at u and v are merged via a root-root edge into a single graph, a GNN forward pass produces h_u and h_v from the two root positions, and the decoder scores the pair. Preserves the joint-computation property of full-graph LP within a small merged subgraph. Exported as `CompGraphLinkPredictor` for back-compat; supports GCN/GIN/GraphSAGE and warns when `num_layers != step_num` (receptive field should match merged-tree depth). Also defines tree-model LP detectors: `CGTXGBoostLinkPredictor` (raw root features → Hadamard → XGBoost) and `CGTXGBGraphLinkPredictor(CGTXGBoostLinkPredictor)` which adds `GIN_noparam` over the merged tree and enforces `num_layers == step_num`. Both dispatched via `CGT_LP_TREE_MODELS` in `scripts/benchmark/link_benchmark.py`; `SUPPORTED_MODELS` (and `CGT_LP_SUPPORTED_MODELS`) include both `XGBoost` and `XGBGraph`. Hyperparam grids reused from `GADBench/utils.py` `param_space['XGBoost' | 'XGBGraph']`.
* `scripts/benchmark/models/cross_graph_link_predictor.py`: Cross-graph link predictors (GNN + XGBoost + XGBGraph) — train on synthetic edges, test on original edges. `CrossGraphXGBoostLinkPredictor` uses raw node features per graph; `CrossGraphXGBGraphLinkPredictor` applies `GIN_noparam` per graph first.

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

**Val/test divergence curves**: every per-epoch GNN detector emits `val_auprc_curve` / `test_auprc_curve` on its returned `test_score`; both benchmark mains aggregate them into a per-(source, model, epoch) CSV alongside the scalar results — `divergence_curves.csv` for anomaly, `link_divergence_curves.csv` for link prediction (suffixed `__{phase_tag}_{eval_mode}` when array-sharded). Sources include `original`, `synthetic-graph`, `real-subsampled-graph` (anomaly only), `original-cg`, `original-cg-quantized`, and `synthetic-cgt`. XGB detectors stay no-curve (single-fit). Plotted via `scripts/benchmark/divergence_curves.ipynb` — set `TASK = 'link_prediction'` to point it at the link output layout.

**Quantization-isolating baseline (`original-cg-quantized`)**: third CG row in both `evaluate_models_cgt` (`anomaly_benchmark.py`) and `evaluate_link_models_cgt` (`link_benchmark.py`). Identical to `original-cg` except train/val tree-node features are replaced by their *assigned* cluster centers (`cluster_centers[cluster_ids[node]]`), reusing the exact vocabulary CGT was trained on. Built by `build_original_cg_quantized_datasets` (anomaly, full tree walk quantized) and `build_quantized_dgl_graph` (link, only train/val *root* features replaced — mirrors `build_synthetic_dgl_graph`'s scope). Decomposes the apparent synthetic-vs-real gap into two effects: `(original-cg → original-cg-quantized)` = pure K-quantization/denoising; `(original-cg-quantized → synthetic-cgt)` = pure CGT generation contribution. Requires `cluster_ids` in the `.pt` (added at `CGT/generator/gpt/gpt.py:train_and_generate` return); old artifacts hard-fail with a retrain message via `_load_cluster_ids`. The link benchmark's `--eval_mode` accepts `original_cg_quantized` to run only this row; `both` runs all three.

---

## Feature Normalization in Cross-Graph Evaluation

When BiGG trains with feature normalization (e.g., `-normalize zscore`), the synthetic graph's features are in normalized space. The benchmarks must apply the same transform to the original graph's features before testing, otherwise the model trains on one feature distribution and tests on another.

* `bigg/bigg/extension/preprocessing.py`: `normalize_features()` returns `(features, stats)` — stats dict contains the method and parameters (e.g., mean/std for zscore, sorted_values for quantile/cdf). `apply_normalization()` applies saved stats to new data. `invert_normalization()` reverses lossless normalizations (zscore, minmax, quantile, cdf; errors on row).
* Normalization methods: `zscore` (zero mean, unit variance), `minmax` ([0,1] scaling), `row` (L2 row norm, lossy), `quantile` (rank-based inverse normal transform — any distribution to N(0,1)), `cdf` (empirical CDF — any distribution to Uniform[0,1]; couples to a sigmoid+BCE continuous head in the model).
* `bigg/bigg/extension/pipeline.py`: Saves `_norm_stats.pt` alongside the synthetic graph. When `--binary_feat` is enabled, also saves `_binary_idx.pt`.
* `scripts/benchmark/bench_utils.py`: `apply_normalization()` duplicated here for benchmark imports. When stats contain `binary_idx`, only non-binary columns are normalized.
* Both anomaly and link prediction benchmarks: load stats if the `_norm_stats.pt` file exists, apply to original graph features before cross-graph evaluation. If no stats file exists (non-normalized runs), features are used as-is.

This only affects the BiGG full-graph path. The CGT pipeline enforces one invariant — every feature the GNN consumes has unit L2 norm — in three explicit places: (1) `CGT/task/utils/utils.py` L2-normalizes the real feature matrix before k-means and GPT training; (2) `cluster_feats` in `CGT/generator/cluster.py` row-normalizes the cluster centres after the size-repair step so saved `.pt` files contain exactly-unit-norm centres (the appended `empty_id` row stays at zero norm intentionally); (3) the eval framework re-normalizes real features at test time — anomaly detection inside `build_cgt_datasets` / `build_original_cg_datasets` / `build_original_cg_quantized_datasets`, and link prediction once at the top of `evaluate_link_models_cgt` (in `scripts/benchmark/link_benchmark.py`), so the original-CG baseline, the quantized hybrid graph from `build_quantized_dgl_graph`, and the synthetic-CGT hybrid graph from `build_synthetic_dgl_graph` all see unit-norm features.
