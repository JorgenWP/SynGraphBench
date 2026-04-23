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
* `scripts/benchmark/anomaly_benchmark.py`: Project-level anomaly detection benchmark comparing original vs. synthetic data. Supports multi-trial CGT evaluation: when per-trial `.pt` files exist (`{stem}_t0.pt` through `{stem}_t9.pt`), each trial loads a different file with its own train/val split, matching the split-varying behaviour of the original-data baseline. For BiGG subsampled runs, additionally evaluates a `real-subsampled-graph` source that trains on the real forest-fire subsamples saved under `{syn_path}/training_subsamples/` and tests on the full original graph — sharing subsampling + cross-graph testing with the synthetic path so the delta isolates generative-model fidelity from the subsampling effect.
* `scripts/benchmark/models/cross_graph_detector.py`: Cross-graph detectors (GNN + XGBGraph) — train on a source graph (synthetic or real-subsampled), test on original.
* **Hyperparameters:** `CGT/args.py` or `GADBench/benchmark.py`.
* **Label leakage prevention:** BiGG's `--mask_test_labels` flag excludes test node labels (split 0) from label loss during training. Without this, BiGG can memorize test labels, artificially inflating anomaly detection scores when synthetic-trained GNNs are tested on original test nodes.

---

## Link Prediction (extension)

An extension added to this project that reuses existing GNN architectures for edge existence prediction. Rather than predicting node labels, the GNNs produce node embeddings (via an `output_emb=True` flag), which are then scored pairwise by a lightweight edge decoder.

**Design principle:** Minimal new code — existing GNN models are reused unchanged by toggling `output_emb=True`, and only a thin `BaseGNNLinkPredictor` wrapper is added on top.

* `GADBench/link_benchmark.py`: Link prediction benchmark (epochs, patience hyperparameters here).
* `GADBench/link_utils.py`: `LinkDataset` — edge splitting, negative sampling, model registry.
* `GADBench/models/link_prediction/link_predictor.py`: `BaseGNNLinkPredictor` and `XGBGraphLinkPredictor` — edge decoder and training loop (decoder architecture hyperparameters here).
* `GADBench/models/link_prediction/cgt_link_predictor.py`: CGT computation-graph link prediction.
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

---

## Feature Normalization in Cross-Graph Evaluation

When BiGG trains with feature normalization (e.g., `-normalize zscore`), the synthetic graph's features are in normalized space. The benchmarks must apply the same transform to the original graph's features before testing, otherwise the model trains on one feature distribution and tests on another.

* `bigg/bigg/extension/preprocessing.py`: `normalize_features()` returns `(features, stats)` — stats dict contains the method and parameters (e.g., mean/std for zscore, sorted_values for quantile/cdf). `apply_normalization()` applies saved stats to new data. `invert_normalization()` reverses lossless normalizations (zscore, minmax, quantile, cdf; errors on row).
* Normalization methods: `zscore` (zero mean, unit variance), `minmax` ([0,1] scaling), `row` (L2 row norm, lossy), `quantile` (rank-based inverse normal transform — any distribution to N(0,1)), `cdf` (empirical CDF — any distribution to Uniform[0,1]; couples to a sigmoid+BCE continuous head in the model).
* `bigg/bigg/extension/pipeline.py`: Saves `_norm_stats.pt` alongside the synthetic graph. When `--binary_feat` is enabled, also saves `_binary_idx.pt`.
* `scripts/benchmark/bench_utils.py`: `apply_normalization()` duplicated here for benchmark imports. When stats contain `binary_idx`, only non-binary columns are normalized.
* Both anomaly and link prediction benchmarks: load stats if the `_norm_stats.pt` file exists, apply to original graph features before cross-graph evaluation. If no stats file exists (non-normalized runs), features are used as-is.

This only affects the BiGG full-graph path. CGT handles its own L2 normalization internally (`build_cgt_datasets` in `bench_utils.py`).
