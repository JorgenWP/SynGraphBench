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
* `scripts/benchmark/anomaly_benchmark.py`: Project-level anomaly detection benchmark comparing original vs. synthetic data.
* **Hyperparameters:** `CGT/args.py` or `GADBench/benchmark.py`.

---

## Link Prediction (extension)

An extension added to this project that reuses existing GNN architectures for edge existence prediction. Rather than predicting node labels, the GNNs produce node embeddings (via an `output_emb=True` flag), which are then scored pairwise by a lightweight edge decoder.

**Design principle:** Minimal new code — existing GNN models are reused unchanged by toggling `output_emb=True`, and only a thin `BaseGNNLinkPredictor` wrapper is added on top.

* `GADBench/link_benchmark.py`: Link prediction benchmark (epochs, patience hyperparameters here).
* `GADBench/link_utils.py`: `LinkDataset` — edge splitting, negative sampling, model registry.
* `GADBench/models/link_prediction/link_predictor.py`: `BaseGNNLinkPredictor` — edge decoder and training loop (decoder architecture hyperparameters here).
* `GADBench/models/link_prediction/cgt_link_predictor.py`: Placeholder for CGT-based link prediction (not yet implemented).

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
