---
name: generative-models
description: The two generative paradigms (BiGG and CGT), their mechanisms, output formats, key files, and Conda environments.
---

# SynGraphBench — Generative Models

## The Two Paradigms

A central conceptual distinction in this project is **how** synthetic data is produced. BiGG and CGT operate at fundamentally different levels of abstraction, which determines what they output, how that output is evaluated, and what kind of utility/privacy trade-off they represent.

**BiGG output** is a whole DGL graph → evaluated with standard whole-graph GNNs (`--synthetic_type graph`).
**CGT output** is a `.pt` file of cluster centers and sequence indices → evaluated with computation-graph GNNs (`--synthetic_type comp-graph`).
The two paradigms are **not directly comparable** on a common evaluation framework; the benchmark script handles each path separately.

---

## BiGG — Whole-Graph Generation

BiGG generates a **complete, new graph** — both topology (edges) and node features — from scratch. The output is a full DGL graph that is a drop-in replacement for the original. The original graph is discarded entirely during evaluation; downstream models train and are tested purely on the generated graph.

The generative model captures the joint distribution of graph structure and node attributes, and samples from it to produce a synthetic counterpart.

* **Key Mechanisms:** Decomposes graph generation into a sequence of binary tree decisions, processed efficiently via a custom C++ extension (`tree_clib`). Two modes: a conditional model (features + labels) and a structure-only baseline.
* **Output format:** A full DGL graph stored as a file under `datasets/synthetic/bigg/<dataset>/<task>/`.
* **Important Files:**
  * `bigg/extension/pipeline.py`: Conditional model training (features + labels).
  * `bigg/extension/pipeline_structure_only.py`: Structure-only baseline.
  * `bigg/data_process/`: Dataset preparation scripts.
* **Environment:** `bigg` Conda environment (Python 3.9, PyTorch 2.4.1).
  * **Agent Alert:** Requires compiling the `tree_clib` C++ extension via `make` before running. This is handled in `scripts/env_setups/bigg.sh`. Modern CUDA architectures are patched via `sed` in that script.

---

## CGT — Computation-Graph Generation

CGT operates at a different level. Rather than generating a new graph, it **generates synthetic node feature distributions** while leaving the original graph topology unchanged. The key idea is that GNNs aggregate information via local computation trees (computation graphs) rooted at each node. CGT learns to generate sequences of feature vectors that match the distribution of these computation trees.

Concretely, CGT uses **DP-k-means** to cluster the real node features into `k` cluster centers. These centers — not the raw features — are what is "synthesized" and shared. During evaluation, the original graph's training and validation node features are replaced by the nearest cluster center, and a GNN is trained on this feature-masked graph and tested on the unmasked original test nodes.

* **Key Mechanisms:** Operates on minibatches of computation graph sequences (not the full graph). DP-k-means provides differential privacy. A Transformer learns to generate realistic sequences of cluster assignments.
* **Output format:** A `.pt` file containing cluster centers, generated sequence indices, and train/val/test node ID mappings. Stored under `datasets/synthetic/cgt/<dataset>/<task>/`.
* **Important Files:**
  * `CGT/train.py`: Training script.
  * `CGT/test.py`: Generation and evaluation script.
  * `CGT/args.py`: Hyperparameter configurations.
* **Environment:** `CGT` Conda environment (Python 3.11). Setup via `scripts/env_setups/cgt_setup.sh`.
