---
name: project-overview
description: A comprehensive overview of the SynGraphBench project, detailing its structure, core components, and execution flow for developers and agents.
---

# SynGraphBench - Developer & Agent Skill Guide

## 1. Project Overview

**SynGraphBench** is a benchmarking suite designed to evaluate the trade-off between data utility and privacy in synthetic graph generation. It measures how well synthetic graph data performs compared to real data on downstream machine learning tasks, and analyzes how strict privacy guarantees (like k-anonymity) degrade this performance.

The project is an amalgamation of three distinct, previously published research repositories, orchestrated together via custom shell scripts to form a complete benchmarking pipeline.

## 2. The Two Generative Paradigms

A central conceptual distinction in this project is **how** synthetic data is produced. The two generative models (CGT and BiGG) operate at fundamentally different levels of abstraction, which determines what they output, how that output is evaluated, and what kind of utility/privacy trade-off they represent.

### Whole-Graph Generation (BiGG)

BiGG generates a **complete, new graph** — both topology (edges) and node features — from scratch. The output is a full DGL graph that is a drop-in replacement for the original. The original graph is discarded entirely during evaluation; downstream models train and are tested purely on the generated graph.

This is the more intuitive paradigm: the generative model captures the joint distribution of graph structure and node attributes, and samples from it to produce a synthetic counterpart.

### Computation-Graph Generation (CGT)

CGT operates at a different level. Rather than generating a new graph, it **generates synthetic node feature distributions** while leaving the original graph topology unchanged. The key idea is that GNNs aggregate information via local computation trees (computation graphs) rooted at each node. CGT learns to generate sequences of feature vectors that match the distribution of these computation trees, rather than the full graph.

Concretely, CGT uses **DP-k-means** to cluster the real node features into `k` cluster centers. These centers — not the raw features — are what is "synthesized" and shared. During evaluation, the original graph's training and validation node features are replaced by the nearest cluster center, and a GNN is trained on this feature-masked graph and tested on the unmasked original test nodes.

**Why this matters for the benchmark:**
* BiGG output is a whole DGL graph → evaluated with standard whole-graph GNNs (`--synthetic_type graph`).
* CGT output is a `.pt` file of cluster centers and sequence indices → evaluated with computation-graph GNNs that can ingest these sequences directly (`--synthetic_type comp-graph`).
* The two paradigms are not directly comparable on a common evaluation framework; the benchmark script handles each path separately.

## 3. Core Sub-Repositories

The repository integrates three independent sub-systems. **Note for Agents:** Because these are distinct research projects, they have conflicting dependencies. *Always ensure you are using the correct Conda environment for the specific sub-repo you are interacting with.*

### A. CGT (Computation Graph Transformer)

* **Purpose:** A generative model that synthesizes node feature distributions for large-scale graphs using a Transformer over computation-graph sequences. Designed for privacy-preserving feature synthesis, not topology generation.
* **Key Mechanisms:** Operates on minibatches of computation graph sequences (not the full graph). DP-k-means clusters real features into `k` centers, providing differential privacy. The Transformer learns to generate realistic sequences of these cluster assignments.
* **Output format:** A `.pt` file containing cluster centers, generated sequence indices, and train/val/test node ID mappings. Stored under `datasets/synthetic/cgt/<dataset>/`.
* **Important Files:**
  * `CGT/train.py`: Training script.
  * `CGT/test.py`: Generation and evaluation script.
  * `CGT/args.py`: Hyperparameter configurations.
* **Environment:** `CGT` Conda environment (Python 3.11). Setup via `scripts/env_setups/cgt_setup.sh`.

### B. BiGG (Scalable Deep Generative Modeling)

* **Purpose:** A whole-graph generative model that autoregressively generates sparse graphs including both structure (edges) and node features/labels.
* **Key Mechanisms:** Decomposes graph generation into a sequence of binary tree decisions, processed efficiently via a custom C++ extension (`tree_clib`). The project uses two modes: a conditional model (features + labels) and a structure-only baseline.
* **Output format:** A full DGL graph stored as a file under `datasets/synthetic/bigg/<dataset>/`. The generated graph is a complete stand-alone dataset.
* **Important Files:**
  * `bigg/extension/pipeline.py`: Conditional model training (features + labels).
  * `bigg/extension/pipeline_structure_only.py`: Structure-only baseline.
  * `bigg/data_process/`: Dataset preparation scripts.
* **Environment:** `bigg` Conda environment (Python 3.9, PyTorch 2.4.1). *Agent Alert:* Requires compiling the `tree_clib` C++ extension via `make` before running (handled in `scripts/env_setups/bigg.sh`).

### C. GADBench (Graph Anomaly Detection Benchmark)

GADBench serves as the **downstream evaluation framework** for both generative paradigms. It supports two distinct tasks:

**Anomaly Detection (native):** The original GADBench capability. Trains GNN-based classifiers to identify anomalous nodes. Supports 25+ models (GCN, GIN, BWGNN, etc.) across fully-supervised, semi-supervised, and inductive settings. This is the primary benchmark task for comparing real vs. synthetic data utility.

**Link Prediction (added extension):** An extension added to this project that reuses the same GNN architectures for edge existence prediction. Rather than predicting node labels, the GNNs produce node embeddings (via an `output_emb=True` flag), which are then scored pairwise by a lightweight edge decoder. This was implemented to broaden the downstream evaluation beyond node classification.

The key design principle of the extension is **minimal new code**: existing GNN models are reused unchanged by toggling `output_emb=True`, and only a thin `BaseGNNLinkPredictor` wrapper (with edge splitting, negative sampling, and a decoder) is added on top.

* **Important Files:**
  * `GADBench/benchmark.py`: Native anomaly detection benchmark.
  * `GADBench/link_benchmark.py`: Link prediction benchmark (extension).
  * `GADBench/link_utils.py`: `LinkDataset` — edge splitting, negative sampling, model registry.
  * `GADBench/models/link_prediction/link_predictor.py`: `BaseGNNLinkPredictor` — edge decoder and training loop.
  * `GADBench/models/link_prediction/cgt_link_predictor.py`: Placeholder for CGT-based link prediction (not yet implemented).
  * `GADBench/random_search.py`: Hyperparameter tuning.
  * `scripts/benchmark/benchmark.py`: Project-level benchmark comparing original vs. synthetic data.
* **Environment:** `GADBench` Conda environment (Python 3.10, PyTorch 1.13.1, DGL).

## 4. Link Prediction Extension — Key Design Details

The link prediction extension in `GADBench/link_utils.py` and `GADBench/models/link_prediction/link_predictor.py` is worth understanding in depth, as it involves several non-trivial design decisions:

**Edge splitting with connectivity preservation:** Edges are split into train/val/test sets, but a minimum spanning tree (via NetworkX) is computed first. Spanning-tree edges are never moved to val/test splits — this guarantees the training graph remains connected, which is critical for GNN message passing.

**Negative sampling:**
* `random` — uniform sampling with collision detection (vectorized hashing, 10-attempt retry per sample).
* `hard` — 2-hop random walks to generate structurally plausible negatives (harder for the model to distinguish).
* Val/test negatives are fixed at dataset creation for reproducibility. Training negatives are resampled each epoch to prevent overfitting to specific negatives.

**Edge decoders:**
* `dot` — simple dot product `(h[u] * h[v]).sum(dim=-1)`. No extra parameters; relies entirely on the GNN embedding quality.
* `mlp` — learnable scoring on Hadamard product: `Linear(h) → ReLU → Dropout → Linear(1)`. Adds capacity at the cost of extra parameters.

**Metrics:** AUROC, AUPRC, and Recall@K (where K = number of positive test edges).

## 5. Folder Structure

```text
SynGraphBench/
├── README.md               # Main project documentation
├── scripts/                # CENTRAL HUB FOR EXECUTION
│   ├── env_setups/         # Conda environment creation scripts
│   ├── train/              # Scripts to train generative models
│   │   ├── train_bigg.sh           # Train BiGG (conditional: features + labels)
│   │   ├── train_bigg_structure.sh # Train BiGG (structure-only baseline)
│   │   ├── train_bigg.slurm        # SLURM job template
│   │   ├── train_bigg_structure.slurm
│   │   └── train_cgt.sh            # Train CGT generative model
│   ├── benchmark/
│   │   ├── run_benchmark.sh        # Shell wrapper for anomaly detection benchmark
│   │   ├── benchmark.py            # Project-level benchmark (original vs. synthetic)
│   │   └── bench_utils.py          # Arg parsing, data loading, CGT helpers
│   └── test/               # Quick test/example scripts
├── datasets/
│   ├── original/           # Original DGL datasets (reddit, tolokers, amazon, …)
│   └── synthetic/
│       ├── cgt/            # CGT outputs: .pt files with cluster centers + sequence indices
│       │   └── <dataset>/
│       │       └── e<epochs>_k<clusters>_d<depth>_f<fanout>.pt
│       └── bigg/           # BiGG outputs: full DGL graph files
│           └── <dataset>/
│               └── <variant_hyperparams>
├── results/                # Evaluation outputs (CSVs, XLSX)
├── GADBench/               # Anomaly Detection + Link Prediction Sub-repo
│   ├── benchmark.py
│   ├── link_benchmark.py
│   ├── link_utils.py
│   └── models/
│       ├── anomaly_detection/   # Native GNN detectors
│       │   ├── detector.py
│       │   └── cgt_detector.py
│       └── link_prediction/     # Extension
│           ├── link_predictor.py
│           └── cgt_link_predictor.py   # Placeholder
├── CGT/                    # CGT Sub-repo
└── bigg/                   # BiGG Sub-repo
```

### Synthetic Dataset Naming Convention

All synthetic outputs follow the structure `datasets/synthetic/<generative_model>/<dataset>/<file_name>`. The dataset name is encoded in the directory, so filenames contain only the arguments that define the generated data.

| Generator | Type | Example path |
|-----------|------|--------------|
| `cgt` | Cluster centers + sequence indices (`.pt`) | `synthetic/cgt/reddit/e50_k512_d2_f5.pt` |
| `bigg` | Full DGL graph — conditional (features + labels) | `synthetic/bigg/tolokers/blksize_1024_b_1_lr_0.001_epochs_50` |
| `bigg` | Structure-only baseline | `synthetic/bigg/tolokers/structure_blksize_128_lr_0.001_epochs_100` |

## 6. Execution Flow

**The Pipeline:**

1. **Baseline Evaluation:** Run GADBench on `datasets/original/` to get real-data performance.
2. **Synthetic Generation:** Train CGT or BiGG on real data; outputs go to `datasets/synthetic/`.
3. **Utility Evaluation:** Run GADBench on `datasets/synthetic/` and compare against baselines.
4. **Privacy Evaluation:** Apply k-anonymity, generate private synthetic data, measure performance drop.

**Run from anywhere:** All shell scripts use `cd "$(dirname "$0")/../.."` to navigate to the project root automatically.

### Key Scripts

**`bash scripts/benchmark/run_benchmark.sh [datasets] [models] [trials]`**
Anomaly detection benchmark. Defaults: `reddit`, `GCN,GIN,GraphSAGE,XGBGraph`, `1` trial. Calls `scripts/benchmark/benchmark.py`.

`scripts/benchmark/benchmark.py` has two evaluation modes, selected via `--synthetic_type`:
* `graph` — loads a full DGL graph from `synthetic/bigg/`; trains/tests standard GNNs.
* `comp-graph` — loads a CGT `.pt` file from `synthetic/cgt/`; trains computation-graph GNNs on synthetic sequences and tests on original graph test nodes.

**`bash scripts/train/train_bigg.sh [DATASET] [BLKSIZE] [BSIZE] [LR] [EMBED_DIM] [EPOCHS]`**
Train BiGG conditional model (features + labels). Defaults: `tolokers 1024 1 0.001 256 50`. Checkpoints saved to `checkpoints/bigg/${DATASET}_blk${BLKSIZE}_b${BSIZE}_lr${LR}_e${EPOCHS}`.

**`bash scripts/train/train_bigg_structure.sh [DATASET] [BLKSIZE] [BSIZE] [LR] [EMBED_DIM] [EPOCHS]`**
Train BiGG structure-only baseline. Defaults: `tolokers 128 1 0.001 256 100`. Checkpoints saved with `structure_` prefix.

**`bash scripts/train/train_cgt.sh`**
Train CGT on specified datasets (currently `reddit`). Calls `CGT/train.py`.

**Environment setup:**
```bash
bash scripts/env_setups/bigg.sh           # Creates bigg env, compiles tree_clib C++ extension
bash scripts/env_setups/cgt_setup.sh      # Creates CGT env from CGT/cgt_env.yml
bash scripts/env_setups/gadbench_setup.sh # Creates GADBench env with DGL + ML libraries
```

## 7. Quick Heuristics for the Agent

1. **Which synthetic type to use?** BiGG → `--synthetic_type graph`. CGT → `--synthetic_type comp-graph`. They are not interchangeable.
2. **Changing anomaly detection hyperparameters?** `CGT/args.py` or `GADBench/benchmark.py`.
3. **Changing link prediction hyperparameters?** `GADBench/link_benchmark.py` (epochs, patience) and `GADBench/models/link_prediction/link_predictor.py` (decoder architecture).
4. **Fixing C++ compilation errors?** `scripts/env_setups/bigg.sh` and `bigg/bigg/model/tree_clib/Makefile`. Modern CUDA architectures are patched via `sed` in the setup script.
5. **Adding a new dataset?** Place original in `datasets/original/`. Synthetic outputs go to `datasets/synthetic/<generator>/<dataset>/`. Update loading utilities in `CGT/task/utils/utils.py`, `GADBench/benchmark.py`, or `GADBench/link_utils.py` as appropriate.
6. **BiGG output naming?** Saved to `datasets/synthetic/bigg/<dataset>/` with the filename encoding only hyperparameters (no dataset prefix). Conditional: `blksize_{blksize}_b_{batch_size}_lr_{lr}_epochs_{epochs}`. Structure-only: `structure_blksize_{blksize}_lr_{lr}_epochs_{epochs}`. CGT: `e{epochs}_k{clusters}_d{depth}_f{fanout}.pt` under `datasets/synthetic/cgt/<dataset>/`.
7. **Benchmark `--synthetic_name`?** Pass the filename stem (without dataset prefix). E.g. `--generator bigg --synthetic_name blksize_1024_b_1_lr_0.001_epochs_50` resolves to `synthetic/bigg/<dataset>/blksize_1024_b_1_lr_0.001_epochs_50`.
