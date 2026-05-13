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

BiGG generates a **complete, new graph** — both topology (edges) and node features — from scratch. The output is a full DGL graph that is a drop-in replacement for the original at *training* time only: the downstream GNN trains and validates on the synthetic graph, but is tested on the **original graph's test nodes/edges** via `CrossGraphGNNDetector` / `CrossGraphLinkPredictor` (`scripts/benchmark/models/cross_graph_*.py`). This matches the CGT path's evaluation protocol — both paradigms are scored against the same real test set.

The generative model captures the joint distribution of graph structure and node attributes, and samples from it to produce a synthetic counterpart.

* **Key Mechanisms:** Decomposes graph generation into a sequence of binary tree decisions, processed efficiently via a custom C++ extension (`tree_clib`). Two modes: a conditional model (features + labels) and a structure-only baseline.
* **Data Augmentation (optional):** Two training-time augmentation strategies, both disabled by default:
  * *Gaussian noise* (`noise_std`): Adds random noise to hidden state during training to improve generalization. Default `0.0` (disabled).
  * *Scheduled self-sampling* (`ss_max_prob`, `ss_start_epoch`): Gradually replaces teacher-forced inputs with the model's own predictions during training, reducing exposure bias. `ss_max_prob` controls the max probability (default `0.0` = disabled); `ss_start_epoch` controls when ramp-up begins.
* **Label Masking (optional):** `--mask_test_labels` excludes test node labels (split 0) from the label loss during training, preventing data leakage in anomaly detection benchmarks. The model still sees test node features and structure (and uses test labels for teacher forcing / state updates) — only the label *loss* is masked. Appends `_masked` to the save name.
* **Binary Feature Head (optional):** `--binary_feat` auto-detects binary {0,1} feature columns and uses a separate BCE loss + Bernoulli sampling head for them, instead of the Gaussian head. Binary columns also skip normalization. Appends `_binfeat` to the save name.
* **Output format:** A full DGL graph stored as a file under `datasets/synthetic/bigg/<dataset>/<task>/`.
* **Important Files:**
  * `bigg/extension/pipeline.py`: Conditional model training (features + labels).
  * `bigg/extension/pipeline_structure_only.py`: Structure-only baseline.
  * `bigg/data_process/`: Dataset preparation scripts.
* **Environment:** `bigg` Conda environment (Python 3.9, PyTorch 2.4.1).
  * **Agent Alert:** Requires compiling the `tree_clib` C++ extension via `make` before running. This is handled in `scripts/env_setups/bigg_setup.sh`. Modern CUDA architectures are patched via `sed` in that script.

---

## CGT — Computation-Graph Generation

CGT operates at a different level. Rather than generating a new graph, it **generates synthetic node feature distributions** while leaving the original graph topology unchanged. The key idea is that GNNs aggregate information via local computation trees (computation graphs) rooted at each node. CGT learns to generate sequences of feature vectors that match the distribution of these computation trees.

Concretely, CGT uses **DP-k-means** to cluster the real node features into `k` cluster centers. K-means is **fit only on train+val node features** (test features never influence the centers); cluster-id assignment then covers all nodes since computation graphs span the full graph. These centers — not the raw features — are what is "synthesized" and shared. During evaluation, the original graph's training and validation node features are replaced by the nearest cluster center, and a GNN is trained on this feature-masked graph and tested on the unmasked original test nodes.

* **Key Mechanisms:** Operates on minibatches of computation graph sequences (not the full graph). DP-k-means provides differential privacy. A Transformer learns to generate realistic sequences of cluster assignments.
* **Output format:** A `.pt` file containing cluster centers, generated sequence indices, and train/val/test node ID mappings. Stored under `datasets/synthetic/cgt/<dataset>/<task>/`.
* **Important Files:**
  * `CGT/train.py`: Training script.
  * `CGT/test.py`: Generation and evaluation script.
  * `CGT/args.py`: Hyperparameter configurations.
* **Environment:** `CGT` Conda environment (Python 3.11). Setup via `scripts/env_setups/cgt_setup.sh`.

### Tasks

* **`hidden_labels` (anomaly detection)**: uses GADBench mask columns for the train/val/test node split (via `split_ids_from_dgl`). CGT sees all edges; only labels for test nodes are withheld downstream. `ids['train']`/`ids['val']`/`ids['test']` match GADBench split columns.
* **`hidden_links` (link prediction)**: `CGT/train.py` dispatches to `load_dgl_graph_with_hidden_links(args, trial_id, val_ratio, test_ratio)` in `CGT/task/utils/utils.py`, which mirrors `GADBench/link_utils.py:LinkDataset.split(trial_id)` byte-for-byte (seed `3407 + trial_id*10`, MST-protected split). Test edges are stripped from the adjacency before training; val edges remain. There is no node-level train/test concept for link prediction, so `split_node_ids_for_hidden_links(num_nodes, trial_id)` produces an 80/20 node split just to give CGT two non-empty target_id buckets (`gen_train_ids` + `gen_val_ids` together cover every node). The `.pt` records `hidden_test_edges`, `task`, `trial_id`, `val_ratio`, `test_ratio`; `scripts/benchmark/bench_utils.py:_assert_link_pt_alignment` verifies these downstream. Added CLI args: `--val_ratio` (default 0.05), `--test_ratio` (default 0.10) — must match `LinkDataset` defaults.
