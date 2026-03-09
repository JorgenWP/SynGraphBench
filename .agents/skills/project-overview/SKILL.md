---
name: project-overview
description: Describe what this skill does and when to use it. Include keywords that help agents identify relevant tasks.
---

# SynGraphBench - Developer & Agent Skill Guide

## 1. Project Overview

**SynGraphBench** is a benchmarking suite designed to evaluate the trade-off between data utility and privacy in synthetic graph generation. It measures how well synthetic graph data performs compared to real data on downstream machine learning tasks, and analyzes how strict privacy guarantees (like k-anonymity) degrade this performance.

The project is an amalgamation of three distinct, previously published research repositories, orchestrated together via custom shell scripts to form a complete benchmarking pipeline.

## 2. Core Sub-Repositories

The repository integrates three independent sub-systems. **Note for Agents:** Because these are distinct research projects, they have conflicting dependencies. *Always ensure you are using the correct Conda environment for the specific sub-repo you are interacting with.*

### A. CGT (Computation Graph Transformer)

* **Purpose:** A graph generative model used to create privacy-controlled, synthetic substitutes of large-scale real-world graphs.
* **Key Mechanisms:** Operates on minibatches rather than the whole graph. Converts graph distributions into feature vector sequence distributions using a Transformer architecture. Includes a DP-k-means module for differential privacy.
* **Important Files:**
* `CGT/test.py`: The main execution script (prepares models, reads datasets, generates graphs, and evaluates GNNs).
* `CGT/args.py`: Hyperparameter configurations.


* **Environment:** Uses the `cgt_test` Conda environment (Python 3.7, PyTorch 1.13.1).

### B. BiGG (Scalable Deep Generative Modeling)

* **Purpose:** A scalable deep generative model specifically for sparse graphs, capable of autoregressively generating graphs with node and edge features.
* **Key Mechanisms:** Relies on a custom C++ extension (`tree_clib`) for fast tree operations.
* **Important Files:** * `bigg/data_process/`: Scripts to prepare synthetic and SAT datasets.
* `bigg/experiments/`: Contains execution scripts for different graph types (e.g., `run_lobster.sh`, `run_grid.sh`).


* **Environment:** Uses the `bigg` Conda environment (Python 3.9, PyTorch 2.4.1). *Agent Alert:* Requires compiling the `tree_clib` C++ extension via `make` before running (handled in the setup script).

### C. GADBench (Graph Anomaly Detection Benchmark)

* **Purpose:** A suite for evaluating supervised graph anomaly detection models. Used in this pipeline as the downstream predictive task to compare real vs. synthetic graph utility.
* **Key Mechanisms:** Supports 25+ GAD models (GCN, GIN, BWGNN, etc.) across multiple settings (fully-supervised, semi-supervised, inductive). Relies heavily on DGL (Deep Graph Library).
* **Important Files:**
* `GADBench/benchmark.py`: Main benchmarking script.
* `GADBench/random_search.py`: Hyperparameter tuning.


* **Environment:** Uses the `GADBench` Conda environment (Python 3.10, PyTorch 1.13.1, DGL).

## 3. Folder Structure

Understanding the directory routing is critical, as scripts in one folder often call Python files in another.

```text
SynGraphBench/
├── README.md               # Main project documentation
├── scripts/                # CENTRAL HUB FOR EXECUTION
│   ├── env_setups/         # Conda environment creation scripts (bigg.sh, cgt_setup.sh)
│   ├── train/              # Scripts to train generative models (CGT/BiGG)
│   ├── generate/           # Scripts to sample/generate synthetic data
│   ├── evaluate/           # Scripts to evaluate predictive performance 
│   └── test/               # Scripts to run example/test scripts (e.g., run_cgt_test.sh)
├── datasets/               # Data storage
│   ├── original/           # Original datasets (e.g., reddit, cora, citeseer)
│   └── synthetic/          # Generated datasets ready for downstream evaluation
├── GADBench/               # Anomaly Detection Sub-repo
├── CGT/                    # Computation Graph Transformer Sub-repo
└── bigg/                   # BiGG Sub-repo

```

## 4. Execution Flow & Script Usage

The project relies on a decoupled architecture where training, generation, and evaluation are executed via shell scripts located in the `scripts/` folder.

**The Pipeline:**

1. **Baseline Evaluation:** Run predictive models (via `GADBench`) on `datasets/real/`.
2. **Synthetic Generation:** Train generative models (`CGT` or `BiGG`) on real data, then generate outputs into `datasets/synthetic/`.
3. **Utility Evaluation:** Train predictive models from scratch on `datasets/synthetic/` and compare against baselines.
4. **Privacy Evaluation:** Apply k-anonymity (clustering) to real data, train generators, create private synthetic datasets, and measure the performance drop.

**Running Scripts (Agent Guidelines):**

* **Relative Paths:** Scripts in the `scripts/` directory usually expect to be executed from *within* the `scripts/` folder itself. They use relative paths to step back into the sub-repos (e.g., `../CGT/test.py` or `../bigg/bigg/model/tree_clib`).
* **Example Execution (`scripts/test/run_cgt_test.sh`):**
* Uses CGT own pipeline to evaluate GNN performance on synthetic data generated by CGT under different noise levels.
* The rest of the repo should instead use the predictive models from GADBench for evaluation, but this script serves as a quick test of the CGT pipeline.
* Iterates over datasets (`cora`, `citeseer`).
* Iterates over noise levels (`0`, `2`, `4`).
* Calls `python ../CGT/test.py --dataset ... --noise_num ... -n gcn sgc gin`.


* **Setting up Environments:** * If instructed to initialize an environment, navigate to `scripts/env_setups/` and run `bash <script_name>.sh`.
* E.g., `bash env_setups/bigg.sh` creates the conda environment, installs PyTorch, and automatically handles the `sed` and `make` commands to compile the C++ `tree_clib` library.



## 5. Quick Heuristics for the Agent

1. **Changing GNN Hyperparameters?** Look in `CGT/args.py` or `GADBench/benchmark.py`.
2. **Fixing C++ Compilation Errors?** Look at `scripts/env_setups/bigg.sh` and `bigg/bigg/model/tree_clib/Makefile`. Modern CUDA architectures are patched in via `sed` in the setup script.
3. **Adding a New Dataset?** You must place it in `datasets/original/` (or `CGT/data/` if testing CGT directly) and ensure the loading utility functions in the respective sub-repo (e.g., `CGT/task/utils/utils.py` or `GADBench/benchmark.py`) are updated to parse it.