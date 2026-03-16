---
name: project-overview
description: A comprehensive overview of the SynGraphBench project, detailing its structure, core components, and execution flow for developers and agents.
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
* `CGT/train.py`: Training script for the CGT generative model.
* `CGT/test.py`: Test/evaluation script (generates graphs and evaluates GNNs).
* `CGT/args.py`: Hyperparameter configurations.


* **Environment:** Uses the `CGT` Conda environment (Python 3.11). Setup via `scripts/env_setups/cgt_setup.sh`.

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
* `scripts/benchmark/benchmark.py`: Project-level benchmark comparing original vs. synthetic data.


* **Environment:** Uses the `GADBench` Conda environment (Python 3.10, PyTorch 1.13.1, DGL).

## 3. Folder Structure

Understanding the directory routing is critical, as scripts in one folder often call Python files in another.

```text
SynGraphBench/
├── README.md               # Main project documentation
├── scripts/                # CENTRAL HUB FOR EXECUTION
│   ├── env_setups/         # Conda environment creation scripts (bigg.sh, cgt_setup.sh, gadbench_setup.sh)
│   ├── train/              # Scripts to train generative models (e.g., train_cgt.sh)
│   ├── benchmark/          # Scripts and Python modules for benchmarking (run_benchmark.sh, benchmark.py)
│   └── test/               # Quick test/example scripts (e.g., run_cgt_test.sh)
├── datasets/               # Data storage
│   ├── original/           # Original datasets (e.g., reddit, cora, citeseer)
│   └── synthetic/          # Generated datasets ready for downstream evaluation
├── results/                # Evaluation outputs (e.g., CSVs)
├── GADBench/               # Anomaly Detection Sub-repo
├── CGT/                    # Computation Graph Transformer Sub-repo
└── bigg/                   # BiGG Sub-repo
```

## 4. Execution Flow & Script Usage

The project relies on a decoupled architecture where training, generation, and evaluation are executed via shell scripts located in the `scripts/` folder.

**The Pipeline:**

1. **Baseline Evaluation:** Run predictive models (via `GADBench`) on `datasets/original/`.
2. **Synthetic Generation:** Train generative models (`CGT` or `BiGG`) on real data, then generate outputs into `datasets/synthetic/`.
3. **Utility Evaluation:** Train predictive models from scratch on `datasets/synthetic/` and compare against baselines.
4. **Privacy Evaluation:** Apply k-anonymity (clustering) to real data, train generators, create private synthetic datasets, and measure the performance drop.

**Running Scripts (Agent Guidelines):**

* **Run from anywhere:** All shell scripts use `cd "$(dirname "$0")/../.."` at the top to automatically navigate to the project root before executing. This means scripts can be invoked from any working directory — there is no need to `cd` into `scripts/` first.
* **Example: `bash scripts/benchmark/run_benchmark.sh [datasets] [models] [trials]`**
    * Evaluates GNN models on original vs. CGT-generated synthetic graph data using GADBench.
    * Accepts three optional arguments with defaults: datasets (`reddit`), models (`GCN,GIN,GraphSAGE,XGBGraph`), and number of trials (`1`).
    * Activates the `GADBench` Conda environment, then calls `scripts/benchmark/benchmark.py`.
* **Example: `bash scripts/train/train_cgt.sh`**
    * Trains the CGT generative model on specified datasets (currently `reddit`).
    * Activates the appropriate Conda environment and calls `CGT/train.py`.
* **Setting up Environments:** Run setup scripts from any directory, e.g., `bash scripts/env_setups/bigg.sh`.
    * `bigg.sh` creates the conda environment, installs PyTorch, and automatically handles the `sed` and `make` commands to compile the C++ `tree_clib` library.
    * `cgt_setup.sh` creates the CGT environment from `CGT/cgt_env.yml` and installs GPU dependencies.
    * `gadbench_setup.sh` creates the GADBench environment with DGL and ML libraries.



## 5. Quick Heuristics for the Agent

1. **Changing GNN Hyperparameters?** Look in `CGT/args.py` or `GADBench/benchmark.py`.
2. **Fixing C++ Compilation Errors?** Look at `scripts/env_setups/bigg.sh` and `bigg/bigg/model/tree_clib/Makefile`. Modern CUDA architectures are patched in via `sed` in the setup script.
3. **Adding a New Dataset?** You must place it in `datasets/original/` (or `CGT/data/` if testing CGT directly) and ensure the loading utility functions in the respective sub-repo (e.g., `CGT/task/utils/utils.py` or `GADBench/benchmark.py`) are updated to parse it.