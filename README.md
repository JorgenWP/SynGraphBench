# SynGraphBench - Synthetic Graph Benchmarking

SynGraphBench is a benchmarking suite designed to evaluate the performance of graph processing (GNNs and tree ensembles with aggregation) on synthetic graph data. It provides a collection of repositories, for training graph processing models and for generating synthetic graph data. 

## Goal and Methodology

The benchmark focuses on investigating the trade-off between data utility and privacy in synthetic graph generation. Specifically, it evaluates how well synthetic graph data performs compared to real data on various downstream machine learning tasks. A second aspect of the benchmark is to analyze how applying strict privacy guarantees (like k-anonymity) degrades this downstream performance.

This repository is a collection of publicly available repositories used in previous research, here is an overview of the different repositories and their purpose:

| Repository | Purpose | Original repo URL | Research Paper |
|------------|---------|-----|----------------|
| [GADBench](/GADBench/) | A benchmarking suite for evaluating the performance of graph anomaly detection models. | https://github.com/squareRoot3/GADBench | [GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection](https://arxiv.org/pdf/2306.12251) |
| [CGT](/CGT/) | A graph generative model for benchmarking graph neural networks. | https://github.com/minjiyoon/CGT/tree/main | [Graph Generative Model for Benchmarking Graph Neural Networks](https://proceedings.mlr.press/v202/yoon23d/yoon23d.pdf) |
| [BiGG](/bigg/) | Scalable deep generative modeling for sparse graphs. | https://github.com/google-research/google-research/tree/master/bigg | [Scalable Deep Generative Modeling for Sparse Graphs](https://proceedings.mlr.press/v119/dai20b/dai20b.pdf5) |



### Pipeline

The pipeline for evaluating the performance of synthetic graph data on downstream tasks consists of the following steps:

1. **Baseline Evaluation**: Train selected predictive models (GNNs and Tree Ensembles) on real graph datasets and record baseline performance across multiple tasks.
2. **Synthetic Generation**: Train generative models to learn the joint distribution of the graph topology and node features to generate synthetic datasets.
3. **Synthetic Evaluation (Utility)**: Train the same predictive models from scratch on the synthetic datasets and compare their performance to the real-data baselines.
4. **Privacy Evaluation (Utility vs. Privacy)**: Apply k-anonymity to the real data (via feature clustering) before training the generative models. Generate private synthetic datasets, train the predictive models on them, and measure the performance drop to quantify the "cost of privacy."



## Project Structure

```
SynGraphBench/
├── README.md
├── scripts/
│   ├── train/
│   │   ├── cgt.sh
│   │   └── bigg.sh
│   └── generate/
│       ├── cgt.sh
│       └── bigg.sh
├── datasets/
│   ├── original/
│   └── synthetic/
├── GADBench/
├── CGT/
└── BiGG/
```



## Getting Started

The project is set up to run the training of the generative models and generation of synthetic data separately from the training and evaluation of the predictive models. The scripts for running these processes are located in the `scripts/` directory. Before running the scripts, make sure to set up the [necessary environments and dependencies](#environment-setup).

### Prerequisites

- Conda or Miniconda for environment management

### Environment Setup

To set up the different environments for the generative models and the predictive models, you can run the scripts located in the `scripts/env_setups/` directory. These scripts will create conda environments with the required dependencies for the different parts of the pipeline.

Run the following commands to set up the different conda environments:

```bash
# Set up environment for CGT
bash env_setups/cgt_setup.sh

# Set up environment for BiGG
bash env_setups/bigg_setup.sh

# Set up environment for predictive models
bash env_setups/gadbench_setup.sh
```

### Training the Generative Models and Generating Synthetic Data

To train the generative models, you can run the scripts located in the `scripts/train/` directory. These scripts are designed to be run on a cluster using SLURM. Make sure to adjust the SLURM parameters (like job name, time limit, output paths) in the scripts before submitting them to the cluster.

Examples:

```bash
# Train CGT
sbatch scripts/train/train_cgt.slurm

# Train BiGG
sbatch scripts/train/train_bigg.slurm
```

### Training the Predictive Models and Evaluating Performance

To train the predictive models and evaluate their performance on the real and synthetic datasets, you can run the scripts located in the `scripts/benchmark/` directory. These scripts will train the GNNs and Tree Ensembles on the specified datasets and record their performance.

The benchmarking script has the following usage:

```bash
bash scripts/benchmark/run_anomaly_benchmark.sh [datasets] [models] [trials] [generator] [synthetic_dataset_name]
```

#### Examples

Example command for running the benchmark on the CGT datasets:
> This command will train GCN and GIN models on the synthetic reddit dataset (generated by CGT) for 3 trials, and compare their performance to the baselines trained on the original reddit dataset.

```bash
# Train and evaluate predictive models on CGT datasets
bash scripts/benchmark/run_anomaly_benchmark.sh reddit GCN,GIN 3 cgt synthetic_cgt_dataset
```

Example command for running the benchmark on the BiGG datasets:
> This command will train GCN, GIN, GraphSAGE, and XGBGraph models on the synthetic tolokers dataset (generated by BiGG) for 1 trial, and compare their performance to the baseline trained on the original tolokers dataset.

```bash
# Train and evaluate predictive models on BiGG datasets
bash scripts/benchmark/run_anomaly_benchmark.sh tolokers GCN,GIN,GraphSAGE,XGBGraph 1 bigg synthetic_bigg_dataset
```
