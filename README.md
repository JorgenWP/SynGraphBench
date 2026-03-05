# SynGraphBench - Synthetic Graph data Benchmarking

SynGraphBench is a benchmarking suite designed to evaluate the performance of graph processing (GNNs and tree ensembles with aggregation) on synthetic graph data. It provides a collection of repositories, for training graph processing models and for generating synthetic graph data. 

The benchmark focuses on investigating the trade-off between data utility and privacy in synthetic graph generation. Specifically, it evaluates how well synthetic graph data performs compared to real data on various downstream machine learning tasks. A second aspect of the benchmark is to analyze how applying strict privacy guarantees (like k-anonymity) degrades this downstream performance.

This repository is a collection of publicly available repositories used in previous research, here is an overview of the different repositories and their purpose:

| Repository | Purpose | URL | Research Paper |
|------------|---------|-----|----------------|
| GADBench | A benchmarking suite for evaluating the performance of graph anomaly detection models. | https://github.com/squareRoot3/GADBench | [GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection](https://arxiv.org/pdf/2306.12251) |
| CGT | - | https://github.com/minjiyoon/CGT/tree/main | [Graph Generative Model for Benchmarking Graph Neural Networks](https://proceedings.mlr.press/v202/yoon23d/yoon23d.pdf) |
| BiGG | - | https://github.com/google-research/google-research/tree/master/bigg | [Scalable Deep Generative Modeling for Sparse Graphs](https://proceedings.mlr.press/v119/dai20b/dai20b.pdf5) |

