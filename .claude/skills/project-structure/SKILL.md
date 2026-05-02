---
name: project-structure
description: Folder layout, file locations, and synthetic dataset naming conventions for SynGraphBench.
---

# SynGraphBench — Project Structure

## Folder Layout

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
│   │   ├── run_anomaly_benchmark.sh # Shell wrapper for anomaly detection benchmark
│   │   ├── run_link_benchmark.sh   # Shell wrapper for link prediction benchmark
│   │   ├── anomaly_benchmark.py    # Project-level anomaly detection benchmark (original vs. synthetic)
│   │   ├── link_benchmark.py       # Link prediction benchmark
│   │   ├── bench_utils.py          # Arg parsing, data loading, CGT helpers
│   │   ├── bigg_benchmark.slurm            # SLURM template for BiGG anomaly evaluation
│   │   ├── cgt_anomaly_benchmark.slurm     # SLURM template for CGT anomaly evaluation
│   │   ├── cgt_link_benchmark.slurm        # SLURM template for CGT link prediction evaluation
│   │   └── models/
│   │       ├── cross_graph_detector.py         # Cross-graph anomaly (GNN + XGBGraph)
│   │       └── cross_graph_link_predictor.py   # Cross-graph link prediction (GNN + XGBGraph)
│   └── test/               # Quick test/example scripts
├── datasets/
│   ├── original/           # Original DGL datasets (reddit, tolokers, amazon, …)
│   └── synthetic/
│       ├── cgt/            # CGT outputs: .pt files with cluster centers + sequence indices
│       │   └── <dataset>/
│       │       └── <task>/
│       │           └── <variant>/          # Hyperparameter variant grouping trials
│       │               └── <variant>_t<trial_id>.pt
│       └── bigg/           # BiGG outputs: full DGL graph files
│           └── <dataset>/
│               └── <task>/
│                   └── <variant_hyperparams>
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
│           ├── link_predictor.py        # BaseGNNLinkPredictor — edge decoder + training loop
│           └── cgt_link_predictor.py    # Placeholder for CGT link prediction
├── CGT/                    # CGT Sub-repo
└── bigg/                   # BiGG Sub-repo
```

## Synthetic Dataset Naming Convention

All synthetic outputs follow the structure `datasets/synthetic/<generative_model>/<dataset>/<task>/<file_name>`. For CGT, an additional variant subdirectory groups per-trial `.pt` files by hyperparameter config: `datasets/synthetic/cgt/<dataset>/<task>/<variant>/<variant>_t<trial_id>.pt`. The dataset and task are encoded in the directory hierarchy; filenames contain only the arguments that define the generated data.

**Supported tasks:** `hidden_labels` (anomaly detection — labels withheld from the generative model), `hidden_links` (link prediction — test edges withheld from the generative model). The task level exists because the generative model has different information available during training per task, so the generated datasets are fundamentally different.

| Generator | Task folder | Type | Example path |
|-----------|-------------|------|--------------|
| `cgt` | `hidden_labels` | Cluster centers + sequence indices (`.pt`) | `synthetic/cgt/reddit/hidden_labels/reddit_e50_k512_c1_d2_f5_s5000/reddit_e50_k512_c1_d2_f5_s5000_t0.pt` |
| `cgt` | `hidden_links` | Cluster centers + sequence indices (`.pt`) | `synthetic/cgt/reddit/hidden_links/reddit_e50_k512_c1_d2_f5_s5000/reddit_e50_k512_c1_d2_f5_s5000_t0.pt` |
| `bigg` | `hidden_labels` | Full DGL graph — conditional (features + labels) | `synthetic/bigg/tolokers/hidden_labels/blksize_1024_b_1_lr_0.001_epochs_50` |
| `bigg` | `hidden_labels` | Structure-only baseline | `synthetic/bigg/tolokers/hidden_labels/structure_blksize_128_lr_0.001_epochs_100` |

**Filename patterns:**
- CGT: `{variant}/{variant}_t{trial_id}.pt` where variant = `{dataset}_e{epochs}_k{clusters}_c{cluster_size}_d{depth}_f{fanout}_s{cluster_sample_num}`. `cluster_sample_num` is the cap on nodes subsampled when fitting `KMeansConstrained` — it bounds `cluster_num × cluster_size` and affects the kmeans fit, so it's included in the variant to avoid silent collisions across runs with different sample budgets.
- BiGG conditional: `blksize_{blksize}_b_{batch}_lr_{lr}_epochs_{epochs}_noise_{noise}_ss_{ss}_norm_{method}_{bfs}_lw_{cw}_{lw}_{hetero|det}`
- BiGG structure-only: `structure_blksize_{blksize}_lr_{lr}_epochs_{epochs}`

**Normalization stats:** When BiGG trains with feature normalization, a `_norm_stats.pt` file is saved alongside the synthetic graph. This contains the normalization parameters (e.g., mean/std for zscore) needed to transform original graph features at benchmark time.
