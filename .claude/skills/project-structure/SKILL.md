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
│   │   ├── run_benchmark.sh        # Shell wrapper for anomaly detection benchmark
│   │   ├── benchmark.py            # Project-level benchmark (original vs. synthetic)
│   │   └── bench_utils.py          # Arg parsing, data loading, CGT helpers
│   └── test/               # Quick test/example scripts
├── datasets/
│   ├── original/           # Original DGL datasets (reddit, tolokers, amazon, …)
│   └── synthetic/
│       ├── cgt/            # CGT outputs: .pt files with cluster centers + sequence indices
│       │   └── <dataset>/
│       │       └── <task>/
│       │           └── e<epochs>_k<clusters>_d<depth>_f<fanout>.pt
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
│           ├── link_predictor.py
│           └── cgt_link_predictor.py   # Placeholder
├── CGT/                    # CGT Sub-repo
└── bigg/                   # BiGG Sub-repo
```

## Synthetic Dataset Naming Convention

All synthetic outputs follow the structure `datasets/synthetic/<generative_model>/<dataset>/<task>/<file_name>`. The dataset and task are encoded in the directory hierarchy; filenames contain only the arguments that define the generated data.

**Supported tasks:** `hidden_labels` (anomaly detection — labels withheld from the generative model), `hidden_links` (link prediction — test edges withheld from the generative model). The task level exists because the generative model has different information available during training per task, so the generated datasets are fundamentally different.

| Generator | Task folder | Type | Example path |
|-----------|-------------|------|--------------|
| `cgt` | `hidden_labels` | Cluster centers + sequence indices (`.pt`) | `synthetic/cgt/reddit/hidden_labels/e50_k512_d2_f5.pt` |
| `cgt` | `hidden_links` | Cluster centers + sequence indices (`.pt`) | `synthetic/cgt/reddit/hidden_links/e50_k512_d2_f5.pt` |
| `bigg` | `hidden_labels` | Full DGL graph — conditional (features + labels) | `synthetic/bigg/tolokers/hidden_labels/blksize_1024_b_1_lr_0.001_epochs_50` |
| `bigg` | `hidden_labels` | Structure-only baseline | `synthetic/bigg/tolokers/hidden_labels/structure_blksize_128_lr_0.001_epochs_100` |

**Filename patterns:**
- CGT: `e{epochs}_k{clusters}_d{depth}_f{fanout}.pt`
- BiGG conditional: `blksize_{blksize}_b_{batch_size}_lr_{lr}_epochs_{epochs}`
- BiGG structure-only: `structure_blksize_{blksize}_lr_{lr}_epochs_{epochs}`
