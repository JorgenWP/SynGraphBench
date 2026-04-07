---
name: execution-flow
description: End-to-end pipeline steps, key shell scripts with their CLI arguments and defaults, and environment setup commands for SynGraphBench.
---

# SynGraphBench — Execution Flow

## The Pipeline

1. **Baseline Evaluation:** Run GADBench on `datasets/original/` to get real-data performance.
2. **Synthetic Generation:** Train CGT or BiGG on real data; outputs go to `datasets/synthetic/`.
3. **Utility Evaluation:** Run GADBench on `datasets/synthetic/` and compare against baselines.
4. **Privacy Evaluation:** Apply k-anonymity, generate private synthetic data, measure performance drop.

**Run from anywhere:** All shell scripts use `cd "$(dirname "$0")/../.."` to navigate to the project root automatically.

---

## Key Scripts

### Benchmark

**`bash scripts/benchmark/run_anomaly_benchmark.sh [datasets] [models] [trials] [generator] [synthetic_name] [task]`**
Anomaly detection benchmark. Defaults: `reddit`, `GCN,GIN,GraphSAGE,XGBGraph`, `1`, `cgt`, `""` (uses dataset name), `hidden_labels`. Calls `scripts/benchmark/anomaly_benchmark.py`.

`scripts/benchmark/anomaly_benchmark.py` has two evaluation modes, selected via `--synthetic_type`:
* `graph` — loads a full DGL graph from `synthetic/bigg/`; trains/tests standard GNNs.
* `comp-graph` — loads a CGT `.pt` file from `synthetic/cgt/`; trains computation-graph GNNs on synthetic sequences and tests on original graph test nodes.

Examples:
```bash
# CGT on reddit, 3 trials
bash scripts/benchmark/run_anomaly_benchmark.sh reddit GCN,GIN 3 cgt

# BiGG on tolokers (single graph)
bash scripts/benchmark/run_anomaly_benchmark.sh tolokers GCN,GIN 1 bigg blksize_1024_b_1_lr_0.001_epochs_50 hidden_labels

# BiGG subsampled run — benchmark auto-combines subgraph_* files into a block-diagonal graph
bash scripts/benchmark/run_anomaly_benchmark.sh tolokers GCN,GIN 1 bigg blksize_-1_b_1_lr_0.001_epochs_50_..._sub6_size2000_p0.3 hidden_labels
```

**`bash scripts/benchmark/run_link_benchmark.sh [datasets] [models] [trials] [generator] [neg_sampling] [decoder] [synthetic_name]`**
Link prediction benchmark. Defaults: `reddit`, `GCN,GIN,GraphSAGE`, `1`, `cgt`, `random`, `dot`, `""`. Calls `scripts/benchmark/link_benchmark.py`.

* `neg_sampling`: `random` (uniform) or `hard` (2-hop random walks).
* `decoder`: `dot` (dot product, no params) or `mlp` (learnable Hadamard-product scorer).

Examples:
```bash
# CGT on reddit, random negatives, dot decoder
bash scripts/benchmark/run_link_benchmark.sh reddit GCN,GIN 3 cgt random dot

# BiGG on tolokers, MLP decoder
bash scripts/benchmark/run_link_benchmark.sh tolokers GCN,GIN 1 bigg random mlp tolokers_blksize_1024_b_1
```

### Training

**`bash scripts/train/train_bigg.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim] [noise_std] [ss_max_prob] [ss_start_epoch] [bfs_preprocess] [normalize] [loss_weights] [hetero_feat] [mask_test_labels] [subsample] [subsample_size] [burn_prob] [num_subgraphs]`**
Train BiGG conditional model (features + labels). Defaults: `tolokers 1024 1 50 0.001 256 0.0 0.0 0 False none 1,1 false false false 2000 0.3 ""`.
* `noise_std`: Gaussian noise std added to hidden state during training (0.0 = disabled).
* `ss_max_prob`: Max scheduled-sampling probability (0.0 = disabled; uses teacher forcing only).
* `ss_start_epoch`: Epoch at which scheduled sampling begins ramping up.
* `bfs_preprocess`: Apply fixed BFS node ordering before training (`True`/`False`).
* `normalize`: Feature normalisation method (`zscore`, `minmax`, `row`, or `none`).
* `loss_weights`: Comma-separated cont,label weights relative to struct (e.g., `0.1,0.1`).
* `hetero_feat`: `true` for heteroscedastic feature prediction (mean + variance).
* `mask_test_labels`: `true` to exclude test node labels (split 0) from label loss, preventing data leakage in anomaly benchmarks. Appends `_masked` to save name.
* `subsample`: `true` to sample subgraphs via forest fire for VRAM-limited training. Samples are independent (with replacement). Outputs a directory of `subgraph_N` files; benchmark auto-combines via `dgl.batch()`. Appends `_subM_sizeS_pP` to save name.
* `subsample_size`: target nodes per subgraph (default: 2000).
* `burn_prob`: forest fire burn probability — controls subgraph density (default: 0.3).
* `num_subgraphs`: number of subgraphs to generate (default: `ceil(N / subsample_size)`).

**`bash scripts/train/train_bigg_structure.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim]`**
Train BiGG structure-only baseline. Defaults: `tolokers 128 1 100 0.001 256`.
Checkpoints saved with `structure_` prefix.

**`bash scripts/train/train_cgt.sh [dataset] [gpt_epochs] [cluster_num] [cluster_size] [gpt_batch_size] [cg_depth] [cg_fanout]`**
Train CGT on a dataset. Defaults: `reddit 50 512 1 128 2 5`. Calls `CGT/train.py`.
Output saved to `datasets/synthetic/cgt/<dataset>/hidden_labels/<dataset>_e{gpt_epochs}_k{cluster_num}_d{cg_depth}_f{cg_fanout}.pt`.

### Environment Setup

```bash
bash scripts/env_setups/bigg_setup.sh     # Creates bigg env, compiles tree_clib C++ extension
bash scripts/env_setups/cgt_setup.sh      # Creates CGT env from CGT/cgt_env.yml
bash scripts/env_setups/gadbench_setup.sh # Creates GADBench env with DGL + ML libraries
```
