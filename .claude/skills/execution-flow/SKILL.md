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

**`bash scripts/benchmark/run_benchmark.sh [datasets] [models] [trials]`**
Anomaly detection benchmark. Defaults: `reddit`, `GCN,GIN,GraphSAGE,XGBGraph`, `1` trial. Calls `scripts/benchmark/benchmark.py`.

`scripts/benchmark/benchmark.py` has two evaluation modes, selected via `--synthetic_type`:
* `graph` — loads a full DGL graph from `synthetic/bigg/`; trains/tests standard GNNs.
* `comp-graph` — loads a CGT `.pt` file from `synthetic/cgt/`; trains computation-graph GNNs on synthetic sequences and tests on original graph test nodes.

Pass `--generator`, `--task` (`hidden_labels` or `hidden_links`), and `--synthetic_name` (filename stem).
Example: `--generator bigg --task hidden_labels --synthetic_name blksize_1024_b_1_lr_0.001_epochs_50`
resolves to `synthetic/bigg/<dataset>/hidden_labels/blksize_1024_b_1_lr_0.001_epochs_50`.

### Training

**`bash scripts/train/train_bigg.sh [DATASET] [BLKSIZE] [BSIZE] [LR] [EMBED_DIM] [EPOCHS]`**
Train BiGG conditional model (features + labels). Defaults: `tolokers 1024 1 0.001 256 50`.
Checkpoints saved to `checkpoints/bigg/${DATASET}_blk${BLKSIZE}_b${BSIZE}_lr${LR}_e${EPOCHS}`.

**`bash scripts/train/train_bigg_structure.sh [DATASET] [BLKSIZE] [BSIZE] [LR] [EMBED_DIM] [EPOCHS]`**
Train BiGG structure-only baseline. Defaults: `tolokers 128 1 0.001 256 100`.
Checkpoints saved with `structure_` prefix.

**`bash scripts/train/train_cgt.sh`**
Train CGT on specified datasets (currently `reddit`). Calls `CGT/train.py`.

### Environment Setup

```bash
bash scripts/env_setups/bigg.sh           # Creates bigg env, compiles tree_clib C++ extension
bash scripts/env_setups/cgt_setup.sh      # Creates CGT env from CGT/cgt_env.yml
bash scripts/env_setups/gadbench_setup.sh # Creates GADBench env with DGL + ML libraries
```
