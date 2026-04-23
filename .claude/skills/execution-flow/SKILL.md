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
Anomaly detection benchmark. Defaults: `reddit`, `GCN,GIN,GraphSAGE,XGBGraph,XGBoost`, `1`, `cgt`, `""` (uses dataset name), `hidden_labels`. Calls `scripts/benchmark/anomaly_benchmark.py`.

`XGBoost` is the feature-only diagnostic row: trains on synthetic raw features and tests on original raw features, no graph structure either side. A synthetic >> original AUROC gap means the generator over-encoded the label into features, so downstream GNN wins aren't measuring topology use.

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

**`bash scripts/train/train_bigg.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim] [noise_std] [ss_max_prob] [ss_start_epoch] [bfs_preprocess] [normalize] [loss_weights] [hetero_feat] [mask_test_labels] [logvar_floor] [binary_feat]`**
Train BiGG conditional model (features + labels). Defaults: `tolokers 1024 1 50 0.001 256 0.0 0.0 0 False none 1,1 false false -4.0 false`.
* `noise_std`: Gaussian noise std added to hidden state during training (0.0 = disabled).
* `ss_max_prob`: Max scheduled-sampling probability (0.0 = disabled; uses teacher forcing only).
* `ss_start_epoch`: Epoch at which scheduled sampling begins ramping up.
* `bfs_preprocess`: Apply fixed BFS node ordering before training (`True`/`False`).
* `normalize`: Feature normalisation method (`zscore`, `minmax`, `row`, `quantile`, `cdf`, or `none`). `quantile` uses rank-based inverse normal transform (any distribution → N(0,1)). `cdf` uses the empirical CDF (any distribution → Uniform[0,1]) and couples to a sigmoid+BCE continuous head; mutually exclusive with `hetero_feat`, `cat_feat`, `mdn_feat`.
* `loss_weights`: Comma-separated cont,label weights relative to struct (e.g., `0.1,0.1`).
* `hetero_feat`: `true` for heteroscedastic feature prediction (mean + variance).
* `mask_test_labels`: `true` to exclude test node labels (split 0) from label loss, preventing data leakage in anomaly benchmarks. Appends `_masked` to save name.
* `logvar_floor`: Lower clamp for log-variance in hetero_feat mode (default: -4.0).
* `binary_feat`: `true` to auto-detect binary feature columns and use BCE loss + Bernoulli sampling instead of Gaussian head. Appends `_binfeat` to save name. Binary columns skip normalization.

**`bash scripts/train/train_bigg_subsample.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim] [noise_std] [ss_max_prob] [ss_start_epoch] [bfs_preprocess] [normalize] [loss_weights] [hetero_feat] [mask_test_labels] [logvar_floor] [subsample_size] [burn_prob] [num_subgraphs] [binary_feat] [vae_feat] [vae_dim] [kl_weight] [cat_feat] [n_bins] [bin_sigma] [mdn_feat] [mdn_components] [mdn_logsigma_floor]`**
Train BiGG with forest fire subsampling for VRAM-limited training. Same args as above, plus:
* `subsample_size`: target nodes per subgraph (default: 2000).
* `burn_prob`: forest fire burn probability — controls subgraph density (default: 0.3).
* `num_subgraphs`: number of subgraphs to generate (default: `ceil(N / subsample_size)`).
* `vae_feat`: `true` to add a per-node label-agnostic CVAE latent shared across feature decoders (default: false). Appends `_vae{dim}_kl{weight}` to save name.
* `vae_dim`: latent dimensionality when `vae_feat` is on (default: 16).
* `kl_weight`: coefficient on the KL term in the VAE ELBO (default: 1.0). Not subject to dynamic calibration.
* `cat_feat`: `true` to use the AR categorical feature predictor (quantile bins + value-space soft-label cross-entropy). Mutually exclusive with `hetero_feat` and `vae_feat`. Appends `_cat{n_bins}_s{bin_sigma}` to save name and persists `cat_bins.pt` alongside the generated graph(s).
* `n_bins`: number of quantile bins per continuous feature when `cat_feat` is on (default: 32).
* `bin_sigma`: Gaussian soft-label std in feature-value units. Leave empty for auto per-feature sigma (0.5 × median positive spacing of *that* feature's bin centers — adapts across regimes: small for dense continuous, large for binary-like, scales automatically under k-means privacy). Scalar override broadcast to all features.
* `mdn_feat`: `true` to use Mixture Density Network feature head (per-feature K-component Gaussian mixture conditioned on `[h, label_embed]`). Mutually exclusive with `hetero_feat` / `vae_feat` / `cat_feat`. Appends `_mdn{K}` to save name.
* `mdn_components`: number of mixture components per feature when `mdn_feat` is on (default: 8).
* `mdn_logsigma_floor`: lower clamp for MDN component log-sigma (default: -4.0).

**`bash scripts/train/train_bigg_structure.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim]`**
Train BiGG structure-only baseline. Defaults: `tolokers 128 1 100 0.001 256`.
Checkpoints saved with `structure_` prefix.

**`bash scripts/train/train_cgt.sh [dataset] [gpt_epochs] [cluster_num] [cluster_size] [gpt_batch_size] [cg_depth] [cg_fanout] [trial_id] [task]`**
Train CGT on a dataset. Defaults: `reddit 50 512 1 128 2 5 0 hidden_labels`. Calls `CGT/train.py`.
`trial_id` selects which GADBench mask column (0-9) to use for the train/val/test split.
Output saved to `datasets/synthetic/cgt/<dataset>/<task>/<dataset>_e{epochs}_k{clusters}_d{depth}_f{fanout}_t{trial_id}.pt`.

**`bash scripts/train/train_cgt_all_trials.sh [dataset] [gpt_epochs] [cluster_num] [cluster_size] [gpt_batch_size] [cg_depth] [cg_fanout] [num_trials]`**
Train CGT on all GADBench splits (trials 0 to num_trials-1). Defaults: same as `train_cgt.sh`, `num_trials=10`. Loops over `train_cgt.sh` with each trial_id.

### Environment Setup

```bash
bash scripts/env_setups/bigg_setup.sh     # Creates bigg env, compiles tree_clib C++ extension
bash scripts/env_setups/cgt_setup.sh      # Creates CGT env from CGT/cgt_env.yml
bash scripts/env_setups/gadbench_setup.sh # Creates GADBench env with DGL + ML libraries
```
