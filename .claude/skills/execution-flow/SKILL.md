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

**`bash scripts/benchmark/run_anomaly_benchmark.sh [datasets] [models] [trials] [generator] [synthetic_name] [task] [cdf_invert] [seeds_per_split] [dump_per_trial] [tune_test_ratio] [tune_test_seed] [tune_portion]`**
Anomaly detection benchmark. Defaults: `reddit`, `GCN,GIN,GraphSAGE,XGBGraph,XGBoost`, `1`, `cgt`, `""` (uses dataset name), `hidden_labels`, `linear`, `3`, `false`, `""`, `0`, `tune`. Calls `scripts/benchmark/anomaly_benchmark.py`. Trailing four positionals are BO-tuning hooks (see *Per-trial AUPRC dump + tune-mask split* in the evaluation-framework skill).

`seeds_per_split` only applies in BiGG split-bundle mode: each of the bundle's splits is repeated with the same `SEED_LIST[:seeds_per_split]` seeds, so the per-trial variance reflects both seed and split rather than confounding them. Total runs in bundle mode = `#splits × seeds_per_split`. In single-variant mode and CGT mode the flag is ignored and the legacy one-seed-per-trial rotation is used.

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

# BiGG subsampled run (standard: load_subsamples) — benchmark auto-combines subgraph_* files into a block-diagonal graph
bash scripts/benchmark/run_anomaly_benchmark.sh tolokers GCN,GIN 1 bigg blksize_-1_b_1_lr_0.001_epochs_50_..._loadsub_ff_b0.5_M1_split0 hidden_labels

# BiGG split bundle: my_5_splits/ is a directory containing exactly 5 per-split BiGG variant subdirs
# (each tagged *_split{0..4}_n5). Trials is forced to the bundle size; each trial loads its own
# variant + matching original-data split. Use when splits were tuned to different hparams.
bash scripts/benchmark/run_anomaly_benchmark.sh tolokers GCN,GIN,GraphSAGE,XGBGraph,XGBoost 5 bigg my_5_splits hidden_labels
```

**`bash scripts/benchmark/run_link_benchmark.sh [datasets] [models] [trials] [generator] [synthetic_name] [neg_sampling] [decoder] [task]`**
Link prediction benchmark. Defaults: `reddit`, `GCN,GIN,GraphSAGE`, `1`, `cgt`, `""`, `random`, `dot`, `hidden_links`. Calls `scripts/benchmark/link_benchmark.py`.

* `neg_sampling`: `random` (uniform) or `hard` (2-hop random walks).
* `decoder`: `dot` (dot product, no params) or `mlp` (learnable Hadamard-product scorer).

For `generator=cgt`: the benchmark resolves per-trial `.pt` files `{stem}/{stem}_t{t}.pt` for `t in 0..trials-1` (via `resolve_cgt_trial_paths`). If all files exist, each trial loads its own synthetic data; otherwise it falls back to a single `{stem}/{stem}.pt` with a warning. Three comparison rows are emitted per model: `original` (full-graph LP), `original-cg` (merged-CG LP on original graph), `synthetic-cgt` (merged-CG LP on per-trial CGT-generated hybrid graph). Alignment between the .pt's `hidden_test_edges` and the trial's `LinkDataset.split(trial_id)` is asserted before each synthetic-cgt trial.

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
* `normalize`: Feature normalisation method (`zscore`, `minmax`, `row`, `quantile`, `cdf`, or `none`). `quantile` uses rank-based inverse normal transform (any distribution → N(0,1)). `cdf` uses the empirical CDF (any distribution → Uniform[0,1]) and couples to either the default sigmoid+BCE continuous head or `--mdn_feat --mdn_base logit_normal`. Mutually exclusive with `hetero_feat`, `cat_feat`, and `mdn_feat` with the default Gaussian base.
* `loss_weights`: Comma-separated cont,label weights relative to struct (e.g., `0.1,0.1`).
* `hetero_feat`: `true` for heteroscedastic feature prediction (mean + variance).
* `mask_test_labels`: `true` to exclude test node labels (split 0) from label loss, preventing data leakage in anomaly benchmarks. Appends `_masked` to save name.
* `logvar_floor`: Lower clamp for log-variance in hetero_feat mode (default: -4.0).
* `binary_feat`: `true` to auto-detect binary feature columns and use BCE loss + Bernoulli sampling instead of Gaussian head. Appends `_binfeat` to save name. Binary columns skip normalization.

**`bash scripts/train/train_bigg_subsample.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim] [noise_std] [ss_max_prob] [ss_start_epoch] [bfs_preprocess] [normalize] [loss_weights] [hetero_feat] [mask_test_labels] [logvar_floor] [subsample_size] [burn_prob] [num_subgraphs] [binary_feat] [vae_feat] [vae_dim] [kl_weight] [cat_feat] [n_bins] [bin_sigma] [mdn_feat] [mdn_components] [mdn_logsigma_floor] [mdn_base] [kl_schedule] [kl_anneal_epochs] [kl_cycle_epochs] [kl_ramp_ratio] [load_subsamples] [subsampling_config] [split_id] [recal_momentum] [min_subgraph_nodes]`**
Train BiGG on pre-generated subgraphs for VRAM-limited training. The **standard mode** is `load_subsamples=true` — every variant trains on an identical subgraph set from disk so model comparisons are fair. Same args as above, plus:
* `load_subsamples`: `true` (standard) to load pre-computed training subgraphs from `datasets/bigg_subsamples/<dataset>/<subsampling_config>/split<split_id>.pkl` (generated by `experiments/subsample_search/run_grid.py`). When on, `subsample_size` / `burn_prob` / `num_subgraphs` / `subsample_method` are ignored; the runtime forest-fire/metis path is skipped entirely. Save name uses `_loadsub_<cfg>_split<id>`. Default position value is `false` for backwards compat, but production runs set it to `true`.
* `subsampling_config`: `params_tag` of the persisted subsample set (e.g. `ff_b0.5_M1`, `metis_K5`, `bfs_part_mn32`, `bfs_snow_ts500_mn8_dinf_Minf`). Required when `load_subsamples=true`. Pick from the available directories under `datasets/bigg_subsamples/<dataset>/`.
* `split_id`: GADBench split id (0..4) selecting which `split<id>.pkl` to load (default: 0). Each pickle holds one (config × split) trial with multiple subgraphs; one slurm job trains on one split.
* `subsample_size` / `burn_prob` / `num_subgraphs`: **legacy runtime-sampling knobs** (defaults 2000 / 0.3 / auto). Used only in the unsupported `load_subsamples=false` path; ignored under the standard workflow. Save-name suffix in legacy mode is `_sub{N}_size{S}_p{B}`. The capacity benchmark (`train_bigg_capacity.sh`) is the one tool that still exercises this path, because it exists to time the sampler.
* `vae_feat`: `true` to add a per-node label-agnostic CVAE latent shared across feature decoders (default: false). Appends `_vae{dim}_kl{weight}` to save name.
* `vae_dim`: latent dimensionality when `vae_feat` is on (default: 16).
* `kl_weight`: coefficient on the KL term in the VAE ELBO (default: 1.0). Not subject to dynamic calibration.
* `cat_feat`: `true` to use the AR categorical feature predictor (quantile bins + value-space soft-label cross-entropy). Mutually exclusive with `hetero_feat` and `vae_feat`. Appends `_cat{n_bins}_s{bin_sigma}` to save name and persists `cat_bins.pt` alongside the generated graph(s).
* `n_bins`: number of quantile bins per continuous feature when `cat_feat` is on (default: 32).
* `bin_sigma`: Gaussian soft-label std in feature-value units. Leave empty for auto per-feature sigma (0.5 × median positive spacing of *that* feature's bin centers — adapts across regimes: small for dense continuous, large for binary-like, scales automatically under k-means privacy). Scalar override broadcast to all features.
* `mdn_feat`: `true` to use Mixture Density Network feature head (per-feature K-component mixture conditioned on `[h, label_embed, (z)]`). Mutually exclusive with `hetero_feat` / `cat_feat`. Composes with `vae_feat` (z is concatenated into the head's conditioning input). Appends `_mdn{K}_lsf{floor}` (or `_lnmdn{K}_lsf{floor}` for `logit_normal`) to save name.
* `mdn_components`: number of mixture components per feature when `mdn_feat` is on (default: 8).
* `mdn_logsigma_floor`: lower clamp for MDN component log-sigma (default: -4.0).
* `mdn_base`: per-component base distribution — `gaussian` (default; unbounded targets) or `logit_normal` (targets in [0,1]; pairs with `--normalize cdf`). Use `logit_normal` whenever feature targets are CDF-encoded.
* `kl_schedule`: KL annealing schedule for the VAE coefficient — `none` (default; constant `kl_weight`), `linear` (warmup then constant), or `cyclic` (Fu et al. NAACL 2019 — repeated reset+ramp). Use to address posterior collapse when `vae_feat` is on. Schedule produces β ∈ [0, 1]; the effective KL coefficient is `β × kl_weight`. Appends `_klan{N}` (linear) or `_klcyc{C}r{int(R*100)}` (cyclic) to save name when `vae_feat` is on.
* `kl_anneal_epochs`: linear ramp duration (epochs) — β ramps 0 → 1 over this many epochs, then sits at 1. Required when `kl_schedule=linear`.
* `kl_cycle_epochs`: cyclic schedule cycle length (epochs) — β resets to 0 every cycle. Required when `kl_schedule=cyclic`. Paper M = `ceil(epochs / kl_cycle_epochs)`.
* `kl_ramp_ratio`: cyclic ramp fraction within each cycle (paper R, default 0.5). 0.5 → first half of cycle ramps 0→1, second half sits at 1. Lower values = faster ramp, longer fix phase.
* `recal_momentum`: EMA momentum for dynamic loss-weight recalibration in `[0,1]` (default 1.0 disables recalibration — bit-identical to one-time calibration). With `m<1`, each epoch updates `ema = m*ema + (1-m)*current` for struct/cont/bin/label and recomputes `w_cont/w_bin/w_label` from the EMA ratios. `user_w_cont`/`user_w_label` from `loss_weights` still apply as multiplicative priors. KL is unaffected (annealed separately via `kl_schedule`). Effective horizon ≈ `1/(1-m)` epochs: `0.9` ≈ 10 epochs, `0.99` ≈ 100 epochs. Appends `_recalM{value}` to the save name when `m<1`.
* `min_subgraph_nodes`: absolute node-count floor applied to partitions loaded under `load_subsamples=true` (default 0 = no filter). Drops persisted partitions with fewer than N nodes after pickle load, before training. Targets the tiny remnants (2–4 nodes) that disjoint partitioners like `bfs_partition` and `M=1` forest_fire emit when BFS seeds land in isolated subcomponents. METIS / depth-only BFS snowball naturally produce no such remnants. Filter is silent when 0; logs `Filtered N partitions below X nodes (sizes: [...])` when active. The mirror in `experiments/bo_tuning/bigg_invoke.py:_build_args` is the 38th positional — keep it in lockstep when editing the .sh.

`SAVE_MODEL` env-var (default `false`): when `true`, persists `model.state_dict` + reconstruction args (`cmd_args`, `pipeline_args`, `feat_dim`, `num_classes`, `binary_idx`) to `model.pt` alongside the synthetic outputs after training. Subsample branch writes `<out_dir>/model.pt`; single-graph branch writes `<save_dir>/<save_name>_model.pt`. Consumed as an env-var rather than a positional arg; the slurm wrapper exports it before calling the shell script. Used by offline diagnostics in `experiments/diagnostics/`.

`SAVE_MODEL_EVERY` env-var (default `0`, requires `SAVE_MODEL=true`): when `> 0`, also writes a `model_epoch{N}.pt` snapshot every N epochs during training (same payload as `model.pt`, plus `epoch` field). Subsample branch writes `<out_dir>/model_epoch{N}.pt`; single-graph branch writes `<save_dir>/<save_name>_model_epoch{N}.pt`. Lets diagnostics measure overfitting trajectories — e.g. how MDN label-embed sensitivity evolves across training. The final `model.pt` is always written at end of training when `SAVE_MODEL=true`.

**`bash scripts/train/train_bigg_structure.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim]`**
Train BiGG structure-only baseline. Defaults: `tolokers 128 1 100 0.001 256`.
Checkpoints saved with `structure_` prefix.

**`bash scripts/train/train_bigg_capacity.sh [...same first 33 args as train_bigg_subsample.sh...] [subsample_method] [multiplicity_cap] [num_train_subgraphs] [num_gen_subgraphs] [timing_log_path]`**
Run a single capacity-benchmark trial — short calibration run that emits a JSON timing log so the orchestrator can extrapolate "how many subgraphs fit in 1 hour at this size/density?". Same first 33 positions as `train_bigg_subsample.sh`; appends positions 34–38:
* `subsample_method`: `forest_fire` (default) or `metis`. Metis dispatches via `experiments/subsample_search/sampling.py::sample_with_method`.
* `multiplicity_cap`: `m1` / `m2` / `minf` (default `minf` = no cap, preserves legacy forest fire). Ignored for metis (M=1 by construction).
* `num_train_subgraphs`: cap inner training loop at first N partitions (empty = all sampled).
* `num_gen_subgraphs`: cap generation loop at first N partitions (empty = same as train).
* `timing_log_path`: write JSON timing log here (status, train/gen seconds, peak VRAM, partition stats). Always written, even on caught failures (pipeline exits with code 2 + structured `status` field on OOM/error).

Usually invoked by `experiments/bigg_capacity/capacity_benchmark.py`, which orchestrates the (dataset × method × partition_size) sweep, classifies results, and appends rows to a CSV with `extrap_K_at_50ep` and `extrap_K_at_300ep` columns. Run via:
```bash
sbatch scripts/train/train_bigg_capacity.slurm     # full sweep on gpu80g
```


**`bash scripts/train/train_cgt.sh [dataset] [gpt_epochs] [cluster_num] [cluster_size] [gpt_batch_size] [cg_depth] [cg_fanout] [num_trials] [task] [cluster_sample_num]`**
Train CGT on `num_trials` GADBench splits (trials 0 to num_trials-1). Defaults: `reddit 50 512 1 128 2 5 10 hidden_labels 5000`. Calls `CGT/train.py` once per trial; set `num_trials=1` for a single-trial run. Idempotent: trials whose `.pt` already exists are skipped (supports SLURM re-submission after timeout).
`trial_id` (per-trial, from the loop) selects the mask column (for `hidden_labels`) or seeds the edge split (for `hidden_links`). `task=hidden_links` additionally reads `--val_ratio` (default 0.05) and `--test_ratio` (default 0.10) from `CGT/args.py`; these must match the downstream `LinkDataset.split` ratios or alignment assertions fail.
`cluster_sample_num` caps how many nodes are subsampled to fit `KMeansConstrained`; the library requires `cluster_num × cluster_size ≤ cluster_sample_num`, so raise this to use a higher `cluster_size` (k-anonymity) without giving up cluster resolution. Set ≥ graph size to fit on all nodes.
Output saved to `datasets/synthetic/cgt/<dataset>/<task>/<variant>/<variant>_t{trial_id}.pt` where variant = `{dataset}_e{epochs}_k{clusters}_c{cluster_size}_d{depth}_f{fanout}_s{cluster_sample_num}`. `.pt` contains: generated sequences, cluster centers, `ids`, `task`, `trial_id`, and (for `hidden_links`) `hidden_test_edges` + `val_ratio` + `test_ratio` for provenance-checking.

### BO Hyperparameter Tuning

**`python -m experiments.bo_tuning.coordinator --dataset {tolokers|questions|weibo} --mode {shared|per_split} [--split_id N] [--n_trials N] [--max_trials N] [--max_wall_seconds S] [--study_version v1]`**
Bayesian-optimization tuner over BiGG's `lr`, `kl_weight`, `lw_cont`, `lw_label` at privacy=0. Runs entirely in the unified `bigg` conda env — coordinator and benchmark subprocess both (env name in `configs/{dataset}.yaml::benchmark.conda_env`, resolved at runtime via `conda info --json` so the same config works on any host). Resumable: re-launching against the same study DB picks up where it left off; `cleanup_stale.py` (auto-called on start) re-enqueues trials killed by walltime.

* `shared` mode → one Optuna study per dataset; each trial trains all 5 splits and the BO objective is `CVaR_60%(split_gap) − 0.2·n_collapsed_splits`.
* `per_split` mode → 5 independent studies per dataset (one per split); each trial trains 1 split; objective degenerates to `mean_m(gap[m,s]) − 0.2·collapse_indicator`.
* `--max_trials N` caps trials per process (set to 1 for array workers under topology C).
* `--max_wall_seconds S` stops accepting new trials past this wall-clock — pair with SLURM walltime minus a 10 min grace.
* BO selection runs against 50% of test nodes (anomaly-stratified, fixed seed). The other 50% is held out for `experiments.bo_tuning.final_report`.
* Phase 1 (original-data baseline) is identical across every trial in a study, so the coordinator builds it once at startup and caches `experiments/bo_tuning/{dataset}/baselines/{study_version}.csv` + `.json` (config it was built with). Each trial's benchmark runs with `--skip_original`; `run_trial.py` merges the cached rows back in before `compute_objective`. Config mismatch with the cache raises — bump `--study_version` to rebuild. Concurrent jobs (shared + per_split) serialize the build on an `fcntl.LOCK_EX` flock on `<csv>.lock`, write to per-process tmp paths, then `os.replace` atomically.
* BO BiGG outputs land under `experiments/bo_tuning/{dataset}/bigg_cache/{save_name}/`, **not** in the canonical `datasets/synthetic/bigg/{dataset}/hidden_labels/`. The pipeline reads `BIGG_SYNTHETIC_SAVE_ROOT` and writes there when set; the BO coordinator sets it via subprocess env. Cache is shared across shared+per_split studies for a dataset, so warm-start is trained once and reused. Same-(HP, split) collisions serialize on `fcntl.LOCK_EX` flock on `<out_dir>.lock` in `train_one_split`; loser returns `skipped_after_wait`. Canonical pipeline location stays untouched until you manually re-train BiGG with the best HPs post-BO.
* SLURM templates: `experiments/bo_tuning/slurm/bo_coordinator.slurm` (Topology A, persistent — the default), `bo_worker.slurm` (Topology C, array — opt-in for intra-trial parallelism on weibo shared mode if ever needed). The file split is by topology, **not** by mode; both templates accept `MODE=shared|per_split` via env var.
* After the study finishes: `python -m experiments.bo_tuning.aggregate ...` rebuilds `summary.csv` + `best_params.json`; `python -m experiments.bo_tuning.final_report --dataset X --mode shared` re-evaluates best HPs on the held-out test portion with real-data baseline rotated through the same splits.
* Metadata for the thesis: per-trial `metadata.json` (trial dir) + a flat `trial_log.jsonl` carry per-(split, model, seed) AUROC/AUPRC/RecK, gap ratios, base rate, BO source, wall-clock, git commit, SLURM job id.

**Concrete submission cheat sheet (IDUN).** Always submit from project root. After walltime kill, just resubmit the same command — the coordinator resumes from the SQLite study and skips already-trained splits via cached BiGG outputs.

```bash
# Smoke test first — 1 trial, short walltime
sbatch --export=ALL,DATASET=tolokers,MODE=shared,N_TRIALS=1 \
       --time=02:00:00 experiments/bo_tuning/slurm/bo_coordinator.slurm

# Real studies
# tolokers shared (40 trials; ~5 walltime cycles at 12h)
sbatch --export=ALL,DATASET=tolokers,MODE=shared,N_TRIALS=40 \
       experiments/bo_tuning/slurm/bo_coordinator.slurm

# questions shared (40 trials; ~8 walltime cycles)
sbatch --export=ALL,DATASET=questions,MODE=shared,N_TRIALS=40 \
       experiments/bo_tuning/slurm/bo_coordinator.slurm

# weibo per_split — one array submission, 5 tasks. SPLIT_ID is auto-derived
# from SLURM_ARRAY_TASK_ID by the template. %5 caps concurrent tasks. weibo
# per-split BiGG train time is comparable to tolokers (~30 min), not the older
# 90 min figure — the same walltimes work.
sbatch --array=0-4%5 \
       --export=ALL,DATASET=weibo,MODE=per_split,N_TRIALS=20 \
       experiments/bo_tuning/slurm/bo_coordinator.slurm

# Held-out final report after a study is done
sbatch --wrap="python -m experiments.bo_tuning.final_report --dataset tolokers --mode shared"
```

Recommendation per dataset: run **both shared and per_split×5** for tolokers/questions/weibo when you can. weibo per-split BiGG time ≈ tolokers (~30 min); the same `--time=50h` shared / `--time=30h` per_split walltimes work across all three.

### Subsample Search (Cluster)

Three SLURM templates in `scripts/subsample_search/` drive the §3 sampler grid on IDUN. All three run in the unified `bigg` env (which carries the GADBench dep set — xgboost/catboost/sklearn — alongside BiGG; see also the dryrun's `splitsource.py` which loads via `dgl.load_graphs`). Each template declares empty config vars at the top — edit before sbatch:

* **`plan_grid.slurm`** (CPU) — runs `experiments.subsample_search.plan_grid` for one dataset. Vars: `DATASET`, `EXCLUDE_METHODS` (e.g. `metis` for tfinance), `SPLIT_ID`, `GPU_VRAM_GB`, `K_FLOOR`, `BUDGET_S`, `TARGET_FULL_EPOCHS`. Writes `artifacts/grid/plan.csv` rows for that dataset.
* **`run_grid.slurm`** (GPU 80g, 24h) — runs `experiments.subsample_search.run_grid` for one dataset. Vars: `DATASETS`, `EXCLUDE_METHODS`, `METHODS`, `PARAMS_TAGS`, `SPLITS`, `OUT_DIR` (use per-dataset to avoid CSV races with parallel jobs), `PLAN_CSV`, `EPOCHS`, `PATIENCE`. Resumable via `utility.csv`.
* **`compute_baselines.slurm`** (GPU 32g, 4h) — runs `experiments.subsample_search.compute_baselines --datasets $DATASETS`. The script's hardcoded `DATASETS = ['tolokers', 'questions', 'weibo', 'reddit']` default is unchanged; pass yelp/tfinance explicitly via the SLURM var.

Both `plan_grid.py` and `run_grid.py` accept `--exclude_methods <m1,m2,...>` to drop methods after enumeration. METIS scales poorly on tfinance (~21M edges) — set `EXCLUDE_METHODS=metis` on the tfinance templates. `run_grid.py`'s `--exclude_methods` is applied after `--methods`/`--params_tags` so they compose.

### Environment Setup

```bash
bash scripts/env_setups/bigg_setup.sh     # Creates bigg env, compiles tree_clib C++ extension
bash scripts/env_setups/cgt_setup.sh      # Creates CGT env from CGT/cgt_env.yml
bash scripts/env_setups/gadbench_setup.sh # Creates GADBench env with DGL + ML libraries
```
