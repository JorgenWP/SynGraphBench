# BO hyperparameter tuning for BiGG

Bayesian optimization over BiGG's `-learning_rate`, `-kl_weight`, and the
two `-loss_weights` components. Tuning happens at privacy=0 so the later
privacy sweep doesn't conflate capacity sensitivity with privacy sensitivity.

## Environment assumptions

The coordinator runs in the **`bigg`** conda env (Optuna + BiGG installed
there). The benchmark subprocess runs in the **`GADBench`** conda env
(xgboost, dgl, the rest). Configs reference the benchmark env *by name*
(`conda_env: GADBench`); the coordinator resolves the absolute python
binary via `conda info --json` at startup, so the same configs work on any
host — local laptop or IDUN. If the cluster's env is named differently,
edit the YAML's `conda_env`, or override with `python: /abs/path` to
bypass the lookup.

## Invariants

- BO sees only **50%** of test nodes (anomaly-stratified, fixed seed). The
  other half is reserved for the final report — never selected against.
- Objective = `CVaR_60%(split_gap) − 0.2 · n_collapsed_splits` where
  `split_gap = mean_models(auprc_synth / auprc_real)`. CVaR over splits is
  the mean of the worst 3 of 5 — heavy-tail aware, no σ estimation noise.
- Downstream classifier HPs are **not tuned** (would advantage synthetic
  unfairly). 3 seeds per (split, classifier) so BO doesn't chase seed noise.
- Per-split BiGG outputs land under a **BO-scoped per-dataset cache** at
  `experiments/bo_tuning/{dataset}/bigg_cache/{save_name}/` so trial
  scaffolding doesn't accumulate inside the canonical pipeline location.
  Trial bundles are symlinks into this cache. Two trials with the same
  HPs share the cache (including across the shared and per_split[i]
  studies for a given dataset — warm-start is trained once and reused
  6×). Implemented via `BIGG_SYNTHETIC_SAVE_ROOT` env var read by
  `bigg/bigg/extension/pipeline.py`. `train_one_split` holds an
  `fcntl.LOCK_EX` flock on `<out_dir>.lock` while writing, so concurrent
  jobs that collide on the same (HP, split) serialize cleanly: the
  loser waits, re-checks on lock release, and returns
  `skipped_after_wait`. The canonical
  `datasets/synthetic/bigg/{dataset}/hidden_labels/` stays untouched
  during BO — you re-train BiGG once with the best HPs after the study
  (see `final_report.py`).
- The original-data baseline (Phase 1 of `anomaly_benchmark.py`) is
  identical across every trial in a study. The coordinator builds it once
  at study startup, caches `experiments/bo_tuning/{dataset}/baselines/{study_version}.csv`
  alongside `{study_version}.json` (the config it was built with), and
  passes `--skip_original` to each trial's benchmark. Trials merge the
  cached rows back in before `compute_objective`. A config mismatch
  between cache and current YAML raises — bump `--study_version` to
  rebuild deliberately.

## Files

| File | Purpose |
|---|---|
| `search_space.py` | Optuna distributions + per-dataset warm-start defaults |
| `objective.py` | Composite CVaR + collapse-penalty scoring of a trial's per-trial CSV |
| `bigg_invoke.py` | Byte-exact save_name builder + subprocess wrapper around `train_bigg_subsample.sh` |
| `run_trial.py` | One trial: train splits → bundle → benchmark → score |
| `coordinator.py` | Open/resume Optuna study, ask→tell loop |
| `cleanup_stale.py` | Mark stale RUNNING trials FAILED + re-enqueue (recovery from killed coordinator) |
| `aggregate.py` | Rebuild flat `summary.csv` + `best_params.json` from study DB |
| `final_report.py` | After BO ends, re-evaluate best HPs on the held-out test portion |
| `configs/{dataset}.yaml` | Per-dataset fixed HPs, search-space overrides, benchmark args |
| `slurm/bo_coordinator.slurm` | Persistent coordinator job (Topology A) |
| `slurm/bo_worker.slurm` | Array worker (Topology C — opt-in for weibo if you ever need intra-trial parallelism) |

## Output layout

```
experiments/bo_tuning/{dataset}/
  bigg_cache/{save_name}/           # BO-scoped BiGG outputs (shared across studies)
  baselines/{study_version}.csv     # cached original-data benchmark
  baselines/{study_version}.json    # config sig (cache-mismatch guard)
  {shared|per_split}/
    study.db                        # per_split: split{N}/study.db
    trial_log.jsonl                 # one JSON line per completed trial
    summary.csv                     # rebuilt by aggregate.py
    best_params.json                # rebuilt by aggregate.py
    trials/trial_{NNNN}/
      bundle/                       # symlinks into bigg_cache/
      benchmark/
        evaluation_results.csv      # aggregated mean/std (legacy schema)
        per_trial_results.csv       # raw per-(split, model, seed) AUPRC/AUROC/RecK
      logs/{split0..4,benchmark}.{out,err}
      metadata.json                 # full thesis metadata blob
    final_report/
      held_out_final_report.csv     # held-out portion, after final_report.py
```

## Usage

Manual (no SLURM):
```
conda activate bigg
python -m experiments.bo_tuning.coordinator \
    --dataset tolokers --mode shared --n_trials 40
```

Per-split mode (one study per split):
```
python -m experiments.bo_tuning.coordinator \
    --dataset weibo --mode per_split --split_id 0 --n_trials 20
```

After the study finishes, the held-out report:
```
python -m experiments.bo_tuning.final_report \
    --dataset tolokers --mode shared
```

Rebuild the summary at any time:
```
python -m experiments.bo_tuning.aggregate \
    --study_db experiments/bo_tuning/tolokers/shared/study.db \
    --study_name bo_bigg_tolokers_shared_v1 \
    --out_dir experiments/bo_tuning/tolokers/shared
```

## SLURM

See `slurm/bo_coordinator.slurm`. The default coordinator walltime is 12 h
and trials run sequentially. If the job hits walltime, just resubmit —
`cleanup_stale.py` re-enqueues the killed trial and Optuna continues.

### Concurrent shared + per_split

For per_split, prefer one array submission over five separate sbatches:

```bash
# Shared HP (one job)
sbatch --export=ALL,DATASET=tolokers,MODE=shared,N_TRIALS=40,STUDY_VERSION=v1 \
       slurm/bo_coordinator.slurm

# Per-split HP (one array job, 5 tasks). SPLIT_ID auto-derived from
# SLURM_ARRAY_TASK_ID. %5 caps simultaneous tasks at 5.
sbatch --array=0-4%5 \
       --export=ALL,DATASET=tolokers,MODE=per_split,N_TRIALS=40,STUDY_VERSION=v1 \
       slurm/bo_coordinator.slurm
```

Both can be submitted back-to-back; they share the per-dataset
`bigg_cache/` and `baselines/` directories safely:

- **Baseline cache**: whichever job acquires the flock first builds it
  (one benchmark run, ~10–30 min); everyone else waits and hits the cache.
- **BiGG cache**: same-(HP, split) collisions (mostly warm-start trial 0)
  serialize on `<save_name>.lock`; the loser logs `skipped_after_wait`.
- **Optuna DBs**: distinct files per (mode, split), no contention.

After BO finishes and you've picked best HPs, re-train BiGG **without**
`BIGG_SYNTHETIC_SAVE_ROOT` to land the deliverable graph in the
canonical `datasets/synthetic/bigg/{dataset}/hidden_labels/` location:

```bash
bash scripts/train/train_bigg_subsample.sh <dataset> ... <best HPs>
```

## Resumability — what survives a kill?

| Failure | Recovery |
|---|---|
| Coordinator killed mid-trial | `cleanup_stale.py` (auto, on next start) marks RUNNING trials >6 h old as FAILED and re-enqueues their params (use `STALE_AFTER_SEC=600` env override on resubmit if you know the prior job is dead). |
| Killed after 3 of 5 splits trained | Per-split BiGG output dirs persist by their HP-determined name → next resume detects them and skips re-training. |
| Benchmark crashed after train succeeded | Bundle symlinks persist; re-run only re-executes the benchmark. |
| SQLite contention (per-split mode = 5 concurrent writers per dataset) | Each per-split study has its own DB file, so contention only matters in topology C. |

## Methodology

See the plan file at `/home/eirik/.claude/plans/sounds-good-we-also-majestic-sunrise.md`.
