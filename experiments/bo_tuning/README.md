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
- Per-split BiGG outputs land in their canonical
  `datasets/synthetic/bigg/{dataset}/hidden_labels/{save_name}/` directory —
  trial bundles are symlinks to those. Two trials that land on the same HPs
  share the cache.

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
experiments/bo_tuning/{dataset}/{shared|per_split}/
  study.db                          # per_split: split{N}/study.db
  trial_log.jsonl                   # one JSON line per completed trial
  summary.csv                       # rebuilt by aggregate.py
  best_params.json                  # rebuilt by aggregate.py
  trials/trial_{NNNN}/
    bundle/                         # symlinks to canonical BiGG outputs
    benchmark/
      evaluation_results.csv        # aggregated mean/std (legacy schema)
      per_trial_results.csv         # raw per-(split, model, seed) AUPRC/AUROC/RecK
    logs/{split0..4,benchmark}.{out,err}
    metadata.json                   # full thesis metadata blob
  final_report/
    held_out_final_report.csv       # held-out portion, after final_report.py
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

## Resumability — what survives a kill?

| Failure | Recovery |
|---|---|
| Coordinator killed mid-trial | `cleanup_stale.py` (auto, on next start) marks RUNNING trials >6 h old as FAILED and re-enqueues their params. |
| Killed after 3 of 5 splits trained | Per-split BiGG output dirs persist by their HP-determined name → next resume detects them and skips re-training. |
| Benchmark crashed after train succeeded | Bundle symlinks persist; re-run only re-executes the benchmark. |
| SQLite contention (per-split mode = 5 concurrent writers per dataset) | Each per-split study has its own DB file, so contention only matters in topology C. |

## Methodology

See the plan file at `/home/eirik/.claude/plans/sounds-good-we-also-majestic-sunrise.md`.
