---
name: slurm-jobs
description: How to submit jobs on the IDUN HPC cluster via SLURM — workflow, template anatomy, dataset reference, and CGT-training argument guide. Use when the user asks to "run", "submit", "train", or "benchmark" anything that needs the cluster.
---

# SLURM Jobs on IDUN

## Workflow

The user keeps **one persistent template per script type** under `scripts/<area>/<job>.slurm`. To run a job:

1. **Edit the template in place** — change SBATCH directives and bash variables for the run.
2. **`sbatch scripts/<area>/<job>.slurm`** — note the returned job ID.
3. **For multiple variants** of the same job (e.g., k=20, 50, 100): edit → sbatch → edit → sbatch, sequentially in one session. One template file, many submissions. Submit immediately after each edit so the SBATCH metadata at submit time matches the variant.

## Available templates

CGT training and benchmarking:

| Path | What it submits |
|---|---|
| `scripts/train/train_cgt.slurm` | CGT generative training + synthetic generation, all `NUM_TRIALS` in one job |
| `scripts/benchmark/cgt_anomaly_benchmark.slurm` | GADBench anomaly detection on CGT synthetic |
| `scripts/benchmark/cgt_link_benchmark.slurm` | GADBench link prediction on CGT synthetic |
| `scripts/pipeline/run_cgt_pipeline.slurm` | End-to-end CGT pipeline (train + benchmark) |

BiGG training and benchmarking:

| Path | What it submits |
|---|---|
| `scripts/train/train_bigg.slurm` | BiGG conditional training (full graph) |
| `scripts/train/train_bigg_subsample.slurm` | BiGG with forest-fire subsampling |
| `scripts/train/train_bigg_subsample_grid.slurm` | BiGG subsample hyperparameter sweep |
| `scripts/train/train_bigg_structure.slurm` | BiGG structure-only baseline |
| `scripts/benchmark/bigg_benchmark.slurm` | GADBench on BiGG synthetic |

## SBATCH directives to update per job

```
#SBATCH --account=share-ie-idi          # keep
#SBATCH --job-name="<DESCRIPTIVE>"      # see naming below
#SBATCH --time=HH:MM:SS                 # estimate generously; see time-estimate guide below

#SBATCH --partition=GPUQ                # keep
#SBATCH --gres=gpu:1                    # keep
#SBATCH --constraint=gpu16g             # bump to gpu40g for large-graph training if VRAM tight
#SBATCH --mem=10G                       # system RAM (NOT GPU memory); raise for large datasets

#SBATCH --output=output/<area>/<phase>/<dataset>/<task>/output_<description>.txt
#SBATCH --error=output/<area>/<phase>/<dataset>/<task>/error_<description>.err

#SBATCH --mail-user=<username>@stud.ntnu.no         # The email to send notifications to; ask user for this if not already set
```

**Output log layout**: nest under `<dataset>/<task>/`. **Do not** encode the task in the filename — it lives in the directory path. So `output/cgt/train/weibo/hidden_labels/output_all_trials_k20.txt`, not `output/cgt/train/weibo/output_hidden_labels_k20.txt`.

**Job name convention**: `<GENERATOR>_<task>_<train/benchmark>_<dataset>_<job-specific-description>`, e.g., `CGT_ANOMALY_TRAIN_weibo_k20_c367` or `CGT_LINK_BENCHMARK_tolokers_k512_dot_random`. The default template usually has placeholder text like `<dataset>` — replace it.

## Submitting and monitoring

```bash
sbatch scripts/train/train_cgt.slurm           # returns "Submitted batch job <JOBID>"
squeue                                         # see queued/running jobs
squeue -o "%.10i %.50j %.10M %.10l %.8T %R"    # wider format
scancel <JOBID>                                # cancel
tail -f output/.../output_<description>.txt    # watch a running job
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode  # post-mortem
```

## Dataset reference

`GADBench/readme.md` (table at the bottom) lists all 12 datasets with **#Nodes, #Edges, #Dim, Anomaly ratio, Train ratio, Relation Concept, Feature Type**. Consult it when sizing `CLUSTER_SAMPLE_NUM`, time limits, or memory.

## Dataset splits

Splits are needed for sizing clustering constraints and for any sanity-checking of train/val/test counts. They depend on the **task**.

### `hidden_labels` (anomaly detection) — GADBench-defined splits

| Dataset | train | val | test | \|train+val\| |
|---|---|---|---|---|
| reddit | 4,393 | 2,196 | 4,395 | 6,589 |
| weibo | 3,362 | 1,681 | — | 5,043 |
| tolokers | 5,879 | 2,939 | 2,940 | 8,818 |
| yelp (YelpChi) | 32,167 | 4,595 | 9,192 | 36,762 |

For other datasets, consult log output from previous runs to see the actual splits. 

### `hidden_links` (link prediction) — different split scheme

Link prediction has no inherent node-level test set; all nodes are observed. CGT instead does a deterministic **80/20 train/val node split** for GPT training (`CGT/task/utils/utils.py:202`) and strips a fraction of edges (default `test_ratio=0.10`, `val_ratio=0.05` — see `utils.py:124`) as the test/val edge sets. So:

- `|train+val|` ≈ **full graph node count** (much larger than for `hidden_labels`).
- Node-level `test = 0` in the printed split (`hidden_links node split (for GPT training): train=N, val=M, test=0`).
- The "test set" is a small slice of edges, not nodes — it is much smaller than the anomaly-detection node test set.

---

## Clustering parameters

Used by any pipeline that fits constrained k-means on train+val nodes. Three knobs:

- `CLUSTER_SIZE` — minimum cluster size (k-anonymity bound). User-chosen: 1 (no privacy), 20, 50, 100, …
- `CLUSTER_NUM` — number of clusters.
- `CLUSTER_SAMPLE_NUM` — internal subsample of train+val used to fit k-means.

### Constraint

**`CLUSTER_NUM × CLUSTER_SIZE ≤ |train+val|`.** `CLUSTER_SAMPLE_NUM` does NOT enter this bound — k-means fits only on the train+val set (`CGT/generator/cluster.py:71-85`). Violations fail with `ValueError: The product of size_min and n_clusters cannot exceed the number of samples` (`k_means_constrained_.py:325`).

### Picking values

- `CLUSTER_SAMPLE_NUM`: ≥ full graph node count. Lower values silently drop train+val nodes from the fit.
- `CLUSTER_NUM`: default **512**. Reduce to `floor(|train+val| / CLUSTER_SIZE)` only when 512 violates the bound. Don't maximise — at low `CLUSTER_SIZE` on small graphs this collapses toward one node per cluster.

| Dataset (\|train+val\|) | k=20 | k=50 | k=100 |
|---|---|---|---|
| weibo (5,043) | 252 | 100 | 50 |
| yelp (36,762) | 512 | 512 | 367 |

See **Dataset splits** above for `|train+val|` per dataset/task.

---

## Running CGT training

Edit **`scripts/train/train_cgt.slurm`**. The body calls `scripts/train/train_cgt.sh` which loops `python CGT/train.py` over `NUM_TRIALS` trials. Each call trains the GPT model **and** generates the synthetic dataset for that trial in one shot.

The training script (`train_cgt.sh:36-42`) **skips trials whose output `.pt` already exists**. If a job hits the time limit, re-submit the same template — it picks up where it left off. This makes generous-but-not-extreme time limits the right choice: better to re-submit once than over-allocate cluster hours.

### Arguments to set

```bash
DATASET="<dataset_key>"          # e.g., "weibo", "yelp", "reddit"
GPT_EPOCHS=50                    # standard
GPT_BATCH_SIZE=128               # standard
CLUSTER_SIZE=<k>                 # see Clustering parameters
CLUSTER_NUM=<n>                  # see Clustering parameters
CLUSTER_SAMPLE_NUM=<s>           # see Clustering parameters
CG_DEPTH=2                       # standard
CG_FANOUT=5                      # standard
NUM_TRIALS=10                    # standard for full GADBench evaluation
TASK="hidden_labels"             # or "hidden_links" for link prediction
```

### Dataset naming (from `scripts/train/train_cgt.sh:31`)

```
VARIANT = ${DATASET}_e${EPOCHS}_k${CLUSTER_NUM}_c${CLUSTER_SIZE}_d${DEPTH}_f${FANOUT}_s${SAMPLE_NUM}
SAVE_PATH = datasets/synthetic/cgt/${DATASET}/${TASK}/${VARIANT}/${VARIANT}_t${trial}.pt
```

**In dataset filenames `k` = CLUSTER_NUM, `c` = CLUSTER_SIZE.** This is counterintuitive — when the user says "k=20", they mean **CLUSTER_SIZE=20** (the k-anonymity value), which appears as `c20` in the dataset filename.

### Time estimates (10 trials, GPT_EPOCHS=50)

| Dataset | Per trial | All 10 trials | Recommended `--time` |
|---|---|---|---|
| reddit (~11k nodes) | ~20 min | ~3.5 h | `05:00:00` |
| weibo (~8k) | ~20 min | ~3.5 h | `05:00:00` |
| yelp (~46k) | ~80 min | ~13 h | `18:00:00` |
| Larger (>100k) | extrapolate | extrapolate | extrapolate, then add buffer |

Buffers above account for KMeansConstrained slowdown at high `CLUSTER_SIZE` and tight `NUM × SIZE / SAMPLE` ratios. If a job times out, the idempotent skip lets you re-submit safely.

## Running CGT benchmarking

Two templates, one per task: anomaly detection and link prediction.

### Shared arguments

```bash
DATASETS="<dataset_key>"               # e.g., "weibo", "yelp", "reddit"
MODELS="<comma-separated>"             # see supported models below
TRIALS="10"                            # must be ≤ NUM_TRIALS actually trained
SYNTHETIC_NAME="<variant_stem>"        # exactly the VARIANT produced by training
                                       # (e.g., weibo_e50_k512_c1_d2_f5_s8405) — NO _t<trial> suffix
```

`SYNTHETIC_NAME` is the directory name under `datasets/synthetic/cgt/<dataset>/<task>/`. The benchmark loads `<variant>_t0.pt … <variant>_t{TRIALS-1}.pt` from it. If `SYNTHETIC_NAME` is empty, the script falls back to the dataset key.

**Supported models differ by task** (`scripts/benchmark/anomaly_benchmark.py`, `link_benchmark.py:53`):

| Task | Supported models |
|---|---|
| Anomaly detection | `GCN, GIN, GraphSAGE, XGBoost, XGBGraph` |
| Link prediction | `GCN, GIN, GraphSAGE, XGBGraph` (no plain `XGBoost`) |

Output logs follow the same nesting as training: `output/cgt/benchmark/<dataset>/<task>/<description>_output.txt`.

### Anomaly detection benchmark

Edit **`scripts/benchmark/cgt_anomaly_benchmark.slurm`**. No extra arguments beyond the shared set. Typical `--time`: `04:30:00` (single dataset, 10 trials, full model set).

### Link prediction benchmark

Edit **`scripts/benchmark/cgt_link_benchmark.slurm`**. Two extra arguments:

```bash
NEG_SAMPLING="hard"     # random | hard  (defaults to random in run_link_benchmark.sh)
DECODER="mlp"           # dot | mlp      (defaults to dot)
```

Typical `--time`: `12:00:00` (link prediction is heavier — more train-time computation and `hard` neg sampling further increases cost).
