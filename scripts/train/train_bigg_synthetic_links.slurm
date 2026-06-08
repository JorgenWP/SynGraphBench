#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="BiGG_SYN_LINKS"
#SBATCH --time=01:30:00

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1

#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --array=0-4%5       # 5 splits (no K dimension); max 5 concurrent

#SBATCH --mail-user=eirimhau@stud.ntnu.no
#SBATCH --mail-type=FAIL

# Generate the NON-anonymized synthetic BiGG graphs for the hidden_links task.
# These are the Delta_syn reference for the LP privacy table: identical
# generative config to the anonymized bundles (same per-split BO-tuned HPs)
# EXCEPT no k-anonymity clustering. The 5 array tasks (one per split) all write
# their per-split variant dir into ONE shared bundle dir, which the link
# benchmark then consumes in bundle mode.
#
# Prerequisite: scripts/preprocess/build_hidden_links_subsamples.slurm must have
# been run for this dataset using the SAME subsampling_config as
# experiments/bo_tuning/configs/<dataset>.yaml (fixed_hp.subsampling_config), so
# the pruned pkls and held_out_edges sidecar exist at
# datasets/bigg_subsamples/<dataset>/hidden_links/.

# === EDIT BEFORE SUBMITTING ===
DATASET=""             # e.g. tolokers / weibo / questions / amazon / reddit / yelp
TASK=hidden_links      # hidden_links (this script's purpose) | hidden_labels
# ==============================

if [ -z "$DATASET" ] || [ -z "$TASK" ]; then
  echo "ERROR: DATASET and TASK must be set at the top of this script." >&2
  exit 1
fi

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job:    $SLURM_JOB_NAME"
echo "ID of job:      $SLURM_JOB_ID"
echo "Array task ID:  $SLURM_ARRAY_TASK_ID"
echo "Dataset:        $DATASET"
echo "Task:           $TASK"
echo "Nodes:          $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1
conda activate bigg

# Array index == split id (0..4)
SPLIT=$SLURM_ARRAY_TASK_ID

echo "Cell: split=$SPLIT  (non-anonymized)"
echo ""

# Resolve config: helper prints bash-eval-friendly KEY=value lines. Omitting
# --K selects non-anon mode (empty cache block + synthetic_baseline SAVE_ROOT).
HELPER_OUT=$(python scripts/train/launch_anonymized_args.py \
  --dataset "$DATASET" --task "$TASK" --split "$SPLIT")
if [ $? -ne 0 ]; then
  echo "ERROR: launch_anonymized_args.py failed for split=$SPLIT" >&2
  echo "$HELPER_OUT" >&2
  exit 1
fi
eval "$HELPER_OUT"

echo "SAVE_ROOT:           $SAVE_ROOT"
echo "CLUSTER_CACHE_DIR:   ${CLUSTER_CACHE_DIR:-(none — non-anon)}"
echo "BO trial_number:     $BO_TRIAL_NUMBER (objective=$BO_OBJECTIVE)"
echo "BO lr:               $BO_LR"
echo "BO kl_weight:        $BO_KL_WEIGHT"
echo "BO lw_cont,lw_label: $BO_LW_CONT,$BO_LW_LABEL"
echo ""

mkdir -p "$SAVE_ROOT"
export BIGG_SYNTHETIC_SAVE_ROOT="$SAVE_ROOT"

bash scripts/train/train_bigg_subsample.sh "${TRAIN_ARGS[@]}"
RC=$?
if [ $RC -ne 0 ]; then
  echo "ERROR: train_bigg_subsample.sh exited with code $RC" >&2
  exit $RC
fi

# Locate THIS split's out_dir. SAVE_ROOT is shared across all 5 array tasks, so
# match on the _split<SPLIT>_ tag rather than picking the first dir.
OUT_DIR=$(ls -d "$SAVE_ROOT"/*_split${SPLIT}_*/ 2>/dev/null | head -n1)
OUT_DIR=${OUT_DIR%/}
if [ -z "$OUT_DIR" ] || [ ! -d "$OUT_DIR" ]; then
  echo "WARN: no _split${SPLIT}_ out_dir under $SAVE_ROOT after training" >&2
else
  python - <<PYEOF
import json, os, subprocess
out_dir = "$OUT_DIR"
git_commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
meta = {
    "dataset": "$DATASET",
    "task": "$TASK",
    "split_id": int("$SPLIT"),
    "K": None,
    "anonymized": False,
    "bo_provenance": {
        "trial_number": "$BO_TRIAL_NUMBER",
        "objective": float("$BO_OBJECTIVE"),
        "lr": float("$BO_LR"),
        "kl_weight": float("$BO_KL_WEIGHT"),
        "lw_cont": float("$BO_LW_CONT"),
        "lw_label": float("$BO_LW_LABEL"),
    },
    "git_commit": git_commit,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
}
with open(os.path.join(out_dir, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"wrote metadata.json to {out_dir}")
PYEOF
fi
