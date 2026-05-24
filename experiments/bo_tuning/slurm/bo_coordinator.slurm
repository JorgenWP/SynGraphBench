#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="BO_BIGG"
#SBATCH --time=12:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu32g
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=output/bo_tuning/coordinator_%j.out
#SBATCH --error=output/bo_tuning/coordinator_%j.err
#SBATCH --mail-user=eirikmhauge123@gmail.com
#SBATCH --mail-type=END,FAIL

# Persistent BO coordinator (Topology A).
#
# Defaults to a 12 h walltime; trials run sequentially. If walltime kills
# the coordinator mid-trial, just resubmit — cleanup_stale.py re-enqueues
# the killed trial on restart.
#
# Override per invocation:
#   sbatch --export=ALL,DATASET=tolokers,MODE=shared,N_TRIALS=40 \
#          experiments/bo_tuning/slurm/bo_coordinator.slurm
#   sbatch --export=ALL,DATASET=weibo,MODE=per_split,SPLIT_ID=0,N_TRIALS=20 \
#          experiments/bo_tuning/slurm/bo_coordinator.slurm
#
# Per-split array submission (recommended for the per_split campaign —
# one sbatch, 5 tasks, one per split). SPLIT_ID is auto-derived from
# SLURM_ARRAY_TASK_ID when unset:
#   sbatch --array=0-4%5 \
#          --export=ALL,DATASET=tolokers,MODE=per_split,N_TRIALS=40 \
#          experiments/bo_tuning/slurm/bo_coordinator.slurm

set -e

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from: $SLURM_SUBMIT_DIR"
echo "Job:          $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Nodes:        $SLURM_JOB_NODELIST"

mkdir -p output/bo_tuning

module purge
module load Anaconda3/2024.02-1
conda activate bigg

DATASET="${DATASET:-tolokers}"
MODE="${MODE:-shared}"
N_TRIALS="${N_TRIALS:-40}"
STUDY_VERSION="${STUDY_VERSION:-v1}"

# Leave 10 min for trial cleanup before walltime.
WALL_BUDGET_SEC="${WALL_BUDGET_SEC:-41400}"

EXTRA_ARGS=""
if [ "$MODE" = "per_split" ]; then
    # Allow --array=0-N invocations to drive SPLIT_ID automatically.
    if [ -z "${SPLIT_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        SPLIT_ID="$SLURM_ARRAY_TASK_ID"
        echo "Derived SPLIT_ID=$SPLIT_ID from SLURM_ARRAY_TASK_ID"
    fi
    if [ -z "${SPLIT_ID:-}" ]; then
        echo "ERROR: per_split mode requires SPLIT_ID (or --array submission)" >&2
        exit 1
    fi
    EXTRA_ARGS="--split_id $SPLIT_ID"
fi

echo "Dataset:       $DATASET"
echo "Mode:          $MODE"
echo "n_trials:      $N_TRIALS"
echo "Wall budget:   $WALL_BUDGET_SEC s"
echo "Study version: $STUDY_VERSION"
echo ""

python -m experiments.bo_tuning.coordinator \
    --dataset "$DATASET" \
    --mode "$MODE" \
    --n_trials "$N_TRIALS" \
    --max_wall_seconds "$WALL_BUDGET_SEC" \
    --study_version "$STUDY_VERSION" \
    $EXTRA_ARGS
