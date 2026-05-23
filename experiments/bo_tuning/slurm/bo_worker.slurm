#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="BO_BIGG_WORKER"
#SBATCH --time=02:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu32g
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=output/bo_tuning/worker_%A_%a.out
#SBATCH --error=output/bo_tuning/worker_%A_%a.err
#SBATCH --array=0-19%4
#SBATCH --mail-user=eirikmhauge123@gmail.com
#SBATCH --mail-type=FAIL

# Topology C: array worker. Each task pulls one trial from the shared
# Optuna SQLite study, runs it, exits. Concurrency capped by %N in the
# array spec above (e.g. 0-19%4 → 20 trials, 4 concurrent).
#
# Recommended for per_split mode on weibo where each split is its own
# study and you want true parallelism across splits.
#
# Override per invocation:
#   sbatch --export=ALL,DATASET=weibo,MODE=per_split,SPLIT_ID=0 \
#          --array=0-19%5 \
#          experiments/bo_tuning/slurm/bo_worker.slurm

set -e

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from: $SLURM_SUBMIT_DIR"
echo "Array job:    $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID"

mkdir -p output/bo_tuning

module purge
module load Anaconda3/2024.02-1
conda activate bigg

DATASET="${DATASET:-tolokers}"
MODE="${MODE:-per_split}"
N_TRIALS="${N_TRIALS:-20}"
STUDY_VERSION="${STUDY_VERSION:-v1}"

EXTRA_ARGS=""
if [ "$MODE" = "per_split" ]; then
    if [ -z "${SPLIT_ID:-}" ]; then
        echo "ERROR: per_split mode requires SPLIT_ID" >&2
        exit 1
    fi
    EXTRA_ARGS="--split_id $SPLIT_ID"
fi

python -m experiments.bo_tuning.coordinator \
    --dataset "$DATASET" \
    --mode "$MODE" \
    --n_trials "$N_TRIALS" \
    --max_trials 1 \
    --study_version "$STUDY_VERSION" \
    $EXTRA_ARGS
