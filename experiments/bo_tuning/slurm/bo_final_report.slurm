#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="BO_FINAL_REPORT"
#SBATCH --time=02:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu32g
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=eirikmhauge123@gmail.com
#SBATCH --mail-type=END,FAIL

# Held-out evaluation of the best BO HPs. Reuses the BiGG cache from the BO
# study; fails fast if a best-trial save_dir is missing (no retraining).
#
# Fill in the three config vars below before submitting. SPLIT_ID is taken
# from SLURM_ARRAY_TASK_ID for per_split mode.
#
# Submission examples:
#   sbatch --array=0-4 experiments/bo_tuning/slurm/bo_final_report.slurm     # per_split
#   sbatch              experiments/bo_tuning/slurm/bo_final_report.slurm     # shared

# ----- edit these before sbatch -----
DATASET=
MODE=
STUDY_VERSION=
# ------------------------------------

set -e

if [ -z "$DATASET" ]; then
    echo "ERROR: DATASET is empty — edit the script before submitting" >&2
    exit 1
fi
if [ -z "$MODE" ]; then
    echo "ERROR: MODE is empty — edit the script before submitting" >&2
    exit 1
fi
if [ -z "$STUDY_VERSION" ]; then
    echo "ERROR: STUDY_VERSION is empty — edit the script before submitting" >&2
    exit 1
fi

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from: $SLURM_SUBMIT_DIR"
echo "Job:          $SLURM_JOB_NAME ($SLURM_JOB_ID, array task ${SLURM_ARRAY_TASK_ID:-none})"
echo "Nodes:        $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1
conda activate bigg

EXTRA_ARGS=""
if [ "$MODE" = "per_split" ]; then
    if [ -z "${SPLIT_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        SPLIT_ID="$SLURM_ARRAY_TASK_ID"
        echo "Derived SPLIT_ID=$SPLIT_ID from SLURM_ARRAY_TASK_ID"
    fi
    if [ -z "${SPLIT_ID:-}" ]; then
        echo "ERROR: per_split mode requires SPLIT_ID (use --array submission)" >&2
        exit 1
    fi
    EXTRA_ARGS="--split_id $SPLIT_ID"
fi

echo "Dataset:        $DATASET"
echo "Mode:           $MODE"
echo "Study version:  $STUDY_VERSION"
echo ""

python -m experiments.bo_tuning.final_report \
    --dataset "$DATASET" \
    --mode "$MODE" \
    --study_version "$STUDY_VERSION" \
    $EXTRA_ARGS
