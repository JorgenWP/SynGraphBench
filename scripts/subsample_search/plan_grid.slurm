#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="SUBSAMPLE_PLAN"
#SBATCH --time=02:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=eirimhau@stud.ntnu.no
#SBATCH --mail-type=FAIL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

# --- Config (edit before sbatch) ---
DATASET=                # e.g. yelp | tfinance | questions | reddit | amazon
EXCLUDE_METHODS=        # e.g. metis (recommended for tfinance)
SPLIT_ID=0              # split id to dry-run on (n/m distributions are split-stable)
GPU_VRAM_GB=80          # VRAM gate; cells with predicted peak VRAM > oom_factor*this are dropped
K_FLOOR=5               # drop iterative cells with predicted K below this
BUDGET_S=3600           # per-cell training+gen budget the cost model is solving for
TARGET_FULL_EPOCHS=300

if [ -z "$DATASET" ]; then
    echo "ERROR: DATASET unset" >&2
    exit 1
fi

module purge
module load Anaconda3/2024.02-1

EXTRA=""
[ -n "$EXCLUDE_METHODS" ] && EXTRA="$EXTRA --exclude_methods $EXCLUDE_METHODS"

conda run --no-capture-output -n GADBench python -m experiments.subsample_search.plan_grid \
    --dataset "$DATASET" \
    --split_id "$SPLIT_ID" \
    --gpu_vram_gb "$GPU_VRAM_GB" \
    --K_floor "$K_FLOOR" \
    --budget_s "$BUDGET_S" \
    --target_full_epochs "$TARGET_FULL_EPOCHS" \
    $EXTRA
