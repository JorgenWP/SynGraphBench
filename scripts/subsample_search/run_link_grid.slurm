#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="SUBSAMPLE_LP"
#SBATCH --time=24:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80g
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=eirimhau@stud.ntnu.no
#SBATCH --mail-type=FAIL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

DATASET=                 # e.g. tolokers
NEG_SAMPLING=            # random | hard
DECODER=                 # dot | mlp
TOP_N=                   # e.g. 10
SPLITS=                  # e.g. "0 1 2 3 4"
VAL_RATIO=               # e.g. 0.1
TEST_RATIO=              # e.g. 0.1
EPOCHS=                  # e.g. 200
PATIENCE=                # e.g. 50

if [ -z "$DATASET" ] || [ -z "$NEG_SAMPLING" ] || [ -z "$DECODER" ]; then
    echo "ERROR: DATASET, NEG_SAMPLING, DECODER must be set" >&2
    exit 1
fi

module purge
module load Anaconda3/2024.02-1

GRID_EXTRA=""
BASE_EXTRA=""
[ -n "$TOP_N" ]      && GRID_EXTRA="$GRID_EXTRA --top_n $TOP_N"
[ -n "$SPLITS" ]     && GRID_EXTRA="$GRID_EXTRA --splits $SPLITS"
[ -n "$SPLITS" ]     && BASE_EXTRA="$BASE_EXTRA --splits $SPLITS"
[ -n "$VAL_RATIO" ]  && GRID_EXTRA="$GRID_EXTRA --val_ratio $VAL_RATIO"
[ -n "$VAL_RATIO" ]  && BASE_EXTRA="$BASE_EXTRA --val_ratio $VAL_RATIO"
[ -n "$TEST_RATIO" ] && GRID_EXTRA="$GRID_EXTRA --test_ratio $TEST_RATIO"
[ -n "$TEST_RATIO" ] && BASE_EXTRA="$BASE_EXTRA --test_ratio $TEST_RATIO"
[ -n "$EPOCHS" ]     && GRID_EXTRA="$GRID_EXTRA --epochs $EPOCHS"
[ -n "$EPOCHS" ]     && BASE_EXTRA="$BASE_EXTRA --epochs $EPOCHS"
[ -n "$PATIENCE" ]   && GRID_EXTRA="$GRID_EXTRA --patience $PATIENCE"
[ -n "$PATIENCE" ]   && BASE_EXTRA="$BASE_EXTRA --patience $PATIENCE"

echo "=== Phase 1: full-graph LP baseline ==="
conda run --no-capture-output -n bigg python -m experiments.subsample_search.link_compute_baselines \
    --datasets "$DATASET" \
    --neg_sampling "$NEG_SAMPLING" \
    --decoder "$DECODER" \
    $BASE_EXTRA

echo "=== Phase 2: top-N cross-graph LP sweep ==="
conda run --no-capture-output -n bigg python -m experiments.subsample_search.link_run_grid \
    --dataset "$DATASET" \
    --neg_sampling "$NEG_SAMPLING" \
    --decoder "$DECODER" \
    $GRID_EXTRA
