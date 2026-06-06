#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="LINK_PRIVACY_BASELINES"
#SBATCH --time=12:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40g
#SBATCH --mem=32G
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

# --- Config (edit before sbatch) ---
DATASETS=                 # e.g. weibo  (comma-separated for multiple)
NEG_SAMPLING=             # random | hard
DECODER=                  # dot | mlp
ANONYMITY_K=              # e.g. "20 50 100 500 1000"  (space-separated)
SPLITS=                   # e.g. "0 1 2 3 4"  (blank = all 5)
VAL_RATIO=                # e.g. 0.1  (blank = script default)
TEST_RATIO=               # e.g. 0.1  (blank = script default)
EPOCHS=                   # e.g. 100  (blank = script default)
PATIENCE=                 # e.g. 50   (blank = script default)
CACHE_ROOT=               # blank = script default (cache/clustering)

if [ -z "$DATASETS" ] || [ -z "$NEG_SAMPLING" ] || [ -z "$DECODER" ] || [ -z "$ANONYMITY_K" ]; then
    echo "ERROR: DATASETS, NEG_SAMPLING, DECODER, ANONYMITY_K must be set" >&2
    exit 1
fi

module purge
module load Anaconda3/2024.02-1

EXTRA=""
[ -n "$SPLITS" ]     && EXTRA="$EXTRA --splits $SPLITS"
[ -n "$VAL_RATIO" ]  && EXTRA="$EXTRA --val_ratio $VAL_RATIO"
[ -n "$TEST_RATIO" ] && EXTRA="$EXTRA --test_ratio $TEST_RATIO"
[ -n "$EPOCHS" ]     && EXTRA="$EXTRA --epochs $EPOCHS"
[ -n "$PATIENCE" ]   && EXTRA="$EXTRA --patience $PATIENCE"
[ -n "$CACHE_ROOT" ] && EXTRA="$EXTRA --cache_root $CACHE_ROOT"

conda run --no-capture-output -n bigg python -m experiments.subsample_search.link_compute_baselines \
    --datasets "$DATASETS" \
    --neg_sampling "$NEG_SAMPLING" \
    --decoder "$DECODER" \
    --anonymity_k $ANONYMITY_K \
    $EXTRA
