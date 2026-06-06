#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="AD_PRIVACY_BASELINES"
#SBATCH --time=04:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
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
ANONYMITY_K=              # e.g. "20 50 100 500 1000"  (space-separated)
CACHE_ROOT=               # blank = script default (cache/clustering)

if [ -z "$DATASETS" ] || [ -z "$ANONYMITY_K" ]; then
    echo "ERROR: DATASETS and ANONYMITY_K must be set" >&2
    exit 1
fi

module purge
module load Anaconda3/2024.02-1

EXTRA=""
[ -n "$CACHE_ROOT" ] && EXTRA="$EXTRA --cache_root $CACHE_ROOT"

conda run --no-capture-output -n bigg python -m experiments.subsample_search.compute_baselines \
    --datasets "$DATASETS" \
    --anonymity_k $ANONYMITY_K \
    $EXTRA
