#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="SUBSAMPLE_BASELINES_tolokers"
#SBATCH --time=01:20:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=eirimhau@stud.ntnu.no
#SBATCH --output=output/subsample/baselines/tolokers.txt
#SBATCH --mail-type=FAIL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

# --- Config (edit before sbatch) ---
DATASETS=tolokers               # e.g. yelp | tfinance (comma-separated for multiple)

if [ -z "$DATASETS" ]; then
    echo "ERROR: DATASETS unset" >&2
    exit 1
fi

module purge
module load Anaconda3/2024.02-1

conda run --no-capture-output -n bigg python -m experiments.subsample_search.compute_baselines \
    --datasets "$DATASETS"
