#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="BiGG_PRUNE_LINKS"
#SBATCH --time=00:30:00

#SBATCH --partition=CPUQ
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

#SBATCH --mail-user=eirimhau@stud.ntnu.no
#SBATCH --mail-type=ALL

# === EDIT BEFORE SUBMITTING ===
DATASET=""              # e.g. tolokers / weibo / questions / amazon / reddit / yelp
SUBSAMPLING_CONFIG=""   # params_tag of the source pickles to prune, e.g. uniform_ts2500_K15
# ==============================

if [ -z "$DATASET" ] || [ -z "$SUBSAMPLING_CONFIG" ]; then
  echo "ERROR: DATASET and SUBSAMPLING_CONFIG must be set at the top of this script." >&2
  exit 1
fi

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job:   $SLURM_JOB_ID"
echo "Dataset:     $DATASET"
echo "Config:      $SUBSAMPLING_CONFIG"

module purge
module load Anaconda3/2024.02-1
conda activate bigg

python scripts/preprocess/build_hidden_links_subsamples.py \
  --dataset "$DATASET" \
  --subsampling_config "$SUBSAMPLING_CONFIG" \
  --splits 0 1 2 3 4
