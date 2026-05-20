#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="BiGG_CAPACITY"
#SBATCH --time=04:00:00              # 4h: room for the full sweep + per-trial 1.5h subprocess timeouts.

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80g          # capacity benchmark targets the 80 GB tier.
#SBATCH --mem=32G                    # metis on yelp/questions needs more host RAM than the default 10G.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --output=output/bigg/capacity/output_%j.txt
#SBATCH --error=output/bigg/capacity/error_%j.txt

#SBATCH --mail-user=<something>@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

mkdir -p output/bigg/capacity

module purge
module load Anaconda3/2024.02-1
conda activate bigg

# --- Sweep configuration ---
DATASETS="tolokers,reddit,weibo,questions,amazon,yelp"
PARTITION_SIZES="500,1500,4000,10000"
METHODS="metis,ff_b0.5_M2"
EPOCHS=5
NUM_TRAIN_SUBGRAPHS=5
NUM_GEN_SUBGRAPHS=1
TARGET_FULL_EPOCHS="50,300"
TIME_BUDGET_SEC=3600
CSV_OUT="experiments/bigg_capacity/results.csv"

python experiments/bigg_capacity/capacity_benchmark.py \
  --datasets "$DATASETS" \
  --partition_sizes "$PARTITION_SIZES" \
  --methods "$METHODS" \
  --epochs "$EPOCHS" \
  --num_train_subgraphs "$NUM_TRAIN_SUBGRAPHS" \
  --num_gen_subgraphs "$NUM_GEN_SUBGRAPHS" \
  --target_full_epochs "$TARGET_FULL_EPOCHS" \
  --time_budget_sec "$TIME_BUDGET_SEC" \
  --csv_out "$CSV_OUT"
