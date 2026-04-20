#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi"
#SBATCH --job-name="CGT_ALL_TRIALS_<dataset>_<description>"
#SBATCH --time=02:30:00             # Time limit (hh:mm:ss)

#SBATCH --partition=GPUQ            # Asking for GPU
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --constraint=gpu16g
#SBATCH --mem=10G                   # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=1                   # Number of compute nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node

#SBATCH --output=output/cgt/train/<dataset>/output_<description>.txt    # Specifying 'stdout'
#SBATCH --error=output/cgt/train/<dataset>/error_<description>.err      # Specifying 'stderr'

#SBATCH --mail-user=<username>@stud.ntnu.no         # The email to send notifications to
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

module load Anaconda3/2024.02-1

# Activate CGT conda environment
conda activate CGT

# Arguments
DATASET="reddit"
GPT_EPOCHS=50
GPT_BATCH_SIZE=128
# Privacy/Clustering (CLUSTER_NUM * CLUSTER_SIZE ≤ CLUSTER_SAMPLE_NUM)
# CLUSTER_SAMPLE_NUM caps how many nodes are subsampled to fit KMeansConstrained.
# Set it >= graph size to fit on all nodes (removes artificial cluster_num cap
# and strengthens the k-anonymity guarantee). Default 5000 matches legacy behaviour.
CLUSTER_SIZE=500
CLUSTER_NUM=10
CLUSTER_SAMPLE_NUM=10984
# ---------------------------
CG_DEPTH=2
CG_FANOUT=5
NUM_TRIALS=10
TASK="hidden_labels"

# Run CGT training on all trials
bash scripts/train/train_cgt_all_trials.sh "$DATASET" "$GPT_EPOCHS" "$CLUSTER_NUM" "$CLUSTER_SIZE" "$GPT_BATCH_SIZE" "$CG_DEPTH" "$CG_FANOUT" "$NUM_TRIALS" "$TASK" "$CLUSTER_SAMPLE_NUM"
