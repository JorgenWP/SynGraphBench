#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi"
#SBATCH --job-name="BiGG_TRAIN_STRUCTURE_128_1"
#SBATCH --time=04:00:00             # Time limit (hh:mm:ss)

#SBATCH --partition=GPUQ            # Asking for GPU
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --constraint=gpu80g
#SBATCH --mem=10G                   # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=2                   # Number of compute nodes
#SBATCH --ntasks-per-node=2         # Number of tasks per node

#SBATCH --output=output/bigg/train/output_structure_blksize_128.txt    # Specifying 'stdout'
#SBATCH --error=output/bigg/train/error_structure_blksize_128.txt      # Specifying 'stderr'

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

# Activate bigg conda environment
conda activate bigg

# Arguments
DATASET="tolokers"
BLKSIZE=128
BSIZE=1
EPOCHS=100

# Run the BiGG structure-only baseline training and generation script
bash scripts/train/train_bigg_structure.sh "$DATASET" "$BLKSIZE" "$BSIZE" "$EPOCHS"
