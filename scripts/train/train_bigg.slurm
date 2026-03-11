#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi"
#SBATCH --job-name="BiGG_TRAIN"
#SBATCH --time=03:00:00             # Time limit (hh:mm:ss)

#SBATCH --partition=GPUQ            # Asking for GPU
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --constraint=gpu40g
#SBATCH --mem=10G                   # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=2                   # Number of compute nodes
#SBATCH --ntasks-per-node=2         # Number of tasks per node

#SBATCH --output=output/bigg/train/output.txt    # Specifying 'stdout'
#SBATCH --error=output/bigg/train/error.txt      # Specifying 'stderr'

#SBATCH --mail-user=jorgenwp@stud.ntnu.no         # The email to send notifications to
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


# Run the BiGG training and generation script
bash scripts/train/train_bigg.sh