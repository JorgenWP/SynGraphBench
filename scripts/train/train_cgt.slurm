#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi" 
#SBATCH --job-name="CGT_TRAIN"
#SBATCH --time=00:45:00             # Time limit (hh:mm:ss)

#SBATCH --partition=GPUQ            # Asking for GPU
#SBATCH --gres=gpu:2                # Number of GPUs 
#SBATCH --mem=10G                   # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=2                   # Number of compute nodes
#SBATCH --ntasks-per-node=2         # Number of tasks per node

#SBATCH --output=output/cgt/train/output.txt    # Specifying 'stdout'
#SBATCH --error=output/cgt/train/error.txt      # Specifying 'stderr'

#SBATCH --mail-user=<email>@stud.ntnu.no         # The email to send notifications to
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
CLUSTER_NUM=512
CLUSTER_SIZE=1
GPT_BATCH_SIZE=128
CG_DEPTH=2
CG_FANOUT=5

# Run the CGT training and generation script
bash scripts/train/train_cgt.sh "$DATASET" "$GPT_EPOCHS" "$CLUSTER_NUM" "$CLUSTER_SIZE" "$GPT_BATCH_SIZE" "$CG_DEPTH" "$CG_FANOUT"
