#!/bin/sh
#SBATCH --account=<account>    # E.g. "ie-idi" if you belong to IDI. See https://www.hpc.ntnu.no/idun/documentation/accounting/ for more info.
#SBATCH --job-name="CGT_CUDA"
#SBATCH --time=01:00:00        # Time limit (hh:mm:ss)

#SBATCH --partition=GPUQ       # Asking for GPU
#SBATCH --gres=gpu:2           # Number of GPUs 
#SBATCH --mem=10G              # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=2              # Number of compute nodes
#SBATCH --ntasks-per-node=2    # Number of tasks per node

#SBATCH --output=output.txt    # Specifying 'stdout'
#SBATCH --error=output.err     # Specifying 'stderr'

#SBATCH --mail-user=<email>    # The email to send notifications to
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

module load Anaconda3/2024.02-1

# Set up the conda environment
# conda env create -f cgt_modern_env.yml
conda activate cgt_modern
# bash install_gpu.sh


# Run the project script
bash ../scripts/train/train_cgt.sh