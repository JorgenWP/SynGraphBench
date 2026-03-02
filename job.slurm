#!/bin/sh
#SBATCH --account=ie-idi    # E.g. "ie-idi" if you belong to IDI. See https://www.hpc.ntnu.no/idun/documentation/accounting/ for more info.
#SBATCH --job-name="GADBench_CUDA"
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
conda create -n GADBench python=3.10
conda activate GADBench

conda install "numpy<2"
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c dglteam/label/cu117 dgl=1.1.2
conda install "mkl<2024.1" "intel-openmp<2024.1"

pip install xgboost pyod scikit-learn sympy pandas catboost bidict openpyxl


# Run the benchmark
python benchmark.py --trial 1 --datasets 0 --models GCN