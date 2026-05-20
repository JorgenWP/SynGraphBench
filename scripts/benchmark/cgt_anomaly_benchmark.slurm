#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi" 
#SBATCH --job-name="CGT_ANOMALY_BENCHMARK_<dataset>_k<k-anonymity>_c<cluster-num>"
#SBATCH --time=04:30:00             # Time limit (hh:mm:ss)

#SBATCH --partition=GPUQ            # Asking for GPU
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --constraint=gpu16g
#SBATCH --mem=10G                   # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=1                   # Number of compute nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node

#SBATCH --output=output/cgt/benchmark/<dataset>/hidden_labels/anomaly_k<k-anonymity>_c<cluster-num>_output.txt     # Specifying 'stdout'
#SBATCH --error=output/cgt/benchmark/<dataset>/hidden_labels/anomaly_k<k-anonymity>_c<cluster-num>_error.txt       # Specifying 'stderr'

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

# Activate benchmark conda environment
conda activate GADBench

# Config
DATASETS="tolokers"
MODELS="GCN,GIN,GraphSAGE,XGBoost,XGBGraph"
TRIALS="10"
SYNTHETIC_NAME=""   # Optional: set to a specific variant, e.g. "reddit_eps0.1"

# Run the benchmark script with CGT synthetic data
bash scripts/benchmark/run_anomaly_benchmark.sh "$DATASETS" "$MODELS" "$TRIALS" "cgt" "$SYNTHETIC_NAME"
