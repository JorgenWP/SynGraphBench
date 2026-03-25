#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi" 
#SBATCH --job-name="BIGG_BENCHMARK"
#SBATCH --time=00:45:00             # Time limit (hh:mm:ss)

#SBATCH --partition=GPUQ            # Asking for GPU
#SBATCH --gres=gpu:2                # Number of GPUs 
#SBATCH --mem=10G                   # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=2                   # Number of compute nodes
#SBATCH --ntasks-per-node=2         # Number of tasks per node

#SBATCH --output=output/bigg/benchmark/output.txt    # Specifying 'stdout'
#SBATCH --error=output/bigg/benchmark/error.txt      # Specifying 'stderr'

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
GENERATOR="bigg"
DATASETS="tolokers"
MODELS="GCN,GIN,GraphSAGE,XGBGraph"
TRIALS="1"
SYNTHETIC_NAME=""   # Optional: set to a specific variant, e.g. "tolokers_blksize_1024_b_1"

# Run the benchmark script with BiGG synthetic data
bash scripts/benchmark/run_anomaly_benchmark.sh "$DATASETS" "$MODELS" "$TRIALS" "$GENERATOR" "$SYNTHETIC_NAME"
