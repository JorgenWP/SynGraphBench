#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi"
#SBATCH --job-name="CGT_LINK_BENCHMARK_<dataset>_k<k-anonymity>_c<cluster-num>_<decoder>_<neg-sampling>"
#SBATCH --time=04:30:00             # Time limit (hh:mm:ss)

#SBATCH --partition=GPUQ            # Asking for GPU
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --constraint=gpu16g
#SBATCH --mem=10G                   # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=1                   # Number of compute nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=8           # CPU cores for DataLoader workers

#SBATCH --output=output/cgt/benchmark/<dataset>/hidden_links/link_k<k-anonymity>_c<cluster-num>_<decoder>_<neg-sampling>_output.txt    # Specifying 'stdout'
#SBATCH --error=output/cgt/benchmark/<dataset>/hidden_links/link_k<k-anonymity>_c<cluster-num>_<decoder>_<neg-sampling>_error.txt      # Specifying 'stderr'

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
DATASETS="reddit"
MODELS="GCN,GIN,GraphSAGE,XGBoost,XGBGraph"
TRIALS="10"
SYNTHETIC_NAME=""       # Optional: variant stem, e.g. "reddit_e50_k512_c1_d2_f5_s5000"

NEG_SAMPLING="random"   # random | hard
DECODER="dot"           # dot | mlp

# Run the link prediction benchmark with CGT synthetic data
bash scripts/benchmark/run_link_benchmark.sh "$DATASETS" "$MODELS" "$TRIALS" "cgt" "$SYNTHETIC_NAME" "$NEG_SAMPLING" "$DECODER"
