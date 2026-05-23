#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi"
#SBATCH --job-name="CGT_LINK_BENCHMARK_<dataset>_k<cluster_size>_c<cluster_num>_<decoder>_<neg>_<description>"
#SBATCH --time=35:00:00             # Per-task time limit (hh:mm:ss)

#SBATCH --partition=GPUQ            # Asking for GPU
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --constraint=gpu16g
#SBATCH --mem=10G                   # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=1                   # Number of compute nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=8           # CPU cores for DataLoader workers

#SBATCH --array=1-2                 # 1 array task per CGT eval mode (match the CONFIGS length below)

# %A = parent job id, %a = array task id. Edit the path placeholders before submitting.
#SBATCH --output=output/cgt/benchmark/<dataset>/<task>/link_k<cluster_size>_c<cluster_num>_<decoder>_<neg>_<description>_t%a_output.txt
#SBATCH --error=output/cgt/benchmark/<dataset>/<task>/link_k<cluster_size>_c<cluster_num>_<decoder>_<neg>_<description>_t%a_error.err

#SBATCH --mail-user=<user>@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job:    $SLURM_JOB_NAME"
echo "ID of job:      $SLURM_JOB_ID"
echo "Array task ID:  $SLURM_ARRAY_TASK_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

module load Anaconda3/2024.02-1

# Activate benchmark conda environment
conda activate GADBench

# Shared config (applies to every array task)
MODELS="GCN,GIN,GraphSAGE,XGBoost,XGBGraph"
TRIALS="<num_trials>"        # e.g. 5
NEG_SAMPLING="random"        # random | hard
DECODER="dot"                # dot | mlp
BATCH_SIZE="4096"            # CGT comp-graph batch size (default 256)
SKIP_PHASE1="1"              # 1 to skip full-graph baseline, 0 to include it

# Per-task config: "<dataset> <synthetic_name> <eval_mode>"
# Add one row per array task; bump --array=1-N to match the row count.
CONFIGS=(
    "<dataset> <synthetic_stem> original_cg"
    "<dataset> <synthetic_stem> synthetic_cgt"
)
IFS=' ' read -r DATASETS SYNTHETIC_NAME EVAL_MODE <<< "${CONFIGS[$((SLURM_ARRAY_TASK_ID - 1))]}"

if [ -z "$EVAL_MODE" ]; then
  echo "ERROR: no config for array task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

echo "Array task $SLURM_ARRAY_TASK_ID: dataset=$DATASETS, synthetic_name=$SYNTHETIC_NAME, eval_mode=$EVAL_MODE"
echo ""

# Run the link prediction benchmark with CGT synthetic data
bash scripts/benchmark/run_link_benchmark.sh \
    "$DATASETS" "$MODELS" "$TRIALS" "cgt" "$SYNTHETIC_NAME" \
    "$NEG_SAMPLING" "$DECODER" "hidden_links" \
    "$EVAL_MODE" "$SKIP_PHASE1" "$BATCH_SIZE"
