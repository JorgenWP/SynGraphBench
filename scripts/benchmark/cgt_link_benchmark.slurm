#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi"
#SBATCH --job-name="CGT_LINK_BENCHMARK_<dataset>_k<cluster_size>_c<cluster_num>_<decoder>_<neg>_<description>"
#SBATCH --time=24:00:00             # Per-task time limit (hh:mm:ss)

#SBATCH --partition=GPUQ            # Asking for GPU
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --constraint=gpu16g
#SBATCH --mem=10G                   # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=1                   # Number of compute nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=8           # CPU cores for DataLoader workers

#SBATCH --array=1-2                 # Match the CONFIGS length below (1 shared original_cg task + N synthetic_cgt tasks)

# %A = parent job id, %a = array task id. Edit the path placeholders before submitting.
#SBATCH --output=output/cgt/benchmark/<dataset>/hidden_links/link_k<cluster_size>_c<cluster_num>_<decoder>_<neg>_<description>_t%a_output.txt
#SBATCH --error=output/cgt/benchmark/<dataset>/hidden_links/link_k<cluster_size>_c<cluster_num>_<decoder>_<neg>_<description>_t%a_error.err

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

# Per-task config: "<dataset> <synthetic_name> <eval_mode> <output_dir|-> <skip_phase1>"
# Use "-" in the output_dir slot to fall back to auto-derive
# (results/evaluate/cgt/<dataset>/hidden_links/<synthetic_name>/).
# skip_phase1: 1 to skip full-graph baseline, 0 to include it.
#
# original_cg is identical across variants with the same cg_depth/cg_fanout —
# run it ONCE under a shared folder and reuse for all variants. Attach Phase 1
# (full-graph GNN baseline) to that same task when it hasn't been run yet.
CONFIGS=(
    "<dataset> <synthetic_stem> original_cg   results/evaluate/cgt/<dataset>/hidden_links/_original_cg_shared 0"
    "<dataset> <synthetic_stem> synthetic_cgt -                                                               1"
)
IFS=' ' read -r DATASETS SYNTHETIC_NAME EVAL_MODE OUTPUT_DIR SKIP_PHASE1 <<< "${CONFIGS[$((SLURM_ARRAY_TASK_ID - 1))]}"

if [ -z "$EVAL_MODE" ]; then
  echo "ERROR: no config for array task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

# Sentinel "-" means use the wrapper's default (auto-derive in link_benchmark.py).
if [ "$OUTPUT_DIR" = "-" ]; then
    OUTPUT_DIR=""
fi

echo "Array task $SLURM_ARRAY_TASK_ID: dataset=$DATASETS, synthetic_name=$SYNTHETIC_NAME, eval_mode=$EVAL_MODE, output_dir=${OUTPUT_DIR:-'(auto)'}, skip_phase1=$SKIP_PHASE1"
echo ""

# Run the link prediction benchmark with CGT synthetic data
bash scripts/benchmark/run_link_benchmark.sh \
    "$DATASETS" "$MODELS" "$TRIALS" "cgt" "$SYNTHETIC_NAME" \
    "$NEG_SAMPLING" "$DECODER" "hidden_links" \
    "$EVAL_MODE" "$SKIP_PHASE1" "$BATCH_SIZE" "$OUTPUT_DIR"
