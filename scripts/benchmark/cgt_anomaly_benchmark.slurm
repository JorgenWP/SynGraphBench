#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi"
#SBATCH --job-name="CGT_ANOMALY_BENCHMARK_<dataset>_k<cluster_size>_c<cluster_num>_<description>"
#SBATCH --time=08:00:00             # Per-task time limit (hh:mm:ss)

#SBATCH --partition=GPUQ            # Asking for GPU
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --mem=20G                   # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=1                   # Number of compute nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node

#SBATCH --array=1-2                 # Match the CONFIGS length below (1 shared original_cg task + N phase2_variant tasks)

# %A = parent job id, %a = array task id. Edit the path placeholders before submitting.
#SBATCH --output=output/cgt/benchmark/<dataset>/hidden_labels/anomaly_k<cluster_size>_c<cluster_num>_<description>_t%a_output.txt
#SBATCH --error=output/cgt/benchmark/<dataset>/hidden_labels/anomaly_k<cluster_size>_c<cluster_num>_<description>_t%a_error.txt

#SBATCH --mail-user=<user>@stud.ntnu.no         # The email to send notifications to
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

module load Anaconda3/2024.02-1

# Activate benchmark conda environment
conda activate GADBench

# Shared config (applies to every array task)
DATASET="<dataset>"
MODELS="GCN,GIN,GraphSAGE,XGBoost,XGBGraph"
TRIALS="<num_trials>"        # e.g. 5

# Per-task config: "<synthetic_name> <eval_mode> <output_dir|-> <skip_phase1>"
# Use "-" in the output_dir slot to fall back to auto-derive
# (results/evaluate/cgt/$DATASET/hidden_labels/<synthetic_name>/).
# skip_phase1: 1 to skip whole-graph baseline, 0 to include it.
#
# original_cg + Phase 1 depend only on the dataset (and CG params shared across
# all k/cluster_num variants) — run them ONCE in a shared folder. phase2_variant
# bundles the two per-variant baselines (original_cg_quantized + synthetic_cgt)
# so a single python process per variant covers both — no redundant .pt load.
CONFIGS=(
    "<synthetic_stem> original_cg    results/evaluate/cgt/${DATASET}/hidden_labels/_original_cg_shared 0"
    "<synthetic_stem> phase2_variant -                                                                 1"
)

IFS=' ' read -r SYNTHETIC_NAME EVAL_MODE OUTPUT_DIR SKIP_PHASE1 <<< "${CONFIGS[$((SLURM_ARRAY_TASK_ID - 1))]}"

if [ -z "$EVAL_MODE" ]; then
  echo "ERROR: no config for array task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

# Sentinel "-" means use the wrapper's default (auto-derive in anomaly_benchmark.py).
if [ "$OUTPUT_DIR" = "-" ]; then
    OUTPUT_DIR=""
fi

echo "Array task $SLURM_ARRAY_TASK_ID: dataset=$DATASET, synthetic_name=$SYNTHETIC_NAME, eval_mode=$EVAL_MODE, output_dir=${OUTPUT_DIR:-'(auto)'}, skip_phase1=$SKIP_PHASE1"
echo ""

# Run the anomaly benchmark with CGT synthetic data.
# Positional arg 7 is cdf_invert — only used by the BiGG full-graph path,
# unused by CGT; pass the wrapper default ("linear") as a placeholder so
# eval_mode/skip_phase1/output_dir line up at positions 8/9/10.
bash scripts/benchmark/run_anomaly_benchmark.sh \
    "$DATASET" "$MODELS" "$TRIALS" "cgt" "$SYNTHETIC_NAME" \
    "hidden_labels" "linear" \
    "$EVAL_MODE" "$SKIP_PHASE1" "$OUTPUT_DIR"
