#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi"
#SBATCH --job-name="CLUSTER_PRECOMPUTE"
#SBATCH --time=12:00:00             # Per-array-task time limit (hh:mm:ss)

#SBATCH --partition=CPUQ            # CPU-only — clustering does not use GPU
#SBATCH --mem=64G                   # Bumped for larger datasets (elliptic ~204K nodes)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # k_means_constrained MCF + sklearn benefit from threads

#SBATCH --array=1-3                 # Array length must equal the DATASETS array below.         

#SBATCH --output=output/cluster/_slurm/precompute_%A_%a.out
#SBATCH --error=output/cluster/_slurm/precompute_%A_%a.err

#SBATCH --mail-user=jorgenwp@stud.ntnu.no
#SBATCH --mail-type=ALL

# Datasets — one array task per entry. Edit and resize #SBATCH --array accordingly.
DATASETS=(
    "tolokers"
    "questions"
    "elliptic"
)
DATASET="${DATASETS[$((SLURM_ARRAY_TASK_ID - 1))]}"
if [ -z "$DATASET" ]; then
    echo "ERROR: no dataset for array task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"

# Redirect the rest of stdout/stderr into a per-dataset log file
mkdir -p "output/cluster/${DATASET}"
mkdir -p "output/cluster/_slurm"
exec > "output/cluster/${DATASET}/precompute.out" 2> "output/cluster/${DATASET}/precompute.err"

echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID -> dataset=$DATASET"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1
conda activate CGT

# (cluster_size, cluster_num) pairs — zipped index-wise. Same length required.
SIZES=(1 50)
NUMS=(512 597)

# Tasks to cover. hidden_links clustering is trial-agnostic (handled inside the
# python script), so the trial list is only consulted for hidden_labels.
TASKS=(hidden_labels hidden_links)
TRIALS=(0 1 2 3 4 5 6 7 8 9)

# Resume-safe: any (dataset, task, trial, k, c) entry whose cache dir already
# contains DONE is skipped. To force re-fitting, add --force to the invocation.
python scripts/cluster/precompute_clusters.py \
    --datasets "$DATASET" \
    --cluster-sizes "${SIZES[@]}" \
    --cluster-nums "${NUMS[@]}" \
    --tasks "${TASKS[@]}" \
    --trials "${TRIALS[@]}"
