#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi"
#SBATCH --job-name="BiGG_SUBSAMPLE_tolokers_grid"
#SBATCH --time=01:30:00             # Per-task time limit (hh:mm:ss)

#SBATCH --partition=GPUQ            # Asking for GPU
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --constraint=gpu80g
#SBATCH --mem=10G                   # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=2                   # Number of compute nodes
#SBATCH --ntasks-per-node=2         # Number of tasks per node

#SBATCH --array=1-12%4              # 12 combos, max 4 running concurrently

#SBATCH --output=output/bigg/train/grid_%A_%a.out   # %A=parent job id, %a=array task id
#SBATCH --error=output/bigg/train/grid_%A_%a.err

#SBATCH --mail-user=eirimhau@stud.ntnu.no
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

# Activate bigg conda environment
conda activate bigg

# Pick the parameter line for this array task (skipping comment / blank lines)
PARAMS_FILE="scripts/train/bigg_tolokers_grid.txt"
LINE=$(grep -v '^[[:space:]]*\(#\|$\)' "$PARAMS_FILE" | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$LINE" ]; then
  echo "ERROR: no params for array task $SLURM_ARRAY_TASK_ID in $PARAMS_FILE" >&2
  exit 1
fi

echo "Combo: $LINE"
echo ""

# Word-split intentional: the line contains positional args for train_bigg_subsample.sh
bash scripts/train/train_bigg_subsample.sh $LINE
