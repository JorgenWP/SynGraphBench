#!/bin/sh
#SBATCH --account=share-ie-idi      # "share-ie-idi" "ie-idi"
#SBATCH --job-name="BiGG_SUBSAMPLE_<something>"
#SBATCH --time=04:00:00             # Time limit (hh:mm:ss)

#SBATCH --partition=GPUQ            # Asking for GPU
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --constraint=gpu80g
#SBATCH --mem=10G                   # Gigabytes of system memory (not GPU memory)
#SBATCH --nodes=2                   # Number of compute nodes
#SBATCH --ntasks-per-node=2         # Number of tasks per node

#SBATCH --output=output/bigg/train/output_<something>.txt    # Specifying 'stdout'
#SBATCH --error=output/bigg/train/error_<something>.txt      # Specifying 'stderr'

#SBATCH --mail-user=<username>@stud.ntnu.no         # The email to send notifications to
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

module load Anaconda3/2024.02-1

# Activate bigg conda environment
conda activate bigg

# Arguments
DATASET="reddit"
BLKSIZE=-1
BSIZE=1
EPOCHS=300
LR=0.001
EMBED_DIM=256
NOISE_STD=0.3
SS_MAX_PROB=0.0
SS_START_EPOCH=0
BFS_PREPROCESS=True
NORMALIZE=zscore        # "zscore", "minmax", "row", or "none"
LOSS_WEIGHTS="0.1,0.1"  # cont,label weights relative to struct (applied after dynamic normalization)
HETERO_FEAT=true        # "true" for heteroscedastic feature prediction (mean + variance), "false" for deterministic MSE
MASK_TEST_LABELS=true   # "true" to exclude test node labels (split 0) from label loss
SUBSAMPLE_SIZE=2000     # target number of nodes per BFS subgraph
BURN_PROB=0.3           # forest fire burn probability (controls subgraph density)
NUM_SUBGRAPHS=""        # number of subgraphs to use (leave empty for auto = ceil(N / subsample_size))

# Run the BiGG subsampled training and generation script
bash scripts/train/train_bigg_subsample.sh "$DATASET" "$BLKSIZE" "$BSIZE" "$EPOCHS" "$LR" "$EMBED_DIM" "$NOISE_STD" "$SS_MAX_PROB" "$SS_START_EPOCH" "$BFS_PREPROCESS" "$NORMALIZE" "$LOSS_WEIGHTS" "$HETERO_FEAT" "$MASK_TEST_LABELS" "$SUBSAMPLE_SIZE" "$BURN_PROB" "$NUM_SUBGRAPHS"
