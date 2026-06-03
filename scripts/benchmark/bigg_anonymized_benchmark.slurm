#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="BIGG_ANON_AD_BENCH"
#SBATCH --time=04:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40g
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=eirimhau@stud.ntnu.no
#SBATCH --mail-type=FAIL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

# --- Config (edit before sbatch) ---
DATASET=                 # e.g. questions  (one dataset per job)
K_LEVELS=                # e.g. "20 50 100 500 1000"  (space-separated anonymity levels)
MODELS=GCN,GIN,GraphSAGE,XGBGraph,XGBoost
SEEDS_PER_SPLIT=3        # matches the BO held-out final report
TUNE_TEST_RATIO=0.5      # held-out half of the BO test-node split
TUNE_TEST_SEED=20260523  # same seed as the BO configs
TUNE_PORTION=heldout

if [ -z "$DATASET" ] || [ -z "$K_LEVELS" ]; then
    echo "ERROR: DATASET and K_LEVELS must be set" >&2
    exit 1
fi

module purge
module load Anaconda3/2024.02-1
conda activate bigg

ANON_DIR="datasets/synthetic/bigg/anonymized/${DATASET}/hidden_labels"
LINK_DIR="datasets/synthetic/bigg/${DATASET}/hidden_labels"
mkdir -p "$LINK_DIR"

for K in $K_LEVELS; do
    echo ""
    echo "============================================================"
    echo " ${DATASET} | k=${K} anonymity"
    echo "============================================================"
    if [ ! -d "${ANON_DIR}/${K}_anonymity" ]; then
        echo "WARN: ${ANON_DIR}/${K}_anonymity missing — skipping" >&2
        continue
    fi
    # Symlink the anonymized bundle into the canonical resolver location so
    # <synthetic_dir>/bigg/<dataset>/hidden_labels/<name> finds it.
    ln -sfn "../../anonymized/${DATASET}/hidden_labels/${K}_anonymity" \
            "${LINK_DIR}/${K}_anon_bundle"

    bash scripts/benchmark/run_anomaly_benchmark.sh \
        "$DATASET" \
        "$MODELS" \
        5 \
        bigg \
        "${K}_anon_bundle" \
        hidden_labels \
        linear \
        "$SEEDS_PER_SPLIT" \
        true \
        "$TUNE_TEST_RATIO" \
        "$TUNE_TEST_SEED" \
        "$TUNE_PORTION"
done
