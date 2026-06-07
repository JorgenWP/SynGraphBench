#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="BIGG_ANON_LP_BENCH"
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
DATASET=                 # e.g. amazon  (one dataset per job)
K_LEVELS=                # e.g. "20 50 100 500 1000"  (space-separated anonymity levels)
MODELS=GCN,GIN,GraphSAGE,XGBGraph
SEEDS_PER_SPLIT=3        # matches the AD anonymized bundle
NEG_SAMPLING=random
DECODER=dot

if [ -z "$DATASET" ] || [ -z "$K_LEVELS" ]; then
    echo "ERROR: DATASET and K_LEVELS must be set" >&2
    exit 1
fi

module purge
module load Anaconda3/2024.02-1
conda activate bigg

ANON_DIR="datasets/synthetic/bigg/anonymized/${DATASET}/hidden_links"
LINK_DIR="datasets/synthetic/bigg/${DATASET}/hidden_links"
mkdir -p "$LINK_DIR"

for K in $K_LEVELS; do
    echo ""
    echo "============================================================"
    echo " ${DATASET} | k=${K} anonymity | link prediction"
    echo "============================================================"
    if [ ! -d "${ANON_DIR}/${K}_anonymity" ]; then
        echo "WARN: ${ANON_DIR}/${K}_anonymity missing — skipping" >&2
        continue
    fi
    # Symlink the anonymized bundle into the canonical resolver location so
    # <synthetic_dir>/bigg/<dataset>/hidden_links/<name> finds it.
    ln -sfn "../../anonymized/${DATASET}/hidden_links/${K}_anonymity" \
            "${LINK_DIR}/${K}_anon_bundle"

    # Args: datasets models trials generator synthetic_name neg_sampling decoder
    #       task seeds_per_split dump_per_trial skip_original cdf_invert
    # trials is overridden to the bundle size automatically.
    bash scripts/benchmark/run_link_benchmark.sh \
        "$DATASET" \
        "$MODELS" \
        5 \
        bigg \
        "${K}_anon_bundle" \
        "$NEG_SAMPLING" \
        "$DECODER" \
        hidden_links \
        "$SEEDS_PER_SPLIT" \
        true \
        true \
        linear
done
