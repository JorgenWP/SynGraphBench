#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="BIGG_SYN_LP_BENCH"
#SBATCH --time=02:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80g
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=eirimhau@stud.ntnu.no
#SBATCH --mail-type=FAIL

# Benchmark the NON-anonymized synthetic BiGG graphs on link prediction. This
# produces the Delta_syn reference for the LP privacy table — the synthetic
# baseline that the anonymized hidden_links results are compared against.
#
# Prerequisite: scripts/train/train_bigg_synthetic_links.slurm must have run all
# 5 splits, producing the bundle at
# datasets/synthetic/bigg/<DATASET>/hidden_links/synthetic_baseline/
# (5 per-split variant dirs). No symlink step is needed — unlike the anonymized
# bundles (which live under .../anonymized/), this bundle is already in the
# canonical resolver location.
#
# Config matches the anonymized LP benchmark (neg_sampling=hard, decoder=mlp,
# 5 models incl. XGBoost, seeds_per_split=1) so Delta_syn is apples-to-apples.
# skip_original=true: the full real-graph baseline already exists at
# experiments/subsample_search/artifacts/baselines_link.csv under the same mlp/hard config.
# eval_real_subsampled=true: also emit real-subsampled-graph rows (train on the real
# pruned training_subsamples, test on the original held-out edges) — the Delta reference
# for the per-model synthetic LP table (LP analog of the AD tab:synthetic-auprc Delta).

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

# --- Config (edit before sbatch) ---
DATASET=                 # e.g. amazon  (one dataset per job)
MODELS=GCN,GIN,GraphSAGE,XGBGraph,XGBoost
SEEDS_PER_SPLIT=1        # matches the anonymized LP bundle
NEG_SAMPLING=hard
DECODER=mlp
BUNDLE=synthetic_baseline   # dir name written by train_bigg_synthetic_links.slurm

if [ -z "$DATASET" ]; then
    echo "ERROR: DATASET must be set" >&2
    exit 1
fi

module purge
module load Anaconda3/2024.02-1
conda activate bigg

BUNDLE_DIR="datasets/synthetic/bigg/${DATASET}/hidden_links/${BUNDLE}"
if [ ! -d "$BUNDLE_DIR" ]; then
    echo "ERROR: bundle missing: $BUNDLE_DIR — run train_bigg_synthetic_links.slurm first" >&2
    exit 1
fi

echo ""
echo "============================================================"
echo " ${DATASET} | non-anonymized synthetic | link prediction"
echo "============================================================"

# Args: datasets models trials generator synthetic_name neg_sampling decoder
#       task seeds_per_split dump_per_trial skip_original cdf_invert eval_real_subsampled
# trials is overridden to the bundle size automatically.
bash scripts/benchmark/run_link_benchmark.sh \
    "$DATASET" \
    "$MODELS" \
    5 \
    bigg \
    "$BUNDLE" \
    "$NEG_SAMPLING" \
    "$DECODER" \
    hidden_links \
    "$SEEDS_PER_SPLIT" \
    true \
    true \
    linear \
    true
