#!/bin/bash
# Build a k-anonymized DGL graph artifact via constrained k-means + centroid replacement.
#
# Usage:
#   bash scripts/kanon.sh [dataset] [k] [cluster_norm] [seed] [trial_id] [max_iter]
#
# Examples:
#   bash scripts/kanon.sh tolokers 10 zscore 0 0
#   bash scripts/kanon.sh questions 20 l2 0 3 100
#
# Output:
#   datasets/kanon/<dataset>/<dataset>_k{k}_{norm}_s{seed}_t{trial_id}.dgl
#   datasets/kanon/<dataset>/<dataset>_k{k}_{norm}_s{seed}_t{trial_id}.meta.pt

set -e

DATASET="${1:-tolokers}"
K="${2:-10}"
CLUSTER_NORM="${3:-zscore}"
SEED="${4:-0}"
TRIAL_ID="${5:-0}"
MAX_ITER="${6:-100}"

source "$(conda info --base)/etc/profile.d/conda.sh"
# k-means-constrained is already installed in the cgt env; reusing it
# avoids an ABI-mismatched install in the bigg env (where it conflicts
# with the numpy 2.x ABI used by the bigg-env wheels).
conda activate cgt

cd "$(dirname "$0")/../bigg"

echo "=== k-anonymity baseline (constrained k-means) ==="
echo "Dataset:       $DATASET"
echo "k (size_min):  $K"
echo "cluster_norm:  $CLUSTER_NORM"
echo "seed:          $SEED"
echo "trial_id:      $TRIAL_ID"
echo "max_iter:      $MAX_ITER"
echo ""

python -m bigg.data_process.kanonymize \
    -dataset "$DATASET" \
    -k "$K" \
    -cluster_norm "$CLUSTER_NORM" \
    -seed "$SEED" \
    -trial_id "$TRIAL_ID" \
    -max_iter "$MAX_ITER"
