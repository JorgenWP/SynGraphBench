#!/bin/bash
# Train BiGG model on a dataset using forest fire subsampling.
#
# Use this script when the full graph does not fit in VRAM. Subgraphs are
# sampled independently via forest fire (with replacement), each trained
# independently. At generation time, one synthetic subgraph is produced per
# training subgraph.
#
# Usage:
#   bash scripts/train/train_bigg_subsample.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim] [noise_std] [ss_max_prob] [ss_start_epoch] [bfs_preprocess] [normalize] [loss_weights] [hetero_feat] [mask_test_labels] [logvar_floor] [subsample_size] [burn_prob] [num_subgraphs]
#
# normalize:        feature normalisation — one of "zscore", "minmax", "row", or "none" (default: none)
# loss_weights:     comma-separated cont,label weights relative to struct, applied after dynamic normalization (default: 1,1)
# hetero_feat:      "true" to enable heteroscedastic feature prediction (mean + variance), "false" for deterministic MSE (default: false)
# mask_test_labels: "true" to exclude test node labels (split 0) from label loss (default: false)
# logvar_floor:     lower clamp for log-variance in hetero_feat mode (default: -4.0)
# subsample_size:   target number of nodes per subgraph (default: 2000)
# burn_prob:        forest fire burn probability — controls subgraph density (default: 0.3)
# num_subgraphs:    number of subgraphs to use (default: auto = ceil(N / subsample_size))
#
# Examples:
#   bash scripts/train/train_bigg_subsample.sh reddit -1 1 300 0.001 256 0.3 0.0 0 True zscore 0.1,0.1 true true -4.0 2000 0.3
#   bash scripts/train/train_bigg_subsample.sh reddit -1 1 300 0.001 256 0.3 0.0 0 True zscore 0.1,0.1 true true -4.0 2000 0.3 3
#

set -e

# Configuration with defaults
DATASET="${1:-tolokers}"
BLKSIZE="${2:-1024}"
BSIZE="${3:-1}"
EPOCHS="${4:-50}"
LR="${5:-0.001}"
EMBED_DIM="${6:-256}"
NOISE_STD="${7:-0.0}"
SS_MAX_PROB="${8:-0.0}"
SS_START_EPOCH="${9:-0}"
BFS_PREPROCESS="${10:-False}"
NORMALIZE="${11:-none}"
LOSS_WEIGHTS="${12:-1,1}"
HETERO_FEAT="${13:-false}"
MASK_TEST_LABELS="${14:-false}"
LOGVAR_FLOOR="${15:--4.0}"
SUBSAMPLE_SIZE="${16:-2000}"
BURN_PROB="${17:-0.3}"
NUM_SUBGRAPHS="${18:-}"

cd "$(dirname "$0")/../../bigg"

echo "=== BiGG Subsampled Training ==="
echo "Dataset:         $DATASET"
echo "Block size:      $BLKSIZE"
echo "Batch size:      $BSIZE"
echo "Epochs:          $EPOCHS"
echo "Learning rate:   $LR"
echo "Embed dim:       $EMBED_DIM"
echo "Noise std:       $NOISE_STD"
echo "SS max prob:     $SS_MAX_PROB"
echo "SS start epoch:  $SS_START_EPOCH"
echo "BFS preprocess:  $BFS_PREPROCESS"
echo "Normalize:       $NORMALIZE"
echo "Loss weights:    $LOSS_WEIGHTS"
echo "Hetero feat:     $HETERO_FEAT"
echo "Mask test labels: $MASK_TEST_LABELS"
echo "Logvar floor:    $LOGVAR_FLOOR"
echo "Subsample size:  $SUBSAMPLE_SIZE"
echo "Burn prob:       $BURN_PROB"
echo "Num subgraphs:   ${NUM_SUBGRAPHS:-auto}"
echo ""

NORM_FLAG=""
if [ "$NORMALIZE" != "none" ]; then
  NORM_FLAG="-normalize $NORMALIZE"
fi

HETERO_FLAG=""
if [ "$HETERO_FEAT" = "true" ]; then
  HETERO_FLAG="--hetero_feat"
fi

MASK_FLAG=""
if [ "$MASK_TEST_LABELS" = "true" ]; then
  MASK_FLAG="--mask_test_labels"
fi

NUM_SUBGRAPHS_FLAG=""
if [ -n "$NUM_SUBGRAPHS" ]; then
  NUM_SUBGRAPHS_FLAG="-num_subgraphs $NUM_SUBGRAPHS"
fi

python -m bigg.extension.pipeline \
  -data_dir "$DATASET" \
  -model_type conditional \
  -gpu 0 \
  -embed_dim "$EMBED_DIM" \
  -bits_compress 0 \
  -bfs_preprocess "$BFS_PREPROCESS" \
  -learning_rate "$LR" \
  -num_epochs "$EPOCHS" \
  -batch_size "$BSIZE" \
  -blksize "$BLKSIZE" \
  -noise_std "$NOISE_STD" \
  -ss_max_prob "$SS_MAX_PROB" \
  -ss_start_epoch "$SS_START_EPOCH" \
  -seed 34 \
  $NORM_FLAG \
  -loss_weights "$LOSS_WEIGHTS" \
  $HETERO_FLAG \
  $MASK_FLAG \
  -logvar_floor "$LOGVAR_FLOOR" \
  --subsample \
  -subsample_size "$SUBSAMPLE_SIZE" \
  -burn_prob "$BURN_PROB" \
  $NUM_SUBGRAPHS_FLAG \
  -save_dir "checkpoints/bigg/${DATASET}_blk${BLKSIZE}_b${BSIZE}_lr${LR}_e${EPOCHS}_noise${NOISE_STD}_ss${SS_MAX_PROB}_norm${NORMALIZE}_bfs${BFS_PREPROCESS}_lw${LOSS_WEIGHTS}_${HETERO_FEAT}_lvf${LOGVAR_FLOOR}_sub${SUBSAMPLE_SIZE}_p${BURN_PROB}"
