#!/bin/bash
# Train BiGG model on a dataset.
#
# Usage:
#   bash scripts/train/train_bigg.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim] [noise_std] [ss_max_prob] [ss_start_epoch] [bfs_preprocess] [normalize] [loss_weights] [hetero_feat]
#
# normalize:    feature normalisation — one of "zscore", "minmax", "row", or "none" (default: none)
# loss_weights: comma-separated cont,label weights relative to struct, applied after dynamic normalization (default: 1,1)
# hetero_feat:  "true" to enable heteroscedastic feature prediction (mean + variance), "false" for deterministic MSE (default: false)
#
# Examples:
#   bash scripts/train/train_bigg.sh tolokers 1024 1 50 0.001 256
#   bash scripts/train/train_bigg.sh reddit 512 2 100 0.0005 128 0.1 0.5 50 True zscore 1,1 true
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

cd "$(dirname "$0")/../../bigg"

echo "=== BiGG Training ==="
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
echo ""

NORM_FLAG=""
if [ "$NORMALIZE" != "none" ]; then
  NORM_FLAG="-normalize $NORMALIZE"
fi

HETERO_FLAG=""
if [ "$HETERO_FEAT" = "true" ]; then
  HETERO_FLAG="-hetero_feat"
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
  -save_dir "checkpoints/bigg/${DATASET}_blk${BLKSIZE}_b${BSIZE}_lr${LR}_e${EPOCHS}_noise${NOISE_STD}_ss${SS_MAX_PROB}_norm${NORMALIZE}_bfs${BFS_PREPROCESS}_lw${LOSS_WEIGHTS}_${HETERO_FEAT}"