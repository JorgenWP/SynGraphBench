#!/bin/bash
# Train BiGG model on a dataset.
#
# Usage:
#   bash scripts/train/train_bigg.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim] [noise_std] [ss_max_prob] [ss_start_epoch] [bfs_preprocess] [normalize] [loss_weights] [hetero_feat] [mask_test_labels] [subsample] [subsample_size] [subsample_k] [num_subgraphs]
#
# normalize:        feature normalisation — one of "zscore", "minmax", "row", or "none" (default: none)
# loss_weights:     comma-separated cont,label weights relative to struct, applied after dynamic normalization (default: 1,1)
# hetero_feat:      "true" to enable heteroscedastic feature prediction (mean + variance), "false" for deterministic MSE (default: false)
# mask_test_labels: "true" to exclude test node labels (split 0) from label loss (default: false)
# subsample:        "true" to partition graph into BFS subgraphs for VRAM-limited training (default: false)
# subsample_size:   target number of nodes per subgraph (default: 2000)
# subsample_k:      max neighbors per BFS step — controls edge density (default: 10)
# num_subgraphs:    number of subgraphs to generate (default: auto = ceil(N / subsample_size))
#
# Examples:
#   bash scripts/train/train_bigg.sh tolokers 1024 1 50 0.001 256
#   bash scripts/train/train_bigg.sh reddit 512 2 100 0.0005 128 0.1 0.5 50 True zscore 1,1 true true
#   bash scripts/train/train_bigg.sh tolokers -1 1 50 0.001 256 0.3 0.0 0 False zscore 0.1,0.1 true false true 2000 10
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
SUBSAMPLE="${15:-false}"
SUBSAMPLE_SIZE="${16:-2000}"
SUBSAMPLE_K="${17:-10}"
NUM_SUBGRAPHS="${18:-}"

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
echo "Mask test labels: $MASK_TEST_LABELS"
echo "Subsample:       $SUBSAMPLE"
if [ "$SUBSAMPLE" = "true" ]; then
  echo "Subsample size:  $SUBSAMPLE_SIZE"
  echo "Subsample k:     $SUBSAMPLE_K"
  echo "Num subgraphs:   ${NUM_SUBGRAPHS:-auto}"
fi
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

SUBSAMPLE_FLAG=""
SUBSAMPLE_EXTRA=""
if [ "$SUBSAMPLE" = "true" ]; then
  SUBSAMPLE_FLAG="--subsample"
  SUBSAMPLE_EXTRA="-subsample_size $SUBSAMPLE_SIZE -subsample_k $SUBSAMPLE_K"
  if [ -n "$NUM_SUBGRAPHS" ]; then
    SUBSAMPLE_EXTRA="$SUBSAMPLE_EXTRA -num_subgraphs $NUM_SUBGRAPHS"
  fi
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
  $SUBSAMPLE_FLAG \
  $SUBSAMPLE_EXTRA \
  -save_dir "checkpoints/bigg/${DATASET}_blk${BLKSIZE}_b${BSIZE}_lr${LR}_e${EPOCHS}_noise${NOISE_STD}_ss${SS_MAX_PROB}_norm${NORMALIZE}_bfs${BFS_PREPROCESS}_lw${LOSS_WEIGHTS}_${HETERO_FEAT}"