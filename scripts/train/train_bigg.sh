#!/bin/bash
# Train BiGG model on a dataset.
#
# Usage:
#   bash scripts/train/train_bigg.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim]
#
# Examples:
#   bash scripts/train/train_bigg.sh tolokers 1024 1 50 0.001 256
#   bash scripts/train/train_bigg.sh reddit 512 2 100 0.0005 128
#

set -e

# Configuration with defaults
DATASET="${1:-tolokers}"
BLKSIZE="${2:-1024}"
BSIZE="${3:-1}"
EPOCHS="${4:-50}"
LR="${5:-0.001}"
EMBED_DIM="${6:-256}"

cd "$(dirname "$0")/../../bigg"

echo "=== BiGG Training ==="
echo "Dataset:         $DATASET"
echo "Block size:      $BLKSIZE"
echo "Batch size:      $BSIZE"
echo "Epochs:          $EPOCHS"
echo "Learning rate:   $LR"
echo "Embed dim:       $EMBED_DIM"
echo ""

python -m bigg.extension.pipeline \
  -data_dir "$DATASET" \
  -model_type conditional \
  -gpu 0 \
  -embed_dim "$EMBED_DIM" \
  -bits_compress 0 \
  -learning_rate "$LR" \
  -num_epochs "$EPOCHS" \
  -batch_size "$BSIZE" \
  -blksize "$BLKSIZE" \
  -seed 34 \
  -save_dir "checkpoints/bigg/${DATASET}_blk${BLKSIZE}_b${BSIZE}_lr${LR}_e${EPOCHS}"