#!/bin/bash
# Train BiGG model on a dataset.
#
# Usage:
#   bash scripts/train/train_bigg.sh [dataset] [blksize] [batch_size] [epochs]
#
# Examples:
#   bash scripts/train/train_bigg.sh tolokers 1024 1 50
#   bash scripts/train/train_bigg.sh reddit 512 2 100
#

set -e

# Configuration with defaults
DATASET="${1:-tolokers}"
BLKSIZE="${2:-1024}"
BSIZE="${3:-1}"
EPOCHS="${4:-50}"

cd "$(dirname "$0")/../../bigg"

echo "=== BiGG Training ==="
echo "Dataset:    $DATASET"
echo "Block size: $BLKSIZE"
echo "Batch size: $BSIZE"
echo "Epochs:     $EPOCHS"
echo ""

python -m bigg.extension.pipeline \
  -data_dir "$DATASET" \
  -model_type conditional \
  -gpu 0 \
  -embed_dim 256 \
  -bits_compress 0 \
  -learning_rate 0.001 \
  -num_epochs "$EPOCHS" \
  -batch_size "$BSIZE" \
  -blksize "$BLKSIZE" \
  -seed 34