#!/bin/bash
# Train BiGG on graph structure only (no node features or labels).
# Use this as a baseline to verify BiGG can learn the topology before
# adding feature/label generation.
#
# Usage:
#   bash scripts/train/train_bigg_structure.sh [dataset] [blksize] [epochs]
#
# Examples:
#   bash scripts/train/train_bigg_structure.sh tolokers 128 100
#

set -e

DATASET="${1:-tolokers}"
BLKSIZE="${2:-128}"
BSIZE="${3:-1}"
EPOCHS="${4:-100}"

cd "$(dirname "$0")/../../bigg"

echo "=== BiGG Structure-Only Baseline ==="
echo "Dataset:    $DATASET"
echo "Block size: $BLKSIZE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs:     $EPOCHS"
echo ""

python -m bigg.extension.pipeline_structure_only \
  -data_dir "$DATASET" \
  -gpu 0 \
  -embed_dim 256 \
  -bits_compress 0 \
  -learning_rate 0.001 \
  -num_epochs "$EPOCHS" \
  -batch_size "$BATCH_SIZE" \
  -blksize "$BLKSIZE" \
  -seed 34 \
  -save_dir "../checkpoints/bigg_structure_${DATASET}"
