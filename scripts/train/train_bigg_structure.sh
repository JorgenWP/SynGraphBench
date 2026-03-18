#!/bin/bash
# Train BiGG on graph structure only (no node features or labels).
# Use this as a baseline to verify BiGG can learn the topology before
# adding feature/label generation.
#
# Usage:
#   bash scripts/train/train_bigg_structure.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim]
#
# Examples:
#   bash scripts/train/train_bigg_structure.sh tolokers 128 1 100 0.001 256
#   bash scripts/train/train_bigg_structure.sh reddit 256 1 150 0.0005 128
#

set -e

DATASET="${1:-tolokers}"
BLKSIZE="${2:-128}"
BSIZE="${3:-1}"
EPOCHS="${4:-100}"
LR="${5:-0.001}"
EMBED_DIM="${6:-256}"

cd "$(dirname "$0")/../../bigg"

echo "=== BiGG Structure-Only Baseline ==="
echo "Dataset:         $DATASET"
echo "Block size:      $BLKSIZE"
echo "Batch size:      $BSIZE"
echo "Epochs:          $EPOCHS"
echo "Learning rate:   $LR"
echo "Embed dim:       $EMBED_DIM"
echo ""

python -m bigg.extension.pipeline_structure_only \
  -data_dir "$DATASET" \
  -gpu 0 \
  -embed_dim "$EMBED_DIM" \
  -bits_compress 0 \
  -learning_rate "$LR" \
  -num_epochs "$EPOCHS" \
  -batch_size "$BSIZE" \
  -blksize "$BLKSIZE" \
  -seed 34 \
  -save_dir "checkpoints/bigg/structure_${DATASET}_blk${BLKSIZE}_b${BSIZE}_lr${LR}_e${EPOCHS}"
