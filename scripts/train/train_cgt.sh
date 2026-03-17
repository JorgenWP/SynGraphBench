#!/bin/bash
# Train CGT model on a dataset.
#
# Usage:
#   bash scripts/train/train_cgt.sh [dataset] [gpt_epochs] [cluster_num] [cluster_size] [gpt_batch_size] [cg_depth] [cg_fanout]
#
# Examples:
#   bash scripts/train/train_cgt.sh reddit 50 512 1 128 2 5
#   bash scripts/train/train_cgt.sh tolokers 100 256 1 64 3 10
#

set -e

# Configuration with defaults
DATASET="${1:-reddit}"
GPT_EPOCHS="${2:-50}"
CLUSTER_NUM="${3:-512}"
CLUSTER_SIZE="${4:-1}"
GPT_BATCH_SIZE="${5:-128}"
CG_DEPTH="${6:-2}"
CG_FANOUT="${7:-5}"

# Navigate to project root
cd "$(dirname "$0")/../.."

echo "=== CGT Training ==="
echo "Dataset:        $DATASET"
echo "GPT epochs:     $GPT_EPOCHS"
echo "Cluster num:    $CLUSTER_NUM"
echo "Cluster size:   $CLUSTER_SIZE"
echo "GPT batch size: $GPT_BATCH_SIZE"
echo "CG depth:       $CG_DEPTH"
echo "CG fanout:      $CG_FANOUT"
echo ""

python CGT/train.py \
    --dataset "$DATASET" \
    --data_dir "datasets/original" \
    --gpt_epochs "$GPT_EPOCHS" \
    --cluster_num "$CLUSTER_NUM" \
    --cluster_size "$CLUSTER_SIZE" \
    --gpt_batch_size "$GPT_BATCH_SIZE" \
    --cg_depth "$CG_DEPTH" \
    --cg_fanout "$CG_FANOUT"
