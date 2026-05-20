#!/bin/bash
# Train CGT on NUM_TRIALS GADBench splits (trials 0 to NUM_TRIALS-1).
# Set num_trials=1 to train a single trial (trial_id=0).
#
# Usage:
#   bash scripts/train/train_cgt.sh [dataset] [gpt_epochs] [cluster_num] [cluster_size] [gpt_batch_size] [cg_depth] [cg_fanout] [num_trials] [task] [cluster_sample_num]
#
# Examples:
#   bash scripts/train/train_cgt.sh reddit 50 512 1 128 2 5
#   bash scripts/train/train_cgt.sh reddit 50 512 1 128 2 5 1 hidden_links
#   bash scripts/train/train_cgt.sh reddit 50 100 100 128 2 5 10 hidden_labels 11000

set -e

DATASET="${1:-reddit}"
GPT_EPOCHS="${2:-50}"
CLUSTER_NUM="${3:-512}"
CLUSTER_SIZE="${4:-1}"
GPT_BATCH_SIZE="${5:-128}"
CG_DEPTH="${6:-2}"
CG_FANOUT="${7:-5}"
NUM_TRIALS="${8:-10}"
TASK="${9:-hidden_labels}"
CLUSTER_SAMPLE_NUM="${10:-5000}"

# Navigate to project root
cd "$(dirname "$0")/../.."

# Skip trials whose synthetic output .pt already exists; makes re-submission
# after a SLURM timeout idempotent. Path format must match CGT/train.py:60-72.
VARIANT="${DATASET}_e${GPT_EPOCHS}_k${CLUSTER_NUM}_c${CLUSTER_SIZE}_d${CG_DEPTH}_f${CG_FANOUT}_s${CLUSTER_SAMPLE_NUM}"
SAVE_DIR="datasets/synthetic/cgt/${DATASET}/${TASK}/${VARIANT}"

for t in $(seq 0 $((NUM_TRIALS - 1))); do
    SAVE_PATH="${SAVE_DIR}/${VARIANT}_t${t}.pt"
    if [ -f "$SAVE_PATH" ]; then
        echo ""
        echo "=========================================="
        echo "  Trial $t / $((NUM_TRIALS - 1))  [SKIP - exists]"
        echo "=========================================="
        continue
    fi
    echo ""
    echo "=========================================="
    echo "  Trial $t / $((NUM_TRIALS - 1))"
    echo "=========================================="
    python CGT/train.py \
        --dataset "$DATASET" \
        --data_dir "datasets/original" \
        --gpt_epochs "$GPT_EPOCHS" \
        --cluster_num "$CLUSTER_NUM" \
        --cluster_size "$CLUSTER_SIZE" \
        --gpt_batch_size "$GPT_BATCH_SIZE" \
        --cg_depth "$CG_DEPTH" \
        --cg_fanout "$CG_FANOUT" \
        --trial_id "$t" \
        --task "$TASK" \
        --cluster_sample_num "$CLUSTER_SAMPLE_NUM"
done
