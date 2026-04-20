#!/bin/bash
# Train CGT on all GADBench splits (trials 0 to NUM_TRIALS-1).
#
# Usage:
#   bash scripts/train/train_cgt_all_trials.sh [dataset] [gpt_epochs] [cluster_num] [cluster_size] [gpt_batch_size] [cg_depth] [cg_fanout] [num_trials] [task] [cluster_sample_num]
#
# Examples:
#   bash scripts/train/train_cgt_all_trials.sh reddit 50 512 1 128 2 5
#   bash scripts/train/train_cgt_all_trials.sh reddit 50 512 1 128 2 5 5 hidden_links
#   bash scripts/train/train_cgt_all_trials.sh reddit 50 100 100 128 2 5 10 hidden_labels 11000

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

SCRIPT_DIR="$(dirname "$0")"

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
    bash "$SCRIPT_DIR/train_cgt.sh" \
        "$DATASET" "$GPT_EPOCHS" "$CLUSTER_NUM" "$CLUSTER_SIZE" \
        "$GPT_BATCH_SIZE" "$CG_DEPTH" "$CG_FANOUT" "$t" "$TASK" \
        "$CLUSTER_SAMPLE_NUM"
done
