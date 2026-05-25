#!/bin/bash
# Evaluate GNN models on link prediction: original vs synthetic graph data.
#
# Usage:
#   bash scripts/benchmark/run_link_benchmark.sh [datasets] [models] [trials] [generator] [synthetic_name] [neg_sampling] [decoder] [task] [eval_mode] [skip_phase1] [batch_size] [output_dir]
#
# Arguments:
#   datasets        Comma-separated dataset names (default: reddit)
#   models          Comma-separated model names (default: GCN,GIN,GraphSAGE)
#   trials          Number of evaluation trials (default: 1)
#   generator       Generative model folder under datasets/synthetic/ (default: cgt)
#                   Supported: cgt, bigg
#   synthetic_name  Exact filename stem for a specific variant (default: uses dataset name)
#   neg_sampling    Negative sampling strategy: random or hard (default: random)
#   decoder         Edge decoder: dot or mlp (default: dot)
#   task            Task subfolder under <dataset>/ (default: hidden_links)
#                   Supported: hidden_labels, hidden_links, structure
#   eval_mode       CGT comp-graph paths to run in Phase 2 (default: both)
#                   Supported: original_cg, synthetic_cgt, both
#   skip_phase1     Set to 1 to skip Phase 1 full-graph baseline (default: 0)
#   batch_size      Batch size for CGT comp-graph training (default: 256)
#   output_dir      Explicit results dir (default: empty → auto-derive
#                   to results/evaluate/{generator}/{dataset}/{task}/{synthetic_name})
#
# Examples:
#   bash scripts/benchmark/run_link_benchmark.sh reddit GCN,GIN 3 cgt
#   bash scripts/benchmark/run_link_benchmark.sh reddit GCN,GIN 3 cgt "" random dot hidden_links
#   bash scripts/benchmark/run_link_benchmark.sh tolokers GCN,GIN 1 bigg blksize_1024_b_1_lr_0.001_epochs_50
#   bash scripts/benchmark/run_link_benchmark.sh tolokers GCN,GIN 1 bigg structure_blksize_128_lr_0.001_epochs_100 random dot structure
#   bash scripts/benchmark/run_link_benchmark.sh tolokers GCN,GIN,GraphSAGE 5 cgt tolokers_e50_k512_c1_d2_f5_s8818 random dot hidden_links original_cg 1 4096

set -e

# Configuration with defaults
DATASETS="${1:-reddit}"
MODELS="${2:-GCN,GIN,GraphSAGE}"
TRIALS="${3:-1}"
GENERATOR="${4:-cgt}"
SYNTHETIC_NAME="${5:-}"
NEG_SAMPLING="${6:-random}"
DECODER="${7:-dot}"
TASK="${8:-hidden_links}"
EVAL_MODE="${9:-both}"
SKIP_PHASE1="${10:-0}"
BATCH_SIZE="${11:-256}"
OUTPUT_DIR="${12:-}"

# Map generator to its synthetic type (evaluation mode)
case "$GENERATOR" in
    cgt)  SYNTHETIC_TYPE="comp-graph" ;;
    bigg) SYNTHETIC_TYPE="graph" ;;
    *)    echo "ERROR: Unknown generator '$GENERATOR'. Supported: cgt, bigg"; exit 1 ;;
esac

# Navigate to project root
cd "$(dirname "$0")/../.."

echo "=== Link Prediction: Original vs Synthetic ==="
echo "Datasets:         $DATASETS"
echo "Models:           $MODELS"
echo "Trials:           $TRIALS"
echo "Generator:        $GENERATOR  (datasets/synthetic/$GENERATOR/)"
echo "Synthetic type:   $SYNTHETIC_TYPE"
echo "Synthetic name:   ${SYNTHETIC_NAME:-'(use dataset name)'}"
echo "Task:             $TASK"
echo "Neg sampling:     $NEG_SAMPLING"
echo "Decoder:          $DECODER"
echo "Eval mode:        $EVAL_MODE"
echo "Skip phase 1:     $SKIP_PHASE1"
echo "Batch size:       $BATCH_SIZE"
echo "Output dir:       ${OUTPUT_DIR:-'(auto-derive)'}"
echo ""

EXTRA_ARGS=""
if [ -n "$SYNTHETIC_NAME" ]; then
    EXTRA_ARGS="--synthetic_name $SYNTHETIC_NAME"
fi
if [ "$SKIP_PHASE1" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_phase1"
fi
if [ -n "$OUTPUT_DIR" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --output_dir $OUTPUT_DIR"
fi

python -u scripts/benchmark/link_benchmark.py \
    --datasets "$DATASETS" \
    --models "$MODELS" \
    --trials "$TRIALS" \
    --generator "$GENERATOR" \
    --synthetic_type "$SYNTHETIC_TYPE" \
    --task "$TASK" \
    --neg_sampling "$NEG_SAMPLING" \
    --decoder "$DECODER" \
    --eval_mode "$EVAL_MODE" \
    --batch_size "$BATCH_SIZE" \
    $EXTRA_ARGS
