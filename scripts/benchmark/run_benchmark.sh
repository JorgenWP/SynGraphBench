#!/bin/bash
# Evaluate GNN models on original vs synthetic graph data.
#
# Usage:
#   bash scripts/benchmark/run_benchmark.sh [datasets] [models] [trials] [generator] [synthetic_name] [task]
#
# Arguments:
#   datasets        Comma-separated dataset names (default: reddit)
#   models          Comma-separated model names (default: GCN,GIN,GraphSAGE,XGBGraph)
#   trials          Number of evaluation trials (default: 1)
#   generator       Generative model folder under datasets/synthetic/ (default: cgt)
#                   Supported: cgt, bigg
#   synthetic_name  Exact filename stem for a specific variant (default: uses dataset name)
#   task            Task subfolder under <dataset>/ (default: hidden_labels)
#                   Supported: hidden_labels, hidden_links, structure
#
# Examples:
#   bash scripts/benchmark/run_benchmark.sh reddit GCN,GIN 3 cgt
#   bash scripts/benchmark/run_benchmark.sh tolokers GCN,GIN 1 bigg blksize_1024_b_1_lr_0.001_epochs_50
#   bash scripts/benchmark/run_benchmark.sh tolokers GCN,GIN 1 bigg structure_blksize_128_lr_0.001_epochs_100 structure

set -e

# Configuration with defaults
DATASETS="${1:-reddit}"
MODELS="${2:-GCN,GIN,GraphSAGE,XGBGraph}"
TRIALS="${3:-1}"
GENERATOR="${4:-cgt}"
SYNTHETIC_NAME="${5:-}"
TASK="${6:-hidden_labels}"

# Map generator to its synthetic type (evaluation mode)
case "$GENERATOR" in
    cgt)  SYNTHETIC_TYPE="comp-graph" ;;
    bigg) SYNTHETIC_TYPE="graph" ;;
    *)    echo "ERROR: Unknown generator '$GENERATOR'. Supported: cgt, bigg"; exit 1 ;;
esac

# Navigate to project root
cd "$(dirname "$0")/../.."

echo "=== GNN Evaluation: Original vs Synthetic ==="
echo "Datasets:         $DATASETS"
echo "Models:           $MODELS"
echo "Trials:           $TRIALS"
echo "Generator:        $GENERATOR  (datasets/synthetic/$GENERATOR/)"
echo "Synthetic type:   $SYNTHETIC_TYPE"
echo "Synthetic name:   ${SYNTHETIC_NAME:-'(use dataset name)'}"
echo "Task:             $TASK"
echo ""

EXTRA_ARGS=""
if [ -n "$SYNTHETIC_NAME" ]; then
    EXTRA_ARGS="--synthetic_name $SYNTHETIC_NAME"
fi

python scripts/benchmark/benchmark.py \
    --datasets "$DATASETS" \
    --models "$MODELS" \
    --trials "$TRIALS" \
    --generator "$GENERATOR" \
    --synthetic_type "$SYNTHETIC_TYPE" \
    --task "$TASK" \
    $EXTRA_ARGS
