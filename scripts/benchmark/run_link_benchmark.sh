#!/bin/bash
# Evaluate GNN models on link prediction: original vs synthetic graph data.
#
# Usage:
#   bash scripts/benchmark/run_link_benchmark.sh [datasets] [models] [trials] [generator] [neg_sampling] [decoder] [synthetic_name]
#
# Arguments:
#   datasets        Comma-separated dataset names (default: reddit)
#   models          Comma-separated model names (default: GCN,GIN,GraphSAGE)
#   trials          Number of evaluation trials (default: 1)
#   generator       Generative model folder under datasets/synthetic/ (default: cgt)
#   neg_sampling    Negative sampling strategy: random or hard (default: random)
#   decoder         Edge decoder: dot or mlp (default: dot)
#   synthetic_name  Exact filename stem for a specific variant (default: uses dataset name)
#
# Examples:
#   bash scripts/benchmark/run_link_benchmark.sh reddit GCN,GIN 3 cgt random dot
#   bash scripts/benchmark/run_link_benchmark.sh reddit GCN,GIN 3 cgt hard mlp
#   bash scripts/benchmark/run_link_benchmark.sh tolokers GCN,GIN 1 bigg random mlp tolokers_blksize_1024_b_1

set -e

# Configuration with defaults
DATASETS="${1:-reddit}"
MODELS="${2:-GCN,GIN,GraphSAGE}"
TRIALS="${3:-1}"
GENERATOR="${4:-cgt}"
NEG_SAMPLING="${5:-random}"
DECODER="${6:-dot}"
SYNTHETIC_NAME="${7:-}"

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
echo "Neg sampling:     $NEG_SAMPLING"
echo "Decoder:          $DECODER"
echo "Synthetic name:   ${SYNTHETIC_NAME:-'(use dataset name)'}"
echo ""

EXTRA_ARGS=""
if [ -n "$SYNTHETIC_NAME" ]; then
    EXTRA_ARGS="--synthetic_name $SYNTHETIC_NAME"
fi

python scripts/benchmark/link_benchmark.py \
    --datasets "$DATASETS" \
    --models "$MODELS" \
    --trials "$TRIALS" \
    --generator "$GENERATOR" \
    --synthetic_type "$SYNTHETIC_TYPE" \
    --neg_sampling "$NEG_SAMPLING" \
    --decoder "$DECODER" \
    $EXTRA_ARGS
