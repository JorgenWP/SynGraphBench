#!/bin/bash
# Evaluate GNN models on original vs synthetic graph data.
#
# Usage:
#   bash scripts/benchmark/run_benchmark.sh [datasets] [models] [trials] [synthetic_model]
#
# Examples:
#   bash scripts/benchmark/run_benchmark.sh reddit GCN,GIN 3 cgt
#   bash scripts/benchmark/run_benchmark.sh reddit GCN,GIN 3 bigg
#

set -e

# Configuration with defaults
DATASETS="${1:-reddit}"
MODELS="${2:-GCN,GIN,GraphSAGE,XGBGraph}"
TRIALS="${3:-1}"
SYNTHETIC_MODEL="${4:-cgt}"

# Map synthetic model to its data type
case "$SYNTHETIC_MODEL" in
    cgt)  SYNTHETIC_TYPE="cgt" ;;
    bigg) SYNTHETIC_TYPE="graph" ;;
    *)    echo "ERROR: Unknown synthetic model '$SYNTHETIC_MODEL'. Supported: cgt, bigg"; exit 1 ;;
esac

# Navigate to project root
cd "$(dirname "$0")/../.."

echo "=== GNN Evaluation: Original vs Synthetic ==="
echo "Datasets:         $DATASETS"
echo "Models:           $MODELS"
echo "Trials:           $TRIALS"
echo "Synthetic model:  $SYNTHETIC_MODEL"
echo "Synthetic type:   $SYNTHETIC_TYPE"
echo ""

python scripts/benchmark/benchmark.py \
    --datasets "$DATASETS" \
    --models "$MODELS" \
    --trials "$TRIALS" \
    --synthetic_model "$SYNTHETIC_MODEL" \
    --synthetic_type "$SYNTHETIC_TYPE"
