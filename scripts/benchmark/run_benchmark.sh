#!/bin/bash
# Evaluate GNN models on original vs CGT-synthetic graph data.
#
# Usage:
#   bash scripts/benchmark/run_benchmark.sh [datasets] [models] [trials] 
#


set -e

# Configuration with defaults
DATASETS="${1:-reddit}"
MODELS="${2:-GCN,GIN,GraphSAGE,XGBGraph}"
TRIALS="${3:-1}"

# Navigate to project root
cd "$(dirname "$0")/../.."

# Activate GADBench environment 
eval "$(conda shell.bash hook)"
conda activate GADBench

echo "=== GNN Evaluation: Original vs Synthetic ==="
echo "Datasets: $DATASETS"
echo "Models:   $MODELS"
echo "Trials:   $TRIALS"
echo ""

python scripts/benchmark/benchmark.py \
    --datasets "$DATASETS" \
    --models "$MODELS" \
    --trials "$TRIALS"
