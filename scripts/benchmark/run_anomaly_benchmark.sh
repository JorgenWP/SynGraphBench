#!/bin/bash
# Evaluate GNN models on original vs synthetic graph data.
#
# Usage:
#   bash scripts/benchmark/run_anomaly_benchmark.sh [datasets] [models] [trials] [generator] [synthetic_name] [task] [cdf_invert] [eval_mode] [skip_phase1] [output_dir]
#
# Arguments:
#   datasets        Comma-separated dataset names (default: reddit)
#   models          Comma-separated model names (default: GCN,GIN,GraphSAGE,XGBGraph,XGBoost)
#                   XGBoost is the feature-only diagnostic row (trains on syn raw features,
#                   tests on orig raw features — no graph either side).
#   trials          Number of evaluation trials (default: 1)
#   generator       Generative model folder under datasets/synthetic/ (default: cgt)
#                   Supported: cgt, bigg
#   synthetic_name  Exact filename stem for a specific variant (default: uses dataset name)
#   task            Task subfolder under <dataset>/ (default: hidden_labels)
#                   Supported: hidden_labels, hidden_links, structure
#   cdf_invert      Inversion strategy for cdf-normalized runs (default: linear)
#                   linear  — linearly interpolate between adjacent sorted training values
#                   nearest — snap predicted ranks to the closest sorted training value
#                             (keeps inverted output on the empirical support; cache key
#                             is suffixed so switching modes doesn't reuse stale caches)
#   eval_mode       CGT comp-graph paths to run in Phase 2 (default: both)
#                   Supported: original_cg, original_cg_quantized, synthetic_cgt,
#                              phase2_variant, both
#                   phase2_variant = quantized + synthetic_cgt in one process
#                   (the per-variant subset for an array job).
#   skip_phase1     Set to 1 to skip Phase 1 whole-graph baseline (default: 0)
#   output_dir      Explicit results dir (default: empty → auto-derive
#                   to results/evaluate/{generator}/{dataset}/{task}/{synthetic_name})
#
# Examples:
#   bash scripts/benchmark/run_anomaly_benchmark.sh reddit GCN,GIN 3 cgt
#   bash scripts/benchmark/run_anomaly_benchmark.sh tolokers GCN,GIN 1 bigg blksize_1024_b_1_lr_0.001_epochs_50
#   bash scripts/benchmark/run_anomaly_benchmark.sh tolokers GCN,GIN 1 bigg structure_blksize_128_lr_0.001_epochs_100 structure
#   bash scripts/benchmark/run_anomaly_benchmark.sh reddit GCN,GIN,GraphSAGE 1 bigg blksize_-1_b_1_lr_0.001_epochs_300_noise_0.1_ss_0.0_BFSPRE_False_reverted hidden_labels
#   # Array-job shared task (Phase 1 + original-cg only, written to a shared dir):
#   bash scripts/benchmark/run_anomaly_benchmark.sh tolokers GCN,GIN,GraphSAGE,XGBoost,XGBGraph 5 cgt tolokers_e50_k512_c1_d2_f5_s8818 hidden_labels linear original_cg 0 results/evaluate/cgt/tolokers/hidden_labels/_original_cg_shared
#   # Array-job per-variant task (quantized + synthetic_cgt, skip Phase 1, auto output dir):
#   bash scripts/benchmark/run_anomaly_benchmark.sh tolokers GCN,GIN,GraphSAGE,XGBoost,XGBGraph 5 cgt tolokers_e50_k512_c1_d2_f5_s8818 hidden_labels linear phase2_variant 1

set -e

# Configuration with defaults
DATASETS="${1:-reddit}"
MODELS="${2:-GCN,GIN,GraphSAGE,XGBGraph,XGBoost}"
TRIALS="${3:-1}"
GENERATOR="${4:-cgt}"
SYNTHETIC_NAME="${5:-}"
TASK="${6:-hidden_labels}"
CDF_INVERT="${7:-linear}"
EVAL_MODE="${8:-both}"
SKIP_PHASE1="${9:-0}"
OUTPUT_DIR="${10:-}"

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
echo "CDF invert:       $CDF_INVERT"
echo "Eval mode:        $EVAL_MODE"
echo "Skip phase 1:     $SKIP_PHASE1"
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

python -u scripts/benchmark/anomaly_benchmark.py \
    --datasets "$DATASETS" \
    --models "$MODELS" \
    --trials "$TRIALS" \
    --generator "$GENERATOR" \
    --synthetic_type "$SYNTHETIC_TYPE" \
    --task "$TASK" \
    --cdf_invert "$CDF_INVERT" \
    --eval_mode "$EVAL_MODE" \
    $EXTRA_ARGS
