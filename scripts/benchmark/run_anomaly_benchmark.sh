#!/bin/bash
# Evaluate GNN models on original vs synthetic graph data.
#
# Usage:
#   bash scripts/benchmark/run_anomaly_benchmark.sh [datasets] [models] [trials] [generator] [synthetic_name] [task] [cdf_invert] [seeds_per_split] [dump_per_trial] [tune_test_ratio] [tune_test_seed] [tune_portion]
#
# Arguments:
#   datasets        Comma-separated dataset names (default: reddit)
#   models          Comma-separated model names (default: GCN,GIN,GraphSAGE,XGBGraph,XGBoost)
#                   XGBoost is the feature-only diagnostic row (trains on syn raw features,
#                   tests on orig raw features — no graph either side).
#   trials          Number of evaluation trials (default: 1). In BiGG split-bundle mode
#                   this is overridden to the number of variants in the bundle.
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
#   seeds_per_split Number of seeds to repeat per split in BiGG bundle mode (default: 3).
#                   Total runs in bundle mode = #splits * seeds_per_split. Reusing the same
#                   SEED_LIST[:seeds_per_split] across every split decouples seed and split
#                   variance. Ignored in single-variant mode and CGT mode.
#   dump_per_trial  "true"/"false" (default: false). When true, write raw per-
#                   (split, classifier, seed) AUROC/AUPRC/RecK rows to
#                   per_trial_results.csv alongside evaluation_results.csv.
#   tune_test_ratio Optional float in (0,1). When set, restrict the per-split
#                   test mask to an anomaly-stratified subsample of this
#                   fraction (BO selection uses the "tune" half; held-out is
#                   reserved for the final report).
#   tune_test_seed  Seed for the stratified tune/heldout split (default: 0).
#   tune_portion    "tune" (default) or "heldout"; ignored unless tune_test_ratio is set.
#
# GADBench split id (trial_id) is auto-inferred from a BiGG synthetic_name
# containing _loadsub_..._split{N}_n... and is NOT exposed as a positional
# argument — BiGG's training subsample for split N excludes only that split's
# test nodes, so a misaligned trial_id silently leaks training data. To
# override (legacy stems / cross-split debugging), call the Python script
# directly with --trial_id.
#
# BiGG split bundles: if synthetic_name points to a directory whose
# immediate children are per-split BiGG variant directories (each containing
# subgraph_* files and 'split{N}' in its name — typically symlinks or moves
# of independently-tuned variants), the benchmark switches to bundle mode:
# trials is overridden to the number of variants, each trial loads its own
# variant + split, and the original-data baseline rotates through the same
# split ids. Cache files (_combined_v2, _real_sub_combined_v3, _norm_stats.pt)
# are written inside the bundle dir, one per variant. This is the right
# mode when each split was tuned to different hyper-parameters.
#
# Examples:
#   bash scripts/benchmark/run_anomaly_benchmark.sh reddit GCN,GIN 3 cgt
#   bash scripts/benchmark/run_anomaly_benchmark.sh tolokers GCN,GIN 1 bigg blksize_1024_b_1_lr_0.001_epochs_50
#   bash scripts/benchmark/run_anomaly_benchmark.sh tolokers GCN,GIN 1 bigg structure_blksize_128_lr_0.001_epochs_100 structure
#   bash scripts/benchmark/run_anomaly_benchmark.sh reddit GCN,GIN,GraphSAGE 1 bigg blksize_-1_b_1_lr_0.001_epochs_300_noise_0.1_ss_0.0_BFSPRE_False_reverted hidden_labels
#   # Bundle: <task_dir>/my_5_splits/ contains 5 sub-dirs, each named *_split{0..4}_n5
#   bash scripts/benchmark/run_anomaly_benchmark.sh tolokers GCN,GIN,GraphSAGE,XGBGraph,XGBoost 5 bigg my_5_splits hidden_labels

set -e

# Configuration with defaults
DATASETS="${1:-reddit}"
MODELS="${2:-GCN,GIN,GraphSAGE,XGBGraph,XGBoost}"
TRIALS="${3:-1}"
GENERATOR="${4:-cgt}"
SYNTHETIC_NAME="${5:-}"
TASK="${6:-hidden_labels}"
CDF_INVERT="${7:-linear}"
SEEDS_PER_SPLIT="${8:-3}"
DUMP_PER_TRIAL="${9:-false}"
TUNE_TEST_RATIO="${10:-}"
TUNE_TEST_SEED="${11:-0}"
TUNE_PORTION="${12:-tune}"

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
echo "Seeds/split:      $SEEDS_PER_SPLIT  (BiGG bundle mode only)"
echo ""

EXTRA_ARGS=""
if [ -n "$SYNTHETIC_NAME" ]; then
    EXTRA_ARGS="--synthetic_name $SYNTHETIC_NAME"
fi
if [ "$DUMP_PER_TRIAL" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --dump_per_trial"
fi
if [ -n "$TUNE_TEST_RATIO" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --tune_test_ratio $TUNE_TEST_RATIO --tune_test_seed $TUNE_TEST_SEED --tune_portion $TUNE_PORTION"
fi

python -u scripts/benchmark/anomaly_benchmark.py \
    --datasets "$DATASETS" \
    --models "$MODELS" \
    --trials "$TRIALS" \
    --generator "$GENERATOR" \
    --synthetic_type "$SYNTHETIC_TYPE" \
    --task "$TASK" \
    --cdf_invert "$CDF_INVERT" \
    --seeds_per_split "$SEEDS_PER_SPLIT" \
    $EXTRA_ARGS
