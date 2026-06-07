#!/bin/bash
# Evaluate GNN models on link prediction: original vs synthetic graph data.
#
# Usage:
#   bash scripts/benchmark/run_link_benchmark.sh [datasets] [models] [trials] [generator] [synthetic_name] [neg_sampling] [decoder] [task] [seeds_per_split] [dump_per_trial] [skip_original] [cdf_invert]
#
# Arguments:
#   datasets        Comma-separated dataset names (default: reddit)
#   models          Comma-separated model names (default: GCN,GIN,GraphSAGE)
#   trials          Number of evaluation trials (default: 1). In BiGG split-bundle
#                   mode this is overridden to the number of variants in the bundle.
#   generator       Generative model folder under datasets/synthetic/ (default: cgt)
#                   Supported: cgt, bigg
#   synthetic_name  Exact filename stem for a specific variant (default: uses dataset name)
#   neg_sampling    Negative sampling strategy: random or hard (default: random)
#   decoder         Edge decoder: dot or mlp (default: dot)
#   task            Task subfolder under <dataset>/ (default: hidden_links)
#                   Supported: hidden_labels, hidden_links, structure
#   seeds_per_split Seeds to repeat per split in BiGG bundle mode (default: 3).
#                   Total runs in bundle mode = #splits * seeds_per_split. The same
#                   seeds are reused across splits to decouple seed/split variance.
#   dump_per_trial  "true"/"false" (default: false). When true, write raw per-
#                   (split, model, seed) AUROC/AUPRC/RecK rows to per_trial_results.csv.
#   skip_original   "true"/"false" (default: false). Skip the original-data baseline
#                   (Phase 1) — use when the baseline was already computed.
#   cdf_invert      cdf inversion strategy when combining subgraphs (default: linear)
#                   Supported: linear, nearest
#
# BiGG split bundles: if synthetic_name points to a directory whose immediate
# children are per-split BiGG variant dirs (each containing subgraph_* files and
# 'split{N}' in its name), the benchmark switches to bundle mode: trials is
# overridden to the number of variants, each combined and tested on the original
# graph's held-out edges for its split. Results auto-land at
# results/evaluate/<generator>/<task>/<dataset>/<synthetic_name>/.
#
# GADBench split id (trial_id) is auto-inferred from a BiGG synthetic_name
# containing _loadsub_..._split{N}_n... and is NOT exposed as a positional
# argument — BiGG's training subsample for split N excludes only that split's
# test nodes, so a misaligned trial_id silently leaks training data. To
# override (legacy stems / cross-split debugging), call the Python script
# directly with --trial_id.
#
# Examples:
#   bash scripts/benchmark/run_link_benchmark.sh reddit GCN,GIN 3 cgt
#   bash scripts/benchmark/run_link_benchmark.sh reddit GCN,GIN 3 cgt "" random dot hidden_links
#   bash scripts/benchmark/run_link_benchmark.sh tolokers GCN,GIN 1 bigg blksize_1024_b_1_lr_0.001_epochs_50
#   bash scripts/benchmark/run_link_benchmark.sh tolokers GCN,GIN 1 bigg structure_blksize_128_lr_0.001_epochs_100 random dot structure

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
SEEDS_PER_SPLIT="${9:-3}"
DUMP_PER_TRIAL="${10:-false}"
SKIP_ORIGINAL="${11:-false}"
CDF_INVERT="${12:-linear}"

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
echo "Seeds/split:      $SEEDS_PER_SPLIT  (BiGG bundle mode only)"
echo "Skip original:    $SKIP_ORIGINAL"
echo "CDF invert:       $CDF_INVERT"
echo ""

EXTRA_ARGS=""
if [ -n "$SYNTHETIC_NAME" ]; then
    EXTRA_ARGS="--synthetic_name $SYNTHETIC_NAME"
fi
if [ "$DUMP_PER_TRIAL" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --dump_per_trial"
fi
if [ "$SKIP_ORIGINAL" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_original"
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
    --seeds_per_split "$SEEDS_PER_SPLIT" \
    --cdf_invert "$CDF_INVERT" \
    $EXTRA_ARGS
