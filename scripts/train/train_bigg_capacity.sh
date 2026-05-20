#!/bin/bash
# Train BiGG for the capacity benchmark — short calibration trial that emits a
# JSON timing log so the orchestrator can extrapolate "how many subgraphs fit
# in 1 hour at this size/density?". Mirrors train_bigg_subsample.sh but adds
# four trailing args for the capacity-specific flags.
#
# Usage:
#   bash scripts/train/train_bigg_capacity.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim] [noise_std] [ss_max_prob] [ss_start_epoch] [bfs_preprocess] [normalize] [loss_weights] [hetero_feat] [mask_test_labels] [logvar_floor] [subsample_size] [burn_prob] [num_subgraphs] [binary_feat] [vae_feat] [vae_dim] [kl_weight] [cat_feat] [n_bins] [bin_sigma] [mdn_feat] [mdn_components] [mdn_logsigma_floor] [mdn_base] [kl_schedule] [kl_anneal_epochs] [kl_cycle_epochs] [kl_ramp_ratio] [subsample_method] [multiplicity_cap] [num_train_subgraphs] [num_gen_subgraphs] [timing_log_path] [recal_momentum]
#
# Capacity-specific args:
#   subsample_method:    "forest_fire" or "metis" (default: forest_fire)
#   multiplicity_cap:    "m1" / "m2" / "minf" — only meaningful for forest_fire (default: minf)
#   num_train_subgraphs: cap inner training loop at first N partitions (default: empty = all)
#   num_gen_subgraphs:   cap generation loop at first N partitions (default: empty = same as train)
#   timing_log_path:     write JSON timing log to this path (default: empty = no log)
#   recal_momentum:      EMA momentum for dynamic loss-weight recalibration in [0,1]. 1.0
#                        (default) disables recalibration; 0.9/0.99 = ~10/~100 epoch horizon.

set -e

DATASET="${1:-tolokers}"
BLKSIZE="${2:--1}"
BSIZE="${3:-1}"
EPOCHS="${4:-5}"
LR="${5:-0.001}"
EMBED_DIM="${6:-256}"
NOISE_STD="${7:-0.0}"
SS_MAX_PROB="${8:-0.0}"
SS_START_EPOCH="${9:-0}"
BFS_PREPROCESS="${10:-False}"
NORMALIZE="${11:-zscore}"
LOSS_WEIGHTS="${12:-0.1,0.1}"
HETERO_FEAT="${13:-true}"
MASK_TEST_LABELS="${14:-true}"
LOGVAR_FLOOR="${15:--10.0}"
SUBSAMPLE_SIZE="${16:-2000}"
BURN_PROB="${17:-0.5}"
NUM_SUBGRAPHS="${18:-}"
BINARY_FEAT="${19:-false}"
VAE_FEAT="${20:-false}"
VAE_DIM="${21:-16}"
KL_WEIGHT="${22:-1.0}"
CAT_FEAT="${23:-false}"
N_BINS="${24:-32}"
BIN_SIGMA="${25:-}"
MDN_FEAT="${26:-false}"
MDN_COMPONENTS="${27:-8}"
MDN_LOGSIGMA_FLOOR="${28:--4.0}"
MDN_BASE="${29:-gaussian}"
KL_SCHEDULE="${30:-none}"
KL_ANNEAL_EPOCHS="${31:-0}"
KL_CYCLE_EPOCHS="${32:-0}"
KL_RAMP_RATIO="${33:-0.5}"
SUBSAMPLE_METHOD="${34:-forest_fire}"
MULTIPLICITY_CAP="${35:-minf}"
NUM_TRAIN_SUBGRAPHS="${36:-}"
NUM_GEN_SUBGRAPHS="${37:-}"
TIMING_LOG_PATH="${38:-}"
RECAL_MOMENTUM="${39:-1.0}"

cd "$(dirname "$0")/../../bigg"

echo "=== BiGG Capacity Trial ==="
echo "Dataset:               $DATASET"
echo "Subsample method:      $SUBSAMPLE_METHOD"
echo "Multiplicity cap:      $MULTIPLICITY_CAP"
echo "Subsample size:        $SUBSAMPLE_SIZE"
echo "Burn prob:             $BURN_PROB"
echo "Num subgraphs (K):     ${NUM_SUBGRAPHS:-auto}"
echo "Num train subgraphs:   ${NUM_TRAIN_SUBGRAPHS:-all}"
echo "Num gen subgraphs:     ${NUM_GEN_SUBGRAPHS:-same as train}"
echo "Epochs:                $EPOCHS"
echo "Timing log path:       ${TIMING_LOG_PATH:-<not set>}"
echo "Recal momentum:        $RECAL_MOMENTUM"
echo ""

NORM_FLAG=""
if [ "$NORMALIZE" != "none" ]; then
  NORM_FLAG="-normalize $NORMALIZE"
fi

HETERO_FLAG=""
if [ "$HETERO_FEAT" = "true" ]; then
  HETERO_FLAG="--hetero_feat"
fi

MASK_FLAG=""
if [ "$MASK_TEST_LABELS" = "true" ]; then
  MASK_FLAG="--mask_test_labels"
fi

NUM_SUBGRAPHS_FLAG=""
if [ -n "$NUM_SUBGRAPHS" ]; then
  NUM_SUBGRAPHS_FLAG="-num_subgraphs $NUM_SUBGRAPHS"
fi

BINARY_FLAG=""
if [ "$BINARY_FEAT" = "true" ]; then
  BINARY_FLAG="--binary_feat"
fi

VAE_FLAG=""
if [ "$VAE_FEAT" = "true" ]; then
  VAE_FLAG="--vae_feat"
fi

CAT_FLAG=""
if [ "$CAT_FEAT" = "true" ]; then
  CAT_FLAG="--cat_feat"
fi

BIN_SIGMA_FLAG=""
if [ -n "$BIN_SIGMA" ]; then
  BIN_SIGMA_FLAG="-bin_sigma $BIN_SIGMA"
fi

MDN_FLAG=""
if [ "$MDN_FEAT" = "true" ]; then
  MDN_FLAG="--mdn_feat"
fi

NUM_TRAIN_SUBGRAPHS_FLAG=""
if [ -n "$NUM_TRAIN_SUBGRAPHS" ]; then
  NUM_TRAIN_SUBGRAPHS_FLAG="-num_train_subgraphs $NUM_TRAIN_SUBGRAPHS"
fi

NUM_GEN_SUBGRAPHS_FLAG=""
if [ -n "$NUM_GEN_SUBGRAPHS" ]; then
  NUM_GEN_SUBGRAPHS_FLAG="-num_gen_subgraphs $NUM_GEN_SUBGRAPHS"
fi

TIMING_LOG_FLAG=""
if [ -n "$TIMING_LOG_PATH" ]; then
  TIMING_LOG_FLAG="-timing_log_path $TIMING_LOG_PATH"
fi

python -m bigg.extension.pipeline \
  -data_dir "$DATASET" \
  -model_type conditional \
  -gpu 0 \
  -embed_dim "$EMBED_DIM" \
  -bits_compress 0 \
  -bfs_preprocess "$BFS_PREPROCESS" \
  -learning_rate "$LR" \
  -num_epochs "$EPOCHS" \
  -batch_size "$BSIZE" \
  -blksize "$BLKSIZE" \
  -noise_std "$NOISE_STD" \
  -ss_max_prob "$SS_MAX_PROB" \
  -ss_start_epoch "$SS_START_EPOCH" \
  -seed 34 \
  $NORM_FLAG \
  -loss_weights "$LOSS_WEIGHTS" \
  $HETERO_FLAG \
  $MASK_FLAG \
  -logvar_floor "$LOGVAR_FLOOR" \
  --subsample \
  -subsample_size "$SUBSAMPLE_SIZE" \
  -burn_prob "$BURN_PROB" \
  $NUM_SUBGRAPHS_FLAG \
  $BINARY_FLAG \
  $VAE_FLAG \
  -vae_dim "$VAE_DIM" \
  -kl_weight "$KL_WEIGHT" \
  $CAT_FLAG \
  -n_bins "$N_BINS" \
  $BIN_SIGMA_FLAG \
  $MDN_FLAG \
  -mdn_components "$MDN_COMPONENTS" \
  -mdn_logsigma_floor "$MDN_LOGSIGMA_FLOOR" \
  -mdn_base "$MDN_BASE" \
  -kl_schedule "$KL_SCHEDULE" \
  -kl_anneal_epochs "$KL_ANNEAL_EPOCHS" \
  -kl_cycle_epochs "$KL_CYCLE_EPOCHS" \
  -kl_ramp_ratio "$KL_RAMP_RATIO" \
  -subsample_method "$SUBSAMPLE_METHOD" \
  -multiplicity_cap "$MULTIPLICITY_CAP" \
  $NUM_TRAIN_SUBGRAPHS_FLAG \
  $NUM_GEN_SUBGRAPHS_FLAG \
  $TIMING_LOG_FLAG \
  -recal_momentum "$RECAL_MOMENTUM" \
  -save_dir "checkpoints/bigg/capacity_${DATASET}_${SUBSAMPLE_METHOD}_${MULTIPLICITY_CAP}_size${SUBSAMPLE_SIZE}_K${NUM_SUBGRAPHS:-auto}"
