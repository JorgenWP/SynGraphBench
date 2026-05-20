#!/bin/bash
# Train BiGG model on a dataset using forest fire subsampling.
#
# Use this script when the full graph does not fit in VRAM. Subgraphs are
# sampled independently via forest fire (with replacement), each trained
# independently. At generation time, one synthetic subgraph is produced per
# training subgraph.
#
# Usage:
#   bash scripts/train/train_bigg_subsample.sh [dataset] [blksize] [batch_size] [epochs] [lr] [embed_dim] [noise_std] [ss_max_prob] [ss_start_epoch] [bfs_preprocess] [normalize] [loss_weights] [hetero_feat] [mask_test_labels] [logvar_floor] [subsample_size] [burn_prob] [num_subgraphs] [binary_feat] [vae_feat] [vae_dim] [kl_weight] [cat_feat] [n_bins] [bin_sigma] [mdn_feat] [mdn_components] [mdn_logsigma_floor] [mdn_base] [kl_schedule] [kl_anneal_epochs] [kl_cycle_epochs] [kl_ramp_ratio] [load_subsamples] [subsampling_config] [split_id] [recal_momentum]
#
# normalize:        feature normalisation — one of "zscore", "minmax", "row", "quantile", or "none" (default: none)
# loss_weights:     comma-separated cont,label weights relative to struct, applied after dynamic normalization (default: 1,1)
# hetero_feat:      "true" to enable heteroscedastic feature prediction (mean + variance), "false" for deterministic MSE (default: false)
# mask_test_labels: "true" to exclude test node labels (split 0) from label loss (default: false)
# logvar_floor:     lower clamp for log-variance in hetero_feat mode (default: -4.0)
# subsample_size:   target number of nodes per subgraph (default: 2000)
# burn_prob:        forest fire burn probability — controls subgraph density (default: 0.3)
# num_subgraphs:    number of subgraphs to use (default: auto = ceil(N / subsample_size))
# binary_feat:      "true" to auto-detect binary features and use BCE loss + Bernoulli sampling (default: false)
# vae_feat:         "true" to add a per-node label-agnostic CVAE latent shared across feature decoders (default: false)
# vae_dim:          latent dimensionality when vae_feat is on (default: 16)
# kl_weight:        coefficient on the KL term in the VAE ELBO (default: 1.0)
# cat_feat:         "true" to use AR categorical feature predictor (quantile bins + value-space soft labels).
#                   Mutually exclusive with hetero_feat and vae_feat (default: false)
# n_bins:           number of quantile bins per continuous feature when cat_feat is on (default: 32)
# bin_sigma:        Gaussian soft-label std in feature-value units. Leave empty for auto
#                   (0.5 x median bin-center spacing) (default: auto)
# mdn_feat:         "true" to use Mixture Density Network feature head (per-feature K-component
#                   mixture). Mutually exclusive with hetero_feat/cat_feat. Composes with vae_feat (default: false)
# mdn_components:   number of mixture components per feature when mdn_feat is on (default: 8)
# mdn_logsigma_floor: lower clamp for MDN component log-sigma (default: -4.0)
# mdn_base:         per-component base distribution when mdn_feat is on — "gaussian" (unbounded targets)
#                   or "logit_normal" (targets in [0,1]; pairs with --normalize cdf) (default: gaussian)
# kl_schedule:      KL annealing schedule for the VAE coefficient — "none", "linear", or "cyclic"
#                   (default: none). Addresses posterior collapse when vae_feat is on.
# kl_anneal_epochs: linear ramp duration in epochs (required when kl_schedule=linear). β goes 0 →
#                   kl_weight over this many epochs, then stays. (default: 0)
# kl_cycle_epochs:  cyclic schedule cycle length in epochs (required when kl_schedule=cyclic).
#                   β resets to 0 every cycle. (default: 0)
# kl_ramp_ratio:    fraction of cycle spent ramping when kl_schedule=cyclic (paper R, default 0.5).
#                   0.5 → first 50%% of cycle ramps 0→1, last 50%% sits at 1.
# load_subsamples:  "true" to load pre-computed training subgraphs from
#                   datasets/bigg_subsamples/<dataset>/<subsampling_config>/split<split_id>.pkl
#                   instead of sampling at runtime. When on, subsample_size/burn_prob/num_subgraphs
#                   are ignored (default: false).
# subsampling_config: params_tag of the persisted subsample set (e.g. "ff_b0.5_M1", "metis_K5").
#                   Required when load_subsamples=true.
# split_id:         GADBench split id (0..4) selecting which split<id>.pkl to load (default: 0).
# recal_momentum:   EMA momentum for dynamic loss-weight recalibration in [0,1]. 1.0 (default)
#                   disables recalibration (one-time calibration only — bit-identical to pre-feature
#                   behavior). Lower values recompute w_cont/w_bin/w_label each epoch from EMA-
#                   smoothed component magnitudes; typical 0.9 (~10-epoch horizon) or 0.99
#                   (~100-epoch horizon). user_w_cont/user_w_label from loss_weights still apply.
#
# Examples:
#   bash scripts/train/train_bigg_subsample.sh reddit -1 1 300 0.001 256 0.3 0.0 0 True zscore 0.1,0.1 true true -4.0 2000 0.3
#   bash scripts/train/train_bigg_subsample.sh reddit -1 1 300 0.001 256 0.3 0.0 0 True zscore 0.1,0.1 true true -4.0 2000 0.3 3 true
#   bash scripts/train/train_bigg_subsample.sh tolokers -1 1 300 0.001 256 0.3 0.0 0 True zscore 0.1,0.1 true true -10.0 500 0.2 15 false true 16 1.0
#   bash scripts/train/train_bigg_subsample.sh tolokers -1 1 300 0.001 256 0.3 0.0 0 True zscore 0.1,0.1 false true -10.0 500 0.2 15 false false 16 1.0 true 32
#   bash scripts/train/train_bigg_subsample.sh tolokers -1 1 300 0.001 256 0.3 0.0 0 True zscore 0.1,0.1 true true -10.0 500 0.2 15 false false 16 1.0 false 32 "" false 8 -4.0 gaussian none 0 0 0.5 true ff_b0.5_M1 0
#

set -e

# Configuration with defaults
DATASET="${1:-tolokers}"
BLKSIZE="${2:-1024}"
BSIZE="${3:-1}"
EPOCHS="${4:-50}"
LR="${5:-0.001}"
EMBED_DIM="${6:-256}"
NOISE_STD="${7:-0.0}"
SS_MAX_PROB="${8:-0.0}"
SS_START_EPOCH="${9:-0}"
BFS_PREPROCESS="${10:-False}"
NORMALIZE="${11:-none}"
LOSS_WEIGHTS="${12:-1,1}"
HETERO_FEAT="${13:-false}"
MASK_TEST_LABELS="${14:-false}"
LOGVAR_FLOOR="${15:--4.0}"
SUBSAMPLE_SIZE="${16:-2000}"
BURN_PROB="${17:-0.3}"
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
LOAD_SUBSAMPLES="${34:-false}"
SUBSAMPLING_CONFIG="${35:-}"
SPLIT_ID="${36:-0}"
RECAL_MOMENTUM="${37:-1.0}"
SAVE_MODEL="${SAVE_MODEL:-false}"
SAVE_MODEL_EVERY="${SAVE_MODEL_EVERY:-0}"

cd "$(dirname "$0")/../../bigg"

echo "=== BiGG Subsampled Training ==="
echo "Dataset:         $DATASET"
echo "Block size:      $BLKSIZE"
echo "Batch size:      $BSIZE"
echo "Epochs:          $EPOCHS"
echo "Learning rate:   $LR"
echo "Embed dim:       $EMBED_DIM"
echo "Noise std:       $NOISE_STD"
echo "SS max prob:     $SS_MAX_PROB"
echo "SS start epoch:  $SS_START_EPOCH"
echo "BFS preprocess:  $BFS_PREPROCESS"
echo "Normalize:       $NORMALIZE"
echo "Loss weights:    $LOSS_WEIGHTS"
echo "Hetero feat:     $HETERO_FEAT"
echo "Mask test labels: $MASK_TEST_LABELS"
echo "Logvar floor:    $LOGVAR_FLOOR"
echo "Subsample size:  $SUBSAMPLE_SIZE"
echo "Burn prob:       $BURN_PROB"
echo "Num subgraphs:   ${NUM_SUBGRAPHS:-auto}"
echo "Binary feat:     $BINARY_FEAT"
echo "VAE feat:        $VAE_FEAT"
echo "VAE dim:         $VAE_DIM"
echo "KL weight:       $KL_WEIGHT"
echo "Cat feat:        $CAT_FEAT"
echo "N bins:          $N_BINS"
echo "Bin sigma:       ${BIN_SIGMA:-auto}"
echo "MDN feat:        $MDN_FEAT"
echo "MDN components:  $MDN_COMPONENTS"
echo "MDN lσ floor:    $MDN_LOGSIGMA_FLOOR"
echo "MDN base:        $MDN_BASE"
echo "KL schedule:     $KL_SCHEDULE"
echo "KL anneal eps:   $KL_ANNEAL_EPOCHS"
echo "KL cycle eps:    $KL_CYCLE_EPOCHS"
echo "KL ramp ratio:   $KL_RAMP_RATIO"
echo "Load subsamples: $LOAD_SUBSAMPLES"
echo "Subsampling cfg: ${SUBSAMPLING_CONFIG:-(none)}"
echo "Split id:        $SPLIT_ID"
echo "Recal momentum:  $RECAL_MOMENTUM"
echo "Save model:      $SAVE_MODEL"
echo "Save every:      $SAVE_MODEL_EVERY"
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

SAVE_MODEL_FLAG=""
if [ "$SAVE_MODEL" = "true" ]; then
  SAVE_MODEL_FLAG="--save_model"
fi

# Subsample source: runtime sampling (default) vs loading pre-computed partitions.
if [ "$LOAD_SUBSAMPLES" = "true" ]; then
  if [ -z "$SUBSAMPLING_CONFIG" ]; then
    echo "ERROR: load_subsamples=true requires subsampling_config (positional arg 35)." >&2
    exit 1
  fi
  SUBSAMPLE_FLAGS="--load_subsamples -subsampling_config $SUBSAMPLING_CONFIG -split_id $SPLIT_ID"
  SUBSAMPLE_TAG="loadsub_${SUBSAMPLING_CONFIG}_split${SPLIT_ID}"
else
  SUBSAMPLE_FLAGS="--subsample -subsample_size $SUBSAMPLE_SIZE -burn_prob $BURN_PROB $NUM_SUBGRAPHS_FLAG"
  SUBSAMPLE_TAG="sub${SUBSAMPLE_SIZE}_p${BURN_PROB}"
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
  $SUBSAMPLE_FLAGS \
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
  -recal_momentum "$RECAL_MOMENTUM" \
  $SAVE_MODEL_FLAG \
  -save_model_every "$SAVE_MODEL_EVERY" \
  -save_dir "checkpoints/bigg/${DATASET}_blk${BLKSIZE}_b${BSIZE}_lr${LR}_e${EPOCHS}_noise${NOISE_STD}_ss${SS_MAX_PROB}_norm${NORMALIZE}_bfs${BFS_PREPROCESS}_lw${LOSS_WEIGHTS}_${HETERO_FEAT}_lvf${LOGVAR_FLOOR}_bin${BINARY_FEAT}_vae${VAE_FEAT}_vd${VAE_DIM}_kl${KL_WEIGHT}_cat${CAT_FEAT}_nb${N_BINS}_mdn${MDN_FEAT}_k${MDN_COMPONENTS}_recalM${RECAL_MOMENTUM}_${SUBSAMPLE_TAG}"
