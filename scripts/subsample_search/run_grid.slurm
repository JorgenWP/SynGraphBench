#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="SUBSAMPLE_RUN_tolokers"
#SBATCH --time=02:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1

#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=eirimhau@stud.ntnu.no
#SBATCH --output=output/subsample/search/tolokers.txt
#SBATCH --mail-type=FAIL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

DATASETS=tolokers               # e.g. yelp | tfinance 
EXCLUDE_METHODS=        # e.g. metis 
METHODS=                # optional positive filter (mutually exclusive with --params_tags)
PARAMS_TAGS=            # optional explicit params_tag filter (overrides METHODS)
SPLITS=                 # e.g. "0 1 2 3 4" (default: 0 1 2 3 4)
OUT_DIR=                # e.g. experiments/subsample_search/artifacts/grid_weibo
PLAN_CSV=               # defaults to <OUT_DIR>/plan.csv; pass empty string ("") to force DEFAULT_K
EPOCHS=200
PATIENCE=50

if [ -z "$DATASETS" ]; then
    echo "ERROR: DATASETS unset" >&2
    exit 1
fi

module purge
module load Anaconda3/2024.02-1

EXTRA=""
[ -n "$EXCLUDE_METHODS" ] && EXTRA="$EXTRA --exclude_methods $EXCLUDE_METHODS"
[ -n "$METHODS" ]        && EXTRA="$EXTRA --methods $METHODS"
[ -n "$PARAMS_TAGS" ]    && EXTRA="$EXTRA --params_tags $PARAMS_TAGS"
[ -n "$SPLITS" ]         && EXTRA="$EXTRA --splits $SPLITS"
[ -n "$OUT_DIR" ]        && EXTRA="$EXTRA --out_dir $OUT_DIR"
[ -n "$PLAN_CSV" ]       && EXTRA="$EXTRA --plan_csv $PLAN_CSV"

conda run --no-capture-output -n bigg python -m experiments.subsample_search.run_grid \
    --datasets "$DATASETS" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    $EXTRA
