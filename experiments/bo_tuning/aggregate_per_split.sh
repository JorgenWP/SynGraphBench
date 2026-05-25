#!/bin/bash
# Aggregate per-split Optuna studies and assemble the thesis-ready rollup.
#
# Step 1: rebuild summary.csv + best_params.json from each split's study.db
#         (writes back into experiments/bo_tuning/{dataset}/per_split/split{N}/).
# Step 2: copy per-split artifacts + derive cross-split rollups under
#         experiments/results/BO_Tuning/{dataset}/per_split/.
#
# CPU-only, no GPU needed. Requires the `bigg` conda env (optuna + pandas +
# matplotlib).
#
# Usage:
#   bash experiments/bo_tuning/aggregate_per_split.sh <dataset> [study_version]
# Example:
#   bash experiments/bo_tuning/aggregate_per_split.sh weibo v1

set -euo pipefail

DATASET="${1:?usage: aggregate_per_split.sh <dataset> [study_version]}"
STUDY_VERSION="${2:-v1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

for s in 0 1 2 3 4; do
  DB="experiments/bo_tuning/${DATASET}/per_split/split${s}/study.db"
  if [[ ! -f "$DB" ]]; then
    echo "skip split${s}: no study.db at $DB"
    continue
  fi
  python -m experiments.bo_tuning.aggregate \
      --study_db "$DB" \
      --study_name "bo_bigg_${DATASET}_per_split_${STUDY_VERSION}_split${s}" \
      --out_dir "experiments/bo_tuning/${DATASET}/per_split/split${s}"
done

python -m experiments.bo_tuning.build_results \
    --dataset "$DATASET" \
    --mode per_split
