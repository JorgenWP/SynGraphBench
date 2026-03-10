#!/bin/bash

set -e
set -x

# --- Configuration ---
DATASETS=("reddit")
data_length=${#DATASETS[@]}

# Navigate to project root
cd "$(dirname "$0")/../.."

for ((i=0;i<$data_length;i++))
do
    python CGT/train.py \
        --dataset "${DATASETS[$i]}" \
        --data_dir "datasets/original"
done
