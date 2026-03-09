#!/bin/bash

set -e
set -x

# --- Configuration ---
DATA_DIR="../datasets/original"
CGT_DIR="../CGT"
PYTHON_SCRIPT="train.py"

DATASETS=("reddit")
data_length=${#DATASETS[@]}

for ((i=0;i<$data_length;i++))
do
    python $CGT_DIR/$PYTHON_SCRIPT --dataset "${DATASETS[$i]}" --data_dir "$DATA_DIR"
done
