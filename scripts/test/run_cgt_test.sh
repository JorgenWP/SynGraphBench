#!/bin/bash

set -e
set -x

DATA_DIR="CGT/data/"
CGT_DIR="CGT/"
PYTHON_SCRIPT="test.py"

DATASETS=("cora" "citeseer")
data_length=${#DATASETS[@]}

GNN_MODELS=("gcn" "sgc" "gin")
TASK="aggregation"

# Navigate to project root
cd "$(dirname "$0")/../.."

# Experiment: effects of noise to aggregation strategies
NOISES=(0 2 4)
noise_length=${#NOISES[@]}

for ((i=0;i<$data_length;i++))
do
    for ((j=0;j<$noise_length;j++))
    do
        python $CGT_DIR/$PYTHON_SCRIPT \
            --dataset "${DATASETS[$i]}" \
            --noise_num ${NOISES[$j]} \
            --data_dir "$DATA_DIR" \
            --task_name "$TASK" \
            -n "${GNN_MODELS[@]}"
    done
done