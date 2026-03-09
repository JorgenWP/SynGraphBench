#!/bin/bash

set -e
set -x

ENV_NAME="GADBench"

source "$(conda info --base)/etc/profile.d/conda.sh"

# Set up the conda environment
conda create -n $ENV_NAME python=3.10
conda activate $ENV_NAME

conda install "numpy<2"
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c dglteam/label/cu117 dgl=1.1.2
conda install "mkl<2024.1" "intel-openmp<2024.1"

pip install xgboost pyod scikit-learn sympy pandas catboost bidict openpyxl

echo "GADBench environment setup complete. To activate, run: conda activate $ENV_NAME"
