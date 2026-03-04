#!/bin/bash

set -e

ENV_NAME="bigg"
PYTHON_VERSION="3.9"

conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Initialize conda in the script so 'conda activate' works
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

conda install -y -c conda-forge gcc_linux-64=15.2.0 gxx_linux-64=15.2.0

conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install -y numpy scipy networkx tqdm -c conda-forge

pip install pyemd
