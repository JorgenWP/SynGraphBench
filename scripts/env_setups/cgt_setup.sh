#!/bin/bash

set -e
set -x

ENV_NAME="cgt_test"
CGT_DIR="../CGT"

source "$(conda info --base)/etc/profile.d/conda.sh"

# Create and activate the CGT environment
conda env create -f $CGT_DIR/cgt_env.yml -n $ENV_NAME
conda activate $ENV_NAME

# Install CGT dependencies
source $CGT_DIR/install_gpu.sh

echo "CGT environment setup complete. To activate, run: conda activate $ENV_NAME"
