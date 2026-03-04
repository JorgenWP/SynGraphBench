#!/bin/bash

set -e

ENV_NAME="bigg"
PYTHON_VERSION="3.9"

conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Initialize conda in the script so 'conda activate' works
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

conda install -y -c conda-forge gcc_linux-64 gxx_linux-64

conda install -y pytorch=2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install -y numpy scipy networkx tqdm -c conda-forge

pip install pyemd

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

TREE_CLIB_DIR="../bigg/bigg/model/tree_clib"

if [ ! -d "$TREE_CLIB_DIR/build" ]; then
    echo "Build directory not found. Compiling tree_clib C++ extension..."
    
    # Navigate to the clib directory
    cd "$TREE_CLIB_DIR"

    # Automate the Makefile fix: replace the old GPU architectures with modern ones
    sed -i 's/CUDA_ARCH := -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70/CUDA_ARCH := -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89/g' Makefile

    # Compile using the CUDA_HOME path that we verified works on your system
    make clean && make CUDA_HOME=/usr

    # Return to the original directory
    cd -
else
    echo "tree_clib build directory already exists. Skipping compilation."
fi
