#!/bin/bash

# 1. Install PyTorch 2.3.1 (Stable + CUDA 12.1)
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 2. Install DGL (Deep Graph Library)
# We point to the CUDA 12.1 repository.
conda install -c dglteam/label/th23_cu121 dgl

# 3. Install PyTorch Geometric (PyG)
# Modern PyG installs cleanly with pip.
pip install torch_geometric

# 4. Install PyG Optional Dependencies (Scatter/Sparse)
# These must match the PyTorch version (2.5.0) and CUDA version (12.4).
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

pip install numpy scipy scikit-learn pandas networkx tqdm ipdb wandb ogb k-means-constrained ortools jinja2 tensorboard
