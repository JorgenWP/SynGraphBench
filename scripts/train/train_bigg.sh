#!/bin/bash

# run_pipeline.sh
cd "$(dirname "$0")/../../bigg"

DATASET="original/tolokers"
BLKSIZE=-1
BSIZE=1
EPOCHS=50

python -m bigg.extension.pipeline \
  -data_dir $DATASET \
  -model_type independent \
  -gpu 0 \
  -embed_dim 256 \
  -bits_compress 128 \
  -learning_rate 0.001 \
  -num_epochs $EPOCHS \
  -batch_size $BSIZE \
  -blksize $BLKSIZE \
  -seed 34