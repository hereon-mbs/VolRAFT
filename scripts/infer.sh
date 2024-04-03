#!/bin/bash

###------  Parameters ------###

# folder path to dataset
DATASET_PATH="volraft_datasets/volume_960x1280x1280/2021_11008741_syn0155_5R_Ti_4w_000_fs402"

# dataset name
DATASET_NAME="2021_11008741_syn0155_5R_Ti_4w_000_fs402"

# folder path to checkpoint
CHECKPOINT_PATH="./checkpoints/volraft_config_120/checkpoint_20240119_184617_980292"

# label
JOBID="volraft_config_120"

# output folder
OUTPUT_FOLDER="./results"

# patch size
FULL_MARGIN=(20 40 40)

# Batch size for inference - volraft_config_120
INFER_BATCH_SIZE=300 # for 32GB

# Number of overlap
NUM_OVERLAPS=7

###------  Tasks ------###
python infer.py \
    $DATASET_PATH \
    $CHECKPOINT_PATH \
    $DATASET_NAME \
    --jobid $JOBID \
    --output_folder $OUTPUT_FOLDER \
    --full_margin "${FULL_MARGIN[@]}" \
    --infer_batch_size $INFER_BATCH_SIZE \
    --num_overlaps $NUM_OVERLAPS \
    --debug
