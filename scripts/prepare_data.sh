#!/bin/bash

###------  Parameters ------###
# file path to the measurement YAML
MEASUREMENTS_PATH="./datasets/measurements.yaml"

# file path to the flow YAML
FLOW_PATH="./datasets/flow_synthetic.yaml"

# file path to the target dataset YAML
TARGET_PATH="./datasets/datasets.yaml"

# Destination folder path of the volume
DEST_VOLUME_FOLDER="volraft_datasets/volume_960x1280x1280"

# Destination folder path of the patch
DEST_PATCH_FOLDER="volraft_datasets/patch_60x80x80"

# output folder
OUTPUT_FOLDER="./results"

# label
JOBID='prepare_data'

# seed set as 0 for random seed
SEED=0

# patch size
PATCH_SIZE=(1 60 80 80)

# patch stride
PATCH_STRIDE=(1 30 40 40)

###------  Tasks ------###
python prepare_data.py \
    $MEASUREMENTS_PATH \
    $FLOW_PATH \
    $TARGET_PATH \
    $DEST_VOLUME_FOLDER \
    $DEST_PATCH_FOLDER \
    --jobid $JOBID \
    --output_folder $OUTPUT_FOLDER \
    --seed $SEED \
    --patch_size "${PATCH_SIZE[@]}" \
    --patch_stride "${PATCH_STRIDE[@]}" \
    --cpu \
    --debug
