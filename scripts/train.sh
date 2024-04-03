#!/bin/bash

###------  Parameters ------###
# folder path to dataset
DATASET_PATH="./datasets/datasets_test.yaml"

# output folder
OUTPUT_FOLDER="./results"

# Configuration file
CONFIG='./models/config/volraft_config_120.yaml'

# label
JOBID='volraft_config_120'

###------  Tasks ------###
python train.py \
    $DATASET_PATH \
    --config $CONFIG \
    --jobid $JOBID \
    --output_folder $OUTPUT_FOLDER \
    --debug
