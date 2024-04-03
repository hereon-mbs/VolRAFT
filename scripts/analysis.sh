#!/bin/bash

###------  Parameters ------###

# slice to analyze
ANALYSIS_PATH="analysis.yaml"

# ########## --- Dataset --- #############
# dataset name
DATASET_NAME="2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004"

# whether this dataset include ground truth flow
HAS_GT_FLOW=1

# folder path to the original watershed folder
MASK_WATERSHED_PATH="volraft_measurements/syn0154_103L_Mg5Gd_4w_000"

# folder path to dataset
DATASET_PATH="volraft_datasets/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004"

# folder path to result of inference
RESULT_PATH="<path/to/infer/result>" # TODO: change to your own path

# folder path to mbsoptflow-vanilla
MBSVAN_PATH="volraft_datasets/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004/mbsoptflow"

# folder path to mbsoptflow-optimal
MBSOPT_PATH="volraft_datasets/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs004/mbsoptflow_opt18"

# label
JOBID="volraft_config_120"

# output folder
OUTPUT_FOLDER="./results"

# seed 42 is very common
SEED=42

# Maximum range of flow field 
MAX_FLOW=24.0

###------  Tasks ------###
python analysis.py \
    $ANALYSIS_PATH \
    $MASK_WATERSHED_PATH \
    $DATASET_PATH \
    $RESULT_PATH \
    $MBSVAN_PATH \
    $MBSOPT_PATH \
    $DATASET_NAME \
    --has_gt_flow $HAS_GT_FLOW \
    --jobid $JOBID \
    --output_folder $OUTPUT_FOLDER \
    --seed $SEED \
    --max_flow $MAX_FLOW \
    --debug
