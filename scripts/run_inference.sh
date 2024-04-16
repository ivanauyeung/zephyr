#!/bin/bash

seed=0
num_workers=1
port=29465
checkpoint_name=Null

# Scheduling Parameters 
NUM_GPU=1
NUM_CPU=16
GPU_NAME=A100
DEVICE_NUMBERS="3"
NUMEXPR_MAX_THREADS=128
EXP_NAME="hpx64_gru_1x1${seed}"
JOB_NAME="inference_${EXP_NAME}"
RUN_DIR="/home/disk/brume/adod/zephyr"
SOURCE_FILE="/home/disk/brume/adod/.bashrc"
CONDA_ENV="zephyr-1.1"

###############################################################################
##################   BOILER PLATE CODE TO QUEUE BATCH JOB    ##################
###############################################################################

# Change directory and activate environment
cd ${RUN_DIR}
source ${SOURCE_FILE}
source activate ${CONDA_ENV}

# Compile RUN_CMD
RUN_CMD="python -u scripts/create_208_forecasts.py"

# Run model
srun -u --ntasks=${NUM_GPU} \
    "--job-name=${JOB_NAME}" \
    "--output=${OUTPUT_FILE}" \
    "--error=${OUTPUT_FILE}" \
    --ntasks-per-node=${NUM_GPU} \
    --gres=gpu:${GPU_NAME}:${NUM_GPU} \
    --cpu_bind=sockets \
    -c $(( ${NUM_CPU} / ${NUM_GPU} )) \
    bash -c "
    export PYTHONPATH=${RUN_DIR}
    export WORLD_RANK=\${SLURM_PROCID}
    export HDF5_USE_FILE_LOCKING=False
    export CUDA_VISIBLE_DEVICES=${DEVICE_NUMBERS}
    export HYDRA_FULL_ERROR=1 
    ${RUN_CMD}" &