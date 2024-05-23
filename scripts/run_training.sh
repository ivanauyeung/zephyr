#!/bin/bash

# RUN_CMD="python -u scripts/train.py num_workers=8 
# port=29450 
# learning_rate=2e-4 
# batch_size=8 
# experiment_name=a_hpx64_unet_136-68-34_cnxt_skip_dil_gru_6h_300 
# model=hpx_rec_unet 
# model/modules/blocks@model.encoder.conv_block=conv_next_block 
# model/modules/blocks@model.decoder.conv_block=conv_next_block 
# model/modules/blocks@model.decoder.output_layer=basic_conv_block 
# model.encoder.n_channels=[136,68,34] 
# model.decoder.n_channels=[34,68,136] 
# trainer.max_epochs=300 
# data=era5_hpx64_7var_6h_24h 
# data.dst_directory=/home/disk/quicksilver2/karlbam/Data/DLWP/HPX64 
# data.prefix=era5_0.25deg_3h_HPX64_1979-2021_ 
# data.prebuilt_dataset=True 
# data.module.drop_last=True 
# trainer/lr_scheduler=cosine 
# trainer/optimizer=adam 
# model.enable_healpixpad=True"

# # Run configuration
# NUM_GPU=1
# NUM_CPU=16
# GPU_NAME=A100
# DEVICE_NUMBERS="3"

# # Command to run model on 
# srun -u --ntasks=${NUM_GPU} \
#      --ntasks-per-node=${NUM_GPU} \
#      --gres=gpu:${GPU_NAME}:${NUM_GPU} \
#      --cpu_bind=sockets \
#      -c $(( ${NUM_CPU} / ${NUM_GPU} )) \
#      bash -c "
#      export WORLD_RANK=\${SLURM_PROCID}
#      export HDF5_USE_FILE_LOCKING=False
#      export CUDA_VISIBLE_DEVICES=${DEVICE_NUMBERS}
#      export HYDRA_FULL_ERROR=1 
#      ${RUN_CMD}"

# Model Configuration

# experiment_name=a_hpx64_unet_136-68-34_cnxt_skip_dil_gru_6h_300 
# model=hpx_rec_unet 
# model/modules/blocks@model.encoder.conv_block=conv_next_block 
# model/modules/blocks@model.decoder.conv_block=conv_next_block 
# model/modules/blocks@model.decoder.output_layer=basic_conv_block 
# model.encoder.n_channels=[136,68,34] 
# model.decoder.n_channels=[34,68,136] 
# trainer.max_epochs=300 
# data=era5_hpx64_7var_6h_24h 
# data.dst_directory=/home/disk/quicksilver2/karlbam/Data/DLWP/HPX64 
# data.prefix=era5_0.25deg_3h_HPX64_1979-2021_ 
# data.prebuilt_dataset=True 
# data.module.drop_last=True 
# trainer/lr_scheduler=cosine 
# trainer/optimizer=adam 
# model.enable_healpixpad=True"

seed=0
num_workers=1
port=29465
learning_rate=2e-4
checkpoint_name=Null
batch_size=14
model="hpx_rec_unet"
encoder_conv_block="conv_next_block"
decoder_conv_block="conv_next_block"
output_layer="basic_conv_block"
decoder="rec_unet_dec"
encoder_n_channels="[180,90,90]"
decoder_n_channels="[90,90,180]"
max_epochs=200
data="era5_hpx64_7var_6h_24h"
data_prefix="era5_0.25deg_3h_HPX64_1979-2021_"
prebuilt_dataset=True
drop_last=True
lr_scheduler="cosine"
max_norm=0.25
optimizer="adam"
enable_healpixpad=True

# Scheduling Parameters 
NUM_GPU=1
NUM_CPU=16
GPU_NAME=A100
DEVICE_NUMBERS="2"
NUMEXPR_MAX_THREADS=128
EXP_NAME="hpx64_gru_3x3${seed}"
OUTPUT_DIR="/home/disk/brume/adod/zephyr/outputs/${EXP_NAME}"
OUTPUT_FILE="${OUTPUT_DIR}/output.out"
JOB_NAME="train_${EXP_NAME}"
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

# Create output directory if it does not exist
mkdir -p ${OUTPUT_DIR}

# Compile RUN_CMD
RUN_CMD="python -u scripts/train.py seed=${seed} \
num_workers=${num_workers} port=${port} \
learning_rate=${learning_rate} \
checkpoint_name=${checkpoint_name} \
batch_size=${batch_size} \
experiment_name=${EXP_NAME} model=${model} \
 model/modules/blocks@model.encoder.conv_block=${encoder_conv_block} \
 model/modules/blocks@model.decoder.conv_block=${decoder_conv_block} \
 model/modules/blocks@model.decoder.output_layer=${output_layer} \
  model.encoder.n_channels=${encoder_n_channels} \
  model.decoder.n_channels=${decoder_n_channels} \
  trainer.max_epochs=${max_epochs} \
  data=${data} \
  data.prefix=${data_prefix} \
  data.prebuilt_dataset=${prebuilt_dataset} \
  data.module.drop_last=${drop_last} \
  trainer/lr_scheduler=${lr_scheduler} \
  trainer/optimizer=${optimizer} \
  model.enable_healpixpad=${enable_healpixpad}"

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