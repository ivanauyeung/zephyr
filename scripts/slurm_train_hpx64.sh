#!/bin/bash

RUN_CMD="python -u scripts/train.py num_workers=8 port=11222 learning_rate=1e-3 batch_size=32 experiment_name=hpx64to64_04-04-24_01 model=hpx_unet model.encoder.n_channels=[64,128,256] model.decoder.n_channels=[256,128,64] trainer.max_epochs=250 data=era5_hpx64_9var_6h_24h_pd data.prebuilt_dataset=True data.module.drop_last=True trainer/lr_scheduler=plateau trainer/optimizer=adam model.enable_healpixpad=True"

# Run configuration
NUM_GPU=1
NUM_CPU=16
GPU_NAME=A100
DEVICE_NUMBERS="0"

# Command to run model on 
srun -u --ntasks=${NUM_GPU} \
     --ntasks-per-node=${NUM_GPU} \
     --gres=gpu:${GPU_NAME}:${NUM_GPU} \
     --cpu_bind=sockets \
     -c $(( ${NUM_CPU} / ${NUM_GPU} )) \
     bash -c "
     export WORLD_RANK=\${SLURM_PROCID}
     export HDF5_USE_FILE_LOCKING=False
     export CUDA_VISIBLE_DEVICES=${DEVICE_NUMBERS}
     export HYDRA_FULL_ERROR=1 
     ${RUN_CMD}"
