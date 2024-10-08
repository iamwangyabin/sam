#!/bin/bash

base_dir="./output_dir"
mkdir -p "${base_dir}"

export CUDA_VISIBLE_DEVICES=0

# Define common parameters
common_params=(
    "--model" "SAMPrompting"
    "--world_size" "1"
    "--batch_size" "1"
    "--data_path" "/home/jwang/ybwork/dataset_generator/edit/00115/train.json"
    "--epochs" "200"
    "--lr" "1e-4"
    "--image_size" "512"
    "--if_resizing"
    "--min_lr" "5e-7"
    "--weight_decay" "0.05"
    "--edge_mask_width" "7"
    "--test_data_path" "/home/jwang/ybwork/dataset_generator/edit/00115/train.json"
    "--warmup_epochs" "2"
    "--output_dir" "${base_dir}/"
    "--log_dir" "${base_dir}/"
    "--accum_iter" "8"
    "--seed" "42"
    "--test_period" "4"
)

# Run the training script
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    ./train.py \
    "${common_params[@]}" \
    2> "${base_dir}/error.log" \
    1> "${base_dir}/logs.log"