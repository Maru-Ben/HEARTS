#!/bin/bash

# Base parameters for all runs
BASE_PARAMS="--contrast_bipartite_edge True \
    --accelerator gpu \
    --devices 1 \
    --replace_sampler_ddp False \
    --accumulate_grad_batches 4 \
    --batch_size 16 \
    --save_every_n_epochs 1 \
    --save_top_k 1"

# Checkpoint path for finetuning
BASE_CHECKPOINT="./checkpoints/hytrel/contrast_pretrained/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt"

echo "Starting training sequence..."
echo "=========================================="

# Santos Dataset
echo "Processing Santos Dataset..."

echo "1/2: Training Santos from scratch..."
CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/hytrel/run_pretrain.py \
    --data_path './data/santos/' \
    --gradient_clip_val 2.0 \
    --base_learning_rate 5e-5 \
    --max_epoch 50 \
    --checkpoint_dir 'checkpoints/hytrel/santos_contrast_scratch' \
    --contrast_bipartite_edge True \
    --accelerator gpu \
    --devices 1 \
    --replace_sampler_ddp False \
    --accumulate_grad_batches 4 \
    --batch_size 16 \
    --save_every_n_epochs 1 \
    --save_top_k 1

# echo "2/2: Finetuning Santos (classic)..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/hytrel/run_pretrain.py \
#     --data_path './data/santos/' \
#     --gradient_clip_val 2.0 \
#     --base_learning_rate 5e-5 \
#     --max_epoch 10 \
#     --checkpoint_path "$BASE_CHECKPOINT" \
#     --checkpoint_dir 'checkpoints/hytrel/santos_contrast_finetuned' \
#     $BASE_PARAMS


# echo "Santos Dataset Complete"
# echo "=========================================="

# # TUS Dataset
# echo "Processing TUS Dataset..."

# echo "1/2: Training TUS from scratch..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/hytrel/run_pretrain.py \
#     --data_path './data/tus/' \
#     --gradient_clip_val 2.0 \
#     --base_learning_rate 5e-5 \
#     --max_epoch 50 \
#     --checkpoint_dir 'checkpoints/hytrel/tus_contrast_scratch' \
#     $BASE_PARAMS

# echo "2/2: Finetuning TUS (classic)..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/hytrel/run_pretrain.py \
#     --data_path './data/tus/' \
#     --gradient_clip_val 2.0 \
#     --base_learning_rate 5e-5 \
#     --max_epoch 10 \
#     --checkpoint_path "$BASE_CHECKPOINT" \
#     --checkpoint_dir 'checkpoints/hytrel/tus_contrast_finetuned' \
#     $BASE_PARAMS


# echo "TUS Dataset Complete"
# echo "=========================================="


# # TUS LARGE Dataset
# echo "Processing TUS LARGE Dataset..."

# echo "1/2: Training TUS LARGE from scratch..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/hytrel/run_pretrain.py \
#     --data_path './data/tusLarge/' \
#     --gradient_clip_val 2.0 \
#     --base_learning_rate 5e-5 \
#     --max_epoch 10 \
#     --checkpoint_dir 'checkpoints/hytrel/tusLarge_contrast_scratch' \
#     $BASE_PARAMS

# echo "2/2: Finetuning TUS LARGE (classic)..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/hytrel/run_pretrain.py \
#     --data_path './data/tusLarge/' \
#     --gradient_clip_val 2.0 \
#     --base_learning_rate 5e-5 \
#     --max_epoch 10 \
#     --checkpoint_path "$BASE_CHECKPOINT" \
#     --checkpoint_dir 'checkpoints/hytrel/tusLarge_contrast_finetuned' \
#     $BASE_PARAMS


# echo "TUS LARGE Dataset Complete"
# echo "=========================================="

# # Pylon Dataset
# echo "Processing Pylon Dataset..."

# echo "1/2: Training Pylon from scratch..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/hytrel/run_pretrain.py \
#     --data_path './data/pylon/' \
#     --gradient_clip_val 2.0 \
#     --base_learning_rate 5e-5 \
#     --max_epoch 10 \
#     --checkpoint_dir 'checkpoints/hytrel/pylon_contrast_scratch' \
#     $BASE_PARAMS

# echo "2/2: Finetuning Pylon (classic)..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/hytrel/run_pretrain.py \
#     --data_path './data/pylon/' \
#     --gradient_clip_val 2.0 \
#     --base_learning_rate 5e-5 \
#     --max_epoch 10 \
#     --checkpoint_path "$BASE_CHECKPOINT" \
#     --checkpoint_dir 'checkpoints/hytrel/pylon_contrast_finetuned' \
#     $BASE_PARAMS



# echo "Pylon Dataset Complete"
# echo "=========================================="

# # echo "3/3: Finetuning Santos with LoRA..."
# # CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
# #     --data_path './data/santos/' \
# #     --gradient_clip_val 1.0 \
# #     --base_learning_rate 1e-5 \
# #     --checkpoint_path "$BASE_CHECKPOINT" \
# #     --checkpoint_dir './checkpoints/santos_contrast_lora' \
# #     --use_lora True \
# #     --lora_r 8 \
# #     --lora_alpha 16 \
# #     --lora_dropout 0.1 \
# #     $BASE_PARAMS