#!/usr/bin/env sh

# total number of GPUs to use.
n_gpus=
# directory containing LLaMA backbone weights downloaded from facebook.
llama_model_path=


torchrun --nproc_per_node ${n_gpus} chat_demo.py \
  --model_name Llama65B_bias_scale_norm_tuning \
  --model_path checkpoints/chat_llama65b_bias_scale_norm_tuning_v1.pth \
  --llama_model_path "${llama_model_path}" 

