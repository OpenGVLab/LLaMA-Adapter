#!/usr/bin/env sh

# slurm partition name.
partition=
# total number of GPUs to use.
n_gpus=
# directory containing LLaMA backbone weights downloaded from facebook.
llama_model_path=


srun -p "${partition}" --gres=gpu:${n_gpus} -n ${n_gpus} --ntasks-per-node ${n_gpus} --unbuffered \
  python -u chat_demo.py \
    --model_name Llama65B_bias_scale_norm_tuning \
    --model_path checkpoints/chat_llama65b_bias_scale_norm_tuning_v1.pth \
    --llama_model_path "${llama_model_path}" 

