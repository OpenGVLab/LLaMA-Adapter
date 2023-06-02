#!/usr/bin/bash

python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=8 --use_env \
 main_finetune.py --batch_size 4 \
 --epochs 4 --warmup_epochs 1 --blr 10e-4 --weight_decay 0.02 \
 --output_dir ./output/finetune \
 --resume ckpts/llama_7B_clip_ViTL_prompt32layer_3layerfusion_4096dim_32_32gpu_all_data_epoch3_instruction_prompt_every_gate_1e_4/checkpoint-13.pth \
 --llama_path /data1/llma &>finetune_output.log &