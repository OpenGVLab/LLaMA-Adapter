#!/usr/bin/bash

RESUME_PATH="$1"
OUTPUT_DIR="$2"

python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=8 --use_env \
 main_finetune.py --batch_size 4 \
 --epochs 4 --warmup_epochs 1 --blr 10e-4 --weight_decay 0.02 \
 --output_dir "$OUTPUT_DIR" \
 --resume "$RESUME_PATH" \
 --llama_path /data1/llma