#!/usr/bin/bash

LLAMA_PATH="$1"
PRETRAINED_PATH="$2" # path to pre-trained checkpoint
CONFIG="$3"
OUTPUT_DIR="$4"

mkdir -p "$OUTPUT_DIR"

python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=8 --use_env \
 main_finetune.py --data_config "$CONFIG" --batch_size 4 \
 --epochs 4 --warmup_epochs 1 --blr 10e-4 --weight_decay 0.02 \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR" \
 --pretrained_path "$PRETRAINED_PATH" \
 &>> "$OUTPUT_DIR"/output.log &