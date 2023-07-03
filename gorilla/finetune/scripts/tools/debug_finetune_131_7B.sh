load_dir="$1"
save_dir="$2"

exp_name=tool/get_consolidated_ckpt
mkdir -p output/"$exp_name"
mkdir -p "$save_dir"

python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=8 --use_env \
tools/get_consolidated_ckpt.py \
--llama_type llama \
--llama_config /data1/llma/7B/params.json \
--data_parallel sdp --model_parallel_size 1 \
--load_dir "$load_dir" --save_dir "$save_dir" \
2>&1 | tee -a output/"$exp_name"/output.log
