data_parallel="$1"
mp="$2"
llama_path="/data1/llma/7B"

exp_name=finetune_"$data_parallel"_mp"$mp"_bsz2_accum_4_gpu8_lr_00002_warmup1_epoch3_max_len512_gorilla_torchhub_consolidate
mkdir -p output/"$exp_name"

python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=8 --use_env main_finetune.py \
--llama_type llama --weight_decay 0.1 --output_dir output/"$exp_name" \
--accum_iter 4 --batch_size 2 --warmup_epochs 1 --epochs 3 --lr 0.00002 --min_lr 0.000005 --clip_grad 2 \
--llama_config "$llama_path"/params.json \
--data_parallel "$data_parallel" --model_parallel_size "$mp" \
--max_words 512 --data_config configs/finetune/gorilla_th.yaml --llama_tokenizer_path "$llama_path"/tokenizer.model \
--pretrained_path "$llama_path" --pretrained_type meta_ori --checkpointing \
--save_consolidated 2>&1 | tee -a output/"$exp_name"/output.log
