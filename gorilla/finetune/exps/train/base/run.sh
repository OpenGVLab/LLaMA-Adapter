exp_name="train/base_wamrup2000_lr_decay_1800000_batch16_32gpu_accmulation_4_lr_0005_adam_095_09_weight_decay_0_dot_1_clip_2"
mkdir -p output/"$exp_name"

srun -p alpha_vl --gres=gpu:8 --cpus-per-task 16 -n32 \
--ntasks-per-node=8 --quotatype=reserved python -u main_pretrain.py --batch_size 16 \
--llama_type llama --weight_decay 0.1 --output_dir output/"$exp_name" \
--accum_iter 4 --warmup_iters 2000 --lr_decay_iters 1800000 --lr 0.0005 --min_lr 0.00005 --clip_grad 2 \
2>&1 | tee -a output/"$exp_name"/output.log
