torchrun --nproc_per_node 1 example.py \
--ckpt_dir /data1/llma/7B \
--tokenizer_path /data1/llma/tokenizer.model \
--adapter_path adapter_prefix10_prompt_layer30.pth

