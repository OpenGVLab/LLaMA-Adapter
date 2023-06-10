torchrun --nproc_per_node 1 example.py \
--ckpt_dir /data1/llma/7B \
--tokenizer_path /data1/llma/tokenizer.model \
--adapter_path llama_adapter_len10_layer30_release.pth \
--quantizer False

