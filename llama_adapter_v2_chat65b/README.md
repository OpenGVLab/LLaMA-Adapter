# LLaMA-Adapter-V2-65B chat demo

## News

* [Apr 28, 2023] Initial release.

## Hardware requirements

The current implementation stores the model weights in FP16 and key-value caches of up to 2048 tokens all on GPUs, which add up to about 160GB of GPU memory requirement. Unlike the [original LLaMA implementation](https://github.com/facebookresearch/llama), the demo does not require an equal number of GPUs and the checkpoint splits: You can now choose to run on fewer GPUs with larger memory as long as the total memory is >= 160GB and #heads is divisible by #GPUs (e.g., 2 * A100-80GB, 4 * A100-40GB or 8 * V100-32GB are all supported).

As tensor parallelism is used, high speed GPU interconnect (e.g., NVLink) is recommended for better GPU utilization. Splitting the model beyond one node is thus not recommended.

We will support quantization in the near future, which will further reduce the memory usage by 2-4x and thus make it runable on a single GPU. Stay tuned for the next release!

## Environment

We provide the conda environment in `environment.yml`. Use the following command to install

```
conda env create -f environment.yml
```

## Usage

* Create an environment and install all dependencies as specified in the [Environment](#Environment) section.

* Obtain the LLaMA backbone weights using [this form](https://forms.gle/jk851eBVbX1m5TAv5). Please note that checkpoints from unofficial sources (e.g., BitTorrent) may contain malicious code and should be used with care. Organize the downloaded file in the following structure
  ```
  /path/to/llama_model_weights
  ├── 13B
  │   ├── checklist.chk
  │   ├── consolidated.00.pth
  │   ├── consolidated.01.pth
  │   └── params.json
  ├── 30B
  │   ├── checklist.chk
  │   ├── consolidated.00.pth
  │   ├── consolidated.01.pth
  │   ├── consolidated.02.pth
  │   ├── consolidated.03.pth
  │   └── params.json
  ├── 65B
  │   ├── checklist.chk
  │   ├── consolidated.00.pth
  │   ├── consolidated.01.pth
  │   ├── consolidated.02.pth
  │   ├── consolidated.03.pth
  │   ├── consolidated.04.pth
  │   ├── consolidated.05.pth
  │   ├── consolidated.06.pth
  │   ├── consolidated.07.pth
  │   └── params.json
  ├── 7B
  │   ├── checklist.chk
  │   ├── consolidated.00.pth
  │   └── params.json
  └── tokenizer.model
  ```

  Substitute the root directory `/path/to/llama_model_weights` with your own choice and pass it to the `--llama_model_path` argument in the subsequent steps. 

* Obtain the tuned weights from [our model zoo](checkpoints/model_zoo.md). Put the tuned weights in the `checkpoints/` folder.

* Run the demo

  * Quick run using recommended settings: Set platform-specific information and run with only one click
    * On slurm clusters, run [scripts/srun_chat_llama65b_bias_scale_norm.sh](scripts/srun_chat_llama65b_bias_scale_norm.sh). This runs the `Llama65B_bias_scale_norm_tuning` model. The options need to be set in the script are: slurm partition name (partition=...), #GPUs (n_gpus=...) and LLaMA backbone weights path (llama_model_path=...).
    * On a single node with multiple GPUs, run [scripts/torchrun_chat_llama65b_bias_scale_norm.sh](scripts/torchrun_chat_llama65b_bias_scale_norm.sh). This runs the `Llama65B_bias_scale_norm_tuning` model. The options need to be set in the script are: #GPUs (n_gpus=...) and LLaMA backbone weights path (llama_model_path=...).
  * Advanced run
    * `chat_demo.py` is the entry point of the demo.
    * The demo is supposed to be launched using distributed bootstrappers to use multiple GPUs. The supported bootstrappers are `slurm` and `torchrun`. For the usage with the bootstrappers see the example scripts ([scripts/srun_chat_llama65b_bias_scale_norm.sh](scripts/srun_chat_llama65b_bias_scale_norm.sh) for slurm, [scripts/torchrun_chat_llama65b_bias_scale_norm.sh](scripts/torchrun_chat_llama65b_bias_scale_norm.sh) for torchrun).
    * Several options can be set using command line arguments. The required arguments are `--model_name` for architecture selection, and `--llama_model_path` for finding the LLaMA backbone weights. `--model_path` is used to specify the checkpoint of the adapted part of the model weights; it can also be left blank which will leave the adapted parts at the initialization state / the same as the backbone (usually meaning using the unadapted model). There are also arguments for the generation hyper-parameters (e.g., `--temperature`, `--top_p`, `--max_new_tokens`). For a detailed explanation, use the help message (`python chat_demo.py -h`).

* Interact with the demo. The demo uses the terminal to take users' inputs and show models' responses. After the `Human:` prompt shows up, type in your sentences and hit the Enter key. Wait some time (a few seconds to a few minutes depending on your hardware) for the response. After the response shows up you may make another input.

* Use Ctrl+C to exit the demo at any time.

## Known issues

* Some users may experience the error `RuntimeError: Expected is_sm80 to be true, but got false.` (Mostly sm_86 GPU users, including A6000, A5000 and 3090). This is because we changed the attention module to use `torch.nn.functional.scaled_dot_product_attention` if it exists, but a [dispatch logic error](https://github.com/pytorch/pytorch/issues/94883) in PyTorch = 2.0.0 causes failure on some GPU architectures. The affected users can upgrade to PyTorch >= 2.1.0 or the nightly build, in which the bug is fixed.

## Disclaimer

The released model weights are trained for research purposes only. The model may generate improper responses, including but not limited to false or biased information, offensive information to certain groups, or information that is not compliant with local laws and regulations in certain countries or regions, as the pretrained model weights and the finetuning data are collected from the Internet without extensive manual intervention. The generated contents do not necessarily reflect our views or positions, and should be used at the end-user's own risk.

## Acknowledgement

This repo benefits from [LLaMA](https://github.com/facebookresearch/llama), ShareGPT and [FastChat](https://github.com/lm-sys/FastChat). Thanks for their wonderful works.
