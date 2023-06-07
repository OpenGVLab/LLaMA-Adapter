# ImageBind-LLM

## News

* [June 5, 2023] Support [3D point cloud input](https://github.com/ZrrSkywalker/Point-Bind) and image generation output. Release stable-version checkpoint.
* [June 2, 2023] Release fine-tuning code and beta-version checkpoint.
* [May 29, 2023] Initial release.


## Setup

* Setup up a new conda env. Install ImageBind and other necessary packages.
  ```bash
  # create conda env
  conda create -n imagebind_LLM python=3.9 -y
  conda activate imagebind_LLM
  # install ImageBind
  cd ImageBind
  pip install -r requirements.txt
  # install other dependencies
  cd ../
  pip install -r requirements.txt
  ```

* Obtain the LLaMA backbone weights using [this form](https://forms.gle/jk851eBVbX1m5TAv5). Please note that checkpoints from unofficial sources (e.g., BitTorrent) may contain malicious code and should be used with care. Organize the downloaded file in the following structure
  ```
  /path/to/llama_model_weights
  ├── 7B
  │   ├── checklist.chk
  │   ├── consolidated.00.pth
  │   └── params.json
  └── tokenizer.model
  ```
  
* The current stable version of ImageBind-LLM is built upon [Open-Chinese-LLaMA](https://github.com/OpenLMLab/OpenChineseLLaMA) for better multilingual support. The following command downloads a pre-processed delta-version patch and automatically merges it into LLaMA weights:
  ```bash
  python get_chinese_llama.py --llama_dir=/path/to/llama_model_weights
  ```
  After running, the Open-Chinese-LLaMA weights will be recovered in `/path/to/llama_model_weights`:
  ```
  /path/to/llama_model_weights
  ├── 7B
  ├── 7B_chinese
  └── tokenizer.model
  ```
  
* Other dependent resources will be automatically downloaded at runtime. 
## Inference

* Here is a simple script for multi-modal inference with ImageBind-LLM:

  ```python
  import ImageBind.data as data
  import llama
  
  
  llama_dir = "/path/to/LLaMA"
  
  # checkpoint will be automatically downloaded
  model = llama.load("7B", llama_dir, knn=True)
  model.eval()
  
  inputs = {}
  image = data.load_and_transform_vision_data(["examples/girl.jpg"], device='cuda')
  inputs['Image'] = [image, 1]
  audio = data.load_and_transform_audio_data(['examples/girl_bgm.wav'], device='cuda')
  inputs['Audio'] = [audio, 1]
  
  results = model.generate(
      inputs,
      [llama.format_prompt("Guess the girl's mood based on the background music and explain the reason?")],
      max_gen_len=256
  )
  result = results[0].strip()
  print(result)
  ```

* Powered by the amazing [Point-Bind](https://github.com/ZrrSkywalker/Point-Bind) project, ImageBind-LLM can also receive 3D point cloud data.    We provide several point cloud samples in `examples/`.

  ```python
  inputs = {}
  point = data.load_and_transform_point_cloud_data(["examples/airplane.pt"], device='cuda')
  inputs['Point'] = [point, 1]
  
  results = model.generate(
      inputs,
      [llama.format_prompt("Describe the 3D object in detail:")],
      max_gen_len=256
  )
  result = results[0].strip()
  print(result)
  ```

## Demo
**Highly recommend trying out our web demo, which incorporates all features currently supported by ImageBind-LLM**



* Run the following command to host the demo locally:
  ``` bash
  python gradio_app.py --llama_dir /path/to/llama_model_weights
  ```

* The official online demo will come very soon

## Pre-traininig & Fine-tuning
See [train.md](docs/train.md)

## Core Contributors
[Peng Gao](https://scholar.google.com/citations?user=_go6DPsAAAAJ&hl=zh-CN), [Jiaming Han](https://csuhan.com), [Chris Liu](https://github.com/ChrisLiu6), [Ziyi Lin](https://github.com/linziyi96), [Renrui Zhang](https://github.com/ZrrSkywalker)

## Acknowledgement
+ [OpenChineseLLaMA](https://github.com/OpenLMLab/OpenChineseLLaMA)
+ [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
+ [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
+ [LLaVA](https://github.com/haotian-liu/LLaVA)
+ [llama](https://github.com/facebookresearch/llama)
+ [Point-Bind](https://github.com/ZrrSkywalker/Point-Bind)
+ [diffusers](https://github.com/huggingface/diffusers)