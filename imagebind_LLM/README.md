# ImageBind-LLM

## News

* [June 5, 2023] Release ImageBind-LLM for 3D point cloud input by [Point-Bind](https://github.com/ZrrSkywalker/Point-Bind).
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

* Download the pre-train weight of [ImageBind with 3D encoder](https://drive.google.com/file/d/1twRymNwVxZ_DG4TQ4m0VMi87j_2LAS8j/view?usp=sharing) that extends ImageBind with 3D point cloud modality, and organize the downloaded file in the following structure
  ```
  LLaMA-Adapter/imagebind_LLM/
  ├── ckpts
      ├── imagebind_w3D.pth
      ├── 7B-beta.pth (download while running)
      └── knn.index (download while running)
  ```
## Inference

Here is a simple inference script for ImageBind-LLM:

```python
import ImageBind.data as data
import llama


llama_dir = "/path/to/LLaMA"

# checkpoint will be automatically downloaded
model = llama.load("7B-beta", llama_dir, knn=True)
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

Here is a simple inference script for ImageBind-LLM testing on 3D point clouds via Point-Bind. We provide several point cloud samples in `examples/`.


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
Run the following command to host a local demo page:
``` bash
python gradio_app.py --llama_dir /path/to/llama_model_weights
```

## Fine-tuning
```
. exps/finetune.sh <pre-trained checkopint path> <output path>
```
