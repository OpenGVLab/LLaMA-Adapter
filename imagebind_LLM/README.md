# ImageBind-LLM

## News

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


## Demo
Run the following command to host a local demo page:
``` bash
python gradio_app.py --llama_dir /path/to/llama_model_weights
```

## Fine-tuning
```
. exps/finetune.sh <pre-trained checkopint path> <output path>
```