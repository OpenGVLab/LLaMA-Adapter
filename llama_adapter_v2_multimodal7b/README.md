# LLaMA-Adapter-V2 Multi-modal

## News
* [Oct 11, 2023] Release LLaMA-Adapter V2.1 and evaluation on MME.
* [July 5, 2023] Release pre-traininig and fine-tuning codes.
* [May 26, 2023] Initial release.


## Setup

* setup up a new conda env and install necessary packages.
  ```bash
  conda create -n llama_adapter_v2 python=3.8 -y
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

Here is a simple inference script for LLaMA-Adapter V2. The pre-trained model will be downloaded directly from [Github Release](https://github.com/OpenGVLab/LLaMA-Adapter/releases/tag/v.2.0.0).

```python
import cv2
import llama
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "/path/to/LLaMA/"

# choose from BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21
model, preprocess = llama.load("BIAS-7B", llama_dir, llama_type="7B", device=device)
model.eval()

prompt = llama.format_prompt("Please introduce this painting.")
img = Image.fromarray(cv2.imread("../docs/logo_v1.png"))
img = preprocess(img).unsqueeze(0).to(device)

result = model.generate(img, [prompt])[0]

print(result)
```

The output will look like the following:
```
The painting features a cute white lama, or llama, standing on a wooden floor. The llama is holding a variety of tools and accessories, such as a paintbrush, a pencil, a ruler, a pair of scissors, and a paint can. The llama is dressed in a suit, which adds a touch of sophistication to the scene. The painting is a creative and whimsical representation of a person or animal holding various tools and accessories, making it an interesting and unique piece of art.
```

## Evaluation
Check [eval.md](./docs/eval.md) for details.

## Online demo

We provide an online demo at [OpenGVLab](http://llama-adapter.opengvlab.com).

You can also start it locally with:
```bash
python gradio_app.py
```

## Models

You can check our models by running:
```python
import llama
print(llama.available_models())
```

Now we provide `BIAS-7B` which fine-tunes the `bias` and `norm` parameters of LLaMA, and `LORA-BIAS-7B` which fine-tunes the `bias`, `norm` and `lora` parameters of LLaMA. We will include more pretrained models in the future, such as the LoRA fine-tuning model `LORA-7B` and partial-tuning model `PARTIAL-7B`.

## Pre-traininig & Fine-tuning
See [train.md](docs/train.md)
