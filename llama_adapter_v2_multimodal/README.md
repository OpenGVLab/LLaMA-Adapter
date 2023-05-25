# LLaMA-Adapter-V2 Multi-modal

## News

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

## Usage

Here is a simple inference script for LLaMA-Adapter V2. The pre-trained model will be downloaded directly from [Github Release](https://github.com/ZrrSkywalker/LLaMA-Adapter/releases/tag/v.2.0.0).

```python
import cv2
import llama
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "/path/to/LLaMA/"

model, preprocess = llama.load("BIAS-7B", llama_dir, device)

prompt = llama.format_prompt("Please introduce this painting.")
img = Image.fromarray(cv2.imread("../docs/logo_v1.png"))
img = preprocess(img).unsqueeze(0).to(device)

result = model.generate(img, [prompt])[0]

print(result)
```

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

Now we provide `BIAS-7B`, which fine-tunes the `bias` and `norm` parameters of LLaMA. We will include more pretrained models in the future, such as the LoRA fine-tuning model `LoRA-7B` and partial-tuning model `PARTIAL-7B`.