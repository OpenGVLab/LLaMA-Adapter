# ImageBind-LLM

## News

* [May 29, 2023] Initial release.


## Setup

* setup up a new conda env. Install ImageBind and other necessary packages.
  ```bash
  # create conda env
  conda create -n imagebind_LLM python=3.8 -y
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

## Usage

Here is a simple inference script for ImageBind-LLM. The pre-trained model will be released soon.

```python
import ImageBind.data as data
import llama

llama_dir = "/path/to/LLaMA/"

model = llama.load("7B", llama_dir)
input = data.load_and_transform_vision_data(["example_imgs/funny-photo.jpg"], device='cuda')
results = model.generate(input, [llama.format_prompt("Explain why this image is funny")], input_type="vision")
result = results[0].strip()
print(result)
```