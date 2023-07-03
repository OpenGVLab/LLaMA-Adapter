# Environment Setup

* Setup up a new conda env and install required packages
  ```bash
  # create conda env
  conda create -n minigpt python=3.10 -y
  conda activate minigpt
  # install packages
  pip install -r requirements.txt
  ```

* This project relies on [apex](https://github.com/NVIDIA/apex), which, unfortunately, you need to compile from source. Please follow the [official instructions](https://github.com/NVIDIA/apex#from-source) to compile.
  * Some experience to compile successfully:
    1. `git clone https://github.com/NVIDIA/apex`
    2. make sure the version of CUDA on your machine is eqaul to the version with which your installed pytorch is built.
    3. make sure `pip >= 23.1`, otherwise run `pip install --upgrade pip`
    4. `pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./`

* LLaMA ckpt preparation. Please request access to the pre-trained LLaMA from [this form](https://forms.gle/jk851eBVbX1m5TAv5), and orgnize the diractory like:
  ```
  {path/to/llama}/
  |- consolidated.00.pth
  |- params.json
  |- tokenizer.model
  ```

* **The `torchhub_train.json` from [Gorilla official repository](https://github.com/ShishirPatil/gorilla/tree/main/data/apibench) has different format compared to `tensorflow_train.json` and `huggingface_train.json`, so currently we didn't conduct experiments on it.**


# Full finetune

## model training

* First specify `llama_path` in `finetune/scripts/finetune/finetune_7B_gorilla_{tf,hf,th}.sh`

* Then run script:
  ```bash
  cd finetune
  bash scripts/finetune/finetune_7B_gorilla_{tf,hf,th}.sh sdp 1
  ```
  Last parameter is model parallel size, and you can increase it if GPU memory is low. For A100/A6000, you can leave it 1.



## Inference

* To evaluate model performance, first we need generate responses with finetuned model.

* First copy `params.json` to model folder:
  ```bash
  cp {path/to/llama}/params.json finetune/output/{exp_name}/{epoch*}/
  ```

* Then run command:
  ```bash
  cd inference
  torchrun --nproc_per_node 1 gorilla_inference_full_finetune.py --dataset_path ../gorilla-main/eval/eval-data/questions/{tensorflowhub, huggingface, torchhub}/questions_{tensorflowhub, huggingface, torchhub}_0_shot.jsonl --ckpt_dir ../finetune/output/{exp_name}/{epoch*}/ --tokenizer_path {path/to/llama}/tokenizer.model
  ```
  **Note**: `ckpt_dir` should be a **FOLDER**, not a `.pth` file. Only inference on one GPU is supported.



# LLaMA adapter finetune

## model training

* First specify `llama_path` in `alpaca_finetuning_v1/finetune_{tf,hf,th}.sh`

* Then run script:
  ```bash
  cd alpaca_finetuning_v1
  bash finetune_{tf,hf,th}.sh
  ```
  Note: `--blr` is the base learning rate, we have `lr = blr * eff_batch_size / 256` in `alpaca_finetuning_v1/finetuning.py` line 237. Adjust it when you change the GPU number.

* After adapter finetune, we can extract adapter parameters from checkpoint. Run:
  ```bash
  python extract_adapter_from_checkpoint.py --model_path ./checkpoint/{exp_name}/{pth_file}
  ```

## Inference

* Run command:
  ```bash
  cd inference
  torchrun --nproc_per_node 1 gorilla_inference_llama_adapter_v1.py --ckpt_dir {path/to/llama} --tokenizer_path {path/to/llama}/tokenizer.model --adapter_path ../alpaca_finetuning_v1/checkpoint/{exp_name}/{adapter_pth_file} --dataset_path ../gorilla-main/eval/eval-data/questions/{tensorflowhub, huggingface, torchhub}/questions_{tensorflowhub, huggingface, torchhub}_0_shot.jsonl
  ```
  **Note**: `ckpt_dir` should be a **FOLDER**, not a `.pth` file. Only inference on one GPU is supported.



# Evaluation

* Run Gorilla official evaluation code by:
  ```bash
  cd gorilla-main/eval/eval-scripts/

  # For full finetune
  python ast_eval_{tf,hf,th}.py --api_dataset ../../data/api/{tensorflowhub_api, huggingface_api, torchhub_api}.jsonl --apibench ../../data/apibench/{tensorflow,huggingface,torchhub}_eval.json --llm_responses ../../../finetune/output/{exp_name}/{epoch*}/model_prediction_results.jsonl
  
  # For llama-adapter
  python ast_eval_{tf,hf,th}.py --api_dataset ../../data/api/{tensorflowhub_api, huggingface_api, torchhub_api}.jsonl --apibench ../../data/apibench/{tensorflow,huggingface,torchhub}_eval.json --llm_responses ../../../alpaca_finetuning_v1/checkpoint/{exp_name}/model_prediction_results.jsonl
  ```



# Results

Our finetuned LLaMA-adapter models and their predictions can be found in [this link](https://drive.google.com/drive/folders/1PN5QjOlMVnmSSFi68CubvQfGYeodvO8w?usp=sharing).

| Methods       | TensorFlow Hub      | TensorFlow Hub     | HuggingFace         | HuggingFace        |
| ------------- | ------------------- | ------------------ | ------------------- | ------------------ |
|               | overall $\uparrow$ | hallu $\downarrow$ | overall $\uparrow$ | hallu $\downarrow$ |
| Official      | 83.79               | 5.40               | 71.68               | 10.95              |
| Full finetune | 88.02               | 1.02               | 69.69               | 10.29              |
| LLaMA-adapter | 86.90               | 0.74               | 63.62               | 11.83              |





