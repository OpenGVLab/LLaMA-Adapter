The training process of LLaMA-Adapter V2 consists of the pre-training and fine-tuning phases. 

## Pre-training
### Data
* We use multiple datasets with **image-text pairs** for pre-training. The texts are English-only.

* For each dataset, the meta file should be organized in the `.csv` format as following:

  ```
  url		caption
  /path/to/image1		caption1
  /path/to/image2		caption2
  ...
  ```

  Alternatively, you may modify the [`PretrainDataset`](/data/dataset.py) implementation to adapt to your own meta file format.

* Write a `.yaml` config file to specify the datasets for pre-training:
  ```
  META:
    - '/path/to/cc3m.csv'
    - '/path/to/coco.csv'
    ...
  ```

### Start pre-training

We are now ready to start pre-training (please make sure that the original LLaMA weights are available in `/path/to/llama_model_weights`). 

```bash
. exps/pretrain.sh /path/to/llama_model_weights /path/to/pretrain-data-config.yaml /output/path
```



## Fine-tuning

### Data

* We fine-tune LLaMA-Adapter V2 on text-only as well as image-text instruction following datasets.

* The following lists the datasets we use for training our release weights:

  | Name                     | Link                                                         |
  | ------------------------ | ------------------------------------------------------------ |
  | alpaca_gpt4_data.json    | [File Link](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) |
  | alpaca_gpt4_data_zh.json | [File Link](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json) |
  | llava_instruct_150k.json | [File Link](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/raw/main/llava_instruct_150k.json) |
  | alpaca_data_zh_51k.json  | [File Link](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/data/alpaca_data_zh_51k.json) |

* Similar to pre-training, write a `.yaml` config file to specify the datasets for fine-tuning:

  ```
  META:
    - '/path/to/alpaca_gpt4_data.json'
    - '/path/to/alpaca_gpt4_data_zh.json'
    ...
  ```

### Start fine-tuning

```bash
. exps/finetune.sh \
 /path/to/llama_model_weights /path/to/pre-trained/checkpoint.pth \
 /path/to/finetune-data-config.yaml /output/path
```

