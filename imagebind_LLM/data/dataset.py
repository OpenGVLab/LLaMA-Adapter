import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from llama import Tokenizer
import copy
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

class CaptionCOCO(Dataset):
    def __init__(self, transform, max_words=30, tokenizer_path=None):
        ann = json.load(open('/data1/LLaVA/LLaVA-Instruct-150K/llava_instruct_150k_single_turn.json'))
        ann += json.load(open('/home/pgao/GPT-4-LLM/data/alpaca_gpt4_data.json'))
        self.ann = ann
        self.transform = transform
        self.max_words = max_words
        tokenizer = Tokenizer(model_path=tokenizer_path)
        self.tokenizer1 = tokenizer
    def __len__(self):
        return len(self.ann)

    def get_qa(self, orig_qa):
        qa = orig_qa.replace('\n\n', '\n')
        qa_list = qa.split('\n')
        qa_list = [sentence[6:] for sentence in qa_list]
        return qa_list

    def __getitem__(self, index):
        data_item = self.ann[index]
        if 'image' in data_item.keys():
            filename = data_item['image']
            question = data_item['conversations'][0]['value']
            answer = data_item['conversations'][1]['value']

            image = Image.open(filename).convert('RGB')
            image = self.transform(image)
            input1 = {'instruction': question}
        else:
            image = torch.zeros(3, 224, 224)
            input1 = {'instruction': data_item['instruction'], 'input': data_item['input']}
            answer = data_item['output']
        input1 = PROMPT_DICT['prompt_no_input'].format_map(input1)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer1.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer1.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image
