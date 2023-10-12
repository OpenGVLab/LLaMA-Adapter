import os
import glob
import argparse
from tqdm import tqdm
import PIL
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import cv2
from llama.llama_adapter import LLaMA_adapter

DATA_DIR = "./MME_Benchmark_release_version"

def get_image(image):
    if type(image) is str:
        try:
            return Image.open(image).convert("RGB")
        except Exception as e:
            print(f"Fail to read image: {image}")
            exit(-1)
    elif type(image) is Image.Image:
        return image
    elif type(image) is PIL.JpegImagePlugin.JpegImageFile:
        return image
    elif type(image) is PIL.PngImagePlugin.PngImageFile:
        return image
    elif type(image) is PIL.MpoImagePlugin.MpoImageFile:
        return image
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")


class MMEDataset(Dataset):
    def __init__(
        self,
        dataset_name
    ):
        self.dataset_name = dataset_name
        self.dataset = []
        jpg_sets = ["artwork", "celebrity", "color", "count", "existence", "landmark", "OCR", "position", "posters", "scene"]
        png_sets = ["code_reasoning", "commonsense_reasoning", "numerical_calculation", "text_translation"]
        image_suffix = '.jpg' if dataset_name in jpg_sets else ".png"

        assert (dataset_name in jpg_sets) or (dataset_name in png_sets), f"Invalid dataset name for MME benchmark: {dataset_name}"

        if os.path.exists(f"{DATA_DIR}/{dataset_name}/images") and os.path.exists(f"{DATA_DIR}/{dataset_name}/questions_answers_YN"):
            question_files = os.listdir(f"{DATA_DIR}/{dataset_name}/questions_answers_YN")
            for question_file in question_files:
                image_file_name = os.path.join(DATA_DIR, dataset_name, "images", question_file.replace('.txt', image_suffix))
                with open(os.path.join(DATA_DIR, dataset_name, "questions_answers_YN", question_file), 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        try:
                            question, gt_answer = line.replace('\n', '').split('\t')
                            self.dataset.append({
                                "image_path": image_file_name,
                                "gt_answers": gt_answer,
                                "question": question
                            })
                        except:
                            pass

        else:
            question_files = glob.glob(f"{DATA_DIR}/{dataset_name}/*.txt")
            for question_file in question_files:
                image_file_name = question_file.replace(".txt", image_suffix)
                with open(question_file, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        try:
                            question, gt_answer = line.replace('\n', '').split('\t')
                            self.dataset.append({
                                "image_path": image_file_name,
                                "gt_answers": gt_answer,
                                "question": question
                            })
                        except:
                            pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_args_parser():
    parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
    # Model parameters
    parser.add_argument('--llama_path', default='/path/to/llama', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='directory containing pre-trained checkpoints')
    parser.add_argument('--lora', default=16, type=int)
    parser.add_argument('--output_path', default='/path/to/output_results', type=str)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    llama_dir = args.llama_path
    llama_type = '7B'
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')
    
    model_path = args.pretrained_path
    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')

    w_bias = True
    w_lora = args.lora > 0
    print('Lora:', w_lora)
    lora_rank = args.lora
    model = LLaMA_adapter(
        llama_ckpt_dir, llama_tokenzier_path,
        max_seq_len=512, max_batch_size=1,
        clip_model='ViT-L/14',
        v_embed_dim=768, v_depth=8,
        v_num_heads=16, v_mlp_ratio=4.0,
        query_len=10, query_layer=31,
        w_bias=w_bias,
        w_lora=w_lora,
        lora_rank=lora_rank,
        w_new_gate=w_lora,  # for compatibility
        phase='finetune')

    load_result = model.load_state_dict(ckpt['model'], strict=False)
    print(load_result)

    model = model.to(device)
    model.half()
    model.eval()
    preprocess = model.clip_transform

    prompt_format = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request using a single word or phrase.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

    def multi_modal_generate(
        img_path: str,
        prompt: str,
        max_gen_len=30,
        temperature: float = 0,
        top_p: float = 0.75,
    ):
        img = Image.fromarray(cv2.imread(img_path))
        img = preprocess(img).unsqueeze(0).half().to(device)
        prompt = prompt_format.format_map({'instruction': prompt})

        result = model.generate(img, [prompt], 
                                max_gen_len=max_gen_len, 
                                temperature=temperature, 
                                top_p=top_p)
        return result[0]


    result = {}
    dataset_names = ["artwork", "celebrity", "color", "count", "existence", "OCR", "position", "posters", "scene", "code_reasoning", "commonsense_reasoning", "numerical_calculation", "text_translation", "landmark"] # landmark (03d5e3bfc958be38.jpg)
    answer_path = args.output_path
    batch_size = 1

    print("Starting...")
    for dataset_name in dataset_names:
        dataset = MMEDataset(dataset_name)

        predictions = []
        with torch.no_grad():
            for data in tqdm(dataset, desc=f"Inferencing {dataset_name}"):
                pred = multi_modal_generate(data['image_path'], data['question'])            
                predictions.append({'image_path': data['image_path'], 'question': data['question'], 'answer': pred, 'gt_answers': data['gt_answers']})

        os.makedirs(answer_path, exist_ok=True)
        prediction_file = os.path.join(answer_path, f"{dataset_name}.txt")
        out_datas = [
            f"{data['image_path']}\t{data['question']}\t{data['gt_answers']}\t{data['answer']}"
            for data in predictions
        ]
        with open(prediction_file, 'w') as f:
            f.write('\n'.join(out_datas))