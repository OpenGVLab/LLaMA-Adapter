# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
from tqdm import tqdm

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def format_prompt(instruction, input=None):
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
    if input is None or input=='':
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})

def get_questions(question_file):
 
    # Load questions file
    question_jsons = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            question_jsons.append(line)

    return question_jsons


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("consolidated*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint = {key[5:]:val for key, val in checkpoint.items() if key.startswith('llma.')}
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    load_result = model.load_state_dict(checkpoint, strict=False)
    print(load_result)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    dataset_path: str = '../gorilla-main/eval/eval-data/questions/{tensorflowhub, huggingface, torchhub}/questions_{tensorflowhub, huggingface, torchhub}_0_shot.jsonl',
    inference_batch_size = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    questions_json = get_questions(dataset_path)

    # ans_jsons = []
    # for i, line in enumerate(tqdm(questions_json)):
    #     ques_json = json.loads(line)
    #     idx = ques_json["question_id"]
    #     prompt = ques_json["text"]
    #     formated_prompt = format_prompt(prompt)
    #     results = generator.generate(
    #         [formated_prompt], max_gen_len=256, temperature=temperature, top_p=top_p
    #     )
    #     ans_jsons.append(
    #         {
    #             "question_id": idx,
    #             "questions": prompt,
    #             "text": results[0],
    #         }
    #     )

    ans_jsons = []
    batch_idx = []
    batch_prompt = []
    batch_formated_prompt = []
    question_num = len(questions_json)
    for i, line in enumerate(tqdm(questions_json)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        prompt = ques_json["text"]
        formated_prompt = format_prompt(prompt)
        batch_idx.append(idx)
        batch_prompt.append(prompt)
        batch_formated_prompt.append(formated_prompt)
        if (i+1) % inference_batch_size == 0 or i == question_num:
            results = generator.generate(
                batch_formated_prompt, max_gen_len=256, temperature=temperature, top_p=top_p
            )
            for i in range(len(batch_idx)):
                ans_jsons.append(
                    {
                        "question_id": batch_idx[i],
                        "questions": batch_prompt[i],
                        "text": results[i],
                    }
                )
            batch_idx = []
            batch_prompt = []
            batch_formated_prompt = []

    # Write output to file
    with open(os.path.join(ckpt_dir, 'model_prediction_results.jsonl'), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")






if __name__ == "__main__":
    fire.Fire(main)
