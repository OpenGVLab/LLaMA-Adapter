import json

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
import pdb
pdb.set_trace()

datas = json.load(open('/home/pgao/stanford_alpaca/stanford_alpaca/alpaca_data.json'))
prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in datas
        ]


targets = [f"{example['output']}" for example in datas]
examples = [s + t for s, t in zip(sources, targets)]
for strings in (examples, sources):
    print(strings)

