import torch
from collections import OrderedDict
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ori", required=True, type=str,
    help="Name of or path to LLaMAAdapter pretrained checkpoint",
)
parser.add_argument(
    "--target", default=None,
    help="target position for the ckpt",
)
args = parser.parse_args()

ori_ckpt_path = Path(args.ori)
target_ckpt_path = ori_ckpt_path.with_stem("converted_" + ori_ckpt_path.stem)

ckpt = torch.load(ori_ckpt_path, map_location='cpu')

replace_dict = {
    'llma': 'llama'
}
renamed_ckpt = {}
for key, val in ckpt['model'].items():
    for replace_key, replace_val in replace_dict.items():
        key = key.replace(replace_key, replace_val)
    renamed_ckpt[key] = val


new_ckpt = {}
discarded  = []

for key, val in renamed_ckpt.items():
    if key.startswith('image_bind.'):
        discarded.append(key)
    elif key.startswith("llama.") and "bias" not in key and "gate" not in key and "lora" not in key and "norm" not in key:
        discarded.append(key)
    else:
        new_ckpt[key] = val

to_remove = ['prefix_projector_norm.weight', 'prefix_projector_norm.bias']
for _ in to_remove:
    if _ in new_ckpt:
        del new_ckpt[_]


print(f"discarded: {discarded}")
print(f"saved: {list(new_ckpt.keys())}")

new_ckpt = {'model': new_ckpt}
torch.save(new_ckpt, target_ckpt_path)