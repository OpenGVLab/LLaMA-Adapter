# Script for obtaining Chinese LLaMA weights from the OpenChineseLLaMA project (https://github.com/OpenLMLab/OpenChineseLLaMA)
# Due to the License of LLaMA, we only provide a delta-version patch
# Adding the patch to the original LLaMA weights makes the Chinese LLaMA weights
import os
import sys
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
import shutil
import torch
import argparse
from util.misc import download

parser = argparse.ArgumentParser()
parser.add_argument(
    "--llama_dir", default="/path/to/llama", type=str,
    help="Path to official LLaMA weights",
)
args = parser.parse_args()

ori_path = os.path.join(args.llama_dir, '7B')
delta_path = os.path.join(args.llama_dir, '7B_chinese_delta')
new_path = os.path.join(args.llama_dir, '7B_chinese')


download('https://huggingface.co/Cxxs/Open-Chinese-LLaMA/resolve/main/7B_chinese_delta/consolidated.00.pth',
         delta_path)
download('https://huggingface.co/Cxxs/Open-Chinese-LLaMA/resolve/main/7B_chinese_delta/params.json',
         delta_path)

os.makedirs(new_path, exist_ok=True)
shutil.copyfile(os.path.join(delta_path, 'params.json'), os.path.join(new_path, 'params.json'))

ori_dict = torch.load(os.path.join(ori_path, 'consolidated.00.pth'), map_location="cpu")
delta_dict = torch.load(os.path.join(delta_path, 'consolidated.00.pth'), map_location="cpu")
new_dict = {}
for key in ori_dict:
    new_value = (ori_dict[key].float() + delta_dict[key].float()).half()
    new_dict[key] = new_value

torch.save(new_dict, os.path.join(new_path, 'consolidated.00.pth'))