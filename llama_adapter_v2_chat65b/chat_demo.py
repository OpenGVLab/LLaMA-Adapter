import argparse
import os
import sys

import torch
import torch.distributed as dist

import fairscale.nn.model_parallel.initialize as fs_init

from conversation import conv_templates, SeparatorStyle
import models_llama_adapter
from util import misc
from llama import LLaMA, Tokenizer


def load_model(args, load_8bit=False):
    model = models_llama_adapter.__dict__[args.model_name](args)
    model.eval()
    if args.model_path is None:
        print('Warning: not loading instruct tuned weights.')
    else:
        print('Using instruct tuned weights from:', args.model_path)
        checkpoint = torch.load(args.model_path, map_location='cpu')
        for k, v in checkpoint['model'].items():
            if k.endswith('.wq_bias') or \
                    k.endswith('.wk_bias') or \
                    k.endswith('.wv_bias') or \
                    k.endswith('.wo_scale') or \
                    k.endswith('.w1_bias') or \
                    k.endswith('.w3_bias') or \
                    k.endswith('.w2_scale'):
                assert v.ndim == 1
                mp_size = fs_init.get_model_parallel_world_size()
                mp_rank = fs_init.get_model_parallel_rank()
                shard_size = v.size(0) // mp_size
                checkpoint['model'][k] = v[shard_size * mp_rank: shard_size * (mp_rank + 1)]
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)

    generator = LLaMA(
        model,
        Tokenizer(model_path=os.path.join(args.llama_model_path, 'tokenizer.model')),
    )

    return generator


@torch.inference_mode()
def generate_stream(model, params):
    """Adapted from fastchat/serve/model_worker.py::generate_stream"""

    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 0.95))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    with torch.cuda.amp.autocast():
        decoded = model.generate(
            [prompt],
            max_gen_len=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    decoded = decoded[0]

    pos = decoded.find(stop_str)
    if pos != -1: 
        decoded = decoded[:pos]

    return decoded

def main(args):
    misc.init_distributed_mode()
    fs_init.initialize_model_parallel(dist.get_world_size())
    torch.manual_seed(1)

    # Model
    model = load_model(args)

    # Chat
    conv = conv_templates[args.conv_template].copy()
    while True:
        if dist.get_rank() == 0:
            try:
                sys.stdout.write(f"\n{conv.roles[0]}: ")
                sys.stdout.flush()
                inp = input()
            except EOFError:
                inp = ""
            dist.broadcast_object_list([inp], src=0)
        else:
            recv_obj = [None]
            dist.broadcast_object_list(recv_obj, src=0)
            inp = recv_obj[0]

        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        params = {
            "model": 'LLaMA-Adapter',
            "prompt": prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        if dist.get_rank() == 0:
            sys.stdout.write(f"{conv.roles[1]}: "); sys.stdout.flush()
        outputs = generate_stream(model, params)
        outputs = outputs.strip()
        if dist.get_rank() == 0:
            sys.stdout.write(outputs + '\n'); sys.stdout.flush()

        conv.messages[-1][-1] = outputs


        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument("--model_path", type=str)
    parser.add_argument('--llama_model_path', type=str, required=True)
    parser.add_argument("--conv_template", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
