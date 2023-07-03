# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])


import argparse
import datetime
import json
import numpy as np
import time
from pathlib import Path
import functools
from functools import partial

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

from fairscale.nn.model_parallel import initialize as fs_init

from apex.optimizers import FusedAdam

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from model.meta import MetaModel
from model.LLM.llama import Attention, FeedForward
from engine_finetune import train_one_epoch, val_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # Model parameters
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL', choices=['llama', 'revllama'],
                        help='Name of model to train')
    parser.add_argument('--reversible_grad', action='store_true', default=False,
                        help='Whether to use reversible grad')

    parser.add_argument('--llama_config', default='params.json', type=str,
                        help='Path to llama model config')

    parser.add_argument('--load_dir', default='/path/to/sharded', type=str,
                        help='path to sharded')
    parser.add_argument('--save_dir', default='/path/to/full', type=str,
                        help='path to full')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.05)')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--model_parallel_size', type=int, default=1)
    parser.add_argument('--data_parallel', type=str, choices=['ddp', 'sdp', 'fsdp'], default='sdp')
    parser.add_argument('--precision', type=str, choices=['fp16', 'bf16', 'tf32'], default='bf16')
    parser.add_argument('--checkpointing', action="store_true", default=False,
                        help="enable gradient checkopointing")

    return parser


def main(args):
    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device('cuda')

    global_rank = misc.get_rank()
    mp_rank = fs_init.get_model_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()
    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_group = fs_init.get_data_parallel_group()

    # define the model
    model = MetaModel(args.llama_type, args.reversible_grad, args.llama_config)

    mixed_precision_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "tf32": torch.float32,
    }[args.precision]
    model = FSDP(
        model,
        process_group=fs_init.get_data_parallel_group(),
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=[Attention, FeedForward],
        ),
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=True,
        mixed_precision=MixedPrecision(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=mixed_precision_dtype,
            buffer_dtype=mixed_precision_dtype,
        ),
        sharding_strategy={
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "ddp": ShardingStrategy.NO_SHARD,
            "fsdp": ShardingStrategy.FULL_SHARD,
        }[args.data_parallel],
        device_id=device
    )

    param_groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = FusedAdam(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                              state_dict_config=ShardedStateDictConfig(offload_to_cpu=True)):
        load_path = os.path.join(
            args.load_dir,
            f"checkpoint.{global_rank:05d}-of-{misc.get_world_size():05d}.pth",
        )
        state_dict = torch.load(load_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])


    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        consolidated_model_state_dict = model.state_dict()
        optim_state_dict = FSDP.optim_state_dict(model, optimizer)

        to_save = {
            "model": consolidated_model_state_dict,
            "optimizer": optim_state_dict,
        }
        consolidated_model_save_path = os.path.join(
            args.save_dir,
            f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
        )
        if dp_rank == 0:
            torch.save(to_save, consolidated_model_save_path)

    # others common
    keys = ["epoch", "iter", "scaler", "args", "dataset_state"]
    other_to_save = {
        k: state_dict[k] for k in keys if k in state_dict
    }
    other_save_path = os.path.join(
            args.save_dir,
            f"other-{misc.get_rank():02d}-of-{misc.get_world_size():02d}.pth",
        )
    torch.save(other_to_save, other_save_path)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
