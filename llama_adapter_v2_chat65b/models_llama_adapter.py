import functools
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import fairscale.nn.model_parallel.initialize as fs_init

from llama import ModelArgs, Tokenizer, LLaMA, Transformer


def _load_and_redistribute_checkpoint(llama_model_path, model_name):
    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))

    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)

    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()
    if mp_world_size == len(checkpoints):
        print('same number of shards of checkpoints and training, loading directly...')
        dist.barrier()
        print('[rank=%d, mp_rank=%d] loading from %s' % (dist.get_rank(), mp_rank, checkpoints[mp_rank]), force=True)
        checkpoint = torch.load(checkpoints[mp_rank], map_location='cpu')
    else:
        print('different number of shards of checkpoints and training, redistributing...')
        if dist.get_rank() == 0:
            loaded = []
            for x in checkpoints:
                print('loading from', x)
                loaded.append(torch.load(x, map_location='cpu'))
            
            full_state_dict = {}
            split_dims = {}

            def add_weight_with_split_dim(name, dim):
                if dim < 0: # bcast without split
                    full_state_dict[name] = loaded[0][name].clone()
                else:
                    full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
                for x in loaded:
                    del x[name]
                split_dims[name] = dim

            add_weight_with_split_dim('tok_embeddings.weight', 1)
            add_weight_with_split_dim('norm.weight', -1)
            add_weight_with_split_dim('output.weight', 0)
            for i in range(params['n_layers']):
                print('gathering layer %d of %d' % (i, params['n_layers']))
                layer_prefix = f'layers.{i}.'
                bcast_names = [
                        'attention_norm.weight',
                        'ffn_norm.weight',
                        ]
                column_parallel_names = [
                        'attention.wq.weight',
                        'attention.wk.weight',
                        'attention.wv.weight',
                        'feed_forward.w1.weight',
                        'feed_forward.w3.weight',
                        ]
                row_parallel_names = [
                        'attention.wo.weight',
                        'feed_forward.w2.weight',
                        ]
                for key in bcast_names:
                    add_weight_with_split_dim(layer_prefix + key, -1)
                for key in column_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 0)
                for key in row_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 1)

            full_state_dict_meta = dict((k, v.shape) for k, v in full_state_dict.items())
            dist.broadcast_object_list([full_state_dict_meta, split_dims], src=0)

        else: # dist.get_rank() != 0
            recv_objs = [None, None]
            dist.broadcast_object_list(recv_objs, src=0)
            full_state_dict_meta, split_dims = recv_objs

        local_state_dict = {}
        for k in sorted(full_state_dict_meta.keys()):
            print('redistributing weights: %s' % k)
            if dist.get_rank() == 0:
                value = full_state_dict[k].cuda().half()
                del full_state_dict[k]
            else:
                value = torch.empty(full_state_dict_meta[k], device='cuda', dtype=torch.half)
            dist.broadcast(value, src=0)
            value = value.cpu()
            if split_dims[k] < 0:
                local_state_dict[k] = value
            else:
                dim = split_dims[k]
                assert dim >= 0 and dim < value.ndim and value.size(dim) % mp_world_size == 0
                shard_size = value.size(dim) // mp_world_size
                shard_st, shard_ed = shard_size * mp_rank, shard_size * (mp_rank + 1)
                # TODO: make more general
                if dim == 0:
                    value = value[shard_st: shard_ed]
                elif dim == 1:
                    value = value[:, shard_st: shard_ed]
                else:
                    raise NotImplementedError()
                local_state_dict[k] = value.clone()

        checkpoint = local_state_dict

    return checkpoint, tokenizer, params


def Llama_adapter(args, model_name, adapter_len=0, adapter_layer=0, add_bias=False, add_scale=False, train_norm=False, **kwargs):
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(args.llama_model_path, model_name)

    model_args: ModelArgs = ModelArgs(
        # caching configuration
        max_seq_len=args.max_seq_len if hasattr(args, 'max_seq_len') else 2048,

        adapter_len=adapter_len,
        adapter_layer=adapter_layer,
        add_bias=add_bias,
        add_scale=add_scale,
        train_norm=train_norm,

        # other args
        **params
    )

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_llama_adapter = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    missing_keys, unexpected_keys = model_llama_adapter.load_state_dict(checkpoint, strict=False)

    for i in range(model_args.n_layers):
        if i < model_args.n_layers - adapter_layer:
            del model_llama_adapter.layers[i].attention.gate

    for name, param in model_llama_adapter.named_parameters():
        requires_grad = \
            name.endswith('.gate') or \
            name == 'adapter_query' or \
            (train_norm and '_norm.' in name) or \
            name.endswith('.added_bias') or \
            name.endswith('.added_scale')
            
        if requires_grad:
            param.data = param.data.float()
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return model_llama_adapter


# set recommended archs
Llama65B_bias_scale_norm_tuning = functools.partial(Llama_adapter, model_name='65B', add_bias=True, add_scale=True, train_norm=True)
