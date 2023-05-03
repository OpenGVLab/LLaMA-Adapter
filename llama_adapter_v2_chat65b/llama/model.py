# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, ParallelEmbedding, RowParallelLinear
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_seq_len: int = 2048

    adapter_len: int = 0
    adapter_layer: int = 0

    add_bias: bool = False
    add_scale: bool = False
    train_norm: bool = False


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return (output * self.weight).type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def forward_linear_with_scale_and_bias(x, module, scale=None, bias=None):
    if scale is not None:
        x = x * scale
    x = module(x)
    if bias is not None:
        x = x + bias
    return x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads
        self.dim = args.dim

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim, args.dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )

        self.gate = torch.nn.Parameter(torch.zeros(1, args.n_heads, 1, 1))
        self.head_start = self.n_local_heads * fs_init.get_model_parallel_rank()
        self.head_end = self.n_local_heads * (fs_init.get_model_parallel_rank() + 1)

        self.cache_enabled = False
        self.cache_k, self.cache_v = None, None

        if args.add_bias:
            self.wq_bias, self.wk_bias, self.wv_bias = [
                nn.Parameter(torch.zeros([self.n_local_heads * self.head_dim])) for _ in range(3)
            ]
            self.wo_bias = nn.Parameter(torch.zeros([args.dim]))
        else:
            self.wq_bias = self.wk_bias = self.wv_bias = self.wo_bias = None

        if args.add_scale:
            self.wq_scale, self.wk_scale, self.wv_scale = [nn.Parameter(torch.ones([args.dim])) for _ in range(3)]
            self.wo_scale = nn.Parameter(torch.ones([self.n_local_heads * self.head_dim]))
        else:
            self.wq_scale = self.wk_scale = self.wv_scale = self.wo_scale = None

    def enable_cache(self):
        self.cache_enabled = True

    def disable_cache(self):
        self.cache_enabled = False
        self.cache_k, self.cache_v = None, None

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None
    ):
        bsz, seqlen, _ = x.shape
        xq = forward_linear_with_scale_and_bias(x, self.wq, self.wq_scale, self.wq_bias)
        xk = forward_linear_with_scale_and_bias(x, self.wk, self.wk_scale, self.wk_bias)
        xv = forward_linear_with_scale_and_bias(x, self.wv, self.wv_scale, self.wv_bias)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = forward_linear_with_scale_and_bias(adapter, self.wk, self.wk_scale, self.wk_bias)
            adapter_k = adapter_k.view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = forward_linear_with_scale_and_bias(adapter, self.wv, self.wv_scale, self.wv_bias)
            adapter_v = adapter_v.view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_k = adapter_k.transpose(1, 2)
            adapter_v = adapter_v.transpose(1, 2)
        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.cache_enabled:
            if self.cache_k is None:
                assert start_pos == 0
                self.cache_k, self.cache_v = keys, values
            else:
                assert self.cache_k.size(2) >= start_pos
                self.cache_k = torch.cat([self.cache_k[:, :, :start_pos], keys], dim=2)
                self.cache_v = torch.cat([self.cache_v[:, :, :start_pos], values], dim=2)
                keys, values = self.cache_k, self.cache_v

        output = self._forward_scaled_dot_product_attention(xq, keys, values, mask)
        if adapter is not None:
            output += self.gate[
                :, self.head_start : self.head_end
            ].tanh().half() * self._forward_scaled_dot_product_attention(xq, adapter_k, adapter_v)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return forward_linear_with_scale_and_bias(output, self.wo, self.wo_scale, self.wo_bias)

    def _forward_scaled_dot_product_attention(self, q, k, v, mask=None):
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(q, k, v, mask >= 0 if mask is not None else None)
        else:
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            output = torch.matmul(scores, v)
            return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        add_bias: bool = False,
        add_scale: bool = False,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim, self.hidden_dim = dim, hidden_dim
        mp_size = fs_init.get_model_parallel_world_size()
        mp_rank = fs_init.get_model_parallel_rank()
        self.mp_dim_st = self.hidden_dim // mp_size * mp_rank
        self.mp_dim_ed = self.hidden_dim // mp_size * (mp_rank + 1)

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)

        if add_bias:
            self.w1_bias, self.w3_bias = [nn.Parameter(torch.zeros([hidden_dim // mp_size])) for _ in range(2)]
            self.w2_bias = nn.Parameter(torch.zeros([dim]))
        else:
            self.w1_bias = self.w2_bias = self.w3_bias = None

        if add_scale:
            self.w1_scale, self.w3_scale = [nn.Parameter(torch.ones([dim])) for _ in range(2)]
            self.w2_scale = nn.Parameter(torch.ones([hidden_dim // mp_size]))
        else:
            self.w1_scale = self.w2_scale = self.w3_scale = None

    def forward(self, x):
        return forward_linear_with_scale_and_bias(
            F.silu(forward_linear_with_scale_and_bias(x, self.w1, self.w1_scale, self.w1_bias))
            * forward_linear_with_scale_and_bias(x, self.w3, self.w3_scale, self.w3_bias),
            self.w2,
            self.w2_scale,
            self.w2_bias,
        )


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            add_bias=args.add_bias,
            add_scale=args.add_scale,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None
    ):

        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = ParallelEmbedding(params.vocab_size, params.dim, init_method=lambda x: x)

        if params.adapter_len * params.adapter_layer > 0:
            self.adapter_query = nn.Embedding(params.adapter_len * params.adapter_layer, params.dim)
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(params.dim, params.vocab_size, bias=False, init_method=lambda x: x)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

    def forward(self, examples, labels):
        _bsz, seqlen = examples.shape

        h = self.tok_embeddings(examples)
        freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
        start_pos = 0

        if self.adapter_len * self.adapter_layer > 0:
            for layer in self.layers[: -1 * self.adapter_layer]:
                h = layer(h, start_pos, freqs_cis, mask)
            adapter_index = 0
            adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)
            for layer in self.layers[-1 * self.adapter_layer :]:
                h = layer(h, start_pos, freqs_cis, mask, adapter[adapter_index].half())
                adapter_index = adapter_index + 1

        else:
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h)
        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()

        c_loss = self.criterion(output, labels)
        return c_loss

    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        if self.adapter_len * self.adapter_layer > 0:
            adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)
        if seqlen == 1:
            mask = None
        elif start_pos == 0:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)
        else:
            raise NotImplementedError()

        for i, layer in enumerate(self.layers):
            adapter_index = i - (len(self.layers) - self.adapter_layer)
            h = layer(h, start_pos, freqs_cis, mask, adapter[adapter_index].half() if adapter_index >= 0 else None)

        h = self.norm(h)
        output = self.output(h[:, -1, :])
        return output.float()

    def enable_cache(self):
        for layer in self.layers:
            layer.attention.enable_cache()

    def disable_cache(self):
        for layer in self.layers:
            layer.attention.disable_cache()
