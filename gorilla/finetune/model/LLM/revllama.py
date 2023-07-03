# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
import copy

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear
from torch.autograd import Function

from apex.normalization import FusedRMSNorm as RMSNorm


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    reversible_gradient: bool = False


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


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

        self.flash = True

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv


        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        if self.flash:
           output = F.scaled_dot_product_attention(xq, keys, values, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
           scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
           if mask is not None:
              scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
           scores = F.softmax(scores.float(), dim=-1).type_as(xq)
           output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
                   1, 2
                   ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=False
        )

    # @torch.compile
    def _silu_gating(self, x, y):
        return F.silu(x) * y

    def forward(self, x):
        return self.w2(self._silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.seeds = {}

    def set_seed(self, key: str):
        """
        For activation recompute
        """
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            seed = int(torch.seed() % sys.maxsize)
        self.seeds[key] = seed
        torch.manual_seed(seed)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None
    ):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        if self.training:
            self.set_seed("F")  # seed for the F function
        f_x2 = self.f_forward(x2, start_pos, freqs_cis, mask, prompt)
        y1 = x1 + f_x2

        if self.training:
            self.set_seed("G") # seed for the G function
        g_y1 = self.g_forward(y1)
        y2 = x2 + g_y1
        return torch.cat([y1, y2], dim=-1)

    def f_forward(self, x, start_pos, freqs_cis, mask, prompt):
        return self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)

    def g_forward(self, x):
        return self.feed_forward(self.ffn_norm(x))

    def backward_pass(
        self,
        y,
        dy,
        start_pos,
        freqs_cis,
        mask,
        prompt
    ):
        assert self.training, (
            "If you want to train ReversibleModel, make sure to put the model into training mode."
        )
        y1, y2 = torch.chunk(y, 2, dim=-1)
        dy1, dy2 = torch.chunk(dy, 2, dim=-1)
        with torch.enable_grad():
            y1.requires_grad = True
            torch.manual_seed(self.seeds["G"])
            g_y1 = self.g_forward(y1)
            g_y1.backward(dy2)

        with torch.no_grad():
            x2 = y2 - g_y1
            del g_y1
            dy1 = dy1 + y1.grad
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            torch.manual_seed(self.seeds["F"])
            f_x2 = self.f_forward(x2, start_pos, freqs_cis, mask, prompt)
            f_x2.backward(dy1)

        with torch.no_grad():
            x1 = y1 - f_x2
            del f_x2, y1
            dy2 = dy2 + x2.grad
            x2.grad = None
            x2 = x2.detach()

        return torch.cat([x1, x2], dim=-1), torch.cat([dy1, dy2], dim=-1)


class RevBackProp(Function):
    @staticmethod
    def forward(
        ctx,
        x,
        layers,
        start_pos,
        freqs_cis,
        mask,
        prompt
    ):
        with torch.no_grad():
            for layer in layers:
                x = layer(x.detach(), start_pos, freqs_cis, mask, prompt)

        ctx.save_for_backward(x.detach())
        ctx.layers = layers
        ctx.start_pos = start_pos
        ctx.freqs_cis = freqs_cis
        ctx.mask = mask
        ctx.prompt = prompt
        return x

    @staticmethod
    def backward(ctx, dy):
        y, = ctx.saved_tensors
        for layer in ctx.layers[::-1]:
            y, dy = layer.backward_pass(
                y,
                dy,
                ctx.start_pos,
                ctx.freqs_cis,
                ctx.mask,
                ctx.prompt
            )
        return dy, None, None, None, None, None


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.reversible_gradient = params.reversible_gradient # If false, use vanilla gradient

    @staticmethod
    def vanilla_forward(h, layers, start_pos, freqs_cis, mask, prompt):
        for _, layer in enumerate(layers):
            h = layer(h, start_pos, freqs_cis, mask, prompt)
        return h

    def forward(self, examples):
        _bsz, seqlen = examples.shape
        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
        start_pos = 0
        prompt = None
        h = torch.cat([h, h], dim=-1)

        if not self.training or not self.reversible_gradient:
            executing_fn = Transformer.vanilla_forward
        else:
            executing_fn = RevBackProp.apply
        h = executing_fn(h, self.layers, start_pos, freqs_cis, mask, prompt)

        h1, h2 = torch.chunk(h, 2, dim=-1)
        h = (h1 + h2) / 2.
        h = self.norm(h)
        output = self.output(h)
        return output



    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        h = torch.cat([h, h], dim=-1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h1, h2 = torch.chunk(h, 2, dim=-1)
        h = (h1 + h2) / 2.
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
