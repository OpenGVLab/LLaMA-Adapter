import torch
import torch.nn as nn
import json
from .tokenizer import Tokenizer
from . import LLM
from global_configs import tokenizer_path


class MetaModel(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, llama_type, reversible_grad: bool, llama_config):
        super().__init__()

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        ModelArgs = LLM.__dict__[llama_type].ModelArgs
        Transformer = LLM.__dict__[llama_type].Transformer

        with open(llama_config, "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=2048, max_batch_size=32, **params
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        if reversible_grad:
            if hasattr(model_args, "reversible_gradient"):
                model_args.reversible_gradient = True
            else:
                raise KeyError (f"{ModelArgs} object has no attribute reversible_gradient")

        model = Transformer(model_args)
        self.llma = model
        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameter count : {count}")

    def forward(self, examples, labels):
        output = self.llma(examples)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
           c_loss = output.mean() * 0
        else:
           c_loss = self.criterion(output.reshape(-1, 32000), labels.flatten())
        pred = 0
        mask = 0
        return c_loss, c_loss, pred, mask