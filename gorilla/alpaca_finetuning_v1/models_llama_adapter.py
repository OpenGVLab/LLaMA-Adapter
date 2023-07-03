import json

import torch

from llama import ModelArgs, Tokenizer, Transformer


def Llama7B_adapter(args, **kwargs):

    llama_model_path = args.llama_model_path
    model_name = "7B"

    checkpoint = torch.load(llama_model_path + "/consolidated.00.pth", map_location="cpu")
    print(llama_model_path + "/consolidated.00.pth")

    with open(llama_model_path + "/params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=32,
        adapter_len=args.adapter_len,
        adapter_layer=args.adapter_layer,
        **params
    )
    tokenizer = Tokenizer(model_path=llama_model_path + "/tokenizer.model")

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_llama_adapter = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model_llama_adapter.load_state_dict(checkpoint, strict=False)

    for name, param in model_llama_adapter.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float()

    for name, param in model_llama_adapter.layers[-1 * args.adapter_layer :].named_parameters():
        if "gate" in name or "adapter" in name:
            param.data = param.data.float()
            param.requires_grad = True

    return model_llama_adapter


# set recommended archs
Llama7B_adapter = Llama7B_adapter
