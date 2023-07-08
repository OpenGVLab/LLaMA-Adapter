import torch

def save(full_model, path, model_type = 'BIAS'):
    if model_type == 'BIAS':
        visual_block_keys = ['norm1', 'attn.qkv', 'attn.proj', 'norm2', 'mlp.fc1', 'mlp.fc2']
        base_keys_extra = ['clip_proj_norm', 'visual_proj_norm', 'visual_proj', 'clip_proj']
        suffixes = ['weight', 'bias']

        base_keys_visual = [f'{key}.{suffix}' for key in visual_block_keys for suffix in suffixes]
        base_keys_llama = ['attention.gate', 'attention.wq.bias', 'attention.wo.bias', 'feed_forward.w1.bias', 'feed_forward.w2.bias', 'feed_forward.w3.bias', 'attention_norm.weight', 'ffn_norm.weight']
        extra_keys = [f'{base_key}.{suffix}' for base_key in base_keys_extra for suffix in suffixes]
        extra_keys.extend(['llama.norm.weight', 'visual_query.weight', 'adapter_query.weight']) 
        
        keys = []
        for i in range(8):
            for base_key in base_keys_visual:
                keys.append(f'visual_blocks.{i}.{base_key}')
        for i in range(32):
            for base_key in base_keys_llama:
                keys.append(f'llama.layers.{i}.{base_key}')
        keys += extra_keys
    ## TODO: Add other model types

    full_model_state_dict = full_model.state_dict()
    small_weights = {key: full_model_state_dict[key] for key in keys}
    wrapped_small_weights = {'model': small_weights}

    # Save the wrapped small weights
    torch.save(wrapped_small_weights, path)