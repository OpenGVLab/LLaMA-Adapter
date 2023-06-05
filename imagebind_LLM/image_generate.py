import llama
import torch
import numpy as np


@torch.inference_mode()
def image_generate(inputs, model: llama.LLaMA_adapter, pipe, prompt, cache_size, cache_t, cache_weight, knn=True, point_scale=5.):

    embeddings = []
    embeddings_weights = []

    for input_type, (input, input_weight) in inputs.items():
        if input_type in ['Image', 'Video']:
            type = 'vision'
        else:
            type = input_type.lower()
        embedding = model.image_bind({type : input}, prenorm=True)[1][type]
        if type == 'point':
            embedding = embedding / point_scale
        embeddings.append(embedding)
        embeddings_weights.append(input_weight)
    embeddings_weights = [x/(sum(embeddings_weights)+1e-6) for x in embeddings_weights]
    embedding = sum([embedding*embedding_weight for embedding, embedding_weight in zip(embeddings, embeddings_weights)])

    if knn:
        index = model.index

        embedding_norm_scale = embedding.norm(dim=-1, keepdim=True)
        embedding = embedding / embedding_norm_scale
        embedding_ori = embedding

        sims, indices = index.search(embedding.detach().cpu(), int(cache_size))
        B = sims.shape[0]
        prototypes = [index.reconstruct(x) for x in indices.reshape(-1, ).tolist()]
        prototypes = np.vstack(prototypes).reshape(B, int(cache_size), -1)  # [N, top_k, 1024]
        sims = torch.tensor(sims, device='cuda')
        prototypes = torch.tensor(prototypes, device='cuda')

        sims = (sims * cache_t).softmax(dim=-1)
        embedding = sims @ prototypes
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        embedding = (1-cache_weight) * embedding_ori + cache_weight * embedding
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        embedding = embedding_norm_scale*embedding

    embedding = torch.squeeze(embedding,0)
    image = pipe(prompt=prompt, image_embeds=embedding).images[0]

    return image