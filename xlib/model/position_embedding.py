import torch
import math


def fourier_embedding(x, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    seps = torch.linspace(-1, 1, half).to(x)
    freqs = torch.exp(-math.log(max_period) * seps) # [C/2,]
    dims = [1] * x.ndim
    args = x.unsqueeze(-1) * freqs.view(*dims, -1)  # [B, N, C, half]
    emb_cos = torch.cos(args)  # [B, N, C, half]
    emb_sin = torch.sin(args)  # [B, N, C, half]
    embedding = torch.cat([emb_cos, emb_sin], dim=-1)  # [B, N, C, dim]
    if dim % 2:
        # pad one more dim with zeros
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    embedding = embedding.view(*x.shape[:-1], -1)
    return embedding
