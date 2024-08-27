import math
from abc import abstractmethod

import torch
from torch import nn

__all__ = ['fourier_embedding']


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def fourier_embedding(timesteps: torch.Tensor, dim, max_period=10000):
    r"""Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.
    Returns:
        embedding (torch.Tensor): [N $\times$ dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class NeuralRepresentation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.param = nn.Parameter(torch.randn(1, dim, 1, 1), requires_grad=True)
        self.scale_layer = nn.Linear(dim, dim)
        self.bias_layer = nn.Linear(dim, dim)
        
        
    def forward(self, emb):
        return self.param * self.scale_layer(emb) + self.bias_layer(emb)
        