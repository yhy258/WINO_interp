"""
    Largely based on https://github.com/JeremieMelo/NeurOLight/blob/main/core/models/layers/fno_conv2d.py
    NeurOLight: A Physics-Agnostic Neural Operator Enabling Parametric Photonic Device Simulation
    MIT License
    Copyright (c) 2022 Jiaqi Gu
"""
from functools import lru_cache
from typing import Tuple

import torch
from torch import nn
from torch.functional import Tensor
from torch.types import Device

__all__ = ["FNOConv2d"]


class FreqLinear(nn.Module):
    def __init__(self, in_channel, modes1, modes2, bias=True):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channel + 4 * modes1 * modes2)
        self.weights = nn.Parameter(scale * torch.randn(in_channel, 2 * modes1 * modes2, dtype=torch.cfloat))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, 2 * modes1 * modes2, dtype=torch.cfloat))
        else:
            self.bias = 0

    def forward(self, x):
        B = x.shape[0]
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, self.modes2)
        return h

class FNOConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int, 
        n_modes: Tuple[int],
        cond_bias: bool,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        https://arxiv.org/pdf/2010.08895.pdf
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_mode_1, self.n_mode_2 = n_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.device = device
        
        self.cond_emb = FreqLinear(cond_channels, self.n_mode_1, self.n_mode_2, bias=cond_bias)
        
        self.scale = 1 / (in_channels * out_channels)
        self.build_parameters()
        self.reset_parameters()

    def build_parameters(self) -> None:
        self.weight_1 = nn.Parameter(
            self.scale * torch.zeros([self.in_channels, self.out_channels, *self.n_modes], dtype=torch.cfloat)
        )
        self.weight_2 = nn.Parameter(
            self.scale * torch.zeros([self.in_channels, self.out_channels, *self.n_modes], dtype=torch.cfloat)
        )

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_1.real)
        nn.init.kaiming_normal_(self.weight_2.real)

    def get_zero_padding(self, size, device):
        bs, h, w = size[0], size[-2], size[-1] // 2 + 1
        return torch.zeros(bs, self.out_channels, h, w, dtype=torch.cfloat, device=device)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # rfft2 : FFT of real signal -> Hemiltonian symmetry -> X_ft[i] = conj(X_ft[-i])
        # shape of x_ft = (BS, C, H, W//2+1)
        
        cond_emb = self.cond_emb(emb)
        emb1 = cond_emb[..., 0]
        emb2 = cond_emb[..., 1]
        batch_size = x.shape[0]
        
        
        x_ft = torch.fft.rfft2(x, norm="ortho")
        
        # Multiply relevant Fourier modes
        out_ft = self.get_zero_padding(x.size(), x.device)

        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # 얼마나의 frequency를 고려할 것인지. 1. : n_mode_1, : n_mode_2 / 2. -n_mode :, : n_mode 
        n_mode_1 = min(out_ft.size(-2)//2, self.n_mode_1)
        n_mode_2 = min(out_ft.size(-1), self.n_mode_2)

        out_ft[..., : n_mode_1, : n_mode_2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[..., : n_mode_1, : n_mode_2]*emb1.unsqueeze(1), self.weight_1
        )
        out_ft[:, :, -n_mode_1 :, : n_mode_2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -n_mode_1 :, : n_mode_2]*emb2.unsqueeze(1), self.weight_2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x

