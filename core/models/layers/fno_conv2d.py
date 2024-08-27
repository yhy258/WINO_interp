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


class FNOConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        sparsity_threshold: float =0,
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
        self.sparsity_threshold = sparsity_threshold
        
        self.device = device

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
        if self.sparsity_threshold > 0:
            self.soft_shrink = nn.Softshrink(lambd=self.sparsity_threshold)

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_1.real)
        nn.init.kaiming_normal_(self.weight_2.real)

    def get_zero_padding(self, size, device):
        bs, h, w = size[0], size[-2], size[-1] // 2 + 1
        return torch.zeros(bs, self.out_channels, h, w, dtype=torch.cfloat, device=device)

    def forward(self, x: Tensor, emb: Tensor=None) -> Tensor:
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # rfft2 : FFT of real signal -> Hemiltonian symmetry -> X_ft[i] = conj(X_ft[-i])
        # shape of x_ft = (BS, C, H, W//2+1)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        
        # Multiply relevant Fourier modes
        out_ft = self.get_zero_padding(x.size(), x.device)

        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # 얼마나의 frequency를 고려할 것인지. 1. : n_mode_1, : n_mode_2 / 2. -n_mode :, : n_mode 
        n_mode_1 = min(out_ft.size(-2)//2, self.n_mode_1)
        n_mode_2 = min(out_ft.size(-1), self.n_mode_2)

        out_ft[..., : n_mode_1, : n_mode_2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[..., : n_mode_1, : n_mode_2], self.weight_1
        )
        out_ft[:, :, -n_mode_1 :, : n_mode_2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -n_mode_1 :, : n_mode_2], self.weight_2
        )

        
        if self.sparsity_threshold > 0 :
            out_ft = torch.stack([out_ft.real, out_ft.imag], dim=-1)
            out_ft = self.soft_shrink(out_ft)
            out_ft = out_ft[..., 0] + 1j*out_ft[..., 1]
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x, out_ft

