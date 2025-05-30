"""
    Largely based on https://github.com/JeremieMelo/NeurOLight/blob/main/core/models/layers/factorfno_conv2d.py
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

__all__ = ["FactorFNOConv2d"]


class FactorFNOConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_mode_1, self.n_mode_2 = n_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.device = device

        self.scale = 1 / (in_channels * out_channels)
        self.build_parameters()
        self.reset_parameters()

    def build_parameters(self) -> None:
        self.weight_1 = nn.Parameter(
            self.scale * torch.zeros([self.in_channels, self.out_channels, self.n_modes[0]], dtype=torch.cfloat)
        )
        self.weight_2 = nn.Parameter(
            self.scale * torch.zeros([self.in_channels, self.out_channels, self.n_modes[1]], dtype=torch.cfloat)
        )

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_1.real)
        nn.init.kaiming_normal_(self.weight_2.real)

    def get_zero_padding(self, size, device):
        return torch.zeros(*size, dtype=torch.cfloat, device=device)

    def _factorfno_forward(self, x, dim = -2):
        if dim == -2:
            x_ft = torch.fft.rfft(x, norm="ortho", dim=-2)
            n_mode = self.n_mode_1
            if n_mode == x_ft.size(-2): # full mode
                out_ft = torch.einsum("bixy,iox->boxy", x_ft, self.weight_1)
            else:
                out_ft = self.get_zero_padding([x.size(0), self.weight_1.size(1), x_ft.size(-2), x_ft.size(-1)], x.device)
 
                out_ft[..., : n_mode, :] = torch.einsum(
                "bixy,iox->boxy", x_ft[..., : n_mode, :], self.weight_1
                )
            x = torch.fft.irfft(out_ft, n=x.size(-2), dim=-2, norm="ortho")
        elif dim == -1:
            x_ft = torch.fft.rfft(x, norm="ortho", dim=-1)
            n_mode = self.n_mode_2
            if n_mode == x_ft.size(-1):
                out_ft = torch.einsum("bixy,ioy->boxy", x_ft, self.weight_2)
            else:
                out_ft = self.get_zero_padding([x.size(0), self.weight_2.size(1), x_ft.size(-2), x_ft.size(-1)], x.device)

                out_ft[..., :n_mode] = torch.einsum(
                "bixy,ioy->boxy", x_ft[..., : n_mode], self.weight_2
                )
            x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1, norm="ortho")
        return x

    def forward(self, x: Tensor) -> Tensor:
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # axis -1, -2 기준 xx xy 구하기. 
        xx = self._factorfno_forward(x, dim=-1)
        xy = self._factorfno_forward(x, dim=-2)
        return xx + xy
