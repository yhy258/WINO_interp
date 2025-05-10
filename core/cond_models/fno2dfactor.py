"""
    Largely based on https://github.com/JeremieMelo/NeurOLight/blob/main/core/models/factorfno_cnn.py
    NeurOLight: A Physics-Agnostic Neural Operator Enabling Parametric Photonic Device Simulation
    MIT License
    Copyright (c) 2022 Jiaqi Gu
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pyutils.activation import Swish
from torch import nn
from torch.functional import Tensor
from torch.types import Device, _size
from torch.utils.checkpoint import checkpoint
# from pyutils.torch_train import set_torch_deterministic
from .layers.factorfno_conv2d import FactorFNOConv2d

__all__ = ["FactorFNO2d"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "GELU",
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "swish":
            self.act_func = nn.SiLU()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x


class FactorFNO2dBlock(nn.Module):
    expansion = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "GELU",
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        with_cp=False,
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.f_conv = FactorFNOConv2d(in_channels, out_channels, n_modes, device=device)
        self.with_cp = with_cp
        # self.norm.weight.data.zero_()
        self.ff = nn.Sequential(
            nn.Linear(out_channels, out_channels * self.expansion),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_channels * self.expansion),
            nn.Linear(out_channels * self.expansion, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        def _inner_forward(x):
            y = x
            b, inc, h, w = x.shape
            x = self.f_conv(x).permute(0, 2, 3, 1).flatten(1, 2)
            x = self.ff(x).reshape(b, h, w, -1).permute(0, 3, 1, 2)
            x = x + y
            return x

        if x.requires_grad and self.with_cp:
            return checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)


class FactorFNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 1+4,
        out_channels: int = 2,
        dim: int = 16,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        act_func: Optional[str] = "GELU",
        mode_list: List[Tuple[int]] = [(20, 20), (20, 20), (20, 20), (20, 20)],
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device('cuda:0')
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % 2 == 0, f"The output channels must be even number larger than 2, but got {out_channels}"
        self.dim = dim
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.hidden_list = hidden_list
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.mode_list = mode_list
        self.device = device
    
        self.build_layers()
        self.reset_parameters()

    def _pos_enc(self, shape, device):
        B, C, W, H = shape
        gridx = torch.arange(0, W, device=device)
        gridy = torch.arange(0, H, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy)
        mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
        return mesh
        
    def encoding(self, eps, dl, wvl, device):
        mesh = self._pos_enc(eps.shape, device)
        mesh = torch.view_as_real(
            torch.exp(
                mesh.mul(dl.div(wvl).mul(1j*2*np.pi)[..., None, None]).mul(eps.data.sqrt())
            )
        )
        return mesh.permute(0, 1, 4, 2, 3).flatten(1, 2) # bs, 4, h, w
        
    def reset_parameters(self, random_state: Optional[int] = None):
        for name, m in self.named_modules():
            if isinstance(m, self._conv):
                m.reset_parameters()
                
    def build_layers(self):
        self.stem = nn.Conv2d(self.in_channels, self.dim, 1)
        kernel_list = [self.dim] + self.kernel_list
        drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))

        features = [
            FactorFNO2dBlock(
                inc,
                outc,
                n_modes,
                kernel_size,
                padding,
                act_func=self.act_func,
                drop_path_rate=drop,
                device=self.device,
                with_cp=False,
            )
            for inc, outc, n_modes, kernel_size, padding, drop in zip(
                kernel_list[:-1],
                kernel_list[1:],
                self.mode_list,
                self.kernel_size_list,
                self.padding_list,
                drop_path_rates,
            )
        ]
        self.features = nn.Sequential(*features)
        hidden_list = [self.kernel_list[-1]] + self.hidden_list
        head = [
            nn.Sequential(
                ConvBlock(inc, outc, kernel_size=1, padding=0, act_func=None, device=self.device),
                nn.Dropout2d(self.dropout_rate),
            )
            for inc, outc in zip(hidden_list[:-1], hidden_list[1:])
        ]
        # 2 channels as real and imag part of the TE field
        head += [
            ConvBlock(
                hidden_list[-1],
                self.out_channels,
                kernel_size=1,
                padding=0,
                act_func=None,
                device=self.device,
            )
        ]

        self.head = nn.Sequential(*head)
        
    def forward(self, x: Tensor, wavelength: Tensor, dl: Tensor):
        # x [bs, inc, h, w] complex
        # wavelength [bs, 1] real
        # dl [bs, 2] real
        
        eps = x * (self.eps_max - self.eps_min) + self.eps_min
        
        grid = self.encoding(eps, dl, wavelength, x.device)
        
        x = torch.cat((x, grid), dim=1)
        
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = torch.view_as_complex(
            x.view(x.size(0), -1, 2, x.size(-2), x.size(-1)).permute(0, 1, 3, 4, 2).contiguous()
        )  # [bs, outc/2, h, w] complex
        return x