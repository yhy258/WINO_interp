"""
    Largely based on https://github.com/JeremieMelo/NeurOLight/blob/main/core/models/factorfno_cnn.py
    NeurOLight: A Physics-Agnostic Neural Operator Enabling Parametric Photonic Device Simulation
    MIT License
    Copyright (c) 2022 Jiaqi Gu
"""

from typing import List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pyutils.activation import nn.SiLU
from torch import nn
from timm.models.layers import DropPath, to_2tuple
from torch.functional import Tensor
from torch.types import Device, _size
from torch.utils.checkpoint import checkpoint
# from pyutils.torch_train import set_torch_deterministic
from .layers.factorfno_conv2d import FactorFNOConv2d
from .layers.fno_conv2d import FNOConv2d

__all__ = ["FactorFNO2d"]


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

def get_act(act='ReLU'): # Default : ReLU()
    
    if act.lower() == "swish":
        return nn.SiLU()
    else:
        return getattr(nn, act)()

class FreqLinear(nn.Module):
    def __init__(self, in_channel, mode, bias=True):
        super().__init__()
        self.mode = mode
        scale = 1 / (in_channel + 4 * mode)
        self.weights = nn.Parameter(scale * torch.randn(in_channel, 2 * mode, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, 2 * mode, dtype=torch.float32))
        else:
            self.bias = 0

    def forward(self, x):
        B = x.shape[0]
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.mode, 2)
        return torch.view_as_complex(h)

class CondFactorFNOConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
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
        
        self.cond_emb1 = FreqLinear(cond_channels, self.n_mode_1)
        self.cond_emb2 = FreqLinear(cond_channels, self.n_mode_2)
        
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

    def _factorfno_forward(self, x, emb, dim = -2):
        if dim == -2:
            x_ft = torch.fft.rfft(x, norm="ortho", dim=-2)
            n_mode = self.n_mode_1
            if n_mode == x_ft.size(-2): # full mode
                out_ft = torch.einsum("bixy,iox->boxy", x_ft*emb[:, None, :, None], self.weight_1)
            else:
                out_ft = self.get_zero_padding([x.size(0), self.weight_1.size(1), x_ft.size(-2), x_ft.size(-1)], x.device)
 
                out_ft[..., : n_mode, :] = torch.einsum(
                "bixy,iox->boxy", x_ft[..., : n_mode, :]*emb[:, None, :, None], self.weight_1
                )
            x = torch.fft.irfft(out_ft, n=x.size(-2), dim=-2, norm="ortho")
        elif dim == -1:
            x_ft = torch.fft.rfft(x, norm="ortho", dim=-1)
            n_mode = self.n_mode_2
            if n_mode == x_ft.size(-1):
                out_ft = torch.einsum("bixy,ioy->boxy", x_ft*emb[:, None, None, :], self.weight_2)
            else:
                out_ft = self.get_zero_padding([x.size(0), self.weight_2.size(1), x_ft.size(-2), x_ft.size(-1)], x.device)

                out_ft[..., :n_mode] = torch.einsum(
                "bixy,ioy->boxy", x_ft[..., : n_mode]*emb[:, None, None, :], self.weight_2
                )
            x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1, norm="ortho")
        return x

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # axis -1, -2 기준 xx xy 구하기. 
        cond_emb1 = self.cond_emb1(emb)
        cond_emb2 = self.cond_emb2(emb)
        xx = self._factorfno_forward(x, cond_emb1, dim=-1)
        xy = self._factorfno_forward(x, cond_emb2, dim=-2)
        return xx + xy



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


class BSConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [(dilation[i] * (kernel_size[i] - 1) + 1) // 2 for i in range(len(kernel_size))]
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=bias,
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class ResStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [(dilation[i] * (kernel_size[i] - 1) + 1) // 2 for i in range(len(kernel_size))]

        self.conv1 = BSConv2d(
            in_channels,
            out_channels // 2,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = BSConv2d(
            out_channels // 2,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x



class FactorFNO2dBlock(nn.Module):
    expansion = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        n_modes: Tuple[int],
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "GELU",
        wp_mult: bool = False,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        freqlinear: bool = False,
        with_cp=False,
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.wp_mult = wp_mult
        self.with_cp = with_cp
        # self.norm.weight.data.zero_()
        self.ff = nn.Sequential(
            nn.Linear(out_channels, out_channels * self.expansion),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_channels * self.expansion),
            nn.Linear(out_channels * self.expansion, out_channels),
        )
        self.freqlinear = freqlinear
        
        if wp_mult:
            self.wp_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.post_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        
        if freqlinear:
            self.f_conv = CondFactorFNOConv2d(in_channels, out_channels, cond_channels, n_modes, device=device)
        else:
            self.f_conv = FactorFNOConv2d(in_channels, out_channels, n_modes, device=device)
    def forward(self, x: Tensor, emb: Tensor = None) -> Tensor:
        def _inner_forward(x):
            y = x
            b, inc, h, w = x.shape
            if self.freqlinear:
                x = self.f_conv(x, emb).permute(0, 2, 3, 1).flatten(1, 2)
            else:
                x = self.f_conv(x).permute(0, 2, 3, 1).flatten(1, 2)
            x = self.ff(x).reshape(b, h, w, -1).permute(0, 3, 1, 2)
            
            if self.wp_mult:
                x = self.post_conv(x*self.wp_conv(emb))
            
            x = x + y
            return x

        if x.requires_grad and self.with_cp:
            return checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)


class FactorFNO2d(nn.Module):
    _conv = (FNOConv2d,)
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        dim: int = 16,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [64],
        act_func: Optional[str] = "GELU",
        mode_list: List[Tuple[int]] = [(20, 20), (20, 20), (20, 20), (20, 20)],
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        eps_min: float = 1.0,
        eps_max: float = 1.46**2,
        wave_prior: bool = False,
        wp_mult: bool = False,
        positional_encoding: bool = False,
        eps_info: bool = False,
        freqlinear: bool = False,
        device: Device = torch.device('cuda:0')
    ):
        super().__init__()
        if len(mode_list) == 2:
            mode_list = [mode_list for i in range(len(kernel_list))]
        if wave_prior and not wp_mult:
            in_channels += 4
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
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.wave_prior = wave_prior
        self.wp_mult = wp_mult
        self.device = device
        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.in_channels += 2
        self.eps_info = eps_info
        self.freqlinear = freqlinear
        
    
        self.build_layers()
        self.reset_parameters()

    def _pos_enc(self, shape, device, normalize=False):
        B, C, W, H = shape
        gridx = torch.arange(0, W, device=device, dtype=torch.float)
        gridy = torch.arange(0, H, device=device, dtype=torch.float)
        if normalize:
            gridx /= W
            gridy /= H
        gridx, gridy = torch.meshgrid(gridx, gridy)
        mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
        return mesh
        
    def encoding(self, eps, dl, wvl, device):
        B, C, W, H = eps.shape
        mesh = self._pos_enc(eps.shape, device)
        
        if self.eps_info:
            mesh = torch.view_as_real(
                torch.exp(
                    mesh.mul(dl.div(wvl).mul(1j*2*np.pi)[..., None, None]).mul(eps.data.sqrt())
                )
            )
        else:
            mesh = torch.view_as_real(
                torch.exp(
                    mesh.mul(dl.div(wvl).mul(1j*2*np.pi)[..., None, None])
                )
            )

        mesh = torch.tensor(mesh.permute(0, 1, 4, 2, 3).flatten(1, 2), dtype=torch.float32) # bs, 4, h, w
        return (mesh+1)/2
    
    def frequency_encoding(self, eps, dl, wvl, device):
        mesh = self._pos_enc(eps.shape, device, normalize=True) # 1, 2, H, W
        _, _, H, W = mesh.shape
        frequency_meshes = torch.arange(1, max(H, W)//2, device=device)[None, :, None, None] # (1, K, 1, 1)
        _, K, _, _ = frequency_meshes.shape
        phase_shift = torch.rand(K, device=device)[None, :, None, None]* torch.pi
        freq_component = mesh[:, None, :, :, :].mul(frequency_meshes[:, :, None, :, :]).mul(2*torch.pi)
        cos_freqs = torch.cos(freq_component + phase_shift[:, :, None, :, :])
        sin_freqs = torch.sin(freq_component + phase_shift[:, :, None, :, :])
        freqs = torch.stack([cos_freqs, sin_freqs], dim=-1)
        freqs = (torch.sum(freqs, dim=1) + K) / (2*K)
        return freqs.permute(0, 1, 4, 2, 3).flatten(1, 2)
    
    def reset_parameters(self, random_state: Optional[int] = None):
        for name, m in self.named_modules():
            if isinstance(m, self._conv):
                # if random_state is not None:
                #     # deterministic seed, but different for different layer, and controllable by random_state
                #     set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
                
    def build_layers(self):
        
        if self.act_func is None:
            act = None
        elif self.act_func.lower() == "swish":
            act = nn.SiLU()
        else:
            act = getattr(nn, self.act_func)()
            
        
        self.stem = nn.Conv2d(
            self.in_channels,
            self.dim,
            1,
            padding=0
        )
        
        self.condition_dim = 0
        if self.freqlinear:
            self.condition_dim = self.dim * 4
            self.cond_embed = nn.Sequential(
                nn.Linear(self.dim, self.condition_dim),
                act,
                nn.Linear(self.condition_dim, self.condition_dim)
            )
        if self.wp_mult:
            self.wpa_stem = nn.Sequential(
                nn.Conv2d(4, self.dim, 1, padding=0),
                act
            )
        
        kernel_list = [self.dim] + self.kernel_list

        drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))

        features = [
            FactorFNO2dBlock(
                inc,
                outc,
                self.condition_dim,
                n_modes,
                kernel_size,
                padding,
                act_func=self.act_func,
                wp_mult=self.wp_mult,
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

        if self.positional_encoding:
            grid1 = self._pos_enc(x.shape, x.device, True)
            x = torch.cat((x, grid1.repeat(repeats=[x.shape[0], 1, 1, 1])), dim=1)
        
        if self.wave_prior:
            grid = self.encoding(eps, dl, wavelength, x.device)
            if self.wp_mult:
                wpa_embed = self.wpa_stem(grid)
            else:
                x = torch.cat((x, grid), dim=1)
        
        x = self.stem(x)
        
        if self.freqlinear:
            wvl = (wavelength - 0.4) / (0.7 - 0.4) # wvl : 0 ~ 1 scaling 해서 넣어주는게 좋음.
            emb = fourier_embedding(wvl, dim=self.dim)[:, 0, :] # wvl : N, 1 -> N, d
            emb = self.cond_embed(emb)
            for b in self.features:
                x = b(x, emb)
        else:
            if self.wp_mult:
                for b in self.features:
                    x = b(x, wpa_embed)
            else:
                x = self.features(x)
        x = self.head(x)
        return x