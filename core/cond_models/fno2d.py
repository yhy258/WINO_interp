"""
    Largely based on https://github.com/JeremieMelo/NeurOLight/blob/main/core/models/fno_cnn.py
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
from timm.models.layers import DropPath
from torch import nn
from torch.functional import Tensor
from torch.types import Device
# from pyutils.torch_train import set_torch_deterministic

__all__ = ["FNO2d"]

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
    def __init__(self, in_channel, modes1, modes2, bias=True):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channel + 4 * modes1 * modes2)
        self.weights = nn.Parameter(scale * torch.zeros(in_channel, 4 * modes1 * modes2, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, 4 * modes1 * modes2, dtype=torch.float32))
        else:
            self.bias = 0

    def forward(self, x):
        B = x.shape[0]
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, self.modes2, 2, 2)
        return torch.view_as_complex(h)
    
class FNOConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int, 
        n_modes: Tuple[int],
        cond_bias: bool = True,
        sparsity_threshold: float = 0.0,
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
        if self.sparsity_threshold > 0 :
            self.soft_shrink = nn.Softshrink(lambd=self.sparsity_threshold)

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
        # print(n_mode_1, n_mode_2)
        out_ft[..., : n_mode_1, : n_mode_2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[..., : n_mode_1, : n_mode_2]*emb1.unsqueeze(1), self.weight_1
        )
        out_ft[:, :, -n_mode_1 :, : n_mode_2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -n_mode_1 :, : n_mode_2]*emb2.unsqueeze(1), self.weight_2
        )
        
        if self.sparsity_threshold > 0 :
            out_ft = torch.stack([out_ft.real, out_ft.imag], dim=-1)
            out_ft = self.soft_shrink(out_ft)
            out_ft = out_ft[..., 0] + 1j*out_ft[..., 1]

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x


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
            return nn.SiLU()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x


class FNO2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        n_modes: Tuple[int],
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "GELU",
        drop_path_rate: float = 0.0,
        sparsity_threshold: float = 0.0,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.f_conv = FNOConv2d(in_channels, out_channels, cond_channels, n_modes, sparsity_threshold, device=device)
        self.norm = nn.BatchNorm2d(out_channels)
        # self.norm.weight.data.zero_()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "swish":
            self.act_func = nn.SiLU()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        x = self.norm(self.conv(x) + self.drop_path(self.f_conv(x, emb)))

        if self.act_func is not None:
            x = self.act_func(x)
        return x


class FNO2d(nn.Module):
    _conv = (FNOConv2d,)
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        dim: int = 16,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list : List[int] = [1, 1, 1, 1],
        padding: int = 8,
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        act_func: Optional[str] = "ReLU",
        mode_list: List[Tuple[int]] = [(20, 20), (20, 20), (20, 20), (20, 20)],
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        sparsity_threshold: float = 0.0,
        device: Device = torch.device('cuda:0')
    ):
        super().__init__()
        
        
        if len(mode_list) == 2:
            mode_list = [mode_list for i in range(len(kernel_list))]
        
        self.in_channels = in_channels + 2
        self.dim = dim
        self.out_channels = out_channels
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.padding = padding
        self.padding_list = padding_list
        self.hidden_list = hidden_list
        self.act_func = act_func
        self.mode_list = mode_list
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.sparsity_threshold = sparsity_threshold
        self.device = device
        self.act = get_act(self.act_func)
        
        self.condition_dim = self.dim * 4
        self.cond_embed = nn.Sequential(
            nn.Linear(dim, self.condition_dim),
            self.act,
            nn.Linear(self.condition_dim, self.condition_dim)
        )
        
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
        
    def reset_parameters(self, random_state: Optional[int] = None):
        for name, m in self.named_modules():
            if isinstance(m, self._conv):
                
                m.reset_parameters()
    
    def build_layers(self):
        self.stem = nn.Conv2d(
            self.in_channels,
            self.dim,
            1,
            padding=0
        )
        
        kernel_list = [self.dim] + self.kernel_list
        drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))
        
        features = [
            FNO2dBlock(
                inc,
                outc,
                self.condition_dim,
                n_modes,
                kernel_size,
                padding,
                sparsity_threshold=self.sparsity_threshold,
                device=self.device,
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
        
        self.features = nn.ModuleList(features)
        hidden_list = [self.kernel_list[-1]] + self.hidden_list
        head = [
            nn.Sequential(
                ConvBlock(inc, outc, kernel_size=1, padding=0, act_func=self.act_func, device=self.device),
                nn.Dropout2d(self.dropout),
            )
            for inc, outc in zip(hidden_list[:-1], hidden_list[1:])
        ]
        
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
        
    def forward(self, x: Tensor, wvl: Tensor, dl: Tensor):
        # z : 0.4 - 0.7
        wvl = (wvl - 0.4) / (0.7 - 0.4) # Normalization
        
        """
            if we have time factor and condition factor... ->
            emb = self.time_embed(fourier_embedding(time, self.hidden_channels))
            if z is not None:
                if self.param_conditioning == "scalar":
                    emb = emb + self.pde_emb(fourier_embedding(z, self.hidden_channels))
        """
        
        emb = fourier_embedding(wvl, dim=self.dim)[:, 0, :]
        emb = self.cond_embed(emb)
        
        grid1 = self._pos_enc(x.shape, x.device, True)
        x = torch.cat([x, grid1.repeat(repeats=[x.shape[0], 1, 1, 1])], dim=1)
        
        x = self.stem(x)
        x = F.pad(x, [0, self.padding, 0, self.padding])
        
        for b in self.features:
            x = b(x, emb)
            
        x = x[..., :-self.padding, :-self.padding]
        x = self.head(x)
        return x # BS, 2, H, W