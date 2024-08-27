"""
    Largely based on https://github.com/JeremieMelo/NeurOLight/blob/main/core/models/neurolight_cnn.py
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
from pyutils.activation import Swish
from timm.models.layers import DropPath, to_2tuple
from torch import nn
from torch.functional import Tensor
from torch.types import Device
from pyutils.torch_train import set_torch_deterministic
from .layers.fno_conv2d import FNOConv2d



__all__ = ["NeurOLight2d"]

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
        kernel_size = 3,
        stride = 1,
        dilation = 1,
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
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x
    

class NeurOLightConv2d(nn.Module):
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
            self.scale
            * torch.zeros([self.in_channels // 2, self.out_channels // 2, self.n_modes[0]], dtype=torch.cfloat)
        )
        self.weight_2 = nn.Parameter(
            self.scale
            * torch.zeros(
                [self.in_channels - self.in_channels // 2, self.out_channels - self.out_channels // 2, self.n_modes[1]],
                dtype=torch.cfloat,
            )
        )

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_1.real)
        nn.init.kaiming_normal_(self.weight_2.real)

    def get_zero_padding(self, size, device):
        return torch.zeros(*size, dtype=torch.cfloat, device=device)

    def _neurolight_forward(self, x, dim=-2):
        if dim == -2:
            x_ft = torch.fft.rfft(x, norm="ortho", dim=-2)
            n_mode = self.n_mode_1
            if n_mode == x_ft.size(-2):  # full mode
                out_ft = torch.einsum("bixy,iox->boxy", x_ft, self.weight_1)
            else:
                out_ft = self.get_zero_padding(
                    [x.size(0), self.weight_1.size(1), x_ft.size(-2), x_ft.size(-1)], x.device
                )
                out_ft[..., :n_mode, :] = torch.einsum("bixy,iox->boxy", x_ft[..., :n_mode, :], self.weight_1)
            x = torch.fft.irfft(out_ft, n=x.size(-2), dim=-2, norm="ortho")
        elif dim == -1:
            x_ft = torch.fft.rfft(x, norm="ortho", dim=-1)
            n_mode = self.n_mode_2
            if n_mode == x_ft.size(-1):
                out_ft = torch.einsum("bixy,ioy->boxy", x_ft, self.weight_2)
            else:
                out_ft = self.get_zero_padding(
                    [x.size(0), self.weight_2.size(1), x_ft.size(-2), x_ft.size(-1)], x.device
                )
                out_ft[..., :n_mode] = torch.einsum("bixy,ioy->boxy", x_ft[..., :n_mode], self.weight_2)
            x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1, norm="ortho")
        return x

    def forward(self, x: Tensor) -> Tensor:
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        xx, xy = x.chunk(2, dim=1)
        xx = self._neurolight_forward(xx, dim=-1)
        xy = self._neurolight_forward(xy, dim=-2)
        x = torch.cat([xx, xy], dim=1)
        return x
    
class NeurOLight2dBlock(nn.Module):
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
        ffn: bool = True,
        ffn_dwconv: bool = True,
        aug_path: bool = True,
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.f_conv = NeurOLightConv2d(in_channels, out_channels, n_modes, device=device)
        self.pre_norm = nn.BatchNorm2d(in_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.with_cp = with_cp
        # self.norm.weight.data.zero_()
        if ffn:
            if ffn_dwconv:
                self.ff = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels * self.expansion, 1),
                    nn.Conv2d(
                        out_channels * self.expansion,
                        out_channels * self.expansion,
                        3,
                        groups=out_channels * self.expansion,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels * self.expansion),
                    nn.GELU(),
                    nn.Conv2d(out_channels * self.expansion, out_channels, 1),
                )
            else:
                self.ff = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels * self.expansion, 1),
                    nn.BatchNorm2d(out_channels * self.expansion),
                    nn.GELU(),
                    nn.Conv2d(out_channels * self.expansion, out_channels, 1),
                )
        else:
            self.ff = None
        if aug_path:
            self.aug_path = nn.Sequential(BSConv2d(in_channels, out_channels, 3), nn.GELU())
        else:
            self.aug_path = None
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        def _inner_forward(x):
            y = x
            if self.ff is not None:
                x = self.norm(self.ff(self.pre_norm(self.f_conv(x))))
                x = self.drop_path(x) + y
            else:
                x = self.act_func(self.drop_path(self.norm(self.f_conv(x))) + y)
            if self.aug_path is not None:
                x = x + self.aug_path(y)
            return x

        return _inner_forward(x)

    
    
class NeurOLight2d(nn.Module):
    _conv = (FNOConv2d,)
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        dim: int = 16,
        padding: int = 8,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list : List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        act_func: Optional[str] = "ReLU",
        mode_list: List[Tuple[int]] = [(20, 20), (20, 20), (20, 20), (20, 20)],
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        eps_min: float = 1,
        eps_max: float = 1.46**2,
        res_stem: bool = True,
        device: Device = torch.device('cuda:0')
    ):
        super().__init__()
        
        
        if len(mode_list) == 2:
            mode_list = [mode_list for i in range(len(kernel_list))]
        
        self.in_channels = in_channels + 4
        
        self.dim = dim
        self.out_channels = out_channels
        self.padding = padding
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.hidden_list = hidden_list
        self.act_func = act_func
        self.mode_list = mode_list
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.res_stem = res_stem
        self.device = device

        
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
        
        mesh = torch.view_as_real(
            torch.exp(
                mesh.mul(dl.div(wvl).mul(1j*2*np.pi)[..., None, None]).mul(eps.data.sqrt())
            )
        )

        mesh = torch.tensor(mesh.permute(0, 1, 4, 2, 3).flatten(1, 2), dtype=torch.float32) # bs, 4, h, w
        return (mesh+1)/2
    
    def reset_parameters(self, random_state: Optional[int] = None):
        for name, m in self.named_modules():
            if isinstance(m, self._conv):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()

    def build_layers(self):
        if self.act_func is None:
            act = None
        elif self.act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            act = getattr(nn, self.act_func)()
        
        if self.res_stem:
            self.stem = ResStem(
                self.in_channels,
                self.dim,
                kernel_size=3,
            )
        else:
            self.stem = nn.Conv2d(
                self.in_channels,
                self.dim,
                1,
                padding=0
            )
                

        kernel_list = [self.dim] + self.kernel_list
        drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))
        
        # FNO2d Block -> NeurOLight block으로 수정.
        features = [
            NeurOLight2dBlock(
                inc,
                outc,
                n_modes,
                kernel_size,
                padding,
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
        
        self.features = nn.Sequential(*features)

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
        eps = x * (self.eps_max - self.eps_min) + self.eps_min  # [bs, inc*2, h, w] real

        wave = self.encoding(eps, dl, wvl, x.device)
       
        x = torch.cat([x, wave], dim=1)
        
        x = self.stem(x)
                    
        
        x = F.pad(x, [0, self.padding, 0, self.padding])
                
        x = self.features(x)
        x = x[..., :-self.padding, :-self.padding]
        x = self.head(x)
        
        return x # BS, 2, H, W'
        