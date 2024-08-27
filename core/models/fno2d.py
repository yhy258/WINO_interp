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
from pyutils.activation import Swish
from timm.models.layers import DropPath, to_2tuple
from torch import nn
from torch.functional import Tensor
from torch.types import Device
from pyutils.torch_train import set_torch_deterministic
from .layers.fno_conv2d import FNOConv2d


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

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class GLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.layer2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.layer3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        return self.layer3(x1*x2)

class FreqLinear(nn.Module):
    def __init__(self, in_channel, modes1, modes2, bias=True):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channel + 4 * modes1 * modes2)
        self.weights = nn.Parameter(scale * torch.randn(in_channel, 4 * modes1 * modes2, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, 4 * modes1 * modes2, dtype=torch.float32))
        else:
            self.bias = 0

    def forward(self, x):
        B = x.shape[0]
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, self.modes2, 2, 2)
        return torch.view_as_complex(h)

class ConditionalCrossChannelAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        out_channels: int,
        act: nn.Module
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=cond_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.act = act
        self.conv_last = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1)
        
    def forward(self, x, emb):
        res = x
        x = self.conv1(x)
        emb = self.conv2(emb)
        x = x * self.act(emb)
        x = self.conv_last(x)
        return res + x
        
    
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
        act_func: str = "GELU",
    ) -> None:
        super().__init__()
        
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [(dilation[i] * (kernel_size[i] - 1) + 1) // 2 for i in range(len(kernel_size))]
        
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()
        
        self.conv1 = BSConv2d(
            in_channels,
            out_channels // 2,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2)

        self.conv2 = BSConv2d(
            out_channels // 2,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act_func(self.bn1(self.conv1(x)))
        x = self.act_func(self.bn2(self.conv2(x)))
        return x
    
class CondFNOConv2d(nn.Module):
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
        return x, out_ft


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
        freqlinear: bool = False,
        waveprior_addition: bool = False,
        waveprior_concatenate: bool = False,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.freqlinear = freqlinear
        self.waveprior_addition = waveprior_addition
        self.waveprior_concatenate = waveprior_concatenate
        

        if freqlinear:
            waveprior_bool = False
            self.f_conv = CondFNOConv2d(in_channels, out_channels, cond_channels, n_modes, device=device)
        else:
            self.f_conv = FNOConv2d(in_channels, out_channels, n_modes, device=device)
        # self.norm = nn.BatchNorm2d(out_channels)
        # self.norm.weight.data.zero_()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    
        if waveprior_addition or waveprior_concatenate:
                self.wpconv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()
        if self.waveprior_concatenate:
            self.wp_act = SimpleGate()
            self.post_conv = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x: Tensor, emb: Tensor = None) -> Tensor:
        # print(x.shape)
        if self.freqlinear:
            out, mode = self.f_conv(x, emb)
            x = self.conv(x) + self.drop_path(out)
            x = self.act_func(x)
        else:
            out, mode = self.f_conv(x)
            x = self.conv(x) + self.drop_path(out)
            x = self.act_func(x)
            if self.waveprior_addition:
                x = x + self.wpconv(emb)
            if self.waveprior_concatenate:
                x = self.post_conv(x * self.wpconv(emb))
        return x

    
    
class FNO2d(nn.Module):
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
        wave_prior: bool = True,
        wp_input_cat: bool = True,
        freqlinear: bool = False,
        waveprior_addition: bool = False,
        waveprior_concatenate: bool = False,
        positional_encoding: bool = True,
        eps_info: bool = False,
        device: Device = torch.device('cuda:0')
    ):
        super().__init__()
        
        
        if len(mode_list) == 2:
            mode_list = [mode_list for i in range(len(kernel_list))]

        if positional_encoding:
            self.in_channels = in_channels+2
        else:
            self.in_channels = in_channels
        
        if wave_prior:
            self.in_channels += 4
                
            
        self.wave_prior = wave_prior
        self.wp_input_cat = wp_input_cat
        
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
        self.waveprior_addition = waveprior_addition
        self.waveprior_concatenate = waveprior_concatenate
        if (waveprior_addition == False) and (waveprior_concatenate == False):
            self.wp_input_cat = True
        if self.wp_input_cat == False:
            if self.wave_prior:
                self.in_channels -= 4

            
        self.freqlinear = freqlinear

        self.positional_encoding = positional_encoding
        self.eps_info = eps_info
        
        
        
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
            
        
            self.stem = nn.Conv2d(
                self.in_channels,
                self.dim,
                1,
                padding=0
            )
            if self.waveprior_addition or self.waveprior_concatenate:
                self.wpa_stem = nn.Sequential(
                    nn.Conv2d(4, self.dim, 1, padding=0),
                    act
                )


        if self.freqlinear:

            self.condition_dim = self.dim * 4
            self.cond_embed = nn.Sequential(
                nn.Linear(self.dim, self.condition_dim),
                act,
                nn.Linear(self.condition_dim, self.condition_dim)
            )
        else:
            self.condition_dim= 0
        

        kernel_list = [self.dim] + self.kernel_list
        drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))
        waveaddition_bools = [self.waveprior_addition] * len(self.mode_list)
        waveconcatenate_bools = [self.waveprior_concatenate] * len(self.mode_list)

        
        if self.freqlinear:
            freqlinear = True
        else:
            freqlinear = False
        features = [
            FNO2dBlock(
                inc,
                outc,
                self.condition_dim,
                n_modes,
                kernel_size,
                padding,
                freqlinear=freqlinear,
                waveprior_addition=wab,
                waveprior_concatenate=wcb,
                device=self.device,
            )
            for inc, outc, n_modes, kernel_size, padding, drop, wab, wcb in zip(
                kernel_list[:-1],
                kernel_list[1:],
                self.mode_list,
                self.kernel_size_list,
                self.padding_list,
                drop_path_rates,
                waveaddition_bools,
                waveconcatenate_bools,
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
        
        
        if self.positional_encoding:
            grid1 = self._pos_enc(x.shape, x.device, True)
            x = torch.cat([x, grid1.repeat(repeats=[x.shape[0], 1, 1, 1])], dim=1)
            
        
        waves = []
        if self.wave_prior:
            wave = self.encoding(eps, dl, wvl, x.device)
            waves.append(wave)
            wave = torch.cat(waves, dim=1)
        
            if self.wp_input_cat:
                x = torch.cat([x, wave], dim=1)
                x = self.stem(x)
            else:
                x = self.stem(x)
                    
            if self.waveprior_addition or self.waveprior_concatenate:
                wpa = self.wpa_stem(wave)
                wpa = F.pad(wpa, [0, self.padding, 0, self.padding])
        else:
            x = self.stem(x)
            
        
        x = F.pad(x, [0, self.padding, 0, self.padding])
                
        if self.freqlinear:
            modes = []          
            wvl = (wvl - 0.4) / (0.7 - 0.4) 
            emb = fourier_embedding(wvl, dim=self.dim)[:, 0, :] # wvl : N, 1 -> N, d
        
            emb = self.cond_embed(emb)
            for b in self.features:
                x = b(x, emb)
                    
        elif self.waveprior_addition or self.waveprior_concatenate:
            for j, b in enumerate(self.features):
                x = b(x, wpa)

        else:
            for m in self.features:
                x = m(x)
                    
        x = x[..., :-self.padding, :-self.padding]
        x = self.head(x)

        return x # BS, 2, H, W'
        