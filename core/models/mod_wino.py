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



__all__ = ["ModulationWINO"]


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

class ComplexAct(nn.Module):
    def __init__(self, act):
        super().__init__()
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        self.act = act

    def forward(self, z):
        return self.act(z.real) + 1.j * self.act(z.imag)

class FreqLinear(nn.Module):
    def __init__(self, in_channel, modes1, modes2, bias=True):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale1 = 1/ (in_channel + 2 * modes1)
        scale2 = 1 / (in_channel + 2 * modes2)

        self.weights1 = nn.Parameter(scale1 * torch.randn(in_channel, 2 * modes1, dtype=torch.float32))
        self.weights2 = nn.Parameter(scale2 * torch.randn(in_channel, 2 * modes2, dtype=torch.float32))
        if bias:
            self.bias1 = nn.Parameter(torch.zeros(1, 2 * modes1, dtype=torch.float32))
            self.bias2 = nn.Parameter(torch.zeros(1, 2 * modes2, dtype=torch.float32))
        else:
            self.bias1 = 0
            self.bias2 = 0
            

    def forward(self, x):
        # x : BS, C
        B = x.shape[0]
        h1 = torch.einsum("tc,cm->tm", x, self.weights1) + self.bias1
        h2 = torch.einsum("tc,cm->tm", x, self.weights2) + self.bias2 # BS, M*2
        # print(x.shape, h1.shape, self.weights1.shape)
        h1 = h1.reshape(B, self.modes1, 2)
        h2 = h2.reshape(B, self.modes2, 2)
        # h = h.reshape(B, self.modes1, self.modes2, 2, 2)
        return torch.view_as_complex(h1), torch.view_as_complex(h2)




class FourierLiftProj(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, device):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_mode_1, self.n_mode_2 = n_modes
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
            
    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_1.real)
        nn.init.kaiming_normal_(self.weight_2.real)
        
    def get_zero_padding(self, size, device):
        bs, h, w = size[0], size[-2], size[-1] // 2 + 1
        return torch.zeros(bs, self.out_channels, h, w, dtype=torch.cfloat, device=device)

    def forward(self, x: Tensor) -> Tensor:
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
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x

class SpectralFourierLiftProj(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_mode_1, self.n_mode_2 = n_modes
        # self.device = device
        
        self.scale = 1 / (in_channels * out_channels)
        self.build_parameters()
        self.reset_parameters()

    def build_parameters(self) -> None:
            self.weight_1 = nn.Parameter(
                self.scale * torch.zeros([self.in_channels, self.out_channels], dtype=torch.cfloat)
            )
            self.weight_2 = nn.Parameter(
                self.scale * torch.zeros([self.in_channels, self.out_channels], dtype=torch.cfloat)
            )
            
    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_1.real)
        nn.init.kaiming_normal_(self.weight_2.real)
        
    # def get_zero_padding(self, size, device):
    #     bs, h, w = size[0], size[-2], size[-1] // 2 + 1
    #     return torch.zeros(bs, self.out_channels, h, w, dtype=torch.cfloat, device=device)

    def forward(self, x: Tensor) -> Tensor:
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # rfft2 : FFT of real signal -> Hemiltonian symmetry -> X_ft[i] = conj(X_ft[-i])
        # shape of x_ft = (BS, C, H, W//2+1)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        
        x_ft = torch.einsum(
            "bixy,io->boxy", x_ft, self.weight_1
        )
        x = torch.fft.irfft2(x_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
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

class FourierBlock(nn.Module): 
    # Num blocks -> deeper layer -> lower blocks.
    # Modes -> deeper layer -> higher modes.
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(self, cond_channels, hidden_size, inner_expand, n_modes, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1, act_func='GELU', in_ch_sf=False, weight_sharing=False, inner_nonlinear=False, bias=False):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()
        self.comp_act_func = ComplexAct(self.act_func)

        self.hidden_size = hidden_size
        self.in_channels = self.out_channels = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 1 / (hidden_size * hidden_size)
        self.n_modes = n_modes
        self.n_mode_1, self.n_mode_2 = n_modes
        self.inner_expand = inner_expand
        self.in_ch_sf = in_ch_sf      
        self.weight_sharing = weight_sharing      
        self.inner_nonlinear = inner_nonlinear

        if self.weight_sharing:
            self.weight_1 = nn.Parameter( # n, i, o, 
                self.scale
                * torch.zeros([self.in_channels // 2 // self.num_blocks, self.out_channels // 2 // self.num_blocks, self.n_modes[0]], dtype=torch.cfloat)
            )
            self.weight_2 = nn.Parameter(
                self.scale
                * torch.zeros(
                    [(self.in_channels - self.in_channels // 2) // self.num_blocks, (self.out_channels - self.out_channels // 2)//self.num_blocks, self.n_modes[1]],
                    dtype=torch.cfloat,
                )
            )
        else:
            self.weight_1 = nn.Parameter( # n, i, o, 
                self.scale
                * torch.zeros([self.num_blocks, self.in_channels // 2 // self.num_blocks, self.out_channels // 2 // self.num_blocks, self.n_modes[0]], dtype=torch.cfloat)
            )
            self.weight_2 = nn.Parameter(
                self.scale
                * torch.zeros(
                    [self.num_blocks, (self.in_channels - self.in_channels // 2) // self.num_blocks, (self.out_channels - self.out_channels // 2)//self.num_blocks, self.n_modes[1]],
                    dtype=torch.cfloat,
                )
            )
            
        if in_ch_sf:
            self.in_weight_1 = nn.Parameter(
                self.scale
                * torch.zeros(
                    [self.in_channels//2, self.out_channels//2],
                    dtype=torch.cfloat,
                )
            )
            self.in_weight_2 = nn.Parameter(
                self.scale
                * torch.zeros(
                    [self.in_channels-self.in_channels//2, self.out_channels-self.out_channels//2],
                    dtype=torch.cfloat,
                )
            )
        else:
            self.in_weight_1 = None
            self.in_weight_2 = None
        
        self.cond_emb = FreqLinear(cond_channels, self.n_mode_1, self.n_mode_2)
        
        
        if self.sparsity_threshold > 0:
            self.soft_shrink = nn.Softshrink(lambd=self.sparsity_threshold)
        else:
            self.soft_shrink = None
        # self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        # self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))

        self.reset_parameters()
            
    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_1.real)
        nn.init.kaiming_normal_(self.weight_2.real)
        if self.in_weight_1 != None:
            nn.init.kaiming_normal_(self.in_weight_1.real)
            nn.init.kaiming_normal_(self.in_weight_2.real)

    def get_zero_padding(self, size, device):
        return torch.zeros(*size, dtype=torch.cfloat, device=device)

    def _fourier_forward(self, x, emb, dim=-2): # BS, C, H, W
        if dim == -2:
            x_ft = torch.fft.rfft(x, norm="ortho", dim=-2) # BS, C, H//2, W
            BS, C, H, W = x_ft.shape
            x_ft = x_ft.reshape(BS, self.num_blocks, C//self.num_blocks, H, W)
            n_mode = self.n_mode_1
            if n_mode == x_ft.size(-2):  # full mode
                if self.weight_sharing:
                    out_ft = torch.einsum("bnixy,iox->bnoxy", x_ft*emb[:, None, None, :, None], self.weight_1)
                else:
                    out_ft = torch.einsum("bnixy,niox->bnoxy", x_ft*emb[:, None, None, :, None], self.weight_1)
                out_ft = out_ft.reshape(BS, -1, H, W)
            else:
                out_ft = self.get_zero_padding(
                    [x.size(0), self.num_blocks, self.weight_1.size(1), x_ft.size(-2), x_ft.size(-1)], x.device
                )
                if self.weight_sharing:
                    out_ft[..., :n_mode, :] = torch.einsum("bnixy,iox->bnoxy", x_ft[..., :n_mode, :]*emb[:, None, None, :, None], self.weight_1)
                else:
                    out_ft[..., :n_mode, :] = torch.einsum("bnixy,niox->bnoxy", x_ft[..., :n_mode, :]*emb[:, None, None, :, None], self.weight_1)
                out_ft = out_ft.reshape(BS, -1, H, W)
            if self.in_ch_sf:
                if self.inner_nonlinear:
                    out_ft = self.comp_act_func(out_ft)
                out_ft = torch.einsum("bixy,io->boxy", out_ft, self.in_weight_1)
            if self.soft_shrink != None:
                out_ft = torch.stack([out_ft.real, out_ft.imag], dim=-1)
                out_ft = self.soft_shrink(out_ft)
                out_ft = out_ft[..., 0] + 1j*out_ft[..., 1]
            x = torch.fft.irfft(out_ft, n=x.size(-2), dim=-2, norm="ortho")
        elif dim == -1:
            x_ft = torch.fft.rfft(x, norm="ortho", dim=-1)
            BS, C, H, W = x_ft.shape
            x_ft = x_ft.reshape(BS, self.num_blocks, C//self.num_blocks, H, W)
            n_mode = self.n_mode_2
            if n_mode == x_ft.size(-1):
                if self.weight_sharing:
                    out_ft = torch.einsum("bnixy,ioy->bnoxy", x_ft*emb[:, None, None, None, :], self.weight_2)
                else:    
                    out_ft = torch.einsum("bnixy,ioy->bnoxy", x_ft*emb[:, None, None, None, :], self.weight_2)
            else:
                out_ft = self.get_zero_padding(
                    [x.size(0), self.num_blocks, self.weight_2.size(1), x_ft.size(-2), x_ft.size(-1)], x.device
                )
                if self.weight_sharing:
                    out_ft[..., :n_mode] = torch.einsum("bnixy,ioy->bnoxy", x_ft[..., :n_mode]*emb[:, None, None, None, :], self.weight_2)
                else:
                    out_ft[..., :n_mode] = torch.einsum("bnixy,nioy->bnoxy", x_ft[..., :n_mode]*emb[:, None, None, None, :], self.weight_2)
                out_ft = out_ft.reshape(BS, -1, H, W)
            if self.in_ch_sf:
                if self.inner_nonlinear:
                    out_ft = self.comp_act_func(out_ft)
                out_ft = torch.einsum("bixy,io->boxy", out_ft, self.in_weight_2)
            if self.soft_shrink != None:
                out_ft = torch.stack([out_ft.real, out_ft.imag], dim=-1)
                out_ft = self.soft_shrink(out_ft)
                out_ft = out_ft[..., 0] + 1j*out_ft[..., 1]
            x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1, norm="ortho")
        return x

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        emb1, emb2 = self.cond_emb(emb) # BS, M1 / BS, M2
        xx, xy = x.chunk(2, dim=1)
        xx = self._fourier_forward(xx, emb2, dim=-1)
        xy = self._fourier_forward(xy, emb1, dim=-2)
        x = torch.cat([xx, xy], dim=1)
        return x
    
    
class ChannelProcessBlock(nn.Module):
    def __init__(self, in_channels, hidden_rate, act_func, dropout=0.0, ffn_simplegate=False, depthwise_conv_module=False):
        super().__init__()

        hidden_dim = int(in_channels * hidden_rate)
                    
        if act_func is None:
            self.act_func = None
        elif ffn_simplegate:
            self.act_func = SimpleGate()
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()
    
        if ffn_simplegate:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                self.act_func,
                nn.Conv2d(hidden_dim//2, in_channels, kernel_size=1),
                nn.Dropout(dropout)
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                self.act_func,
                nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
                nn.Dropout(dropout)
            )
        if depthwise_conv_module:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1),
                nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, padding=1),
                self.act_func,
                nn.Conv2d(hidden_dim, in_channels, 1)
            )
        
    def forward(self, x):
        return self.layer(x)


    
    
class SpectralChannelBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, inner_expand, expand_rate, act_func, mode, num_blocks=8, batch_norm=False, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1, dropout=0.0, drop_path_rate=0.0, fourier_bias=False, ffn_simplegate=False, in_ch_sf=False, weight_sharing=False, inner_nonlinear=False, depthwise_conv_module=False):
        super().__init__()
        self.fno = FourierBlock(cond_channels, in_channels, inner_expand, mode, num_blocks, sparsity_threshold, hard_thresholding_fraction, hidden_size_factor, in_ch_sf=in_ch_sf, weight_sharing=weight_sharing, inner_nonlinear=inner_nonlinear, bias=fourier_bias)

        self.spec = ChannelProcessBlock(in_channels, expand_rate, act_func, dropout, ffn_simplegate, depthwise_conv_module)

            

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(in_channels)
        else:
            self.bn1 = nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
    def forward(self, x, emb=None):
        res = x

        x = self.fno(x, emb)
        
        x = self.bn1(x)
        

        x = self.spec(x)
        x = self.drop_path(x) + res    
        
        return x
        
        
        
class ModulationWINO(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        dim: int = 16,
        padding: int = 8,
        modes: list = [[4, 5], [40, 50]],
        inner_expand: list = [0],
        expand_list: list = [4],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        act_func: Optional[str] = "ReLU",
        num_blocks: int = 4, # The number of fourier blocks.
        hid_num_blocks: int = 8,
        ch_process_block: bool = False,
        pre_spec: bool = True,
        sparsity_threshold : float = 0.01,
        hard_thresholding_fraction: float = 1.,
        hidden_size_factor: int = 1,
        fourier_bias: bool = False,
        stem_dropout: float = 0.0,
        feature_dropout: float = 0.0,
        head_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        eps_min: float = 1,
        eps_max: float = 1.46**2,
        batch_norm: bool = False,
        depthwise_conv_module: bool = False,
        non_linear_stem: bool = False,
        ffn_simplegate: bool = False,
        positional_encoding: bool = True,
        in_ch_sf: bool = False,
        weight_sharing: bool = False,
        inner_nonlinear: bool = False,
        fourier_lift_proj: bool = False,
        lift_proj_modes: list = None,
        eps_info: bool = False,
        device: Device = torch.device('cuda:3')
    ):
        super().__init__()
        if len(expand_list) == 1:
            expand_list = [expand_list for i in range(num_blocks)]
        if len(inner_expand) == 1:
            inner_expand = [inner_expand for i in range(num_blocks)]
            

        if positional_encoding:
            self.in_channels = in_channels+2
        else:
            self.in_channels = in_channels

                
        self.out_channels = out_channels
        self.dim = dim
        self.num_blocks = num_blocks
        self.act_func = act_func
        self.padding = padding
        self.inner_expand = inner_expand
        self.expand_list = expand_list
        self.hid_num_blocks = hid_num_blocks
        self.modes = modes
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.fourier_bias = fourier_bias
        self.ch_process_block = ch_process_block
        self.pre_spec = pre_spec
        self.stem_dropout = stem_dropout
        self.feature_dropout = feature_dropout
        self.head_dropout = head_dropout
        self.drop_path_rate = drop_path_rate
        self.ffn_simplegate = ffn_simplegate
        self.batch_norm = batch_norm
        self.depthwise_conv_module = depthwise_conv_module
        self.hidden_list = hidden_list
        
        self.positional_encoding = positional_encoding
        self.non_linear_stem = non_linear_stem
        self.fourier_lift_proj = fourier_lift_proj
        if lift_proj_modes == None:
            self.lift_proj_modes = [modes, modes]
        else:
            self.lift_proj_modes= lift_proj_modes
        
        self.in_ch_sf = in_ch_sf
        self.weight_sharing = weight_sharing
        self.inner_nonlinear = inner_nonlinear
        self.eps_min = eps_min
        self.eps_max = eps_max
        
        if isinstance(modes[0], int):
            self.modes_schedule = [modes for i in range(num_blocks)]

        if not isinstance(self.hid_num_blocks, list):
            self.hid_num_blocks = [hid_num_blocks for i in range(num_blocks)]

        self.eps_info = eps_info
        self.device = device
        
        
        self.build_layers()
        

        
        
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
    
    
    def build_layers(self):
        
        if self.act_func is None:
            act_func = None
        elif self.act_func.lower() == "swish":
            act_func = Swish()
        else:
            act_func = getattr(nn, self.act_func)()

        self.condition_dim = self.dim * 4
        self.cond_embed = nn.Sequential(
                nn.Linear(self.dim, self.condition_dim),
                act_func,
                nn.Linear(self.condition_dim, self.condition_dim)
            )
        
        if self.non_linear_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(self.in_channels, self.dim//2, 1),
                act_func,
                nn.Conv2d(self.dim//2, self.dim, 1),
            )
        
        elif self.fourier_lift_proj:
            self.stem = SpectralFourierLiftProj(self.in_channels, self.dim, self.lift_proj_modes[0])
            
        else:
            self.stem = nn.Conv2d(
                self.in_channels,
                self.dim,
                1
            )

        drop_path_rates = np.linspace(0, self.drop_path_rate, self.num_blocks)
        if self.in_ch_sf:
            in_ch_sf_list = [True] * len(self.hid_num_blocks)
            for i, hnb in enumerate(self.hid_num_blocks):
                if hnb == 1:
                    in_ch_sf_list[i] = False    
        else:
            in_ch_sf_list = [False] * len(self.hid_num_blocks)
        
        
        # in_channels, out_channels, act_func, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1, dropout=0.0, drop_path_rate=0.0
        
        features = [
            SpectralChannelBlock(
                in_channels=self.dim,
                cond_channels=self.condition_dim,
                inner_expand=iexp[0],
                expand_rate=exp_rate[0],
                act_func=self.act_func,
                mode=mode,
                num_blocks=hnb,
                batch_norm=self.batch_norm,
                sparsity_threshold=self.sparsity_threshold,
                hidden_size_factor=self.hidden_size_factor,
                dropout=self.feature_dropout,
                drop_path_rate=drop_path,
                in_ch_sf=in_ch_sf,
                weight_sharing=self.weight_sharing,
                inner_nonlinear=self.inner_nonlinear,
                fourier_bias=self.fourier_bias,
                depthwise_conv_module=self.depthwise_conv_module
            )
            for iexp, exp_rate, drop_path, hnb, mode, in_ch_sf in zip(self.inner_expand, self.expand_list, drop_path_rates, self.hid_num_blocks, self.modes_schedule, in_ch_sf_list)
        ]
        # else:
        #     features = [
        #         SpatialSpectralBlock(
        #             in_channels=self.dim,
        #             expand_rate=exp_rate[0],
        #             act_func=self.act_func,
        #             mode=None,
        #             num_blocks=hnb,
        #             sparsity_threshold=self.sparsity_threshold,
        #             hard_thresholding_fraction=self.hard_thresholding_fraction,
        #             hidden_size_factor=self.hidden_size_factor,
        #             dropout=self.feature_dropout,
        #             drop_path_rate=drop_path,
        #             fourier_bias=self.fourier_bias,
        #             v2=self.v2
        #         )
        #         for exp_rate, drop_path, hnb in zip(self.expand_list, drop_path_rates, self.hid_num_blocks)
        #     ]
        
        self.features = nn.Sequential(*features)
        
        hidden_list = [self.dim] + self.hidden_list
        
        
        if self.fourier_lift_proj:
            head = [SpectralFourierLiftProj(self.dim, self.out_channels, self.lift_proj_modes[1])]
        else:
            head = [
                nn.Sequential(
                    ConvBlock(inc, outc, kernel_size=1, padding=0, act_func=self.act_func, device=self.device),
                    nn.Dropout2d(self.head_dropout),
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
        
        x = self.stem(x)
        
        x = F.pad(x, [0, self.padding, 0, self.padding])
         
        wvl = (wvl - 0.4) / (0.7 - 0.4) # min wvl : 0.4 / max wvl : 0.7
        emb = fourier_embedding(wvl, dim=self.dim)[:, 0, :]
        emb = self.cond_embed(emb)
        
        for m in self.features:
            x = m(x, emb)
        x = x[..., :-self.padding, :-self.padding]
        x = self.head(x)
        
        return x # BS, 2, H, W
        
