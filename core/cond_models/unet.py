"""
    Based on https://github.com/pdearena/pdearena/blob/main/pdearena/modules/twod_unet.py
    Towards Multi-spatiotemporal-scale Generalized PDE Modeling
    MIT License
    Copyright (c) 2020 Microsoft Corporation.
"""
from typing import List, Optional, Tuple
from torch.types import Device
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
import numpy as np


__all__ = ["CondUNet"]

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

    
def double_conv(in_channels, hidden, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
    )



__all__ = ["FNOConv2d"]


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


def define_down_block(level, blk_num, in_channels, out_channels, condition_dim, act, norm, has_attn=False, cond_bias=False, scale_shift_norm=False, device="cpu"):    
    blocks = []
    for _ in range(blk_num):
        this_block = DownBlock(in_channels, out_channels, cond_channels=condition_dim, has_attn=has_attn, act=act, norm=norm, scale_shift_norm=scale_shift_norm)
        blocks.append(this_block)
        in_channels = out_channels    
    if level < 3:
        down_block = Downsample(out_channels)
    else:
        down_block = nn.Identity()
    return blocks, down_block


def define_up_block(level, blk_num, in_channels, condition_dim, act, norm, has_attn=False, cond_bias=False, scale_shift_norm=False, device='cpu'):
    blocks = []
    
    for _ in range(blk_num):
        this_block = UpBlock(in_channels, in_channels, cond_channels=condition_dim, has_attn=has_attn, act=act, norm=norm, scale_shift_norm=scale_shift_norm)
        blocks.append(this_block)
    out_channels = in_channels // 2
    blocks.append(UpBlock(in_channels, out_channels, cond_channels=condition_dim, has_attn=has_attn, act=act, norm=norm, scale_shift_norm=scale_shift_norm))
    
    if level > 0:
        up_block = Upsample(out_channels)
    else:
        up_block = nn.Identity()
        
    return blocks, up_block

def get_act(act='ReLU'): # Default : ReLU()
    
    if act.lower() == "swish":
        return nn.SiLU()
    else:
        return getattr(nn, act)()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels, act, norm: bool = False, n_groups: int = 1, scale_shift_norm: bool = False):
        super().__init__()
        self.act = act
        self.cond_channels = cond_channels
        self.scale_shift_norm = scale_shift_norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1))
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()
        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            
        self.cond_emb = nn.Linear(cond_channels, 2 * out_channels if scale_shift_norm else out_channels)

            
    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        h = self.conv1(self.act(self.norm1(x)))
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.norm2(h) * (1+scale) + shift
            h = self.conv2(self.act(h))
        else:
            h = h + emb_out
            h = self.conv2(self.act(self.norm2(h)))
            # Add the shortcut connection and return
        return h + self.shortcut(x)
        
class FourierResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int = 16, modes1: int = 16, modes2: int = 16, act=None, norm: bool = False, n_groups: int = 1, cond_bias: bool = True, scale_shift_norm: bool = True, device: Device = torch.device('cuda:0')):
        super().__init__()
        
        self.act = act
        
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale_shift_norm = scale_shift_norm
        
        self.fourier1 = FNOConv2d(in_channels, out_channels, cond_channels, n_modes=(modes1, modes2), cond_bias=cond_bias, device=device)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, padding_mode='zeros')
        self.fourier2 = FNOConv2d(out_channels, out_channels, cond_channels, n_modes=(modes1, modes2), cond_bias=cond_bias, device=device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, padding_mode='zeros')
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        self.cond_emb = nn.Linear(cond_channels, 2 * out_channels if self.scale_shift_norm else out_channels)

        
        
    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        h = self.act(self.norm1(x))
        x1 = self.fourier1(h, emb)
        x2 = self.conv1(h)
        out = x1 + x2
        
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        if self.scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.norm2(out) * (1+scale) + shift
            h = self.act(h)
            x1 = self.fourier2(h, emb)
            x2 = self.conv2(h)
        else:
            out = out + emb_out
            out = self.act(self.norm2(out))
            x1 = self.fourier2(out, emb)
            x2 = self.conv2(out)
        out = x1 + x2 + self.shortcut(x)
        return out

class AttentionBlock(nn.Module):
    """Attention block This is similar to [transformer multi-head
    attention]

    Args:
        n_channels (int): the number of channels in the input
        n_heads (int): the number of heads in multi-head attention
        d_k: the number of dimensions in each head
        n_groups (int): the number of groups for [group normalization][torch.nn.GroupNorm].

    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 1):
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res

class DownBlock(nn.Module):
    """Down block This combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the first half of U-Net at each resolution.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels : int,
        has_attn: bool = False,
        act = None,
        norm: bool = False,
        scale_shift_norm: bool = False
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels, out_channels, cond_channels, act=act, norm=norm, scale_shift_norm=scale_shift_norm
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class FourierDownBlock(nn.Module):
    """Down block This combines [`FourierResidualBlock`][pdearena.modules.twod_unet.FourierResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the first half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        has_attn: bool = False,
        act = None,
        norm: bool = False,
        cond_bias: bool = True,
        scale_shift_norm: bool = True,
        device = 'cpu'
    ):
        super().__init__()
        self.res = FourierResidualBlock(
            in_channels,
            out_channels,
            cond_channels=cond_channels,
            modes1=modes1,
            modes2=modes2,
            act=act,
            norm=norm,
            cond_bias=cond_bias,
            scale_shift_norm=scale_shift_norm,
            device=device
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res(x, emb)
        x = self.attn(x)
        return x

class UpBlock(nn.Module):
    """Up block that combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (str): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        has_attn: bool = False,
        act = None,
        norm: bool = False,
        scale_shift_norm: bool = True
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, cond_channels=cond_channels, act=act, norm=norm, scale_shift_norm=scale_shift_norm)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res(x, emb)
        x = self.attn(x)
        return x
    


class FourierUpBlock(nn.Module):
    """Up block that combines [`FourierResidualBlock`][pdearena.modules.twod_unet.FourierResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the second half of U-Net at each resolution.

    Note:
        We currently don't recommend using this block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        has_attn: bool = False,
        act = None,
        norm: bool = False,
        cond_bias: bool = True,
        scale_shift_norm: bool = True,
        device = 'cpu'
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = FourierResidualBlock(
            in_channels + out_channels,
            out_channels,
            cond_channels=cond_channels,
            modes1=modes1,
            modes2=modes2,
            act=act,
            norm=norm,
            cond_bias=cond_bias,
            scale_shift_norm=scale_shift_norm,
            device=device
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
    """

    def __init__(self, n_channels: int, cond_channels: int, has_attn: bool = False, act = None, norm: bool = False, scale_shift_norm: bool = False):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, cond_channels=cond_channels, act=act, norm=norm, scale_shift_norm=scale_shift_norm)
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(n_channels, n_channels, cond_channels=cond_channels, act=act, norm=norm, scale_shift_norm=scale_shift_norm)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x


class Upsample(nn.Module):
    r"""Scale up the feature map by $2 \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)



class CondUNet(nn.Module):
    _conv = (FNOConv2d,)
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        dim: int = 16,
        blk_num: int = 2,
        act_func: Optional[str] = None,  
        norm: bool = True,
        dropout: float = 0.0,
        has_attn: bool = False,
        cond_bias: bool = True,
        scale_shift_norm: bool = True,
        use_1x1: bool = False,
        device = 'cpu'
        
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % 2 == 0, f"The output channels must be even number larger than 2, but got {out_channels}" # real, image
        self.dim = dim
        self.blk_num = blk_num
        self.act_func = act_func
        self.norm = norm
        self.dropout = dropout
        self.has_attn = has_attn
        self.cond_bias = cond_bias
        self.scale_shift_norm = scale_shift_norm
        self.use_1x1 = use_1x1
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
        
        
    def reset_parameters(self, random_state: Optional[int] = None):
        for name, m in self.named_modules():
            if isinstance(m, self._conv):
                m.reset_parameters()
        
    def adp_pad(self, enc_out, x):
        _, _, s1_p, s2_p = enc_out.shape
        _, _, s1, s2 = x.shape
        pad1, pad2 = s1_p - s1, s2_p - s2
        
        x = F.pad(x, [0,pad2,0,pad1] , "constant", 0)
        return x
    
    
    def build_layers(self):
        dim = self.dim
        if self.use_1x1:
            self.stem = nn.Conv2d(self.in_channels, dim, kernel_size=1)
        else:
            self.stem = nn.Conv2d(self.in_channels, dim, kernel_size=(3,3), padding=(1,1))
        
        downs = nn.ModuleList([])
        
        for i in range(4): # level : 4
            blocks, down = define_down_block(
                i,
                self.blk_num,
                dim,
                dim*2,
                condition_dim=self.condition_dim,
                act=self.act,
                norm=self.norm,
                has_attn=self.has_attn,
                device=self.device
            )
            downs.append(nn.ModuleList([*blocks, down]))
            dim *= 2
        self.downs = downs
        self.middle = MiddleBlock(dim, cond_channels=self.condition_dim, has_attn=False, act=self.act, norm=self.norm)
        
        ups = nn.ModuleList([])
        in_channels = dim
        for j in reversed(range(4)):
            blocks, up = define_up_block(j,
            self.blk_num,
            in_channels,
            condition_dim=self.condition_dim,
            act=self.act,
            norm=self.norm,
            has_attn=self.has_attn,
            device=self.device)
            ups.append(nn.ModuleList([*blocks, up]))
            in_channels = in_channels//2
        self.ups = ups
        
        if self.norm:
            self.gnorm = nn.GroupNorm(8, in_channels)
        else:
            self.gnorm = nn.Identity()
        
        if self.use_1x1:
            self.head = nn.Conv2d(in_channels, self.out_channels, 1)
        else:
            self.head = nn.Conv2d(in_channels, self.out_channels, kernel_size=(3,3), padding=(1,1))
        

    def forward(self, x: Tensor, z: Tensor, dl: Tensor):
        # if len(cond.shape) == 1:
        #     cond = cond[:, None]
        emb = fourier_embedding(z, dim=self.dim)[:, 0, :]
        
        emb = self.cond_embed(emb)
        
        
        x = self.stem(x)

        h = [x]
        for block1, block2, downsample in self.downs:
            blocks = [block1, block2]
            for b in blocks:
                x = b(x, emb)
                h.append(x)
            if not isinstance(downsample, nn.Identity):
                x = downsample(x)
                h.append(x)
            
        x = self.middle(x, emb)
        for block1, block2, block3, upsample in self.ups:
            blocks = [block1, block2, block3]
            for b in blocks:
                enc_out = h.pop()
                x = self.adp_pad(enc_out,x)
                x = torch.cat((x, enc_out), dim=1)
                x = b(x, emb)
                
            x = upsample(x)
            

        x = self.head(self.act(self.gnorm(x))) # out_ch = 2 (real, imag)
        return x

    