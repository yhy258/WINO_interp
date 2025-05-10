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
# from pyutils.activation import Swish
# from pyutils.torch_train import set_torch_deterministic
from timm.models.layers import DropPath, to_2tuple
import numpy as np
from .layers.fno_conv2d import FNOConv2d

__all__ = ["UNet"]

def define_down_block(level, blk_num, in_channels, out_channels, act, norm, has_attn=False, blueprint=False, device="cpu"):    
    blocks = []
    for _ in range(blk_num):
        this_block = DownBlock(in_channels, out_channels, has_attn=has_attn, act=act, norm=norm, blueprint=blueprint)
        blocks.append(this_block)
        in_channels = out_channels    
    if level < 3:
        down_block = Downsample(out_channels)
    else:
        down_block = nn.Identity()
    return blocks, down_block


def define_up_block(level, blk_num, in_channels, act, norm, has_attn=False, blueprint=False, device='cpu'):
    blocks = []
    
    for _ in range(blk_num):
        this_block = UpBlock(in_channels, in_channels, has_attn=has_attn, act=act, norm=norm, blueprint=blueprint)
        blocks.append(this_block)
    out_channels = in_channels // 2
    blocks.append(UpBlock(in_channels, out_channels, has_attn=has_attn, act=act, norm=norm, blueprint=blueprint))
    
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
            self.act_func = nn.Swish()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x

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
        blueprint: bool = False
    ) -> None:
        super().__init__()
        
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [(dilation[i] * (kernel_size[i] - 1) + 1) // 2 for i in range(len(kernel_size))]
        if blueprint:
            self.conv1 = BSConv2d(
                in_channels,
                out_channels // 2,
                kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
            )
            self.conv2 = BSConv2d(
                out_channels // 2,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels//2, kernel_size, stride=1, padding=padding
            )
            self.conv2 = nn.Conv2d(
                out_channels//2, out_channels, kernel_size, stride=1, padding=padding
            )
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.act1 = nn.ReLU(inplace=True)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act, norm: bool = False, n_groups: int = 1, blueprint: bool = False):
        super().__init__()
        self.act = act
        if blueprint:
            self.conv1 = BSConv2d(in_channels, out_channels, kernel_size=3)
            self.conv2 = BSConv2d(out_channels, out_channels, kernel_size=3)
            if in_channels != out_channels:
                self.shortcut = BSConv2d(in_channels, out_channels, kernel_size=3)
            else:
                self.shortcut = nn.Identity()
        else:
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
            
    def forward(self, x: torch.Tensor):
        # First convolution layer
        h = self.conv1(self.act(self.norm1(x)))
        # Second convolution layer
        h = self.conv2(self.act(self.norm2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)
    
class FourierResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int = 16, modes2: int = 16, act=None, norm: bool = False, n_groups: int = 1, device: Device = torch.device('cuda:0')):
        super().__init__()
        
        self.act = act
        
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.fourier1 = FNOConv2d(in_channels, out_channels, n_modes=(modes1, modes2), device=device)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, padding_mode='zeros')
        self.fourier2 = FNOConv2d(out_channels, out_channels, n_modes=(modes1, modes2), device=device)
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
            
    def forward(self, x: torch.Tensor):
        h = self.act(self.norm1(x))
        x1 = self.fourier1(h)
        x2 = self.conv1(h)
        out = x1 + x2
        out = self.act(self.norm2(out))
        x1 = self.fourier2(out)
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
        has_attn: bool = False,
        act = None,
        norm: bool = False,
        blueprint: bool = False
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, act=act, norm=norm, blueprint=blueprint)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
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
        modes1: int = 16,
        modes2: int = 16,
        has_attn: bool = False,
        act = None,
        norm: bool = False,
        device = 'cpu'
    ):
        super().__init__()
        self.res = FourierResidualBlock(
            in_channels,
            out_channels,
            modes1=modes1,
            modes2=modes2,
            act=act,
            norm=norm,
            device=device
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
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
        has_attn: bool = False,
        act = None,
        norm: bool = False,
        blueprint: bool = False,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, act=act, norm=norm, blueprint=blueprint)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
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
        modes1: int = 16,
        modes2: int = 16,
        has_attn: bool = False,
        act = None,
        norm: bool = False,
        device = 'cpu'
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = FourierResidualBlock(
            in_channels + out_channels,
            out_channels,
            modes1=modes1,
            modes2=modes2,
            act=act,
            norm=norm,
            device=device
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
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

    def __init__(self, n_channels: int, has_attn: bool = False, act = None, norm: bool = False):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, act=act, norm=norm)
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(n_channels, n_channels, act=act, norm=norm)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
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



class UNet(nn.Module):
    _conv = (FNOConv2d,)
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        dim: int = 16,
        blk_num: int = 2,
        act_func: Optional[str] = None,  
        norm: bool = True,
        wave_prior: bool = True,
        dropout: float = 0.0,
        has_attn: bool = False,
        use_1x1: bool = True,
        image_shape: Tuple = (32, 256),
        eps_min: float = 1,
        eps_max: float = 1.4623,
        blueprint: bool = False,
        propagate_resonance_prior: bool = False,
        neural_code: bool = False,
        cond_path_flag: bool = False,
        # condition_position: str = "bottleneck",
        # condition_mode: str = "add", # normalization
        device = 'cpu'
        
    ):
        super().__init__()
        

        
        self.in_channels = in_channels+2 # positional encoding
        self.out_channels = out_channels
        assert out_channels % 2 == 0, f"The output channels must be even number larger than 2, but got {out_channels}" # real, image
        self.dim = dim
        self.blk_num = blk_num
        self.act_func = act_func
        self.norm = norm
        self.wave_prior = wave_prior
        if self.wave_prior:
            self.in_channels = in_channels + 4
            self.in_channels += 2
        else:
            cond_path_flag = False
        self.dropout = dropout
        self.has_attn = has_attn
        self.use_1x1 = use_1x1
        self.image_shape = image_shape
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.device = device
        self.act = get_act(self.act_func)
        self.blueprint = blueprint
        self.propagate_resonance_prior = propagate_resonance_prior
        self.neural_code = neural_code
        if self.neural_code:
            self.neural_repr = nn.Parameter(torch.randn(4, 4))
        self.cond_path_flag = cond_path_flag
        if self.cond_path_flag:
            self.in_channels = in_channels + 2 # positional encoding

        
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
                mesh.mul(dl.div(wvl).mul(1j*2*np.pi)[..., None, None])
            )
        )

        mesh = torch.tensor(mesh.permute(0, 1, 4, 2, 3).flatten(1, 2), dtype=torch.float32) # bs, 4, h, w
        if self.propagate_resonance_prior: # y propagate
            mesh = torch.cat([mesh[:, 2:], (mesh[:, 2:]+mesh[:, :2])/2], dim=1) # BS, 4, H, W
        if self.neural_code:
            coded_mesh = torch.einsum('ijk,jl->ilk', mesh.view(mesh.shape[0], 4, -1), self.neural_repr) # bs, 4, hw
            # BS, 4, HW
            mesh = torch.einsum('ijk,jl', coded_mesh, self.neural_repr.permute(1,0))
            mesh = mesh.view(mesh.shape[0], 4, W, H)
        return (mesh + 1)/2
    
        
    def reset_parameters(self, random_state: Optional[int] = None):
        for name, m in self.named_modules():
            if isinstance(m, self._conv):
                # if random_state is not None:
                #     # deterministic seed, but different for different layer, and controllable by random_state
                #     set_torch_deterministic(random_state + sum(map(ord, name)))
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
        
        if self.cond_path_flag:
            self.rem_stem = ResStem(
                dim*2,
                dim*2,
                1,
                blueprint=self.blueprint
            )
            
            self.cond_path = ConvBlock(
                4, dim, kernel_size=1, padding=0, act_func=self.act_func, device=self.device
            )
            dim *= 2

        downs = []
        
        for i in range(4): # level : 4
            blocks, down = define_down_block(i, self.blk_num, dim, dim*2, self.act, norm=self.norm, has_attn=self.has_attn, blueprint=self.blueprint, device=self.device)
            downs.append(nn.ModuleList([*blocks, down]))
            dim *= 2
        
        if self.wave_prior == False:
            self.wvl_layer = nn.Sequential(
                nn.Linear(1, dim//8),
                nn.GELU(),
                nn.Linear(dim//8, dim//2),
                nn.GELU(),
                nn.Linear(dim//2, dim),
            )
        
        self.downs = nn.ModuleList(downs)
        
        self.middle = MiddleBlock(dim, has_attn=False, act=self.act, norm=self.norm)
        
        ups = []
        in_channels = dim

        for j in reversed(range(4)):
            blocks, up = define_up_block(j, self.blk_num, in_channels,  self.act, norm=self.norm, has_attn=self.has_attn, blueprint=self.blueprint, device=self.device)
            ups.append(nn.ModuleList([*blocks, up]))
            in_channels = in_channels//2
        self.ups = nn.ModuleList(ups)
        
        if self.norm:
            self.gnorm = nn.GroupNorm(8, in_channels)
        else:
            self.gnorm = nn.Identity()
        
        if self.use_1x1:
            self.head = nn.Conv2d(in_channels, self.out_channels, 1)
        else:
            self.head = nn.Conv2d(in_channels, self.out_channels, kernel_size=(3,3), padding=(1,1))
        

    def forward(self, x: Tensor, wavelength: Tensor, dl:Tensor):
        eps = x * (self.eps_max - self.eps_min) + self.eps_min
        
        grid1 = self._pos_enc(x.shape, x.device, True)
        x = torch.cat([x, grid1.repeat(repeats=[x.shape[0], 1, 1, 1])], dim=1)
        
        grid = self.encoding(eps, dl, wavelength, x.device)
        
        if self.cond_path_flag:
            grid = self.cond_path(grid)
            x = self.stem(x)

            x = torch.cat((x, grid), dim=1)

            x = self.rem_stem(x)
        else:
            if self.wave_prior:
                x = torch.cat((x, grid), dim=1)
            
            x = self.stem(x)
       

        h = [x]
        for block1, block2, downsample in self.downs:
            blocks = [block1, block2]
            for b in blocks:
                x = b(x)
                h.append(x)
            if not isinstance(downsample, nn.Identity):
                x = downsample(x)
                h.append(x)
                
        if self.wave_prior == False:
            cond = self.wvl_layer(wavelength)[:, :, None, None]
            x += cond
            
            
        x = self.middle(x)
        
        for block1, block2, block3, upsample in self.ups:
            blocks = [block1, block2, block3]
            for b in blocks:
                enc_output = h.pop()
                x = self.adp_pad(enc_output, x) # odd resolution when downsampled
                x = torch.cat((x, enc_output), dim=1)
                x = b(x)
                
            x = upsample(x)
            

        x = self.head(self.act(self.gnorm(x))) # out_ch = 2 (real, imag)
        
        return x
    