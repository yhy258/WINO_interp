"""
    Largely based on https://github.com/JeremieMelo/NeurOLight/blob/main/core/models/unet.py
    NeurOLight: A Physics-Agnostic Neural Operator Enabling Parametric Photonic Device Simulation
    MIT License
    Copyright (c) 2022 Jiaqi Gu
"""

import numpy as np
from typing import List, Optional, Tuple
# from pyutils.torch_train import set_torch_deterministic
import torch
from torch import nn
import torch.nn.functional as F
from torch.functional import Tensor
from torch.types import Device
from .layers.fno_conv2d import FNOConv2d
# from pyutils.activation import Swish
from timm.models.layers import DropPath, to_2tuple

__all__ = ["OrigUNet", "OrigUNet_SFT"]


def get_act(act='ReLU'): # Default : ReLU()
    
    if act.lower() == "swish":
        return nn.SiLU()
    else:
        return getattr(nn, act)()
    
    
def double_conv(in_channels, hidden, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.GELU(),
        nn.Conv2d(hidden, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
    )


class SFT_layer(nn.Module):
    def __init__(self, dim):
        super(SFT_layer, self).__init__()
        
        Relu = nn.ReLU(True)    
        pool = nn.AvgPool2d(2)
        condition_conv1 = nn.Conv2d(1,dim, kernel_size=3, stride=1, padding=1)
        condition_conv2 = nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1)
        condition_conv3 = nn.Conv2d(dim*2, dim*4, kernel_size=3, stride=1, padding=1)

        conditon_conv = [condition_conv1, Relu, pool, condition_conv2, Relu, pool, condition_conv3, Relu]
        self.condition_conv = nn.Sequential(*conditon_conv)

        scale_conv1 = nn.Conv2d(dim*4,dim*4, kernel_size=3, stride=1, padding=1)
        scale_conv2 = nn.Conv2d(dim*4, dim*4, kernel_size=3, stride=1, padding=1)
        scale_conv = [scale_conv1, Relu, scale_conv2, Relu]
        self.scale_conv = nn.Sequential(*scale_conv)

        sift_conv1 = nn.Conv2d(dim*4,dim*4, kernel_size=3, stride=1, padding=1)
        sift_conv2 = nn.Conv2d(dim*4, dim*4, kernel_size=3, stride=1, padding=1)
        sift_conv = [sift_conv1, Relu, sift_conv2, Relu]
        self.sift_conv = nn.Sequential(*sift_conv)

    def forward(self, x, structure):
        structure_condition = self.condition_conv(structure)
        scaled_feature = self.scale_conv(structure_condition) * x
        sifted_feature = scaled_feature + self.sift_conv(structure_condition)

        return sifted_feature


class OrigUNet(nn.Module):
    _conv = (FNOConv2d,)
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        dim: int = 16,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        eps_min: float = 1,
        eps_max: float = 1.4623**2,
        wave_prior: bool = True,
        wave2wave: bool = False,
        shifted:bool = False,
        only_y_direc: bool = False,
        with_cp=False,
    ):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        if wave2wave:
            if only_y_direc:
                self.in_channels = self.out_channels = 2    
            else:
                self.in_channels = self.out_channels = 4 
        else:
            if only_y_direc:
                self.in_channels = in_channels + 2
            else:
                self.in_channels = in_channels + 4
            self.out_channels = out_channels
        

        assert out_channels % 2 == 0, f"The output channels must be even number larger than 2, but got {out_channels}"
        self.dim = dim
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.wave_prior = wave_prior
        self.shifted = shifted
        self.wave2wave = wave2wave
        self.only_y_direc = only_y_direc
        self.with_cp = with_cp

        if not self.wave_prior:
            self.in_channels = in_channels

        # if not self.wave_prior or self.cond_path_flag:
        #     self.in_channels = in_channels
        
        self.device = device
        
        self.build_layers()
        self.reset_parameters()

        self.permittivity_encoder = None
        self.set_trainable_permittivity(False)

    def reset_parameters(self, random_state: Optional[int] = None):
        for name, m in self.named_modules():
            if isinstance(m, self._conv):
                # if random_state is not None:
                #     # deterministic seed, but different for different layer, and controllable by random_state
                #     set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
    
    def build_layers(self):
        dim = self.dim
        if self.wave_prior == False:
            self.wvl_layer = nn.Sequential(
                nn.Linear(1, dim*4),
                nn.GELU(),
                nn.Linear(dim*4, dim*8),
            )
            
        self.dconv_down1 = double_conv(self.in_channels, dim, dim)
        self.dconv_down2 = double_conv(dim, dim * 2, dim * 2)
        self.dconv_down3 = double_conv(dim * 2, dim * 4, dim * 4)
        self.dconv_down4 = double_conv(dim * 4, dim * 8, dim * 8)

        # self.maxpool = nn.MaxPool2d(2)
        self.maxpool = nn.AvgPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample1 = nn.ConvTranspose2d(dim * 8, dim * 8, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(dim * 4, dim * 4, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(dim * 2, dim * 2, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(dim * 12, dim * 8, dim * 4)
        self.dconv_up2 = double_conv(dim * 6, dim * 4, dim * 2)
        self.dconv_up1 = double_conv(dim * 3, dim * 2, dim)
        self.drop_out = nn.Dropout2d(self.dropout_rate)
        self.conv_last = nn.Conv2d(dim, self.out_channels, 1)

    def set_trainable_permittivity(self, mode: bool = True) -> None:
        self.trainable_permittivity = mode

    def requires_network_params_grad(self, mode: float = True) -> None:
        params = self.parameters()
        for p in params:
            p.requires_grad_(mode)
            
            
    def _pos_enc(self, shape, device):
        B, C, W, H = shape
        gridx = torch.arange(0, W, device=device)
        if self.shifted:
            gridx -= 47
        gridy = torch.arange(0, H, device=device)
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
        if self.only_y_direc:
            mesh = mesh[:, 2:, :, :]
        return (mesh+1)/2

    def adp_pad(self, enc_out, x):
        _, _, s1_p, s2_p = enc_out.shape
        _, _, s1, s2 = x.shape
        pad1, pad2 = s1_p - s1, s2_p - s2
        
        x = F.pad(x, [0,pad2,0,pad1] , "constant", 0)
        return x
        

    def forward(self, x: Tensor, wavelength: Tensor, dl: Tensor) -> Tensor:
        # x [bs, inc, h, w] complex
        # wavelength [bs, 1] real
        # grid_step [bs, 2] real
        eps = x * (self.eps_max - self.eps_min) + self.eps_min  # [bs, inc*2, h, w] real
        
        grid = None
        # positional encoding
        if self.wave_prior:
            grid = self.encoding(
                eps, dl, wavelength, x.device
            )  # [bs, 2 or 4 or 8, h, w] real
        if grid is not None:
            x = torch.cat((x, grid), dim=1)  # [bs, inc*2+4, h, w] real

        if self.wave2wave:
            target = grid
            conv1 = self.dconv_down1(grid)
            
        else:
            target = None
            conv1 = self.dconv_down1(x)
            
        
        
        # DNN-based electric field envelop prediction
        
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        if self.wave_prior == False:
            x += self.wvl_layer(wavelength)[:, :, None, None]

        x = self.upsample1(x)
        x = self.adp_pad(conv3,x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample2(x)
        x = self.adp_pad(conv2,x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample3(x)
        x = self.adp_pad(conv1,x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)  # [bs, outc, h, w] real
        # convert to complex frequency-domain electric field envelops
        if self.wave2wave:
            return x, target
        else:
            return x


class OrigUNet_SFT(nn.Module):
    """
    Frequency-domain scattered electric field envelop predictor
    Assumption:
    (1) TE10 mode, i.e., Ey(r, omega) = Ez(r, omega) = 0
    (2) Fixed wavelength. wavelength currently not being modeled
    (3) Only predict Ex_scatter(r, omega)

    Args:
        PDE_NN_BASE ([type]): [description]
    """
    _conv = (FNOConv2d,)
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        dim: int = 16,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        eps_min: float = 1,
        eps_max: float = 1.4623**2,
        wave_prior: bool = True,
        wave2wave: bool = False,
        shifted:bool = False,
        only_y_direc: bool = False,
        load_path: str = None,
        only_sft_train: bool = False,
        with_cp=False,
    ):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        if wave2wave:
            if only_y_direc:
                self.in_channels = self.out_channels = 2    
            else:
                self.in_channels = self.out_channels = 4 
        else:
            if only_y_direc:
                self.in_channels =  2
            else:
                self.in_channels = 4
            self.out_channels = out_channels
        

        assert out_channels % 2 == 0, f"The output channels must be even number larger than 2, but got {out_channels}"
        self.dim = dim
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.wave_prior = wave_prior
        self.shifted = shifted
        self.wave2wave = wave2wave
        self.only_y_direc = only_y_direc
        self.load_path = load_path
        self.only_sft_train = only_sft_train
        
        self.with_cp = with_cp

        if not self.wave_prior:
            self.in_channels = in_channels

        # if not self.wave_prior or self.cond_path_flag:
        #     self.in_channels = in_channels
        
        self.device = device
        
        self.build_layers()
        self.reset_parameters()

        if wave2wave == False:
            self.load_weights(self.load_path, self.only_sft_train)


        self.permittivity_encoder = None
        self.set_trainable_permittivity(False)

    def reset_parameters(self, random_state: Optional[int] = None):
        for name, m in self.named_modules():
            if isinstance(m, self._conv):
                # if random_state is not None:
                #     # deterministic seed, but different for different layer, and controllable by random_state
                #     set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
    
    def build_layers(self):
        dim = self.dim

        self.dconv_down1 = double_conv(self.in_channels, dim, dim)
        self.dconv_down2 = double_conv(dim, dim * 2, dim * 2)
        self.dconv_down3 = double_conv(dim * 2, dim * 4, dim * 4)
        self.dconv_down4 = double_conv(dim * 4, dim * 8, dim * 8)

        # self.maxpool = nn.MaxPool2d(2)
        self.maxpool = nn.AvgPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample1 = nn.ConvTranspose2d(dim * 8, dim * 8, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(dim * 4, dim * 4, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(dim * 2, dim * 2, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(dim * 12, dim * 8, dim * 4)
        self.dconv_up2 = double_conv(dim * 6, dim * 4, dim * 2)
        self.dconv_up1 = double_conv(dim * 3, dim * 2, dim)
        self.drop_out = nn.Dropout2d(self.dropout_rate)
        self.conv_last = nn.Conv2d(dim, self.out_channels, 1)
        
        self.sft = SFT_layer(dim) # if wave2wave == True, -> Dont be used.

    def set_trainable_permittivity(self, mode: bool = True) -> None:
        self.trainable_permittivity = mode

    def requires_network_params_grad(self, mode: float = True) -> None:
        params = self.parameters()
        for p in params:
            p.requires_grad_(mode)
            
            
    def _pos_enc(self, shape, device):
        B, C, W, H = shape
        gridx = torch.arange(0, W, device=device)
        if self.shifted:
            gridx -= 47
        gridy = torch.arange(0, H, device=device)
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
        if self.only_y_direc:
            mesh = mesh[:, 2:, :, :]
        return (mesh+1)/2

    def load_weights(self, model_path, only_sft_train=True):
        
        trainable_layers = [self.upsample1, self.upsample2, self.upsample3,
                            self.dconv_up1, self.dconv_up2, self.dconv_up3, self.conv_last, self.sft]
        
        checkpoint = torch.load(model_path)
        pretrained_dict = checkpoint['model']
    
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(model_dict)
        
        if only_sft_train:
            for p in self.parameters():
                p.requires_grad = False
            for tl in trainable_layers:
                for p in tl.parameters():
                    p.requires_grad = True
            # for p in self.sft.parameters():
            #         p.requires_grad = True

    def adp_pad(self, enc_out, x):
        _, _, s1_p, s2_p = enc_out.shape
        _, _, s1, s2 = x.shape
        pad1, pad2 = s1_p - s1, s2_p - s2
        
        x = F.pad(x, [0,pad2,0,pad1] , "constant", 0)
        return x
        

    def forward(self, structure: Tensor, wavelength: Tensor, dl: Tensor) -> Tensor:
        # x [bs, inc, h, w] complex
        # wavelength [bs, 1] real
        # grid_step [bs, 2] real
        eps = structure * (self.eps_max - self.eps_min) + self.eps_min  # [bs, inc*2, h, w] real
        
        grid = None
        # positional encoding
        grid = self.encoding(
            eps, dl, wavelength, eps.device
        )  # [bs, 2 or 4 or 8, h, w] real
        

        if self.wave2wave:
            target = grid
        else:
            target = None
        # DNN-based electric field envelop prediction
        conv1 = self.dconv_down1(grid)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        if self.wave2wave == False:
            conv3 = self.sft(conv3, structure)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample1(x)
        x = self.adp_pad(conv3,x)
        x = torch.cat([x, conv3], dim=1)
        

        x = self.dconv_up3(x)
        x = self.upsample2(x)
        x = self.adp_pad(conv2,x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample3(x)
        x = self.adp_pad(conv1,x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)  # [bs, outc, h, w] real
        # convert to complex frequency-domain electric field envelops
        if self.wave2wave:            
            return x, target
        else:
            return x

