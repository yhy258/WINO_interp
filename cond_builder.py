"""
    Based on https://github.com/JeremieMelo/NeurOLight/blob/main/core/builder.py
    NeurOLight: A Physics-Agnostic Neural Operator Enabling Parametric Photonic Device Simulation
    MIT License
    Copyright (c) 2022 Jiaqi Gu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# from core.models import *
from core.cond_models import *
from core.losses import *
# from pyutils.optimizer.sam import SAM
# from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from dataset.interpolative_dataset import *
from torch.utils.data import DataLoader


def build_model(opt, model, device):
    print("COND!")
    model = model.lower().split('_')
    if "fno2d" in model:
        model = FNO2d(
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            dim=opt.dim,
            kernel_list=opt.kernel_list,
            kernel_size_list=opt.kernel_size_list,
            padding_list=opt.padding_list,
            hidden_list=opt.hidden_list,
            act_func=opt.act_func,
            mode_list=opt.mode_list,
            dropout=opt.dropout,
            drop_path_rate=opt.drop_path_rate,
            sparsity_threshold=opt.sparsity_threshold,
            device=device
        ).to(device)
        # only unet
    elif 'unet' in model: # if fourier_block=True -> fno Unet
        model = CondUNet(
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            dim=opt.dim,
            blk_num=opt.blk_num,
            act_func=opt.act_func,
            norm=opt.norm,
            has_attn=opt.has_attn,
            dropout=opt.dropout,
            cond_bias=opt.cond_bias,
            scale_shift_norm=True,
            device=device
        ).to(device)
    elif 'fno2dfactor' in model:
        model = FactorFNO2d(
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            dim=opt.dim,
            kernel_list=opt.kernel_list,
            kernel_size_list=opt.kernel_size_list,
            padding_list=opt.padding_list,
            hidden_list=opt.hidden_list,
            act_func=opt.act_func,
            mode_list=opt.mode_list,
            dropout=opt.dropout,
            drop_path_rate=opt.drop_path_rate,
            device=device
        ).to(device)
    else:
        raise Exception('invalid model name')
    return model



def build_optimizer(params, name: str = None, opt=None):
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=opt.optimizer_lr,
            momentum=opt.optimizer_momentum,
            weight_decay=opt.optimizer_weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=opt.optimizer_lr,
            weight_decay=opt.optimizer_weight_decay,
            betas=getattr(opt, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=opt.optimizer_lr,
            weight_decay=opt.optimizer_weight_decay,
        )
    # elif name == "sam_sgd":
    #     base_optimizer = torch.optim.SGD
    #     optimizer = SAM(
    #         params,
    #         base_optimizer=base_optimizer,
    #         rho=getattr(opt, "rho", 0.5),
    #         adaptive=getattr(opt, "adaptive", True),
    #         lr=opt.optimizer_lr,
    #         weight_decay=opt.optimizer_weight_decay,
    #         momenum=0.9,
    #     )
    # elif name == "sam_adam":
    #     base_optimizer = torch.optim.Adam
    #     optimizer = SAM(
    #         params,
    #         base_optimizer=base_optimizer,
    #         rho=getattr(opt, "rho", 0.001),
    #         adaptive=getattr(opt, "adaptive", True),
    #         lr=opt.optimizer_lr,
    #         weight_decay=opt.optimizer_weight_decay,
    #     )
    else:
        raise NotImplementedError(name)

    return optimizer


def build_scheduler(optimizer, name: str = None, opt=None):
    name = (name or opt.scheduler_name).lower()
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(opt.n_epochs), eta_min=float(opt.scheduler_lr_min)
        )
    # elif name == "cosine_warmup":
    #     scheduler = CosineAnnealingWarmupRestarts(
    #         optimizer,
    #         first_cycle_steps=opt.n_epochs,
    #         max_lr=opt.optimizer_lr,
    #         min_lr=opt.scheduler_lr_min,
    #         warmup_steps=int(opt.scheduler_warmup_steps),
    #     )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.scheduler_lr_gamma)
    else:
        raise NotImplementedError(name)

    return scheduler


def build_criterion(name: str = None, criterion_dict=None, criterion_scale=None, device='cpu') -> nn.Module:
    name = name.lower()
    if name == "nll":
        criterion = nn.NLLLoss().to(device)
    elif name == "mse":
        criterion = nn.MSELoss().to(device)
    elif name == "mae":
        criterion = nn.L1Loss().to(device)
    elif name == "ce":
        criterion = nn.CrossEntropyLoss().to(device)
    elif name == "nmse":
        criterion = NMSE().to(device)
    elif name == "nmae":
        criterion = NMAE().to(device)
    else:
        raise NotImplementedError(name)
    return criterion