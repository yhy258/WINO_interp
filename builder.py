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
from core.models import *
# from core.cond_models import *
from core.losses import *
from pyutils.optimizer.sam import SAM
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from dataset.interpolative_dataset import FullSimDataset
from torch.utils.data import DataLoader
from CevicheSim.simulation import *

def build_dataloader(opt, mode):
    if opt.mode == "train":
        shuffle = True
    else:
        shuffle = False
    dataset_root = os.path.join(opt.dataset_root, f'ceviche_train')
    valid_dataset_root = os.path.join(opt.dataset_root, f'ceviche_valid')

    dataset = FullSimDataset(dataset_root, valid_dataset_root, opt.dataset_folder_name, mode=mode, normalize=opt.normalize, step=opt.step)

    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    return dataloader

def build_model(opt, model, device):
    model = model.lower().split('_')
    if "fno2d" in model:
        model = FNO2d(
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            dim=opt.dim,
            padding=opt.padding,
            kernel_list=opt.kernel_list,
            kernel_size_list=opt.kernel_size_list,
            padding_list=opt.padding_list,
            hidden_list=opt.hidden_list,
            act_func=opt.act_func,
            mode_list=opt.mode_list,
            dropout=opt.dropout,
            drop_path_rate=opt.drop_path_rate,
            wave_prior=opt.wave_prior,
            wp_input_cat=opt.wp_input_cat,
            freqlinear=opt.freqlinear,
            waveprior_concatenate=opt.waveprior_concatenate,
            positional_encoding=opt.positional_encoding,
            eps_info=opt.eps_info,
            device=device
        ).to(device)
    elif 'neurolight' in model:
        model = NeurOLight2d(
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            dim=opt.dim,
            padding=opt.padding,
            kernel_list=opt.kernel_list,
            kernel_size_list=opt.kernel_size_list,
            padding_list=opt.padding_list,
            hidden_list=opt.hidden_list,
            act_func=opt.act_func,
            mode_list=opt.mode_list,
            dropout=opt.dropout,
            drop_path_rate=opt.drop_path_rate,
            res_stem=opt.resstem,
            device=device
        ).to(device)
    elif 'wino' in model:
        if opt.wave_prior:
            model = WINO(
                in_channels=opt.in_channels,
                out_channels=opt.out_channels,
                dim=opt.dim,
                num_blocks=opt.num_blocks,
                padding=opt.padding,
                modes=opt.mode_list,
                expand_list=opt.expand_list,
                hidden_list=opt.hidden_list,
                act_func=opt.act_func,
                hid_num_blocks=opt.hid_num_blocks,
                ch_process_block=True,
                wp_addition=False,
                wp_mult=opt.wp_mult,
                pre_spec=True,
                wp_inputcat=opt.wp_inputcat,
                sparsity_threshold=0,
                fourier_bias=opt.fourier_bias,
                stem_dropout=opt.stem_dropout,
                feature_dropout=opt.feature_dropout,
                head_dropout=opt.head_dropout,
                drop_path_rate=opt.drop_path_rate,
                wave_prior=opt.wave_prior,
                batch_norm=False,
                depthwise_conv_module=False,
                non_linear_stem=False,
                in_ch_sf=opt.in_ch_sf,
                weight_sharing=opt.weight_sharing,
                inner_nonlinear=False,
                ffn_simplegate=False,
                positional_encoding=opt.positional_encoding,
                eps_info=opt.eps_info,
                device=device
            ).to(device)
        else:
            model = ModulationWINO(
                in_channels=opt.in_channels,
                out_channels=opt.out_channels,
                dim=opt.dim,
                num_blocks=opt.num_blocks,
                padding=opt.padding,
                modes=opt.mode_list,
                expand_list=opt.expand_list,
                hidden_list=opt.hidden_list,
                act_func=opt.act_func,
                hid_num_blocks=opt.hid_num_blocks,
                ch_process_block=True,
                sparsity_threshold=0,
                fourier_bias=opt.fourier_bias,
                stem_dropout=opt.stem_dropout,
                feature_dropout=opt.feature_dropout,
                head_dropout=opt.head_dropout,
                drop_path_rate=opt.drop_path_rate,
                batch_norm=False,
                depthwise_conv_module=False,
                non_linear_stem=False,
                in_ch_sf=opt.in_ch_sf,
                weight_sharing=opt.weight_sharing,
                inner_nonlinear=False,
                ffn_simplegate=False,
                fourier_lift_proj=False,
                positional_encoding=opt.positional_encoding,
                eps_info=opt.eps_info,
                device=device
            ).to(device)
    elif "unet" in model: # if fourier_block=True -> fno Unet
        model = UNet(
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            dim=opt.dim,
            blk_num=opt.blk_num,
            act_func=opt.act_func,
            norm=opt.norm,
            wave_prior=opt.wave_prior,
            has_attn=opt.has_attn,
            dropout=opt.dropout,
            blueprint=opt.blueprint,
            propagate_resonance_prior=opt.propagate_resonance_prior,
            neural_code=opt.neural_code,
            cond_path_flag=opt.cond_path_flag,
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
            dropout_rate=opt.dropout,
            drop_path_rate=opt.drop_path_rate,
            wave_prior=opt.wave_prior,
            wp_mult=opt.wp_mult,
            positional_encoding=opt.positional_encoding,
            eps_info=opt.eps_info,
            freqlinear=opt.freqlinear, # modulation.
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
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(opt, "rho", 0.5),
            adaptive=getattr(opt, "adaptive", True),
            lr=opt.optimizer_lr,
            weight_decay=opt.optimizer_weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(opt, "rho", 0.001),
            adaptive=getattr(opt, "adaptive", True),
            lr=opt.optimizer_lr,
            weight_decay=opt.optimizer_weight_decay,
        )
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
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=opt.n_epochs,
            max_lr=opt.optimizer_lr,
            min_lr=opt.scheduler_lr_min,
            warmup_steps=int(opt.scheduler_warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.scheduler_lr_gamma)
    else:
        raise NotImplementedError(name)

    return scheduler


def build_criterion(name: str = None, criterion_dict=None, criterion_scale=None, reduction='mean', device='cpu') -> nn.Module:
    name = name.lower()
    if name == "nll":
        criterion = nn.NLLLoss(reduction=reduction).to(device)
    elif name == "mse":
        criterion = nn.MSELoss(reduction=reduction).to(device)
    elif name == "mae":
        criterion = nn.L1Loss(reduction=reduction).to(device)
    elif name == "ce":
        criterion = nn.CrossEntropyLoss(reduction=reduction).to(device)
    elif name == "nmae":
        criterion = NMAE().to(device)
    elif name == "nmse":
        criterion = NMSE().to(device)
    elif name == "fft_mse":
        criterion = SpectralMatchingLoss(norm=False, criterion_mode='mse').to(device)
    elif name == "fft_mae":
        criterion = SpectralMatchingLoss(norm=False, criterion_mode='mae').to(device)
    elif name == "fft_nmse":
        criterion = SpectralMatchingLoss(norm=True, criterion_mode='mse').to(device)
    elif name == "fft_nmae":
        criterion = SpectralMatchingLoss(norm=True, criterion_mode='mae').to(device)
    else:
        raise NotImplementedError(name)
    return criterion
