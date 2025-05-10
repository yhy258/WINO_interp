import os
import math
import logging
from functools import reduce
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torch.types import Device
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


def save_args(args, path):
    file_name = 'args.txt'
    args_path = os.path.join(path, file_name)
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def open_args(path, ipykernel=True):
    argparser = argparse.ArgumentParser()
    
    args = argparser.parse_args(args=[])
    file_name = 'args.txt'
    args_path = os.path.join(path, file_name)
    with open(args_path, 'r') as f:
        args.__dict__ = json.load(f)
    return args


class ModelSaver():
    def __init__(self, opt, best=False):
        self.min_avg = math.inf
        self.opt = opt
        self.best = best
    
    def __call__(self, avg, model, optimizer, scheduler, epoch, loss_dict, log_dir):
        if avg < self.min_avg:
            self.min_avg = avg
        
            self.save(model, optimizer, scheduler, epoch, loss_dict, log_dir)
            
    def save(self, model, optimizer, scheduler, epoch, loss_dict, log_dir):
        if self.best:
            file_name = f'best.pt'
        else:
            file_name = f'epoch{epoch}.pt'
        if scheduler == None:
            torch.save({
                    "epoch" : epoch,
                    "model": model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    'loss_dict' : loss_dict
                }, os.path.join(log_dir, file_name))
        else:
            torch.save({
                    "epoch" : epoch,
                    "model": model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "scheduler" : scheduler.state_dict(),
                    'loss_dict': loss_dict
                }, os.path.join(log_dir, file_name))
    
            


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def line_read_losses_with_inter(line):
    line_chunks = line.split(',')
    interested = line_chunks[-3:]
    train_l = float(interested[0].split(":")[1][1:])
    valid_l = float(interested[1].split(":")[1][1:])
    inter_valid_l = float(interested[2].split(":")[1][1:-2])
    return train_l, valid_l, inter_valid_l


def read_losses_with_inter(path):
    train_losses, valid_losses, inter_valid_losses = [], [], []
    with open(path, 'r') as f:
        for i in range(200):
            this_line = f.readline()
            train_l, valid_l, inter_valid_l = line_read_losses_with_inter(this_line)
            train_losses.append(train_l)
            valid_losses.append(valid_l)
            inter_valid_losses.append(inter_valid_l)
    train_losses = np.stack(train_losses)
    valid_losses = np.stack(valid_losses)
    inter_valid_losses = np.stack(inter_valid_losses)
    return train_losses, valid_losses, inter_valid_losses



def normalize(x):
    if isinstance(x, np.ndarray):
        x_min, x_max = np.percentile(x, 5), np.percentile(x, 95)
        x = np.clip(x, x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    else:
        x_min, x_max = torch.quantile(x, 0.05), torch.quantile(x, 0.95)
        x = x.clamp(x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    return x

# incident angle에 대해서도 확인 할 필요가 있다.
def plot_compare(
    epsilon: Tensor,
    pred_fields: Tensor,
    target_fields: Tensor,
    wavelength: Tensor,
    filepath: str,
    pol: str = "Hz",
    norm: bool = True,
    wavelength_range : list = [950, 1050]
):

    if epsilon.is_complex():
        epsilon = epsilon.real
    
    wl_min, wl_max = wavelength_range
    wavelength = wavelength * (wl_max - wl_min) + wl_min
    
    field_val = pred_fields.data.cpu().numpy()
    target_field_val = target_fields.data.cpu().numpy()
    # eps_r = simulation.eps_r
    eps_r = epsilon.data.cpu().numpy()
    err_field_val = field_val - target_field_val
    
    field_val = field_val.real
    # target_field_val = np.abs(target_field_val)
    target_field_val = target_field_val.real
    err_field_val = np.abs(err_field_val)
    outline_val = np.abs(eps_r)
    
    # vmax = field_val.max()
    vmin = 0.0
    b = field_val.shape[0]
    fig, axes = plt.subplots(3, b, constrained_layout=True, figsize=(5 * b, 3.1))
    if b == 1:
        axes = axes[..., np.newaxis]
    cmap = "magma"
    vmin = np.min(target_field_val.min())
    for i in range(b):
        vmax = np.max(target_field_val[i])
        # vmin = np.min(target_field_val[i])
        if norm:
            h1 = axes[0, i].imshow(normalize(field_val[i]), cmap=cmap, origin="lower")
            h2 = axes[1, i].imshow(normalize(target_field_val[i]), cmap=cmap, origin="lower")
        else:
            h1 = axes[0, i].imshow(field_val[i], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            h2 = axes[1, i].imshow(target_field_val[i], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        h3 = axes[2, i].imshow(err_field_val[i], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        for j in range(3):
            divider = make_axes_locatable(axes[j, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            fig.colorbar([h1, h2, h3][j], label=pol, ax=axes[j, i], cax=cax)
        axes[0, i].title.set_text(
            f"{wavelength[i].item():.2f} nm)"
        )
        # fig.colorbar(h2, label=pol, ax=axes[1,i])
        # fig.colorbar(h3, label=pol, ax=axes[2,i])

    # Do black and white so we can see on both magma and RdBu
    for ax in axes.flatten():
        # ax.contour(outline_val[0], levels=2, linewidths=1.0, colors="w")
        # ax.contour(outline_val[0], levels=2, linewidths=0.5, colors="k")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.savefig(filepath, dpi=150)
    plt.close()

