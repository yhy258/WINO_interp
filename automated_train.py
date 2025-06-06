import os
import yaml
import argparse
from collections import defaultdict
import datetime
# from torchvision.transforms import GaussianBlur
import numpy as np
import torch
import torch.nn as nn
# from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler
# from pyutils.torch_train import set_torch_deterministic
# from pyutils.general import AverageMeter
import cond_builder as cb
from builder import *
from core.losses import NMAE, NMSE
from utils import time2file_name, ModelSaver
import argparse
import wandb
import random
from omegaconf import OmegaConf

def set_torch_deterministic(random_state: int = 0) -> None:
    random_state = int(random_state) % (2**32)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(random_state)
    random.seed(random_state)
    
    
class Meter(object):
    """Base class for Meters."""

    def __init__(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def reset(self):
        raise NotImplementedError

    @property
    def smoothed_value(self) -> float:
        """Smoothed value used for logging."""
        raise NotImplementedError


def safe_round(number, ndigits):
    if hasattr(number, "__round__"):
        return round(number, ndigits)
    elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
        return safe_round(number.item(), ndigits)
    elif np is not None and np.ndim(number) == 0 and hasattr(number, "item"):
        return safe_round(number.item(), ndigits)
    else:
        return number


def type_as(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.to(b)
    else:
        return a


class AverageMeter(Meter):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f", round= None) -> None:
        self.name = name
        self.fmt = fmt
        self.round = round
        self.reset()

    def reset(self):
        self.val = None  # most recent update
        self.sum = 0  # sum from all updates
        self.count = 0  # total n from all updates
        self.avg = 0

    def update(self, val, n=1):
        if val is not None:
            self.val = val
            if n > 0:
                self.sum = type_as(self.sum, val) + (val * n)
                self.count = type_as(self.count, n) + n
        self.avg = self.sum / self.count if self.count > 0 else self.val

    def state_dict(self):
        return {
            "val": self.val,
            "sum": self.sum,
            "count": self.count,
            "round": self.round,
        }

    def load_state_dict(self, state_dict):
        self.val = state_dict["val"]
        self.sum = state_dict["sum"]
        self.count = state_dict["count"]
        self.round = state_dict.get("round", None)

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def complex_mse_loss(output, target):
    return ((output - target)**2).mean(dtype=torch.float32)

def data2phase(data):
    # data.shape : BS, 2, H, W
    data = (data[:, 0, :, :] + 1j*data[:, 1, :, :])[:, None, :, :]
    phase = torch.angle(data)
    return torch.sin(phase) # for continuity


def data2attn(data):
    data = (data[:, 0, :, :] + 1j*data[:, 1, :, :])[:, None, :, :]
    power = data / torch.norm(data, dim=[2,3], keepdim=True)
    power = 1/1j * torch.log(power)
    return power.imag # attenuation.
    

def data2mag(data):
    data = (data[:, 0, :, :] + 1j*data[:, 1, :, :])[:, None, :, :]
    mag = torch.abs(data)
    return mag

def get_design_range(resolution, start=0.4, width=0.275):
    dl = 1e-6/resolution
    NPML = int(1/1e6/dl)
    subtrate_range = (0, 0 + int(start/1e6/dl))
    design_range = (subtrate_range[1],subtrate_range[1] + int(width/1e6/dl))
    return design_range

def get_near_field(data, design_range, resolution):
    # data : B, C, H, W
    start = design_range[1]
    mid_wavelength = 1 # max wavelength - 0.7 : 1µm near field.
    unit_micro = resolution
    distance = int(unit_micro * mid_wavelength)
    param_output = data[:, :, start: start + distance, :] 
    return param_output



# 272, 206
def add_config_to_parser(parser, config):
    for key, value in config.items():
        if isinstance(value, dict) and key != "aux_params":
            # If the value is a nested dictionary, recursively add it to the parser
            subparser = parser.add_argument_group(key)
            add_config_to_parser(subparser, value)
        elif key == "aux_params":
            parser.add_argument(f'--{key}', type=dict, default=value, help=f'{key} (default: {value})')
        else:
            # Assume the value is a string (you can modify this based on your YAML structure)
            parser.add_argument(f'--{key}', type=str, default=value, help=f'{key} (default: {value})')

def yaml_to_args(model='fno2d'):
    parser = argparse.ArgumentParser(description='YAML to argparse', add_help=False)

    ABS_REP_PATH = os.path.abspath(__file__).split(os.path.sep)[:-1]

    yaml_path = os.path.sep + os.path.join(*ABS_REP_PATH, 'configs', model+".yaml")

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    add_config_to_parser(parser, config)
    
    return parser.parse_args()


def train(
    opt,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    epoch: int,
    criterion,
    device: torch.device = torch.device("cuda:0"),
):
    model.train()

    step = epoch * len(train_loader)

    mse_meter = AverageMeter("mse")
    
    data_counter = 0

    for batch_idx, datas in enumerate(train_loader):
        if len(datas) == 7:
            data, ex, ey, target, dl, wavelength, _ = datas
        else:
            data, target, dl, wavelength = datas
        n = data.shape[0]
        data = data.type(torch.float32)
        wavelength = wavelength[:, None].type(torch.float32)
        dl = torch.cat((dl[:, None], dl[:, None]), dim=1)
        
        data = data.to(device)
        wavelength = wavelength.to(device)
        dl = dl.to(device)
        target = target.to(device)
        
        data_counter += data.shape[0]
        
        output = model(data, wavelength, dl)
                
        regression_loss = criterion(output.type(torch.float32), target.type(torch.float32))

        loss = regression_loss
        
        mse_meter.update(regression_loss.item())

        
        optimizer.zero_grad()
        loss.backward()
        
    
        optimizer.step()

        step += 1
        
    
    scheduler.step()

    return mse_meter.avg
    
    
def validate(
    opt,
    model: nn.Module,
    valid_loader: DataLoader,
    mae_criterion,
    mse_criterion,
    nmae_criterion,
    nmse_criterion,
    device: torch.device = torch.device("cuda:0")
):
    model.eval()
    
    cond = opt.cond
    val_loss = 0
    mae_meter = AverageMeter("mse")
    mse_meter = AverageMeter("mse")
    nmae_meter = AverageMeter("mse")
    nmse_meter = AverageMeter("mse")
    structure_mae_meter = AverageMeter("mse")
    structure_mse_meter = AverageMeter("mse")
    structure_nmae_meter = AverageMeter("mse")
    structure_nmse_meter = AverageMeter("mse")
    near_mae_meter = AverageMeter("mse")
    near_mse_meter = AverageMeter("mse")
    near_nmae_meter = AverageMeter("mse")
    near_nmse_meter = AverageMeter("mse")
    
    pred_images = []
    target_images = []
    structure_residue_images = []
    with torch.no_grad():
        for i, datas in enumerate(valid_loader):
            if len(datas) == 7:
                data, ex, ey, target, dl, wavelength, _ = datas
            else:
                data, target, dl, wavelength = datas
            data = data.type(torch.float32)
            
            # print(wavelength)
            wavelength = wavelength[:, None].type(torch.float32)
            
            dl = torch.cat((dl[:, None], dl[:, None]), dim=1)
            data = data.to(device)
            wavelength = wavelength.to(device)
            dl = dl.to(device)
            target = target.to(device)

        
            output = model(data, wavelength, dl)
    
            pred_images.append(
                wandb.Image(
                    output[0,0], caption="Wavelength : {}".format(wavelength[0])
                )
            )
            target_images.append(
                wandb.Image(
                    target[0,0], caption="Wavelength : {}".format(wavelength[0])
                )
            )
            mae_val_loss = mae_criterion(output, target)
            mse_val_loss = mse_criterion(output, target)
            nmae_val_loss = nmae_criterion(output, target)
            nmse_val_loss = nmse_criterion(output, target)
            mae_meter.update(mae_val_loss.item())
            mse_meter.update(mse_val_loss.item())
            nmae_meter.update(nmae_val_loss.item())
            nmse_meter.update(nmse_val_loss.item())        
            
            
            d_range = get_design_range(opt.resolution, opt.design_start, opt.width)
            param_output = output[:, :, d_range[0]:d_range[1], :]
            param_target = target[:, :, d_range[0]:d_range[1], :]
            
            near_output = get_near_field(output, d_range, 40)
            near_target = get_near_field(target, d_range, 40)
            
            structure_residue_images.append(
                wandb.Image(
                    torch.abs(param_output[0,0] - param_target[0,0]), caption="Wavelength : {}".format(wavelength[0])
                )
            )

            structure_mae_val_loss = mae_criterion(param_output, param_target)
            structure_mse_val_loss = mse_criterion(param_output, param_target)
            structure_nmae_val_loss = nmae_criterion(param_output, param_target)
            structure_nmse_val_loss = nmse_criterion(param_output, param_target)
            
            
            near_mae_val_loss = mae_criterion(near_output, near_target)
            near_mse_val_loss = mse_criterion(near_output, near_target)
            near_nmae_val_loss = nmae_criterion(near_output, near_target)
            near_nmse_val_loss = nmse_criterion(near_output, near_target)
            
            structure_mae_meter.update(structure_mae_val_loss.item())
            structure_mse_meter.update(structure_mse_val_loss.item())
            structure_nmae_meter.update(structure_nmae_val_loss.item())
            structure_nmse_meter.update(structure_nmse_val_loss.item())
            
            near_mae_meter.update(near_mae_val_loss.item())
            near_mse_meter.update(near_mse_val_loss.item())
            near_nmae_meter.update(near_nmae_val_loss.item())
            near_nmse_meter.update(near_nmse_val_loss.item())
            
    return mae_meter.avg, mse_meter.avg, nmae_meter.avg, nmse_meter.avg, \
        structure_mae_meter.avg, structure_mse_meter.avg, structure_nmae_meter.avg, structure_nmse_meter.avg, \
            near_mae_meter.avg, near_mse_meter.avg,  near_nmae_meter.avg, near_nmse_meter.avg, pred_images, target_images, structure_residue_images



def init_attr(opt, key, value):
    if hasattr(opt, key) == False:
        setattr(opt, key, value)
    return opt


def main(opt):
    opt.data = 'full_meep'
    opt = init_attr(opt, 'cond', False)
    model = opt.model
    print("Cond mode : ", opt.cond)
    
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    ckpt_base = args.ckpt_base
    ckpt_root = os.path.join(ckpt_base, args.sim_name, args.model)
    model_save_path = os.path.join(ckpt_root, args.save_root.split(os.path.sep)[-1])
    # model_save_path = os.path.join(opt.save_root.split(os.path.sep)[0], opt.sim_name, os.path.sep.join(opt.save_root.split(os.path.sep)[1:]))
    os.makedirs(model_save_path, exist_ok=True)
    # TRAIN
    # Utils initialization.
    project_name= "AUTOMATED WAVE INTERPOLATION"
    wandb.init(
        project = project_name,
        name=opt.model +" data 12000",
    )
    wandb.config.update(opt)

    saver = ModelSaver(opt, best=False)
    best_saver = ModelSaver(opt, best=True)
    
    if torch.cuda.is_available() and len(opt.device) == 1:
        device = f"cuda:{opt.device[0]}"
        device = torch.device(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False
        
    if int(opt.deterministic) == True:
        set_torch_deterministic(int(opt.random_state))
        
    # BUILD DATALOADER
    print("BUILD DATALOADER")
    train_loader = build_dataloader(opt, 'train', 12000)
    valid_loader = build_dataloader(opt, 'valid')
    train_wvl_valid_loader = build_dataloader(opt, 'train_wvl_valid')
   
    data_range_dict = train_loader.dataset.data_range
    
    opt.data_range_dict = data_range_dict
    
    # BUILD MODEL
    print("BUILD MODEL")
    if opt.cond:
        model = cb.build_model(opt, model, device)
    else:
        model = build_model(opt, model, device)
    
    ### optimizer, scheduler, criterion
    print("BUILD OPTIMIZER")
    optimizer = build_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        name=opt.optimizer_name,
        opt=opt,
    )
    
    reduction = 'mean'
    criterion = build_criterion(opt.criterion_name, opt, reduction=reduction)
    criterion = criterion.to(device)
    
    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    nmae_criterion = NMAE()
    nmse_criterion = NMSE()

    scheduler = build_scheduler(optimizer, opt=opt)
    
    train_losses = []
    train_wvl_nmae_param_valid_losses = []
    train_wvl_nmse_param_valid_losses = []
    train_wvl_mae_param_valid_losses = []
    train_wvl_mse_param_valid_losses = []
    train_wvl_nmae_valid_losses = []
    train_wvl_nmse_valid_losses = []
    train_wvl_mae_valid_losses = []
    train_wvl_mse_valid_losses = []
    train_wvl_nmae_near_valid_losses = []
    train_wvl_nmse_near_valid_losses = []
    train_wvl_mae_near_valid_losses = []
    train_wvl_mse_near_valid_losses = []
    
    nmae_valid_losses = []
    nmse_valid_losses = []
    mae_valid_losses = []
    mse_valid_losses = []
    param_nmae_valid_losses = []
    param_nmse_valid_losses = []
    param_mse_valid_losses = []
    param_mae_valid_losses = []
    
    near_nmae_valid_losses = []
    near_nmse_valid_losses = []
    near_mse_valid_losses = []
    near_mae_valid_losses = []
    
    epoch = 0
    train_avg = 1.0
    
    train_avg = 1.0
    
    
    print("START TRAIN!")
    wandb.watch(model, criterion, log='all', log_freq=10)

    for epoch in range(1, int(opt.n_epochs) + 1):
        train_avg = train(
            opt=opt,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            criterion=criterion,
            device=device,
        )
        train_losses.append(train_avg)
       
       
        # Validation for unseen wavelengths during training.
        mae_valid_avg, mse_valid_avg, nmae_valid_avg, nmse_valid_avg,\
            mae_param_valid_avg, mse_param_valid_avg, nmae_param_valid_avg, nmse_param_valid_avg,\
            mae_near_valid_avg, mse_near_valid_avg, nmae_near_valid_avg, nmse_near_valid_avg,\
                pred_images, target_images, structure_residue_image = validate(
            opt=opt,
            model=model,
            valid_loader=valid_loader,
            mae_criterion=mae_criterion,
            mse_criterion=mse_criterion,
            nmae_criterion=nmae_criterion,
            nmse_criterion=nmse_criterion,
            device=device
        )
         
        
        # Validation for seen wavelengths during training.
        train_wvl_mae_valid_avg, train_wvl_mse_valid_avg, train_wvl_nmae_valid_avg, train_wvl_nmse_valid_avg,\
            train_wvl_mae_param_valid_avg, train_wvl_mse_param_valid_avg, train_wvl_nmae_param_valid_avg, train_wvl_nmse_param_valid_avg, \
                train_wvl_mae_near_valid_avg, train_wvl_mse_near_valid_avg, train_wvl_nmae_near_valid_avg, train_wvl_nmse_near_valid_avg,\
                train_wvl_pred_images, train_wvl_target_images, train_wvl_structure_residue_image = validate(
            opt=opt,
            model=model,
            valid_loader=train_wvl_valid_loader,
            mae_criterion=mae_criterion,
            mse_criterion=mse_criterion,
            nmae_criterion=nmae_criterion,
            nmse_criterion=nmse_criterion,
            device=device
        )       
                
        wandb.log({
            'train_loss' : train_avg,
            'mae_valid_loss' : mae_valid_avg,
            'mse_valid_loss' : mse_valid_avg,
            'nmae_valid_loss' : nmae_valid_avg,
            'nmse_valid_loss' : nmse_valid_avg,
            'mae_param_valid_loss' : mae_param_valid_avg,
            'mse_param_valid_loss' : mse_param_valid_avg,
            'nmae_param_valid_loss' : nmae_param_valid_avg,
            'nmse_param_valid_loss' : nmse_param_valid_avg,
            'mae_near_valid_loss' : mae_near_valid_avg,
            'mse_near_valid_loss' : mse_near_valid_avg,
            'nmae_near_valid_loss' : nmae_near_valid_avg,
            'nmse_near_valid_loss' : nmse_near_valid_avg,
            'pred_images' : pred_images,
            'target_images' : target_images,
            'param_residue_image' : structure_residue_image,
            'train_wvl_mae_valid_loss' : train_wvl_mae_valid_avg,
            'train_wvl_mse_valid_loss' : train_wvl_mse_valid_avg,
            'train_wvl_nmae_valid_loss' : train_wvl_nmae_valid_avg,
            'train_wvl_nmse_valid_loss' : train_wvl_nmse_valid_avg,
            'train_wvl_mae_param_valid_loss' : train_wvl_mae_param_valid_avg,
            'train_wvl_mse_param_valid_loss' : train_wvl_mse_param_valid_avg,
            'train_wvl_nmae_param_valid_loss' : train_wvl_nmae_param_valid_avg,
            'train_wvl_nmse_param_valid_loss' : train_wvl_nmse_param_valid_avg,
            'train_wvl_mae_near_valid_loss' : train_wvl_mae_near_valid_avg,
            'train_wvl_mse_near_valid_loss' : train_wvl_mse_near_valid_avg,
            'train_wvl_nmae_near_valid_loss' : train_wvl_nmae_near_valid_avg,
            'train_wvl_nmse_near_valid_loss' : train_wvl_nmse_near_valid_avg,
            'train_wvl_pred_images' : train_wvl_pred_images,
            'train_wvl_target_images' : train_wvl_target_images,
            'train_wvl_param_residue_images' : train_wvl_structure_residue_image,
            })

        
        """
            valid_loss : unseen wavelength
            train_wvl_~ : seen wavelength
        
        """
        valid_loss_dict = {'mae': mae_valid_avg, 'mse' : mse_valid_avg, 'nmae' : nmae_valid_avg, 'nmse': nmse_valid_avg} 
        
        valid_avg = valid_loss_dict[opt.criterion_name]
        train_wvl_nmae_valid_losses.append(train_wvl_nmae_valid_avg)
        train_wvl_nmse_valid_losses.append(train_wvl_nmse_valid_avg)
        train_wvl_mae_valid_losses.append(train_wvl_mae_valid_avg)
        train_wvl_mse_valid_losses.append(train_wvl_mse_valid_avg)
        train_wvl_nmae_param_valid_losses.append(train_wvl_nmae_param_valid_avg)
        train_wvl_nmse_param_valid_losses.append(train_wvl_nmse_param_valid_avg)
        train_wvl_mae_param_valid_losses.append(train_wvl_mae_param_valid_avg)
        train_wvl_mse_param_valid_losses.append(train_wvl_mse_param_valid_avg)
        train_wvl_nmae_near_valid_losses.append(train_wvl_nmae_near_valid_avg)
        train_wvl_nmse_near_valid_losses.append(train_wvl_nmse_near_valid_avg)
        train_wvl_mae_near_valid_losses.append(train_wvl_mae_near_valid_avg)
        train_wvl_mse_near_valid_losses.append(train_wvl_mse_near_valid_avg)
        
        mse_valid_losses.append(mse_valid_avg)
        nmae_valid_losses.append(nmae_valid_avg)
        nmse_valid_losses.append(nmse_valid_avg)
        mae_valid_losses.append(mae_valid_avg)
        mse_valid_losses.append(mse_valid_avg)
        param_nmae_valid_losses.append(nmae_param_valid_avg)
        param_nmse_valid_losses.append(nmse_param_valid_avg)
        param_mse_valid_losses.append(mse_param_valid_avg)
        param_mae_valid_losses.append(mae_param_valid_avg)
        near_nmae_valid_losses.append(nmae_near_valid_avg)
        near_nmse_valid_losses.append(nmse_near_valid_avg)
        near_mae_valid_losses.append(mae_near_valid_avg)
        near_mse_valid_losses.append(mse_near_valid_avg)
        
        
        
        if epoch == opt.n_epochs:
            loss_dict = {'train_loss' : train_losses, 
                            'train_wvl_nmae_valid_loss' : train_wvl_nmae_valid_losses,
                            'train_wvl_nmse_valid_loss' : train_wvl_nmse_valid_losses,
                            'train_wvl_mae_valid_loss' : train_wvl_mae_valid_losses,
                            'train_wvl_mse_valid_loss' : train_wvl_mse_valid_losses,
                            'train_wvl_param_nmae_valid_loss' : train_wvl_nmae_param_valid_losses,
                            'train_wvl_param_nmse_valid_loss' : train_wvl_nmse_param_valid_losses,
                            'train_wvl_param_mae_valid_loss': train_wvl_mae_param_valid_losses,
                            'train_wvl_param_mse_valid_loss': train_wvl_mse_param_valid_losses,
                            'train_wvl_near_nmae_valid_loss' : train_wvl_nmae_near_valid_losses,
                            'train_wvl_near_nmse_valid_loss' : train_wvl_nmse_near_valid_losses,
                            'train_wvl_near_mae_valid_loss': train_wvl_mae_near_valid_losses,
                            'train_wvl_near_mse_valid_loss': train_wvl_mse_near_valid_losses,
                            'nmae_valid_loss' : nmae_valid_losses,
                            'nmse_valid_loss' : nmse_valid_losses,
                            'mae_valid_loss' : mae_valid_losses,
                            'mse_valid_loss' : mse_valid_losses,
                            'param_nmae_valid_loss' : param_nmae_valid_losses,
                            'param_nmse_valid_loss' : param_nmse_valid_losses,
                            'param_mae_valid_loss': param_mae_valid_losses,
                            'param_mse_valid_loss': param_mse_valid_losses,
                            'near_nmae_valid_loss' : near_nmae_valid_losses,
                            'near_nmse_valid_loss' : near_nmse_valid_losses,
                            'near_mae_valid_loss': near_mae_valid_losses,
                            'near_mse_valid_loss': near_mse_valid_losses
                            }
            saver(valid_avg, model, optimizer, scheduler, epoch, loss_dict, model_save_path)
        else:
            loss_dict = None
            best_saver(valid_avg, model, optimizer, scheduler, epoch, loss_dict, model_save_path)
    
    
    
    
if __name__ == "__main__":
    """
        GOAL : 다양한 data setting에 대해서 모두 수행하고 싶음.
        
        시뮬레이션 세팅에 따라 달라지는점. 
        1. Checkpoint root
        2. Dataset root
        3. Design region
        4. Permittivity
    """
    
    eps_dict = {'single_layer': {'eps_min':1.0, 'eps_max': 1.46**2},'triple_layer':{'eps_min':1.0, 'eps_max': 1.46**2}, 'straight_waveguide':{'eps_min':1.0, 'eps_max':2.25}, 'image_sensor': {'eps_min':1.0, 'eps_max':4.0}}
    scaling_dict = {'single_layer': 2.5e13,'image_sensor': 1.33e13, 'triple_layer': 1.5e13, 'straight_waveguide': 1.25e13}
    design_range_dict = {'single_layer': {'design_start':0.4, 'width': 0.12}, 'triple_layer': {'design_start':0.4, 'width': 0.12}, 'straight_waveguide': {'design_start':1.0, 'width':4.85 }, 'image_sensor': {'design_start':0.6, 'width': 3.5}}
    
    parser = argparse.ArgumentParser(description="wino-ablation-study")
    parser.add_argument('--sim_name', type=str, default="single_layer", help='The name of the simulation setting, [triple_layer, straight_waveguide, image_sensor]')
    parser.add_argument('--model', type=str, default="wino", help='The name of the model')
    # args.dataset_root = f'/root/unziped_datasets/{args.sim_name}'
    parser.add_argument('--dataset_root', type=str, default="/root/unziped_datasets/single_layer", help='The root of the dataset')
    parser.add_argument('--ckpt_base', type=str, default="checkpoints", help='The root of the dataset')
    args = parser.parse_args()
    
    config_path = f'configs'
    full_config_path = os.path.join(config_path, args.model+".yaml")
    config = OmegaConf.load(full_config_path)
    print(repr(config))
    
    keys = config.keys()
    for k in keys:
        child_dict = config._get_child(k)
        for ck, cv in child_dict.items():
            args = init_attr(args, ck, cv) # ck가 없으면 cv로 초기화
    
    data_eps_dict = eps_dict[args.sim_name]
    args.eps_min = data_eps_dict['eps_min']
    args.eps_max = data_eps_dict['eps_max']
    args.design_start = design_range_dict[args.sim_name]['design_start']
    args.width = design_range_dict[args.sim_name]['width']
    args.field_scale_factor = scaling_dict[args.sim_name]
    
    main(args)
    
    
    
    
    
