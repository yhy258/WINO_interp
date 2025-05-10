import os
from pathlib import Path
from collections import defaultdict
import sys
import yaml
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset.interpolative_dataset import transpose_data, field_process
from core.losses import NMAE
# from pyutils.torch_train import set_torch_deterministic
import cond_builder as cb
# from pyutils.general import AverageMeter
from builder import *
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

def test_yaml_to_args(config_path, model='fno2d'):
    parser = argparse.ArgumentParser(description='YAML to argparse', add_help=False)
    yaml_path = os.path.join(config_path, model+'.yaml')
    # yaml_path = os.path.join(parent_path, os.path.join('configs', model+'.yaml'))

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    add_config_to_parser(parser, config)
    
    return parser.parse_args()


def get_design_range(resolution, start=0.4, width=0.275):
    dl = 1e-6/resolution
    NPML = int(1/1e6/dl)
    subtrate_range = (0, 0 + int(start/1e6/dl))
    design_range = (subtrate_range[1],subtrate_range[1] + int(width/1e6/dl))
    return design_range



def init_attr(opt, key, value):
    if hasattr(opt, key) == False:
        print(f"{key} -> {value}")
        setattr(opt, key, value)
    return opt


@torch.no_grad()
def train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device = torch.device("cuda:0")
):
    mae_criterion = nn.L1Loss(reduction='none')
    mse_criterion = nn.MSELoss(reduction='none')
    nmae_criterion = NMAE(reduction='none')
    nmse_criterion = NMSE(reduction='none')
    
    
    nmse_losses = defaultdict(list)
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
                
        output = model(data, wavelength, dl)
                
        # mae_loss = mae_criterion(output.type(torch.float32), target.type(torch.float32))
        # mse_loss = mse_criterion(output.type(torch.float32), target.type(torch.float32))
        # nmae_loss = nmae_criterion(output.type(torch.float32), target.type(torch.float32))
        nmse_loss = nmse_criterion(output.type(torch.float32), target.type(torch.float32)) # This is our criterion.
        print(nmse_loss.shape)
        for i, w in enumerate(wavelength.view(-1).detach().cpu()):
            nmse_losses[str(w)].append(nmse_loss[i])
    
    return nmse_losses
        




def main(opt, model='wino', ckpt_root=''):
    parent_path = Path(__file__).parent.parent.resolve()

    if torch.cuda.is_available() and len(opt.device) == 1:
        device = f"cuda:{opt.device[0]}"
        device = torch.device(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(opt.deterministic) == True:
        set_torch_deterministic(int(opt.random_state))
        
    model = build_model(opt, opt.model, device)    
    
    ckpt_root = os.path.join(ckpt_root, opt.save_root.split(os.path.sep)[-1])
    

    if not ckpt_root.startswith('/'):
        model_path = os.path.join(parent_path, ckpt_root, 'best.pt')
    else:
        model_path = os.path.join(ckpt_root, 'best.pt')

    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    
    print("BUILD DATALOADER")
    opt.dataset_root = dataset_root
    train_loader = build_dataloader(opt, 'train', 12000)
    
    nmse_dict = train(model, train_loader, device)
    nmse_dict_new = {}
    for key in nmse_dict.keys():
        nmse_dict_new[key] = torch.tensor(nmse_dict[key])
    
    # train의 경우에는 fields를 저장하지 않는다.
    save_folder= f"retain_batch_train_results/{args.sim_name}/{opt.model}"
    file_name = opt.save_root.split(os.path.sep)[-1]
    os.makedirs(save_folder, exist_ok=True)
    torch.save(nmse_dict_new, os.path.join(save_folder, file_name + ".pt"))
    
    
    
if __name__ == '__main__':
    """
        GOAL : 다양한 data setting에 대해서 모두 수행하고 싶음.
        
        시뮬레이션 세팅에 따라 달라지는점. 
        1. Checkpoint root
        2. Dataset root
        3. Design region
        4. Permittivity
    """
    
    dataset_root_dict = {}
    
    ckpt_base = '/root/WINO_interp/checkpoints'
    eps_dict = {'triple_layer':{'eps_min':1.0, 'eps_max': 1.46**2}, 'straight_waveguide':{'eps_min':1.0, 'eps_max':2.25}, 'image_sensor': {'eps_min':1.0, 'eps_max':4.0}}
    scaling_dict = {'image_sensor': 1.33e13, 'triple_layer': 1.5e13, 'straight_waveguide': 1.25e13}
    
    # ckpt ex :  "/data/joon/Results/WINO/unziped_ckpt/image_sensor/checkpoints/wino/nmse_waveprior_64dim_12layer_256_5060_auggroup4_weightsharing/epoch200.pt"
    # dataset root : '/data/joon/Dataset/WINO/unziped_datasets/double_layer'
    parser = argparse.ArgumentParser(description="wino-ablation-study")
    parser.add_argument('--sim_name', type=str, default="triple_layer", help='The name of the simulation setting, [triple_layer, straight_waveguide, image_sensor]')
    parser.add_argument('--model', type=str, default="wino", help='The name of the model')
    args = parser.parse_args()
    
    
    ckpt_root = os.path.join(ckpt_base, args.sim_name, args.model)
    config_path = f'/root/WINO_interp/configs'
    dataset_root = f'/root/unziped_datasets/{args.sim_name}'
    
    
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
    args.field_scale_factor = scaling_dict[args.sim_name]
    args.dataset_root = dataset_root
    
    
    
    main(args, model=args.model, ckpt_root=ckpt_root)