import os
from pathlib import Path
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset.interpolative_dataset import transpose_data, field_process
from core.losses import NMAE
from pyutils.torch_train import set_torch_deterministic
from meep_train import yaml_to_args
import cond_builder as cb
from pyutils.general import AverageMeter
from builder import *

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

def test_yaml_to_args(parent_path, model='fno2d'):
    parser = argparse.ArgumentParser(description='YAML to argparse', add_help=False)
    yaml_path = os.path.join(parent_path, os.path.join('configs', model+'.yaml'))

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    add_config_to_parser(parser, config)
    
    return parser.parse_args()

def get_design_range(resolution, width=0.275):
    dl = 1e-6/resolution
    NPML = int(1/1e6/dl)
    subtrate_range = (0, 0 + int(0.4/1e6/dl))
    design_range = (subtrate_range[1],subtrate_range[1] + int(width/1e6/dl))
    return design_range

def get_near_field(data, design_range, wavelength):
    # data : B, C, H, W
    start = design_range[1]
    mid_wavelength = wavelength
    unit_micro = 40 # resolution
    distance = int(unit_micro * mid_wavelength)
    param_output = data[:, :, start: start + distance, :] 
    return param_output

def get_dataset(path):
    wvlslab  = list(range(400, 701))
    data_dict = dict()
    for wvl in wvlslab:
        this_path = os.path.join(path, str(wvl))
        epss, ezs, dls, wvls = get_data(this_path)
        this_wvl = wvls[0].item()
        data_dict[str(wvl)] = (epss, ezs, dls, wvls)
    return data_dict # data_dict [wvl] : N, H, W...
        
        
def get_data(path):
    min_eps = 1
    # max_eps = 1.46**2
    max_eps = 1.46**2
    field_scale_factor = 2.5e13
    fns = os.listdir(path)
    epss = []
    ezs = []
    dls = []
    wvls = []
    for fn in fns:
        tp = os.path.join(path, fn)
        data = np.load(tp)
        # print(data)
        ezs.append(data['Ez'])
        full_geo = data['eps']
        epss.append(full_geo)
        dls.append(1/40)
        wvls.append(data['wavelength'])
    epss = np.stack(epss)
    ezs = np.stack(ezs) # N, H, W
    dls = np.stack(dls)
    wvls = np.stack(wvls)
    
    epss, ezs = map(transpose_data, [epss, ezs])
    epss = torch.tensor(epss)[:, None, :, :]
    ezs = field_process(ezs)
    dls = torch.tensor(dls)
    wvls = torch.tensor(wvls)
    
    epss = (epss - min_eps) / (max_eps - min_eps)
    ezs *= field_scale_factor
    
    return epss, ezs, dls, wvls # 동일 wavlenegths
    

    
def wvlwise_test(opt, model, dataset, device, file_name):
    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    nmae_criterion = NMAE()
    nmse_criterion = NMSE()

    wvls  = list(range(400, 701))
    
    mae_dict = dict()
    mse_dict = dict()
    nmae_dict = dict()
    nmse_dict = dict()
    structure_mae_dict = dict()
    structure_mse_dict = dict()
    structure_nmae_dict = dict()
    structure_nmse_dict = dict()
    near_mae_dict = dict()
    near_mse_dict = dict()
    near_nmae_dict = dict()
    near_nmse_dict = dict()
    fields_dict = dict()
    
    
    d_range = get_design_range(opt.resolution, opt.width)
    
    model.eval()
    with torch.no_grad():
        for wvl_num in wvls:
            data, target, dl, wvl = dataset[str(wvl_num)]
            print(data.shape)
            data = data.type(torch.float32)
            # print(wavelength)
            wavelength = wvl[:, None].type(torch.float32)
            
            dl = torch.cat((dl[:, None], dl[:, None]), dim=1)
            data = data.to(device)
            wavelength = wavelength.to(device)
            dl = dl.to(device)
            target = target.to(device)

            
            output = model(data, wavelength, dl)
            
            param_output = output[:, :, d_range[0]:d_range[1], :]
            param_target = target[:, :, d_range[0]:d_range[1], :]
            
            
            #### use for-loop
            near_outputs = []
            near_targets = []
            for i in range(len(output)):
                near_output = get_near_field(output[i:i+1], d_range, wavelength[i, 0].item())
                near_target = get_near_field(target[i:i+1], d_range, wavelength[i, 0].item())
                near_outputs.append(near_output)
                near_targets.append(near_target)
            def get_element_error(criterion, outputs, targets):
                losses = []
                for output, target in zip(outputs, targets):
                    losses.append(criterion(output, target))
                return torch.mean(torch.stack(losses))
                    
            
            
            mae_val_loss = mae_criterion(output, target)
            mse_val_loss = mse_criterion(output, target)
            nmae_val_loss = nmae_criterion(output, target)
            nmse_val_loss = nmse_criterion(output, target)
            structure_mae_val_loss = mae_criterion(param_output, param_target)
            structure_mse_val_loss = mse_criterion(param_output, param_target)
            structure_nmae_val_loss = nmae_criterion(param_output, param_target)
            structure_nmse_val_loss = nmse_criterion(param_output, param_target)
            
            near_mae_val_loss = get_element_error(mae_criterion, near_outputs, near_targets)
            near_mse_val_loss = get_element_error(mse_criterion, near_outputs, near_targets)
            near_nmae_val_loss = get_element_error(nmae_criterion, near_outputs, near_targets)
            near_nmse_val_loss = get_element_error(nmse_criterion, near_outputs, near_targets)
            
            mae_dict[str(wvl_num)] = mae_val_loss.detach().clone().cpu()
            mse_dict[str(wvl_num)] = mse_val_loss.detach().clone().cpu()
            nmae_dict[str(wvl_num)] = nmae_val_loss.detach().clone().cpu()
            nmse_dict[str(wvl_num)] = nmse_val_loss.detach().clone().cpu()
            structure_mae_dict[str(wvl_num)] = structure_mae_val_loss.detach().clone().cpu()
            structure_mse_dict[str(wvl_num)] = structure_mse_val_loss.detach().clone().cpu()
            structure_nmae_dict[str(wvl_num)] = structure_nmae_val_loss.detach().clone().cpu()
            structure_nmse_dict[str(wvl_num)] = structure_nmse_val_loss.detach().clone().cpu()
            near_mae_dict[str(wvl_num)] = near_mae_val_loss.detach().clone().cpu()
            near_mse_dict[str(wvl_num)] = near_mse_val_loss.detach().clone().cpu()
            near_nmae_dict[str(wvl_num)] = near_nmae_val_loss.detach().clone().cpu()
            near_nmse_dict[str(wvl_num)] = near_nmse_val_loss.detach().clone().cpu()
            
            fields_dict[str(wvl_num)] = (output.detach().clone().cpu(), target.detach().clone().cpu())

        save_folder= f"test_wvlwise_results/{opt.model}"
        os.makedirs(save_folder, exist_ok=True)
        torch.save(
            {
                'mae_val_dict' : mae_dict,
                'mse_val_dict' : mse_dict,
                'nmae_val_dict' : nmae_dict,
                'nmse_val_dict' : nmse_dict,
                'structure_mae_val_dict' : structure_mae_dict,
                'structure_mse_val_dict' : structure_mse_dict,
                'structure_nmae_val_dict' : structure_nmae_dict,
                'structure_nmse_val_dict' : structure_nmse_dict,
                'near_mae_val_dict' : near_mae_dict,
                'near_mse_val_dict' : near_mse_dict,
                'near_nmae_val_dict' : near_nmae_dict,
                'near_nmse_val_dict' : near_nmse_dict,
                'fields_dict' : fields_dict,
            }, os.path.join(save_folder, file_name+".pt")
        )



def main(model='wino'):
    parent_path = Path(__file__).parent.parent.resolve()
    
    opt = test_yaml_to_args(parent_path=parent_path, model=model)
    opt.model = model

    
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

    model_path = os.path.join(parent_path, opt.save_root, 'best.pt')

    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint['model'], strict=True)

    data_path = os.path.join(parent_path, opt.test_dataset_root, 'rand_wvl_data')
    dataset = get_dataset(data_path)
    print("DATA OK")
    file_name = opt.save_root.split(os.path.sep)[-1]
    print("TEST START OK")
    wvlwise_test(opt, model, dataset, device, file_name)
    print("TEST END OK")
    
    
    
if __name__ == '__main__':
    for config_name in ["wino", "fno2d", "fno2dfactor", "neurolight", "unet"]:
        main(model=config_name)
        torch.cuda.empty_cache()
