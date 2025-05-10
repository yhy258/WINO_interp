import os
from pathlib import Path
import random
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset.interpolative_dataset import transpose_data, field_process
from core.losses import NMAE
# from pyutils.torch_train import set_torch_deterministic
# from meep_train import yaml_to_args
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

def test_yaml_to_args(parent_path, model='fno2d'):
    parser = argparse.ArgumentParser(description='YAML to argparse', add_help=False)
    yaml_path = os.path.join(parent_path, os.path.join('configs', model+'.yaml'))

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

def get_near_field(data, design_range, wavelength):
    # data : B, C, H, W
    start = design_range[1]
    mid_wavelength = wavelength
    unit_micro = 40 # resolution
    distance = int(unit_micro * mid_wavelength)
    param_output = data[:, :, start: start + distance, :] 
    return param_output

def get_dataset(opt, path):
    wvlslab  = list(range(400, 701))
    data_dict = dict()
    for wvl in wvlslab:
        this_path = os.path.join(path, str(wvl))
        epss, ezs, dls, wvls = get_data(opt, this_path)
        this_wvl = wvls[0].item()
        data_dict[str(wvl)] = (epss, ezs, dls, wvls)
    return data_dict # data_dict [wvl] : N, H, W...
        
        
def get_data(opt, path):
    min_eps = opt.eps_min
    # max_eps = 1.46**2
    max_eps = opt.eps_max
    field_scale_factor = opt.field_scale_factor
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
    

    
# # def wvlwise_test(opt, model, dataset, device, file_name):
# #     mae_criterion = nn.L1Loss()
# #     mse_criterion = nn.MSELoss()
# #     nmae_criterion = NMAE()
# #     nmse_criterion = NMSE()

# #     wvls  = list(range(400, 701))
    
# #     mae_dict = dict()
# #     mse_dict = dict()
# #     nmae_dict = dict()
# #     nmse_dict = dict()
# #     structure_mae_dict = dict()
# #     structure_mse_dict = dict()
# #     structure_nmae_dict = dict()
# #     structure_nmse_dict = dict()
# #     near_mae_dict = dict()
# #     near_mse_dict = dict()
# #     near_nmae_dict = dict()
# #     near_nmse_dict = dict()
# #     fields_dict = dict()
    
    
# #     d_range = get_design_range(opt.resolution, opt.design_start, opt.width)
    
# #     model.eval()
# #     with torch.no_grad():
# #         for wvl_num in wvls:
# #             data, target, dl, wvl = dataset[str(wvl_num)]
# #             print(data.shape)
# #             data = data.type(torch.float32)
# #             # print(wavelength)
# #             wavelength = wvl[:, None].type(torch.float32)
            
# #             dl = torch.cat((dl[:, None], dl[:, None]), dim=1)
# #             data = data.to(device)
# #             wavelength = wavelength.to(device)
# #             dl = dl.to(device)
# #             target = target.to(device)

            
# #             output = model(data, wavelength, dl)
            
# #             param_output = output[:, :, d_range[0]:d_range[1], :]
# #             param_target = target[:, :, d_range[0]:d_range[1], :]
            
            
# #             #### use for-loop
# #             near_outputs = []
# #             near_targets = []
# #             for i in range(len(output)):
# #                 near_output = get_near_field(output[i:i+1], d_range, wavelength[i, 0].item())
# #                 near_target = get_near_field(target[i:i+1], d_range, wavelength[i, 0].item())
# #                 near_outputs.append(near_output)
# #                 near_targets.append(near_target)
# #             def get_element_error(criterion, outputs, targets):
# #                 losses = []
# #                 for output, target in zip(outputs, targets):
# #                     losses.append(criterion(output, target))
# #                 return torch.mean(torch.stack(losses))
                    
            
            
# #             mae_val_loss = mae_criterion(output, target)
# #             mse_val_loss = mse_criterion(output, target)
# #             nmae_val_loss = nmae_criterion(output, target)
# #             nmse_val_loss = nmse_criterion(output, target)
# #             structure_mae_val_loss = mae_criterion(param_output, param_target)
# #             structure_mse_val_loss = mse_criterion(param_output, param_target)
# #             structure_nmae_val_loss = nmae_criterion(param_output, param_target)
# #             structure_nmse_val_loss = nmse_criterion(param_output, param_target)
            
# #             near_mae_val_loss = get_element_error(mae_criterion, near_outputs, near_targets)
# #             near_mse_val_loss = get_element_error(mse_criterion, near_outputs, near_targets)
# #             near_nmae_val_loss = get_element_error(nmae_criterion, near_outputs, near_targets)
# #             near_nmse_val_loss = get_element_error(nmse_criterion, near_outputs, near_targets)
            
# #             mae_dict[str(wvl_num)] = mae_val_loss.detach().clone().cpu()
# #             mse_dict[str(wvl_num)] = mse_val_loss.detach().clone().cpu()
# #             nmae_dict[str(wvl_num)] = nmae_val_loss.detach().clone().cpu()
# #             nmse_dict[str(wvl_num)] = nmse_val_loss.detach().clone().cpu()
# #             structure_mae_dict[str(wvl_num)] = structure_mae_val_loss.detach().clone().cpu()
# #             structure_mse_dict[str(wvl_num)] = structure_mse_val_loss.detach().clone().cpu()
# #             structure_nmae_dict[str(wvl_num)] = structure_nmae_val_loss.detach().clone().cpu()
# #             structure_nmse_dict[str(wvl_num)] = structure_nmse_val_loss.detach().clone().cpu()
# #             near_mae_dict[str(wvl_num)] = near_mae_val_loss.detach().clone().cpu()
# #             near_mse_dict[str(wvl_num)] = near_mse_val_loss.detach().clone().cpu()
# #             near_nmae_dict[str(wvl_num)] = near_nmae_val_loss.detach().clone().cpu()
# #             near_nmse_dict[str(wvl_num)] = near_nmse_val_loss.detach().clone().cpu()
            
# #             fields_dict[str(wvl_num)] = (output.detach().clone().cpu(), target.detach().clone().cpu())

# #         save_folder= f"test_wvlwise_results/{opt.sim_name}/{opt.model}"
# #         os.makedirs(save_folder, exist_ok=True)
# #         torch.save(
# #             {
# #                 'mae_val_dict' : mae_dict,
# #                 'mse_val_dict' : mse_dict,
# #                 'nmae_val_dict' : nmae_dict,
# #                 'nmse_val_dict' : nmse_dict,
# #                 'structure_mae_val_dict' : structure_mae_dict,
# #                 'structure_mse_val_dict' : structure_mse_dict,
# #                 'structure_nmae_val_dict' : structure_nmae_dict,
# #                 'structure_nmse_val_dict' : structure_nmse_dict,
# #                 'near_mae_val_dict' : near_mae_dict,
# #                 'near_mse_val_dict' : near_mse_dict,
# #                 'near_nmae_val_dict' : near_nmae_dict,
# #                 'near_nmse_val_dict' : near_nmse_dict,
# #                 'fields_dict' : fields_dict,
# #             }, os.path.join(save_folder, file_name+".pt")
#         )


def retain_batch_wvlwise_test(opt, model, dataset, device, file_name):
    # mae_criterion = nn.L1Loss('none')
    # mse_criterion = nn.MSELoss('none')
    # nmae_criterion = NMAE('none')
    nmse_criterion = NMSE(reduction='none')

    wvls  = list(range(400, 701))
    
    # mae_dict = dict()
    # mse_dict = dict()
    # nmae_dict = dict()
    nmse_dict = dict()
    # structure_mae_dict = dict()
    # structure_mse_dict = dict()
    # structure_nmae_dict = dict()
    structure_nmse_dict = dict()
    # near_mae_dict = dict()
    # near_mse_dict = dict()
    # near_nmae_dict = dict()
    near_nmse_dict = dict()
    fields_dict = dict()
    
    
    d_range = get_design_range(opt.resolution, opt.design_start, opt.width)
    
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
                return torch.stack(losses)
                    
            
            
            # mae_val_loss = mae_criterion(output, target) # B,
            # mse_val_loss = mse_criterion(output, target)
            # nmae_val_loss = nmae_criterion(output, target)
            nmse_val_loss = nmse_criterion(output, target)
            # structure_mae_val_loss = mae_criterion(param_output, param_target)
            # structure_mse_val_loss = mse_criterion(param_output, param_target)
            # structure_nmae_val_loss = nmae_criterion(param_output, param_target)
            structure_nmse_val_loss = nmse_criterion(param_output, param_target)
            
            # near_mae_val_loss = get_element_error(mae_criterion, near_outputs, near_targets)
            # near_mse_val_loss = get_element_error(mse_criterion, near_outputs, near_targets)
            # near_nmae_val_loss = get_element_error(nmae_criterion, near_outputs, near_targets)
            near_nmse_val_loss = get_element_error(nmse_criterion, near_outputs, near_targets)
            
            # mae_dict[str(wvl_num)] = mae_val_loss.detach().clone().cpu()
            # mse_dict[str(wvl_num)] = mse_val_loss.detach().clone().cpu()
            # nmae_dict[str(wvl_num)] = nmae_val_loss.detach().clone().cpu()
            nmse_dict[str(wvl_num)] = nmse_val_loss.detach().clone().cpu()
            # structure_mae_dict[str(wvl_num)] = structure_mae_val_loss.detach().clone().cpu()
            # structure_mse_dict[str(wvl_num)] = structure_mse_val_loss.detach().clone().cpu()
            # structure_nmae_dict[str(wvl_num)] = structure_nmae_val_loss.detach().clone().cpu()
            structure_nmse_dict[str(wvl_num)] = structure_nmse_val_loss.detach().clone().cpu()
            # near_mae_dict[str(wvl_num)] = near_mae_val_loss.detach().clone().cpu()
            # near_mse_dict[str(wvl_num)] = near_mse_val_loss.detach().clone().cpu()
            # near_nmae_dict[str(wvl_num)] = near_nmae_val_loss.detach().clone().cpu()
            near_nmse_dict[str(wvl_num)] = near_nmse_val_loss.detach().clone().cpu()
            
            fields_dict[str(wvl_num)] = (output.detach().clone().cpu(), target.detach().clone().cpu())

        os.makedirs('retain_batch_test_wvlwise_results', exist_ok=True)
        save_folder= f"retain_batch_test_wvlwise_results/{opt.model}"
        os.makedirs(save_folder, exist_ok=True)
        torch.save(
            {
                # 'mae_val_dict' : mae_dict,
                # 'mse_val_dict' : mse_dict,
                # 'nmae_val_dict' : nmae_dict,
                'nmse_val_dict' : nmse_dict,
                # 'structure_mae_val_dict' : structure_mae_dict,
                # 'structure_mse_val_dict' : structure_mse_dict,
                # 'structure_nmae_val_dict' : structure_nmae_dict,
                'structure_nmse_val_dict' : structure_nmse_dict,
                # 'near_mae_val_dict' : near_mae_dict,
                # 'near_mse_val_dict' : near_mse_dict,
                # 'near_nmae_val_dict' : near_nmae_dict,
                'near_nmse_val_dict' : near_nmse_dict,
                'fields_dict' : fields_dict,
            }, os.path.join(save_folder, file_name+".pt")
        )



# def get_design_range(resolution, start=0.4, width=0.275):
#     dl = 1e-6/resolution
#     NPML = int(1/1e6/dl)
#     subtrate_range = (0, 0 + int(start/1e6/dl))
#     design_range = (subtrate_range[1],subtrate_range[1] + int(width/1e6/dl))
#     return design_range



def init_attr(opt, key, value):
    if hasattr(opt, key) == False:
        print(f"{key} -> {value}")
        setattr(opt, key, value)
    return opt



if __name__== '__main__':
    """
        GOAL : 다양한 data setting에 대해서 모두 수행하고 싶음.
        
        시뮬레이션 세팅에 따라 달라지는점. 
        1. Checkpoint root
        2. Dataset root
        3. Design region
        4. Permittivity
        
    """
    #d_range = get_design_range(opt.resolution, opt.design_start, opt.width)
    
    
    
    parent_path = Path(__file__).parent.parent.resolve()
    dataset_root_dict = {}
    
    
    eps_dict = {'triple_layer':{'eps_min':1.0, 'eps_max': 1.46**2}, 'straight_waveguide':{'eps_min':1.0, 'eps_max':2.25}, 'image_sensor': {'eps_min':1.0, 'eps_max':4.0}}
    scaling_dict = {'image_sensor': 1.33e13, 'triple_layer': 1.5e13, 'straight_waveguide': 1.25e13}
    design_range_dict = {'triple_layer': {'design_start':0.4, 'width': 0.12}, 'straight_waveguide': {'design_start':1.0, 'width':4.85 }, 'image_sensor': {'design_start':0.6, 'width': 3.5}}
    
    # ckpt ex :  "/data/joon/Results/WINO/unziped_ckpt/image_sensor/checkpoints/wino/nmse_waveprior_64dim_12layer_256_5060_auggroup4_weightsharing/epoch200.pt"
    # dataset root : '/data/joon/Dataset/WINO/unziped_datasets/double_layer'
    parser = argparse.ArgumentParser(description="wino-ablation-study")
    parser.add_argument('--sim_name', type=str, default="triple_layer", help='The name of the simulation setting, [triple_layer, straight_waveguide, image_sensor]')
    parser.add_argument('--model', type=str, default="wino", help='The name of the model')
    args = parser.parse_args()
    
    ckpt_base = '/root/WINO_interp/checkpoints'
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
    args.design_start = design_range_dict[args.sim_name]['design_start']
    args.width = design_range_dict[args.sim_name]['width']
    
    ckpt_root = os.path.join(ckpt_root, args.save_root.split(os.path.sep)[-1])
    
    if not ckpt_root.startswith('/'):
        model_path = os.path.join(parent_path, ckpt_root, 'best.pt')
    else:
        model_path = os.path.join(ckpt_root, 'best.pt')

    
    if torch.cuda.is_available() and len(args.device) == 1:
        device = f"cuda:{args.device[0]}"
        device = torch.device(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(args.deterministic) == True:
        set_torch_deterministic(int(args.random_state))
        
        
    #### Model load.
    model = build_model(args, args.model, device)

    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint['model'], strict=True)


    #### Get test_set
    dataset_root = f'/root/unziped_datasets/{args.sim_name}/ceviche_test'
    #### 이건 옮겨서 재설정 필요.
    data_path = os.path.join(dataset_root, 'rand_wvl_data')
    dataset = get_dataset(args, data_path)
    print("DATA OK")
    file_name = args.save_root.split(os.path.sep)[-1]
    print("TEST START OK")
    
    retain_batch_wvlwise_test(args, model, dataset, device, file_name)
    print("TEST END OK")
    

    
