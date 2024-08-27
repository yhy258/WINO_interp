"""
    Based on https://github.com/pdearena/pdearena/blob/main/benchmark/fwdbench.py
    Towards Multi-spatiotemporal-scale Generalized PDE Modeling
    MIT License
    Copyright (c) 2020 Microsoft Corporation.
"""


import torch
import pprint
from thop import profile
import yaml
import os
import argparse
from functools import partial
import time
import timeit
import pickle
from builder import build_model


class Timer:
    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


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

if __name__ == '__main__':
    my_opt = yaml_to_args(model='wino')
    my_opt.model = 'wino'
    opt = my_opt

    # opt = yaml_to_args(model='myfno')
    # opt.model = 'myfno'    
    
    if torch.cuda.is_available() and len(opt.device) == 1:
        device = f"cuda:{opt.device[0]}"
        device = torch.device(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False
        
    model = build_model(opt, opt.model, device)
    
    results = {}
    ft_dict = {}
    mill = 1000000

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    # model_size = total_parameters * precision_megabytes
    model_size = total_parameters / mill
    
    
    print(f"{opt.model} - MODEL SIZE (M) : {model_size}, TRAINABLE PARAMS : {trainable_params}, TOTAL PARAMS : {total_parameters}")

    results[opt.model] = {"num_params": trainable_params, "model_size": model_size}
    del model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
            
    pprint.pprint(results)
