import numpy as np
import natsort

import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import os
import matplotlib.pylab as plt
import ceviche.viz
from ceviche import fdfd_mf_ez, jacobian
from ceviche.optimizers import adam_optimize
from ceviche.modes import insert_mode
from scipy.ndimage import gaussian_filter
# from ceviche.helper import operator_proj, operator_blur 

import collections


class FDFDSim:
    c0 = 299792458.13099605
    eps_0 = 8.85418782e-12
    mu_0 = 1.25663706e-6
    source_amp = 1e-8
    sub_eps = 1. # we don't use subtrate. : Air.
    si_eps = 1.46**2 # we set the material as SiO2. If you use other materials, please edit this relative permittivity value.
    resolution = 40
    
    def __init__(self, without_PML=False):
        # Simulation parameters
        self.without_PML = without_PML 
        self.dl = 1e-6/self.resolution
        self.Nx = int(5.2/1e6/self.dl)
        self.Ny = int((6.85-0.4+0.12)/1e6/self.dl)
        self.NPML = int(1/1e6/self.dl)
        source_loc = self.NPML + int(0.2/1e6/self.dl)
        self.source_loc = source_loc
        subtrate_range = (self.NPML, self.NPML + int(0.4/1e6/self.dl))
        self.design_range = (subtrate_range[1],subtrate_range[1] + int(0.12/1e6/self.dl))

        self.epsr = np.ones((self.Nx, self.Ny))
        self.source = np.zeros((self.Nx, self.Ny), np.complex128)
        self.source[self.NPML:-self.NPML, source_loc] = self.source_amp
        
        self.init_device_setting()
        
    def make_data(self, n, min_eps, max_eps):
        while True:
            design = np.random.uniform(0, 1, (n,))
            design = gaussian_filter(design, sigma=1, mode="constant")
            design = np.array(design > 0.5, dtype="?")
            if not np.all(~design):
                break
        design = design * (max_eps - min_eps) + min_eps
        return design
    
    
    def init_device_setting(self):
        data = self.make_data(self.Nx-2*self.NPML, 1, self.si_eps)
        self.epsr[self.NPML:-self.NPML, self.design_range[0]:self.design_range[1]] = data[:, None]
    
    
    def simulation(self, wavelength, init_device=True):
        self.lambda0 = wavelength/1e6
        self.omega = 2*np.pi*self.c0/self.lambda0 # angular frequency (2pi/s)
        
        if init_device:
            self.init_device_setting()
        
        simulation = ceviche.fdfd_ez(self.omega, self.dl, self.epsr, [self.NPML*2, self.NPML*2])
        Hx, Hy, Ez = simulation.solve(self.source)
        
        if self.without_PML:
            return (self.epsr[self.NPML:-self.NPML, self.NPML:-self.NPML], 
                    Hx[self.NPML:-self.NPML, self.NPML:-self.NPML],
                    Hy[self.NPML:-self.NPML, self.NPML:-self.NPML],
                    Ez[self.NPML:-self.NPML, self.NPML:-self.NPML],
                    self.dl
                    )

        else:
            return (self.epsr, 
                    Hx,
                    Hy,
                    Ez,
                    self.dl
                    )
        
def sampled_wvl_create_datas(save_root, num, sampled_wavelengths):
    wvls = np.random.choice(sampled_wavelengths, size=num, replace=True)
    fdfd = FDFDSim(without_PML=True)
    for n, wvl in enumerate(wvls):
        eps, Hx, Hy, Ez, dl = fdfd.simulation(wvl)
        np.savez_compressed(os.path.join(save_root, f'{n+1}.npz'), eps=eps, Hx=Hx, Hy=Hy, Ez=Ez, dl=dl, wavelength=wvl)

def all_sampled_wvl_create_datas(save_root, num, wavelengths, before_num=0):
    wvl_num = len(wavelengths)
    os.makedirs(save_root, exist_ok=True)
    fdfd = FDFDSim(without_PML=True)
    for n in range(num):
        wvl_idx = n % wvl_num
        wvl = wavelengths[wvl_idx]
        eps, Hx, Hy, Ez, dl = fdfd.simulation(wvl)
        np.savez_compressed(os.path.join(save_root, f'{n+1}.npz'), eps=eps, Hx=Hx, Hy=Hy, Ez=Ez, dl=dl, wavelength=wvl)
        
def same_device_var_wvl(save_root):
    wavelengths = np.arange(0.4, 0.701, 0.001)
    wvl_num = len(wavelengths)
    os.makedirs(save_root, exist_ok=True)
    fdfd = FDFDSim(without_PML=True)
    for n in range(wvl_num):
        eps, Hx, Hy, Ez, dl = fdfd.simulation(wavelengths[n], init_device=False)
        np.savez_compressed(os.path.join(save_root, f'{n+1}.npz'), eps=eps, Hx=Hx, Hy=Hy, Ez=Ez, dl=dl, wavelength=wavelengths[n])
        

def save_all_sampled_wvl_create_datas(save_root, w_num=20):
    wavelengths = np.concatenate([np.arange(0.4, 0.701, 0.001)])
    os.makedirs(save_root, exist_ok=True)
    num = len(wavelengths)*w_num
    all_sampled_wvl_create_datas(save_root, num, wavelengths)
    
    
def save_sampled_wvl_create_datas(save_root, num=12000):
    os.makedirs(save_root, exist_ok=True)
    sampled_wavelengths = np.arange(0.4, 0.701, 0.02)
    sampled_wvl_create_datas(save_root, num, sampled_wavelengths)
    
    
def file_segmenting(save_path, datas, file_names, wvl_labels=None): # save files.
    if wvl_labels == None:
        wvl_labels = list(range(400, 701))
        
    # make dirs
    for wvl in wvl_labels:
        os.makedirs(os.path.join(save_path, str(wvl)), exist_ok=True)
    iterations = len(file_names) // len(wvl_labels)
    for i in range(iterations):
        for j, wl in enumerate(wvl_labels):
            this_data = datas[i*len(wvl_labels)+j]
            this_save_path = os.path.join(save_path, str(wl), f'{i}.npz')
            np.savez(this_save_path, eps=this_data['eps'], Ez=this_data['Ez'], wavelength=this_data['wavelength'])


if __name__ == "__main__":
    dataset_root = 'dataset/data/120nmdata'
    dataset_folder_prefix = 'rand_wvl_data'

    sampled_save_root = os.path.join(dataset_root, f'ceviche_train/{dataset_folder_prefix}')
    all_valid_save_root = os.path.join(dataset_root, f'ceviche_valid/{dataset_folder_prefix}')
    all_test_save_root = os.path.join(dataset_root, f'ceviche_test/{dataset_folder_prefix}')
    save_sampled_wvl_create_datas(sampled_save_root, num=12000) # for train
    save_all_sampled_wvl_create_datas(all_valid_save_root, w_num=20) # for valid
    save_all_sampled_wvl_create_datas(all_test_save_root, w_num=20) # for test
    
    wvl_labels = list(range(400, 701))
    paths = [all_valid_save_root, all_test_save_root]
    labels = ['valid', 'test']
    for lab, path in zip(labels, paths):
        file_list = os.listdir(path)
        file_list = natsort.natsorted(file_list)
        
        datas = []
        for fn in file_list:
            data_path = os.path.join(path, fn)
            data = np.load(data_path)
            datas.append(data)
            
        save_path = os.path.join('dataset/data/120nmdata', f'ceviche_{lab}/{dataset_folder_prefix}')
        file_segmenting(save_path=save_path, datas=datas, file_names=file_list, wvl_labels=wvl_labels)    
