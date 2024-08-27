import os
import torch
from torch.utils.data import Dataset    
import torch.nn as nn
import natsort
import numpy as np

def get_min_max_val_fields(field):
    min_val, max_val = field.min(dim=0, keepdim=True).values.min(dim=2, keepdim=True).values.min(dim=3, keepdim=True).values, field.max(dim=0, keepdim=True).values.max(dim=2, keepdim=True).values.max(dim=3, keepdim=True).values
    return min_val, max_val


def get_condition(folder_name):
    cond = int(folder_name.split('_')[-2])
    return folder_name, cond

def minmaxscaler(datas, data_range=None):
    if data_range == None:
        min_val, max_val = datas.min(), datas.max()
    else:
        min_val, max_val = data_range
        
    return (datas - min_val) / (max_val - min_val), min_val, max_val

def minmaxscaler_field(datas: torch.Tensor, data_range=None):
    # BS, 2, H, W (2 -> real, imag)
    if data_range == None:
        min_val, max_val = datas.min(dim=0, keepdim=True).values.min(dim=2, keepdim=True).values.min(dim=3, keepdim=True).values, datas.max(dim=0, keepdim=True).values.max(dim=2, keepdim=True).values.max(dim=3, keepdim=True).values
    else:
        min_val, max_val = data_range
    
    return (datas - min_val) / (max_val - min_val), min_val, max_val

def transpose_data(data):
    data = np.transpose(data, axes=(0, 2, 1))
    return data

def field_process(field):
    field = torch.tensor(field)
    field = torch.view_as_real(field).permute(0,3,1,2)
    return field

def get_npz_data(data):
    geo = data['mat_geometry']
    ex = data['ex']
    ey = data['ey']
    hz = data['hz']
    return geo, ex, ey, hz

def count_data_nums(path):
    folders = os.listdir(path)
    cnt = 0
    for fd in folders:
        this_num = len(os.listdir(os.path.join(path, fd)))
        cnt += this_num
    return cnt


class ConditionalCevicheProcess():
    field_scale_factor = 2.5e13
    PML_pixel = 40
    resolution = 40
    def __init__(self, path, interp_path, data_num=8000, without_PML=True, step=20):
        self.path = path
        self.data_num = data_num
        self.without_PML = without_PML
        self.min_wvl, self.max_wvl = 400, 700
        self.min_eps, self.max_eps = 1, 1.46**2

        # trainset : load : given cond_interval
        # testset : load : all condition
        
        test_wvls = list(range(self.min_wvl, self.max_wvl+1))
        
        train_wvls = list(range(self.min_wvl, self.max_wvl+1, step))
        
        # train
        epss, ezs, dls, wvls = [], [], [], []
        
        this_path = path
        this_epss, this_ezs, this_dls, this_wvls = self.get_datas(path=this_path,mode='train')
        epss.append(this_epss)
        ezs.append(this_ezs)
        dls.append(this_dls)
        wvls.append(this_wvls)

        
        self.epss = np.concatenate(epss, axis=0)
        self.ezs = np.concatenate(ezs, axis=0)
        self.dls = np.concatenate(dls, axis=0)
        self.wvls = np.concatenate(wvls, axis=0)
        
        epss, ezs, dls, wvls = [], [], [], []
        
        for wvl in test_wvls:
            if wvl not in train_wvls:
                this_path = os.path.join(interp_path, str(wvl))
                this_epss, this_ezs, this_dls, this_wvls = self.get_datas(path=this_path,mode='test')
                epss.append(this_epss)
                ezs.append(this_ezs)
                dls.append(this_dls)
                wvls.append(this_wvls)
        
        self.test_epss = np.concatenate(epss, axis=0)
        self.test_ezs = np.concatenate(ezs, axis=0)
        self.test_dls = np.concatenate(dls, axis=0)
        self.test_wvls = np.concatenate(wvls, axis=0)
        
        epss, ezs, dls, wvls = [], [], [], []
        
        for wvl in test_wvls:
            if wvl in train_wvls:
                this_path = os.path.join(interp_path, str(wvl))
                this_epss, this_ezs, this_dls, this_wvls = self.get_datas(path=this_path,mode='test')
                epss.append(this_epss)
                ezs.append(this_ezs)
                dls.append(this_dls)
                wvls.append(this_wvls)
        
        self.train_wvl_test_epss = np.concatenate(epss, axis=0)
        self.train_wvl_test_ezs = np.concatenate(ezs, axis=0)
        self.train_wvl_test_dls = np.concatenate(dls, axis=0)
        self.train_wvl_test_wvls = np.concatenate(wvls, axis=0)
        
        
        # normalize eps
        self.epss = (self.epss - self.min_eps) / (self.max_eps - self.min_eps)
        self.test_epss = (self.test_epss - self.min_eps) / (self.max_eps - self.min_eps)
        self.train_wvl_test_epss = (self.train_wvl_test_epss - self.min_eps) / (self.max_eps - self.min_eps)
        
        # scaling field
        self.ezs *= self.field_scale_factor
        self.test_ezs *= self.field_scale_factor
        self.train_wvl_test_ezs *= self.field_scale_factor
        
        self.epss, self.ezs, self.dls, self.wvls = self.process(self.epss, self.ezs, self.dls, self.wvls)
        self.test_epss, self.test_ezs, self.test_dls, self.test_wvls = self.process(self.test_epss, self.test_ezs, self.test_dls, self.test_wvls)
        self.train_wvl_test_epss, self.train_wvl_test_ezs, self.train_wvl_test_dls, self.train_wvl_test_wvls = self.process(self.train_wvl_test_epss, self.train_wvl_test_ezs, self.train_wvl_test_dls, self.train_wvl_test_wvls)
        self.data_range_dict = self.get_data_range()
        # minval, maxval (value range)
        # split
        # save function
    
    def split_dataset(self, split=(0.8, 0.2)):

        train_set = (self.epss, self.ezs, self.dls, self.wvls, self.data_range_dict)
        
        test_wvl_set = (self.test_epss, self.test_ezs, self.test_dls, self.test_wvls, self.data_range_dict)
        train_wvl_set = (self.train_wvl_test_epss, self.train_wvl_test_ezs, self.train_wvl_test_dls, self.train_wvl_test_wvls, self.data_range_dict)

        return train_set, test_wvl_set, train_wvl_set
        
    def save(self, save_path, split=(0.8, 0.2)):
        N = self.epss.shape[0]
        train_num = int(N*split[0])
        test_num = int(N*split[1])
        print(train_num, self.data_num)

        train_name = f"train.pt"
        valid_name = f"valid.pt"
        t_wvl_valid_name = f"train_wvl_valid.pt"   
        
        train_path = os.path.join(save_path, train_name)
        valid_path = os.path.join(save_path, valid_name)
        t_wvl_valid_path = os.path.join(save_path, t_wvl_valid_name)
        
        trainset, testset, train_wvl_testset = self.split_dataset(split)
        
        with open(train_path, 'wb') as f:
            torch.save(trainset, f)
        with open(valid_path, 'wb') as f:
            torch.save(testset, f)
        with open(t_wvl_valid_path, 'wb') as f:
            torch.save(train_wvl_testset, f)

    def process(self, epss, ezs, dls, wvls):
        epss, ezs = map(transpose_data, [epss, ezs])
        epss = torch.tensor(epss)[:, None, :, :]
        ezs = field_process(ezs)
        dls = torch.tensor(dls)
        wvls = torch.tensor(wvls)
        return epss, ezs, dls, wvls
    
    def get_datas(self, path, mode='train'):
        files = natsort.natsorted(os.listdir(path))
        n = int(len(files))
        if mode != 'train':
            files = files
        epss = []
        ezs = []
        dls = []
        wvls = []
        nums = 0
        for f in files:
            try :
                data = np.load(os.path.join(path,f))

                if self.without_PML and mode != 'train':
                    ezs.append(data['Ez'][self.PML_pixel:-self.PML_pixel, self.PML_pixel:-self.PML_pixel])
                else:
                    ezs.append(data['Ez'])

                if self.without_PML and mode != 'train':
                    full_geo = data['eps'][self.PML_pixel:-self.PML_pixel, self.PML_pixel:-self.PML_pixel]
                else:
                    full_geo = data['eps']
                epss.append(full_geo)
                dls.append(1/self.resolution)
                wvls.append(data['wavelength'])
            except:
                # print(os.path.join(self.path,f))
                nums += 1
                
        print(nums)
        epss = np.stack(epss)
        ezs = np.stack(ezs)
        dls = np.stack(dls)
        wvls = np.stack(wvls)
        return epss, ezs, dls, wvls

    def get_data_range(self):
        eps_min, eps_max = self.epss.min(), self.epss.max()
        ez_min, ez_max = get_min_max_val_fields(self.ezs)
        wvl_min, wvl_max = self.wvls.min(), self.wvls.max()
        return {'eps_range' : (eps_min, eps_max), 'ez_range' : (ez_min, ez_max), 'wvl_range' : (wvl_min, wvl_max)}



class FullSimDataset(Dataset):
    def __init__(self, path='', test_path="", save_folder_name='rand_wvl_data', mode='train', normalize=False, step=5):
        split = (0.8, 0.2)
        data_path = os.path.join(path, 'rand_wvl_data') # rand_wvl_data - prefix of simulation datas
        files = os.listdir(data_path)
        N = len(files)
        train_num = int(N*split[0])

        train_name = 'train.pt'
        file_path = os.path.join(path, save_folder_name, train_name)

        if not os.path.exists(file_path):
            # when we conduct an EM simulation, we bring simulation observations except PML layers. -> without_PML should be "off".
            proc = ConditionalCevicheProcess(os.path.join(path, 'rand_wvl_data'), interp_path=os.path.join(test_path, 'rand_wvl_data'), data_num=8000, without_PML=False, step=step)
    
            os.makedirs(os.path.join(path,save_folder_name), exist_ok=True)
            proc.save(os.path.join(path,save_folder_name))
        else:
            print("Exist")
            
        if mode == 'train':
            file_name = train_name
        else:
            file_name = f"{mode}.pt"

        save_path = os.path.join(path, save_folder_name, file_name)
        self.epss, self.ezs, self.dls, self.wvls, self.data_range = self.load_files(save_path)

        if normalize:
            self.normalization()
    
    def minmaxscaler(self, data, minval, maxval):
        return (data - minval)/(maxval-minval)
    
    def normalization(self):
        
        eps_min, eps_max = self.data_range['eps_range']
        ez_min, ez_max = self.data_range['ez_range']
        # wvl_min, wvl_max = self.data_range['wvl_range']
        
        self.epss = self.minmaxscaler(self.epss, eps_min, eps_max).type(torch.FloatTensor)
        self.ezs = self.minmaxscaler(self.ezs, ez_min, ez_max).type(torch.FloatTensor)
        # self.wvls = self.minmaxscaler(self.wvls, wvl_min, wvl_max).type(torch.FloatTensor)
        
    def load_files(self, save_path):
        epss, ezs, dls, wvls, data_range = torch.load(save_path)
        
        return epss, ezs, dls, wvls, data_range
    
    def __len__(self):
        return self.epss.shape[0]

    def __getitem__(self, index):
        # epss, exs, eys, hzs, wvls normalization
        # real valued. N, C, H, W
        return self.epss[index], self.ezs[index], self.dls[index], self.wvls[index]



