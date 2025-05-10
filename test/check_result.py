#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

data = torch.load('/root/WINO_interp/retain_batch_train_results/triple_layer/wino/nmse_waveprior_64dim_12layer_256_5060_auggroup4_weightsharing.pt')
# %%
print(data.keys())
# %%
keys = []
for k in data.keys():
    keys.append(k)

# %%
sorted(keys)
# %%
len(keys)
# %%
