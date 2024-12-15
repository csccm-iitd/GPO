
#%%

import os
import sys
sys.path.append("/home/user/Documents/GPO/general")
import gpytorch
from gpytorch.means import MultitaskMean
from gpytorch.kernels import InducingPointKernel, ScaleKernel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from utilities3 import *
from pytorch_wavelets import DWT, IDWT
from pytorch_wavelets import DWT1D, IDWT1D
import scipy
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from utils import *
from _sdd import *

from pytorch_wavelets import DWTForward, DWTInverse
# %%
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f"Device: {device}")
# Disable cuDNN
# torch.backends.cudnn.enabled = False

torch.manual_seed(0)
np.random.seed(0)

# %%
'''   
CONFIG AND DATA 
'''
TRAIN_PATH = '/home/user/Documents/GP_WNO/DATA/NavierStokes_inputs.npy'
TEST_PATH = '/home/user/Documents/GP_WNO/DATA/NavierStokes_outputs.npy'

ntrain = 3000
ntest = 200
# SDD parameters
lr = 0.1
momentum = 0.9
iterations = 20000
B = 24 #8
polyak=2e-3#1e-2
noise_scale=0.002
length_scale = 78 #2

step_size = 50
gamma = 0.75

level = 3 #4
width = 32  #64

r = 2 # 2
h = int(((64 - 1)/r) + 1)
s = h
r1 = 2 # 1
r2 = 1
h1 = int(((64 - 1)/r1) + 1)
h2 = int(((64 - 1)/r2) + 1)
s1 = h1
s2 = h2
# s=s1
# %%
""" Read data """
reader_input = np.load(TRAIN_PATH) #shape: (64, 64, 40000)
reader_output = np.load(TEST_PATH)  #shape: (64, 64, 40000)

reader_input = torch.tensor(reader_input).permute(2,1,0).float() #shape: (40000, 64,64)
reader_output = torch.tensor(reader_output).permute(2,1,0).float()  #shape: (40000, 64,64)

x_train = reader_input[:ntrain,::r1,::r1][:,:s1,:s1]
y_train = reader_output[:ntrain,::r1,::r1][:,:s1,:s1]


x_test = reader_input[30000+ntrain:30000+ntrain+ntest,::r1,::r1][:,:s1,:s1]
y_test = reader_output[30000+ntrain:30000+ntrain+ntest,::r1,::r1][:,:s1,:s1]

#higher resolution
x_train_hres = reader_input[:ntrain,::r2,::r2][:,:s2,:s2]
y_train_hres = reader_output[:ntrain,::r2,::r2][:,:s2,:s2]


x_test_hres = reader_input[30000+ntrain:30000+ntrain+ntest,::r2,::r2][:,:s2,:s2]
y_test_hres = reader_output[30000+ntrain:30000+ntrain+ntest,::r2,::r2][:,:s2,:s2]

#%%
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s1,s1,1)
x_test = x_test.reshape(ntest,s1,s1,1)

#high resoluion 
wp_x_t, wp_x_t_high_freq = wavelet_transform_reduction(x_test_hres.squeeze(-1), x_train.shape[1])
wp_y_t, wp_y_t_high_freq = wavelet_transform_reduction(y_test_hres, y_train.shape[1])

wp_x_train, wp_x_train_high_freq = wavelet_transform_reduction(x_train_hres.squeeze(-1), x_train.shape[1])
wp_y_train, wp_y_train_high_freq = wavelet_transform_reduction(y_test_hres, y_train.shape[1])


x_normalizer_hres = UnitGaussianNormalizer(wp_x_train)
wp_x_train = x_normalizer_hres.encode(wp_x_train)
wp_x_t = x_normalizer_hres.encode(wp_x_t)

y_normalizer_hres = UnitGaussianNormalizer(wp_y_train)
wp_y_train = y_normalizer_hres.encode(wp_y_train)
# y_test_hres = y_normalizer_hres.encode(wp_y_test_hres)

wp_x_train = wp_x_train.reshape(ntrain,s1,s1,1)
wp_x_t = wp_x_t.reshape(ntest,s1,s1,1)

x_t_hres = wp_x_t.reshape(wp_x_t.shape[0],-1).to(device=device)
y_t_hres = wp_y_t.reshape(wp_y_t.shape[0],-1).to(device=device)


# %%
""" Generating Grid and reshaping inputs """

# Grid 

grids = []
grids.append(np.linspace(0, 1, s, dtype=np.float32))
grids.append(np.linspace(0, 1, s, dtype=np.float32))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

# train data
x_tr = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2],x_train.shape[3]).squeeze(-1)

y_tr = y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2])

# test data
x_t = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2],x_test.shape[3]).squeeze(-1)
y_t = y_test.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2])


x_tr=x_tr.to(device)
y_tr=y_tr.to(device)
x_t=x_t.to(device)
y_t=y_t.to(device)



# %%


Kernel = ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)).to(device=device)
Kernel.base_kernel.lengthscale= length_scale

sdd_algo = SDDAlgorithm(Kernel, x_tr, y_tr, iterations=iterations, polyak=polyak,B=B,device=device)
alpha_polyak_trained,_ = sdd_algo.train()

all_predictions = []
with torch.no_grad():
    for i in range(0, len(x_t)):
        x_batch = x_t[i:i+1] 
        predictions_batch = Kernel(x_batch, x_tr) @ alpha_polyak_trained  
        all_predictions.append(predictions_batch)  
y_pred_sdd = torch.cat(all_predictions, dim=0)
torch.save(y_pred_sdd, "/home/user/Documents/GPO/experiments/C5_navier/results/"+'y_pred_sdd_13_8.pt')


#%%
all_predictions_hres = []
with torch.no_grad():
    for i in range(0, len(x_t)):
        x_batch = x_t_hres[i:i+1]  
        predictions_batch = Kernel(x_batch, x_tr) @ alpha_polyak_trained  
        all_predictions_hres.append(predictions_batch)  
y_pred_sdd_hres = torch.cat(all_predictions_hres, dim=0)


#%%
y_pred_sdd = y_pred_sdd.reshape(x_t.shape[0],s,s)
y_pred_sdd = y_normalizer.decode(y_pred_sdd.detach().cpu())
y_pred_sdd = y_pred_sdd.reshape(y_pred_sdd.shape[0], y_pred_sdd.shape[1]*y_pred_sdd.shape[2])

#%%
#high resolution error
y_pred_sdd_hres = y_pred_sdd_hres.reshape(x_t.shape[0],s1,s1)
y_pred_sdd_hres = y_normalizer_hres.decode(y_pred_sdd_hres.detach().cpu())
y_pred_sdd_hres = y_pred_sdd_hres.reshape(y_pred_sdd_hres.shape[0], y_pred_sdd_hres.shape[1]*y_pred_sdd_hres.shape[2])

#%% 

'''PREDICTION ERROR'''
mse_loss = nn.MSELoss()
prediction_error = mse_loss(y_pred_sdd.to(device), y_t)

relative_error = torch.mean(torch.linalg.norm(y_pred_sdd.to(device)-y_t, axis = 1)/torch.linalg.norm(y_t, axis = 1))


print(f'MSE Testing error: {(prediction_error).item()}')
print(f'Mean relative error: {100*relative_error} % ')

# high resolution error
mse_loss = nn.MSELoss()
prediction_error = mse_loss(y_pred_sdd_hres.to(device), y_t_hres)

relative_error = torch.mean(torch.linalg.norm(y_pred_sdd_hres.to(device)- y_t_hres, axis = 1)/torch.linalg.norm(y_t_hres, axis = 1))


print(f'MSE Testing error: {(prediction_error).item()}')
print(f'Mean relative error: {100*relative_error} % ')

