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
from datetime import datetime
from pytorch_wavelets import DWTForward, DWTInverse
# %%
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
print(f"Device: {device}")
# Disable cuDNN
# torch.backends.cudnn.enabled = False
current_time = datetime.now().strftime("%d%m%y_%H%M")

base_dir = '/home/user/Documents/GPO/experiments/C3_darcy_rect/results/'
torch.manual_seed(0)
np.random.seed(0)

# %%

ntrain = 1000
ntest = 200
# SDD parameters
lr = 0.01
momentum = 0.9
iterations = 20000
B = 16 #8
polyak=2e-3#1e-2
noise_scale=0.002
length_scale = 70 #2

step_size = 50
gamma = 0.75

level = 3 #4
width = 32  #64



r = 4 #15 # 5
h = int(((421 - 1)/r) + 1)
s = h

r1 = 15
r2 = 4 #10
h1 = int(((421-1)/r1) + 1)
h2 = int(((421-1)/r2) + 1)
s1=h1
s2=h2


# %%
TRAIN_PATH = '/home/user/Documents/GP_WNO/DATA/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = '/home/user/Documents/GP_WNO/DATA/piececonst_r421_N1024_smooth2.mat'

""" Read data """
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain,::r1,::r1][:,:s1,:s1]
y_train = reader.read_field('sol')[:ntrain,::r1,::r1][:,:s1,:s1]

y_train[:, 0, :] = 0
y_train[:, -1, :] = 0
y_train[:, :, 0] = 0
y_train[:, :, -1] = 0

#higher resolution
x_train_hres = reader.read_field('coeff')[:ntrain,::r2,::r2][:,:s2,:s2]
y_train_hres = reader.read_field('sol')[:ntrain,::r2,::r2][:,:s2,:s2]

y_train_hres[:, 0, :] = 0
y_train_hres[:, -1, :] = 0
y_train_hres[:, :, 0] = 0
y_train_hres[:, :, -1] = 0

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest,::r1,::r1][:,:s1,:s1]
y_test = reader.read_field('sol')[:ntest,::r1,::r1][:,:s1,:s1]

y_test[:, 0, :] = 0
y_test[:, -1, :] = 0
y_test[:, :, 0] = 0
y_test[:, :, -1] = 0

x_test_hres = reader.read_field('coeff')[:ntest,::r2,::r2][:,:s2,:s2]
y_test_hres = reader.read_field('sol')[:ntest,::r2,::r2][:,:s2,:s2]

y_test_hres[:, 0, :] = 0
y_test_hres[:, -1, :] = 0
y_test_hres[:, :, 0] = 0
y_test_hres[:, :, -1] = 0

#%%
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s1,s1,1)
x_test = x_test.reshape(ntest,s1,s1,1)

wp_x_t, wp_x_t_high_freq = wavelet_transform_reduction(x_test_hres.squeeze(-1), x_train.shape[1])
wp_y_t, wp_y_t_high_freq = wavelet_transform_reduction(y_test_hres, y_train.shape[1])

wp_x_train, wp_x_train_high_freq = wavelet_transform_reduction(x_train_hres.squeeze(-1), x_train.shape[1])
wp_y_train, wp_y_train_high_freq = wavelet_transform_reduction(y_test_hres, y_train.shape[1])


x_normalizer_hres = UnitGaussianNormalizer(wp_x_train)
wp_x_train = x_normalizer_hres.encode(wp_x_train)
wp_x_t = x_normalizer_hres.encode(wp_x_t)

y_normalizer_hres = UnitGaussianNormalizer(wp_y_train)
wp_y_train = y_normalizer_hres.encode(wp_y_train)

wp_x_train = wp_x_train.reshape(ntrain,s1,s1,1)
wp_x_t = wp_x_t.reshape(ntest,s1,s1,1)

x_t_hres = wp_x_t.reshape(wp_x_t.shape[0],-1).to(device=device)
y_t_hres = wp_y_t.reshape(wp_y_t.shape[0],-1).to(device=device)


# %%
""" Generating Grid and reshaping inputs """

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
# torch.save(y_pred_sdd,f"/home/user/Documents/GPO/experiments/C3_darcy_rect/results/y_pred_sdd_{current_time}.pt")


#%%
all_predictions_hres = []
with torch.no_grad():
    for i in range(0, len(x_t_hres)):
        x_batch = x_t_hres[i:i+1]  
        predictions_batch = Kernel(x_batch, x_tr) @ alpha_polyak_trained  
        all_predictions_hres.append(predictions_batch)  
y_pred_sdd_hres = torch.cat(all_predictions_hres, dim=0)
# torch.save(y_pred_sdd,f"{base_dir}y_pred_sdd_hres_{current_time}.pt")
#%%
torch.save(y_pred_sdd_hres,f"/home/user/Documents/GPO/experiments/C3_darcy_rect/results/106_y_pred_sdd_hres_{current_time}.pt")


#%%
y_pred_sdd = y_pred_sdd.reshape(x_t.shape[0],s1,s1)
y_pred_sdd = y_normalizer.decode(y_pred_sdd.detach().cpu())
y_pred_sdd = y_pred_sdd.reshape(y_pred_sdd.shape[0], y_pred_sdd.shape[1]*y_pred_sdd.shape[2])

#%%
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
#%%
mse_loss = nn.MSELoss()
prediction_error = mse_loss(y_pred_sdd_hres.to(device), y_t_hres)

relative_error = torch.mean(torch.linalg.norm(y_pred_sdd_hres.to(device)- y_t_hres, axis = 1)/torch.linalg.norm(y_t_hres, axis = 1))


print(f'MSE Testing error: {(prediction_error).item()}')
print(f'Mean relative error: {100*relative_error} % ')


# %%
