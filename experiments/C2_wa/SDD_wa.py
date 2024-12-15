
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
from _sdd import *
from datetime import datetime

# %%
'''  
DEVICE
'''
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print(f"Device: {device}")

# torch.manual_seed(0)
# np.random.seed(0)

current_time = datetime.now().strftime("%d%m%y_%H%M")

base_dir = '/home/user/Documents/GPO/experiments/C2_wa/results/'


# %%
'''   
CONFIG AND DATA 
'''

ntrain = 1000
ntest = 200
s = 40

lr = 0.01 #0.001, 
momentum = 0.9
iterations = 20000
B = 32
length_scale = 6.1
noise_scale = 0.02


# %%
""" Read data """

data_train = np.load('/home/user/Documents/GP_WNO/DATA/train_IC2.npz')
data_test = np.load('/home/user/Documents/GP_WNO/DATA/test_IC2.npz')
x_train, t_train, u_train = data_train["x"], data_train["t"], data_train["u"]  # N x nt x nx
x_test, t_test, u_test = data_test["x"], data_test["t"], data_test["u"]  # N x nt x nx

x_data_train = u_train[:, 0, :]  # N x nx, initial solution # previous time step taken -2
y_data_train = u_train[:, 20, :]  # N x nx, final solution

x_data_test = u_test[:, 0, :]  # N x nx, initial solution
y_data_test = u_test[:, 20, :]  # N x nx, final solution

x_data_train = torch.tensor(x_data_train)
y_data_train = torch.tensor(y_data_train)

x_data_test = torch.tensor(x_data_test)
y_data_test = torch.tensor(y_data_test)


x_train = x_data_train[:ntrain,:]
y_train = y_data_train[:ntrain,:]

x_test = x_data_test[:ntest,:]
y_test = y_data_test[:ntest,:]

x_train = x_train.reshape(ntrain,s,1)
x_test = x_test.reshape(ntest,s,1)

x_tr = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
y_tr = y_train

# test data
x_t = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
y_t = y_test


x_tr=x_tr.to(device)
y_tr=y_tr.to(device)
x_t=x_t.to(device)
y_t=y_t.to(device)



# %%

Kernel = ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)).to(device=device)
Kernel.base_kernel.lengthscale= length_scale
Kernel.outputvariance = noise_scale
sdd_algo = SDDAlgorithm(Kernel,x_tr, y_tr, device=device,iterations=iterations,B=B)
alpha_polyak_trained,_ = sdd_algo.train()


all_predictions = []
with torch.no_grad():
    for i in range(0, len(x_t)):
        x_batch = x_t[i:i+1]  
        predictions_batch = Kernel(x_batch, x_tr) @ alpha_polyak_trained 
        all_predictions.append(predictions_batch)  

y_pred_sdd = torch.cat(all_predictions, dim=0)
# torch.save(y_pred_sdd, f"{base_dir}y_pred_sdd_{current_time}.pt")


#%% 

'''PREDICTION ERROR'''
mse_loss = nn.MSELoss()
prediction_error = mse_loss(y_pred_sdd.to(device), y_t)

relative_error = torch.mean(torch.linalg.norm(y_pred_sdd.to(device)-y_t, axis = 1)/torch.linalg.norm(y_t, axis = 1))


print(f'MSE Testing error: {(prediction_error).item()}')
print(f'Mean relative error: {100*relative_error} % ')

# %%
