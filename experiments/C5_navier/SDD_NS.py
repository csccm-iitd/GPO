
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
# from utils import *
from _sdd import *

# %%
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
print(f"Device: {device}")
# Disable cuDNN
# torch.backends.cudnn.enabled = False

# torch.manual_seed(0)
# np.random.seed(0)


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
B = 8
polyak=2e-2#1e-2
noise_scale=0.002
length_scale = 78

step_size = 50
gamma = 0.75

level = 3 #4
width = 32  #64

r = 2 # 2
h = int(((64 - 1)/r) + 1)
s = h
r1 = 2 # 1
r2 = 2
h1 = int(((64 - 1)/r1) + 1)
h2 = int(((64 - 1)/r2) + 1)
s1 = h1
s2 = h2
s=s1
# %%
""" Read data """
reader_input = np.load(TRAIN_PATH) #shape: (64, 64, 40000)
reader_output = np.load(TEST_PATH)  #shape: (64, 64, 40000)

reader_input = torch.tensor(reader_input).permute(2,1,0).float() #shape: (40000, 64,64)
reader_output = torch.tensor(reader_output).permute(2,1,0).float()  #shape: (40000, 64,64)

x_train = reader_input[:ntrain,::r1,::r1][:,:s1,:s1]
y_train = reader_output[:ntrain,::r2,::r2][:,:s2,:s2]


x_test = reader_input[30000+ntrain:30000+ntrain+ntest,::r1,::r1][:,:s1,:s1]
y_test = reader_output[30000+ntrain:30000+ntrain+ntest,::r2,::r2][:,:s2,:s2]

#%%
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s,s,1)
x_test = x_test.reshape(ntest,s,s,1)


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
# torch.save(y_pred_sdd, "/home/user/Documents/GPO/NS/models/"+'y_pred_sdd.pt')

#%%
y_pred_sdd = y_pred_sdd.reshape(x_t.shape[0],s,s)
y_pred_sdd = y_normalizer.decode(y_pred_sdd.detach().cpu())
y_pred_sdd = y_pred_sdd.reshape(y_pred_sdd.shape[0], y_pred_sdd.shape[1]*y_pred_sdd.shape[2])

#%% 

'''PREDICTION ERROR'''
mse_loss = nn.MSELoss()
prediction_error = mse_loss(y_pred_sdd.to(device), y_t)

relative_error = torch.mean(torch.linalg.norm(y_pred_sdd.to(device)-y_t, axis = 1)/torch.linalg.norm(y_t, axis = 1))


print(f'MSE Testing error: {(prediction_error).item()}')
print(f'Mean relative error: {100*relative_error} % ')

