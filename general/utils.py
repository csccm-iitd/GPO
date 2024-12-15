# %%

import gpytorch
from gpytorch.means import MultitaskMean
from gpytorch.kernels import InducingPointKernel
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
from pytorch_wavelets import DTCWTForward, DTCWTInverse

from pytorch_wavelets import DWTForward, DWTInverse
import numpy as np
import matplotlib.pyplot as plt


#%%

class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        dwt_ = DWT1D(wave="db6", J=self.level, mode="symmetric").to(dummy.device)
        # self.wavelet = 'db3'#'bior1.3'
        # dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode="symmetric") ##
        self.mode_data, _ = dwt_(dummy)
        self.modes1 = self.mode_data.shape[-1]

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        # return torch.einsum("bix,iox->box", input, weights)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute single tree Discrete Wavelet coefficients using some wavelet
        dwt = DWT1D(wave="db6", J=self.level, mode="symmetric").to(x.device)
        x_ft, x_coeff = dwt(x)

        # dwt = DWT1D(wave=self.wavelet, J=self.level, mode="symmetric").to(x.device)
        # x_ft, x_coeff = dwt(x)

        # Multiply the final low pass and high pass coefficients
        out_ft = torch.zeros(
            batchsize, self.out_channels, x_ft.shape[-1], device=x.device
        )
        out_ft[:, :, :] = self.compl_mul1d(x_ft[:, :, :], self.weights1)
        x_coeff[-1] = self.compl_mul1d(x_coeff[-1][:, :, :], self.weights1)

        idwt = IDWT1D(wave="db6", mode="symmetric").to(x.device)
        x = idwt((out_ft, x_coeff))
        # idwt = IDWT1D(wave=self.wavelet, mode="symmetric").to(x.device)
        # x = idwt((out_ft, x_coeff))
        return x


class WNO1d(nn.Module):
    def __init__(self, width, level, dummy_data):
        super(WNO1d, self).__init__()

        self.width = width
        self.level = level
        self.dummy_data = dummy_data
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width)
        # 4 layers of integral operator (k + W) # K = self.conv, w = self.w

        self.conv0 = WaveConv1d(self.width, self.width, self.level, self.dummy_data)
        self.conv1 = WaveConv1d(self.width, self.width, self.level, self.dummy_data)
        self.conv2 = WaveConv1d(self.width, self.width, self.level, self.dummy_data)
        self.conv3 = WaveConv1d(self.width, self.width, self.level, self.dummy_data)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
       # print(f"1 x : {x.shape}")
        # x= x.T
        # x = x.reshape(x.shape[0],1024,-1).float()
        x = x.reshape(x.shape[0],512,-1).float()
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # do padding, if required

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # remove padding, when required
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    

class CustomMean(gpytorch.means.Mean):
    def __init__(self, width, level, dummy_data ):
        super().__init__()
        self.wno = WNO1d(width, level, dummy_data)
        

    def forward(self, x):
        # x: Input data tensor of shape (batch_size, num_inputs)
                
        # Pass the reshaped input through the WNO1d function
        mean_prediction = self.wno(x)
        
        return mean_prediction


class CustomMultitaskMean(MultitaskMean):
    def __init__(self, custom_mean, num_tasks):
        super().__init__(base_means=[gpytorch.means.ConstantMean()], num_tasks=num_tasks)
        self.custom_mean = custom_mean
        

    def forward(self, input):
        mean_prediction = self.custom_mean(input)
        return mean_prediction
    

#%%
    
'''  
Continuous Wavelet transform 2d

'''

""" Def: 2d Wavelet layer """
class CWaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy):
        super(CWaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        # self.wavelet = 'coif2'
        dwt_ = DTCWTForward(J=self.level, biort='near_sym_b', qshift='qshift_b').to(dummy.device)
        mode_data, mode_coef = dwt_(dummy)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        self.modes21 = mode_coef[-1].shape[-3]
        self.modes22 = mode_coef[-1].shape[-2]

        self.scale = (1 / (in_channels * out_channels))
        self.weights0 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights15r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights15c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights45r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights45c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights75r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights75c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights105r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights105c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights135r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights135c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights165r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights165c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))

    # Convolution
    def mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # batchsize = x.shape[0]
        
        #Compute single tree Discrete Wavelet coefficients using some wavelet
        # dwt = DWT(J=self.level, mode='symmetric', wave=self.wavelet).cuda()
        dwt = DTCWTForward(J=self.level, biort='near_sym_b', qshift='qshift_b').to(x.device)
        x_ft, x_coeff = dwt(x)
        
        # Multiply relevant Wavelet modes
        # out_ft = torch.zeros(batchsize, self.out_channels,  x_ft.shape[-2], x_ft.shape[-1], device=x.device)
        out_ft = self.mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights0)
        # Multiply the finer wavelet coefficients        
        x_coeff[-1][:,:,0,:,:,0] = self.mul2d(x_coeff[-1][:,:,0,:,:,0].clone(), self.weights15r)
        x_coeff[-1][:,:,0,:,:,1] = self.mul2d(x_coeff[-1][:,:,0,:,:,1].clone(), self.weights15c)
        x_coeff[-1][:,:,1,:,:,0] = self.mul2d(x_coeff[-1][:,:,1,:,:,0].clone(), self.weights45r)
        x_coeff[-1][:,:,1,:,:,1] = self.mul2d(x_coeff[-1][:,:,1,:,:,1].clone(), self.weights45c)
        x_coeff[-1][:,:,2,:,:,0] = self.mul2d(x_coeff[-1][:,:,2,:,:,0].clone(), self.weights75r)
        x_coeff[-1][:,:,2,:,:,1] = self.mul2d(x_coeff[-1][:,:,2,:,:,1].clone(), self.weights75c)
        x_coeff[-1][:,:,3,:,:,0] = self.mul2d(x_coeff[-1][:,:,3,:,:,0].clone(), self.weights105r)
        x_coeff[-1][:,:,3,:,:,1] = self.mul2d(x_coeff[-1][:,:,3,:,:,1].clone(), self.weights105c)
        x_coeff[-1][:,:,4,:,:,0] = self.mul2d(x_coeff[-1][:,:,4,:,:,0].clone(), self.weights135r)
        x_coeff[-1][:,:,4,:,:,1] = self.mul2d(x_coeff[-1][:,:,4,:,:,1].clone(), self.weights135c)
        x_coeff[-1][:,:,5,:,:,0] = self.mul2d(x_coeff[-1][:,:,5,:,:,0].clone(), self.weights165r)
        x_coeff[-1][:,:,5,:,:,1] = self.mul2d(x_coeff[-1][:,:,5,:,:,1].clone(), self.weights165c)
        
        # Return to physical space        
        # idwt = IDWT(mode='symmetric', wave=self.wavelet).cuda()
        idwt = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(x.device)
        x = idwt((out_ft, x_coeff))
        return x
    

""" The forward operation """
class CWNO2d(nn.Module):
    def __init__(self, width, level, dummy_data):
        super(CWNO2d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.level = level
        self.dummy_data = dummy_data
        self.width = width
        self.padding = 1 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(1, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = CWaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv1 = CWaveConv2d(self.width, self.width, self.level, self.dummy_data)
        # self.conv2 = CWaveConv2d(self.width, self.width, self.level, self.dummy_data)
        # self.conv3 = CWaveConv2d(self.width, self.width, self.level, self.dummy_data)
        # self.conv4 = CWaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv5 = CWaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        # self.w2 = nn.Conv2d(self.width, self.width, 1)
        # self.w3 = nn.Conv2d(self.width, self.width, 1)
        # self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.w5 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 512)
        self.fc2 = nn.Linear(512, 1)
        # self.mu = param()

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding]) # padding, if required

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)
        
        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2
        # x = F.gelu(x)
        
        # x1 = self.conv4(x)
        # x2 = self.w4(x)
        # x = x1 + x2
        # x = F.gelu(x)

        x1 = self.conv5(x)
        x2 = self.w5(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding] # removing padding, when applicable
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x
        

def calculate_wavelet_levels(input_shape, target_shape):
    levels = 0
    while input_shape > target_shape:
        input_shape = (input_shape + 1) // 2  # Equivalent to floor division by 2
        levels += 1
    return levels

def pad_to_power_of_2(data):
    if len(data.shape) == 3:  # 2D case
        N, H, W = data.shape
        new_H = 2 ** int(np.ceil(np.log2(H)))
        new_W = 2 ** int(np.ceil(np.log2(W)))
        padded_data = torch.zeros((N, new_H, new_W))
        padded_data[:, :H, :W] = data
        return padded_data, H, W
    elif len(data.shape) == 2:  # 1D case
        N, L = data.shape
        new_L = 2 ** int(np.ceil(np.log2(L)))
        padded_data = torch.zeros((N, new_L))
        padded_data[:, :L] = data
        return padded_data, L, None

def crop_to_shape(data, H, W):
    if W is not None:  # 2D case
        return data[:, :H, :W]
    else:  # 1D case
        return data[:, :H]

def wavelet_transform_reduction(data, target_shape):
    input_shape = data.shape[-1]
    levels = calculate_wavelet_levels(input_shape, target_shape)
    xfm = DWTForward(J=levels, wave='db1', mode='zero')
    padded_data, original_H, original_W = pad_to_power_of_2(data)
    coeffs = xfm(padded_data.unsqueeze(1) if original_W is not None else padded_data.unsqueeze(1).unsqueeze(1))
    low_freq = coeffs[0]
    if original_W is not None:
        cropped_low_freq = crop_to_shape(low_freq.squeeze(1), target_shape, target_shape)
    else:
        cropped_low_freq = crop_to_shape(low_freq.squeeze(1).squeeze(1), target_shape, None)
    return cropped_low_freq, coeffs[1:]

def inverse_wavelet_transform(low_freq, high_freq, original_H, original_W):
    if high_freq:
        low_freq = low_freq.unsqueeze(1)
    if original_W is not None:
        low_freq_padded = torch.zeros((low_freq.shape[0], 1, 2 * original_H, 2 * original_W))
        low_freq_padded[:, :, :original_H, :original_W] = low_freq
    else:
        low_freq_padded = torch.zeros((low_freq.shape[0], 1, 2 * original_H))
        low_freq_padded[:, :, :original_H] = low_freq
    if high_freq:
        coeffs = (low_freq_padded, high_freq)
    else:
        coeffs = (low_freq_padded, None)
    ifm = DWTInverse(wave='db1', mode='zero')
    reconstructed = ifm(coeffs)
    if original_W is not None:
        return reconstructed.squeeze(1)[:, :original_H, :original_W]
    else:
        return reconstructed.squeeze(1)[:, :original_H]
