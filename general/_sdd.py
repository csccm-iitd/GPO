import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpytorch
import gc
from pytorch_wavelets import DWTForward, DWTInverse



class SDDAlgorithm:
    def __init__(self, Kernel, 
                x_tr, 
                y_tr, 
                device=None, 
                lr=0.001, 
                polyak=1e-3, 
                momentum=0.9, 
                iterations=10000, 
                B=10, 
                noise_scale=0.002):

        self.Kernel = Kernel
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.device = device
        self.lr = lr
        self.polyak = polyak
        self.momentum = momentum
        self.iterations = iterations
        self.B = B
        self.noise_scale = noise_scale
        self.N = len(x_tr)
        self.alpha = torch.zeros((self.N, y_tr.shape[1]), device=device)
        self.alpha_polyak = torch.zeros((self.N, y_tr.shape[1]), device=device)
        self.v = torch.zeros((self.N, y_tr.shape[1]), device=device)

    def g(self, params, idx):
        grad = torch.zeros((self.N, self.y_tr.shape[1]), device=self.device)
        grad[idx] = self.Kernel(self.x_tr[idx], self.x_tr) @ params - self.y_tr[idx] + (self.noise_scale ** 2) * params[idx]
        grad.mul_(self.N / self.B)  # In-place multiplication to scale the gradient
        return grad

    def update(self, params, params_polyak, velocity, idx):
        grad = self.g(params, idx)
        velocity.mul_(self.momentum).sub_(self.lr * grad)  # In-place update of velocity
        params.add_(velocity)  # In-place update of params
        params_polyak.mul_(self.polyak).add_((1.0 - self.polyak) * params)  # In-place update of params_polyak
        return params, params_polyak, velocity

    def compute_loss(self, params):
        predictions = self.Kernel(self.x_tr, self.x_tr) @ params
        mse_loss = torch.mean((predictions - self.y_tr) ** 2)
        prediction_error_norm = torch.linalg.norm(predictions - self.y_tr, axis=1)
        true_value_norm = torch.linalg.norm(self.y_tr, axis=1)
        relative_mse_loss = torch.mean(prediction_error_norm / true_value_norm)
        return mse_loss, relative_mse_loss

    def train(self):
        history = []
        for i in range(self.iterations):
            idx = torch.randperm(self.N, device=self.device)[:self.B]
            self.alpha, self.alpha_polyak, self.v = self.update(self.alpha, self.alpha_polyak, self.v, idx)
            if i % 100 == 0:
                mse_loss, relative_mse_loss = self.compute_loss(self.alpha_polyak)
                history.append((mse_loss.item(), relative_mse_loss.item()))
                print(f"Iteration {i+1}, MSE Loss: {mse_loss.item()}")
            if i % 1000 == 0:
                gc.collect()  # Manual garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear CUDA cache
        return self.alpha_polyak, history



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
    ifm = DWTInverse(wave='db6', mode='symmetric')
    reconstructed = ifm(coeffs)
    if original_W is not None:
        return reconstructed.squeeze(1)[:, :original_H, :original_W]
    else:
        return reconstructed.squeeze(1)[:, :original_H]



