# preprocess.py
import numpy as np
import torch
from typing import Tuple
from config import Config
from utilities3 import *

def load_and_preprocess(config: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, UnitGaussianNormalizer, UnitGaussianNormalizer, torch.Tensor]:

    reader_input = np.load(config.train_path)  # shape: (64, 64, 40000)
    reader_output = np.load(config.test_path)  # shape: (64, 64, 40000)
    
    reader_input = torch.tensor(reader_input).permute(2, 1, 0).float()  # shape: (40000, 64, 64)
    reader_output = torch.tensor(reader_output).permute(2, 1, 0).float()  # shape: (40000, 64, 64)
    
    s = config.width // config.r  # e.g., 64 // 2 = 32
    x_train = reader_input[:config.ntrain, ::config.r, ::config.r][:, :s, :s]
    y_train = reader_output[:config.ntrain, ::config.r, ::config.r][:, :s, :s]
    
    x_test = reader_input[30000 + config.ntrain:30000 + config.ntrain + config.ntest, ::config.r, ::config.r][:, :s, :s]
    y_test = reader_output[30000 + config.ntrain:30000 + config.ntrain + config.ntest, ::config.r, ::config.r][:, :s, :s]
    
    # Normalize
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train_norm = x_normalizer.encode(x_train)
    x_test_norm = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train_norm = y_normalizer.encode(y_train)
    
    # Reshape
    x_train_norm = x_train_norm.reshape(config.ntrain, s*s, 1)
    x_test_norm = x_test_norm.reshape(config.ntest, s*s, 1)
    
    # Reshape for GP
    x_tr = x_train_norm.squeeze(-1)  # shape: (ntrain, s*s)
    y_tr = y_train_norm.reshape(config.ntrain, s*s)
    
    x_t = x_test_norm.squeeze(-1)  # shape: (ntest, s*s)
    y_t = y_test.reshape(config.ntest, s*s)
    
    device = torch.device(f'cuda:{config.device_id}' if torch.cuda.is_available() else 'cpu')
    x_tr = x_tr.to(device)
    y_tr = y_tr.to(device)
    x_t = x_t.to(device)
    y_t = y_t.to(device)
    
    x_train_original = x_train_norm.reshape(config.ntrain, 1, s, s).to(device)
    
    return x_tr, y_tr, x_t, y_t, x_normalizer, y_normalizer, x_train_original
