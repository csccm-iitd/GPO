# preprocess.py
import torch
import numpy as np
from typing import Tuple
from config import Config
from utils import UnitGaussianNormalizer, WNO1d
from utilities3 import MatReader 

def load_and_preprocess(config: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, UnitGaussianNormalizer, UnitGaussianNormalizer, torch.Tensor]:

    dataloader = MatReader(config.data_path)
    x_data = dataloader.read_field("a")[:, ::config.subsampling_rate]
    y_data = dataloader.read_field("u")[:, ::config.subsampling_rate]
    
    ntrain, ntest = config.ntrain, config.ntest
    h = 2**13 // config.subsampling_rate  
    s = h
    
    x_train = x_data[:ntrain, :].reshape(ntrain, s, 1)
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :].reshape(ntest, s, 1)
    y_test = y_data[-ntest:, :]
    
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train_norm = x_normalizer.encode(x_train)
    x_test_norm = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train_norm = y_normalizer.encode(y_train)
    
    x_tr = x_train_norm.reshape(ntrain, s * 1)  # s*1 since it's 1D
    y_tr = y_train_norm  
    
    x_t = x_test_norm.reshape(ntest, s * 1)
    y_t = y_test  
    
    device = torch.device(f"cuda:{config.device_id}" if torch.cuda.is_available() and config.device_id is not None else "cpu")
    x_tr = x_tr.to(device)
    y_tr = y_tr.to(device)
    x_t = x_t.to(device)
    y_t = y_t.to(device)
    
    x_train_original = x_train_norm.reshape(ntrain, 1, s).to(device)
    
    return x_tr, y_tr, x_t, y_t, x_normalizer, y_normalizer, x_train_original
