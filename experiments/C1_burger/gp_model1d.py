# model.py
import sys
import gpytorch
import torch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.kernels import MultitaskKernel, MaternKernel
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.distributions import MultitaskMultivariateNormal
from typing import Any
from utils1 import WNO1d  
from torch import Tensor


class GaussianProcessRegression(ExactGP):
    def __init__(self, 
                 train_x: Tensor, 
                 train_y: Tensor, 
                 likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood, 
                 wno: WNO1d):
        super().__init__(train_x, train_y, likelihood)
        
        num_dims = train_y.shape[-1]  # Number of output dimensions
        
        self.wno = wno
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_dims)
        self.covar_module = MultitaskKernel(
            MaternKernel(nu=2.5, ard_num_dims=num_dims, has_lengthscale=True),
            num_tasks=num_dims,
            rank=2
        )
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)
    
    def forward(self, x: Tensor) -> MultitaskMultivariateNormal:
        nn_proj = self.wno(x)
        nn_proj = self.scale_to_bounds(nn_proj)
        mean_x = self.mean_module(nn_proj)
        covar_x = self.covar_module(nn_proj)
        return MultitaskMultivariateNormal(mean_x, covar_x)