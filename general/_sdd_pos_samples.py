#%%
import torch
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal 
from gpytorch.means import MultitaskMean
from gpytorch.kernels import InducingPointKernel, ScaleKernel, MultitaskKernel, MaternKernel, RBFKernel
import numpy as np
import gc

class Sample_SDDAlgorithm:
    def __init__(self, Kernel, x_tr, f_tr,epsilon=None, device=None, lr=0.001,polyak=1e-3,momentum=0.9, iterations=10000, B=10, noise_scale=0.002):
        self.Kernel = Kernel
        self.x_tr = x_tr
        self.f_tr = f_tr
        self.epsilon = epsilon
        self.device = device
        self.lr = lr
        self.polyak = polyak
        self.momentum = momentum
        self.iterations = iterations
        self.B = B
        self.noise_scale = noise_scale
        self.N = len(x_tr)
        self.V_alpha = torch.zeros((self.N, f_tr.shape[1]), device=device)
        self.V_alpha_polyak = torch.zeros((self.N, f_tr.shape[1]), device=device)
        self.v = torch.zeros((self.N, f_tr.shape[1]), device=device)

    def g(self, params, idx):
        grad = torch.zeros((self.N, self.f_tr.shape[1]), device=self.device)
        grad[idx] = self.Kernel(self.x_tr[idx],self.x_tr) @ params - (self.f_tr+ self.epsilon)[idx] + (self.noise_scale ** 2) * params[idx]
        # grad[idx] = self.Kernel(self.x_tr[idx],self.x_tr) @ params - (self.f_tr+ self.epsilon)[idx] + (self.noise_scale ** 2) * params[idx]
        return (self.N / self.B) * grad

    def update(self, params, params_polyak, velocity, idx):
        # grad = self.g(params, idx)
        velocity = self.momentum * velocity - self.lr * self.g(params, idx)
        params = params + velocity
        params_polyak = self.polyak * params + (1.0 - self.polyak) * params_polyak
        return params, params_polyak, velocity

    def compute_loss(self, params):

        mse_loss = torch.mean((self.Kernel(self.x_tr, self.x_tr) @ params - self.f_tr) ** 2)
        
        # Compute the relative MSE
        prediction_error_norm = torch.linalg.norm(self.Kernel(self.x_tr, self.x_tr) @ params - (self.f_tr+ self.epsilon), axis=1)
        true_value_norm = torch.linalg.norm((self.f_tr+ self.epsilon), axis=1)
        relative_mse_loss = torch.mean(prediction_error_norm / true_value_norm)
        
        return mse_loss, relative_mse_loss

    def train(self):
        for i in range(self.iterations):
            idx = torch.randperm(self.N, device=self.device)[:self.B]
            self.V_alpha, self.V_alpha_polyak, self.v = self.update(self.V_alpha, self.V_alpha_polyak, self.v, idx)

            # Compute and print the losses at each iteration
            mse_loss, relative_mse_loss = self.compute_loss(self.V_alpha_polyak)

            if i%100 ==0:
                print(f"Iteration {i+1}, MSE Loss: {mse_loss.item()}")
            # gc.collect()
        return self.V_alpha_polyak





" Multioutput features: Random Fourier Features"
class RandomFourierFeatures:
    def __init__(self, input_dim, num_features, num_outputs, length_scale=1.0, device='cpu'):
        self.input_dim = input_dim
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.length_scale = length_scale
        self.device = device
        
        # Initialize random weights and biases for each output dimension on the specified device
        self.omega = torch.randn(input_dim, num_features, num_outputs, device=device) / length_scale
        self.b = 2 * torch.pi * torch.rand(num_features, num_outputs, device=device)
    
    def transform(self, X):
        X = X.to(self.device)
        # this needs to be modified
        # X shape: (n_samples, input_dim)
        # X_omega shape: (n_samples, num_features, num_outputs)
        X_omega = torch.einsum('si,ioj->soj', X, self.omega)
        # Reshape to (n_samples, num_features * num_outputs)
        transformed_features = torch.sqrt(torch.tensor(2.0, device=self.device) / self.num_features) * torch.cos(X_omega + self.b)
        return transformed_features.reshape(X.shape[0], -1)

class CustomMultitaskKernel(MultitaskKernel):
    def __init__(self, data_covar_module, num_tasks, rank=1, task_covar_prior=None, device=None, dtype=torch.float32):
        super().__init__(data_covar_module, num_tasks, rank, task_covar_prior)
        random_matrix = torch.randn(num_tasks, num_tasks, device=device, dtype=dtype) * 0.01
        self.task_covar_module.covar_factor.data = (random_matrix + random_matrix.T) / 2
        self.task_covar_module.covar_factor.data.add_(torch.eye(num_tasks, device=device, dtype=dtype) * 0.1)
        self.task_covar_module.var = torch.tensor(0.1, dtype=dtype, device=device)

def draw_sample_prior(x_tr, x_t, output_dim, device=None, noise_scale=0.01):
    ntrain = x_tr.shape[0]
    ntest = x_t.shape[0]
    
    mean_x_tr = torch.zeros(ntrain, output_dim, device=device)
    mean_x_t = torch.zeros(ntest, output_dim, device=device)
    
    covar_module_x_tr = CustomMultitaskKernel(ScaleKernel(MaternKernel(nu=2.5)), num_tasks=output_dim).to(device=device)
    covar_module_x_t = CustomMultitaskKernel(ScaleKernel(MaternKernel(nu=2.5)), num_tasks=output_dim).to(device=device)
    
    # Create and sample from the multitask distributions
    mvn_dist_x_tr = MultitaskMultivariateNormal(mean_x_tr, covariance_matrix=covar_module_x_tr(x_tr))
    f_x_tr = mvn_dist_x_tr.sample()
    
    mvn_dist_x_t = MultitaskMultivariateNormal(mean_x_t, covariance_matrix=covar_module_x_t(x_t))
    f_x_t = mvn_dist_x_t.sample()
    
    # Generate epsilon
    epsilon = noise_scale * torch.eye(ntrain, output_dim, device=device)
    
    return f_x_tr, f_x_t, epsilon


def compute_posterior_fn_sample(x_tr, x_t, output_dim,alpha_polyak, Kernel, B, device):
   
    f_x_tr_sample, f_x_t_sample, epsilon_sample = draw_sample_prior(x_tr, x_t, output_dim, device=device, noise_scale=0.01)

    # Initialize the SDD algorithm with the sampled data
    sdd_algo = Sample_SDDAlgorithm(Kernel, x_tr, f_x_tr_sample, epsilon_sample, device=device, lr=0.001, polyak=1e-3, momentum=0.9, iterations=3000, B=B, noise_scale=0.002)
    V_alpha_polyak_trained = sdd_algo.train()
    
    all_predictions = []
    with torch.no_grad():
        for i in range(0, len(x_t)):
            x_batch = x_t[i:i+1]  
            predictions_batch = Kernel(x_batch, x_tr) @ V_alpha_polyak_trained  # Compute predictions for the batch
            all_predictions.append(predictions_batch)  # Store batch predictions

    v_pred_sdd_sample = torch.cat(all_predictions, dim=0)

    posterior_sample = f_x_t_sample + alpha_polyak - v_pred_sdd_sample
        
  
    print(f"Shape of samples tensor: {posterior_sample.shape}")
    del f_x_tr_sample, f_x_t_sample, epsilon_sample 
    return posterior_sample



''' To initialze the params '''


def initialize_params(nn_wno, length_scale_param=None, noise_scale_param=None, state_dict_path=None, txt_file_path=None):
    if state_dict_path:
        full_state_dict = torch.load(state_dict_path)
        nn_wno_state_dict = {k[len('nn_wno.'):]: v for k, v in full_state_dict.items() if k.startswith('nn_wno.')}
        nn_wno.load_state_dict(nn_wno_state_dict)
        
        if 'length_scale' in full_state_dict:
            length_scale_param.data = full_state_dict['length_scale']
        if 'noise_scale' in full_state_dict:
            noise_scale_param.data = full_state_dict['noise_scale']
        
    if txt_file_path:
        with open(txt_file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                value = float(value)
                if key == 'length_scale':
                    length_scale_param.data = torch.tensor(value)
                elif key == 'noise_scale':
                    noise_scale_param.data = torch.tensor(value)
                elif key.startswith('nn_wno.'):
                    # Remove the 'nn_wno.' prefix to match the model's state dict
                    param_name = key[len('nn_wno.'):]
                    nn_wno.state_dict()[param_name].data = torch.tensor(value)
    
    print("Model and kernel parameters initialized successfully.")

