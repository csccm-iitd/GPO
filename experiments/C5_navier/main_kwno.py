# main.py
#%%
import os
import sys
sys.path.append("/home/user/Documents/GPO/general")
import torch
import numpy as np
from datetime import datetime
from config import Config
from preprocess import load_and_preprocess
from gp_model2d import GaussianProcessRegression
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from utilities3 import *
import wandb
from torch.optim import Adam
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Tuple
from prettytable import PrettyTable  
from misc import display_training_progress, display_final_results, display_config_table


def train_model(model: GaussianProcessRegression, 
               likelihood: MultitaskGaussianLikelihood, 
               train_x: torch.Tensor, 
               train_y: torch.Tensor, 
               config: Config, 
               wandb_run=None) -> GaussianProcessRegression:

    model.train()
    likelihood.train()
    
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(1, config.num_iterations + 1):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, config.num_iterations, loss.item()))

        optimizer.step()
        
        if wandb_run and ((i % 100 == 0) or (i <= 10)):
            wandb_run.log({"Iteration": i, "Loss": loss.item()})
            display_training_progress(i, loss.item())  # Display training progress
    
    return model

def predict(model: GaussianProcessRegression, 
            likelihood: MultitaskGaussianLikelihood, 
            test_x: torch.Tensor, 
            config: Config, 
            wandb_run = None) -> Tuple[np.ndarray, np.ndarray]:

    model.eval()
    likelihood.eval()
    
    all_means = []
    all_variances = []
    
    with torch.no_grad():
        for i in range(test_x.shape[0]):
            x_i = test_x[i:i+1, :]
            observed_pred = model(x_i)
            mean_pred_i = observed_pred.mean.cpu().numpy()
            var_pred_i = observed_pred.variance.cpu().numpy()
            all_means.append(mean_pred_i)
            all_variances.append(var_pred_i)
            

    
    all_means_np = np.concatenate(all_means, axis=0)
    all_variances_np = np.concatenate(all_variances, axis=0)
        
    return all_means_np, all_variances_np

def main() -> None:

    config = Config()
    
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            config=config,
            reinit=True
        )
        wandb_run = wandb.run
    else:
        wandb_run = None 
    
    display_config_table(config)
    
    current_time = datetime.now().strftime("%H%M_%d%m")
    result_dir = os.path.join(config.result_dir, current_time)
    os.makedirs(result_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f'cuda:{config.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if wandb_run:
        wandb_run.log({"Device": str(device)})
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if wandb_run:
        wandb_run.log({"Random Seed": config.seed})
    
    # Preprocess data
    x_tr, y_tr, x_t, y_t, x_normalizer, y_normalizer, x_train_original = load_and_preprocess(config)
    
    # Initialize likelihood and model
    likelihood = MultitaskGaussianLikelihood(num_tasks=y_tr.shape[-1]).to(device)
    model = GaussianProcessRegression(
        train_x=x_tr, 
        train_y=y_tr, 
        likelihood=likelihood, 
        width=config.width, 
        level=config.level, 
        dummy_data=x_train_original  
    ).to(device)
    
    
    model = train_model(model, likelihood, x_tr, y_tr, config, wandb_run)
    
    # Save model state
    os.makedirs(config.model_save_dir, exist_ok=True)
    model_state_path = os.path.join(result_dir, 'model_state_dict.pth')
    torch.save(model.state_dict(), model_state_path)
    model_path = os.path.join(result_dir, 'model.pth')
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")
       
    # Perform prediction
    mean_pred_np, var_pred_np = predict(model, likelihood, x_t, config, wandb_run)
    
    # Reshape and decode predictions
    s = config.width // config.r
    mean_pred = torch.tensor(mean_pred_np).reshape(config.ntest, s, s)
    mean_pred = y_normalizer.decode(mean_pred).reshape(config.ntest, s*s)
    
    var_pred = torch.tensor(var_pred_np).reshape(config.ntest, s, s)
    var_pred = y_normalizer.decode(var_pred).reshape(config.ntest, s*s)
    var_pred = torch.abs(var_pred)
    
    # Compute prediction error
    mse_loss_fn = torch.nn.MSELoss()
    mse_loss = mse_loss_fn(mean_pred.to(device), y_t)
    relative_error = torch.mean(torch.linalg.norm(mean_pred.to(device) - y_t, dim=1) / torch.linalg.norm(y_t, dim=1))
    
    print(f'Mean testing MSE: {mse_loss.item():.6f}')
    print(f'Mean relative error: {100 * relative_error.item():.2f} %')
    
    # Display final results
    display_final_results(mse_loss.item(), relative_error.item(), result_dir)
    
    if wandb_run:
        wandb_run.log({
            "Mean Testing MSE": mse_loss.item(),
            "Mean Relative Error (%)": 100 * relative_error.item()
        })
    
    # Save results
    results = {
        'mean_pred': mean_pred.cpu().numpy(),
        'var_pred': var_pred.cpu().numpy(),
        'mse_loss': mse_loss.item(),
        'relative_error': relative_error.item()
    }
    np.save(os.path.join(result_dir, 'predictions.npy'), results)
    print(f"Results saved to {result_dir}")
    
    if wandb_run:
        wandb_run.log({"Results Saved Path": result_dir})
    
    if wandb_run:
        wandb.finish()

if __name__ == "__main__":
    main()

#%%