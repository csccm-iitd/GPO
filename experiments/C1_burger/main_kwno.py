# main.py
import os
import torch
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
from config import Config
from preprocess import load_and_preprocess
from gp_model1d import GaussianProcessRegression
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from utils import UnitGaussianNormalizer
import wandb
from utils1 import WNO1d
from torch.optim import Adam
from prettytable import PrettyTable
from misc import display_training_progress, display_final_results, display_config_table

def train_model(model: GaussianProcessRegression, 
               likelihood: MultitaskGaussianLikelihood, 
               x_tr: torch.Tensor, 
               y_tr: torch.Tensor, 
               config: Config, 
               wandb_run=None) -> Tuple[list, list]:
    model.train()
    likelihood.train()
    
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    losses_nll, losses_mse = [], []
    
    for i in range(1, config.num_iterations + 1):
        optimizer.zero_grad()
        output = model(x_tr)
        loss_nll = -mll(output, y_tr)
        loss_mse = torch.nn.MSELoss()(output.mean, y_tr)
        loss = loss_nll
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses_nll.append(loss_nll.item())
        losses_mse.append(loss_mse.item())
        
        if (i % 100 == 0) or (i <= 10):
            if wandb_run:
                wandb_run.log({"Iteration": i, "NLL Loss": loss_nll.item(), "MSE Loss": loss_mse.item()})
            display_training_progress(i, loss_nll.item(), loss_mse.item())
    
    return losses_nll, losses_mse

def predict(model: GaussianProcessRegression, 
            likelihood: MultitaskGaussianLikelihood, 
            x_t: torch.Tensor, 
            y_t: torch.Tensor,
            config: Config, 
            y_normalizer: UnitGaussianNormalizer,
            device: torch.device,
            wandb_run=None) -> Tuple[np.ndarray, np.ndarray, float, float]:
    model.eval()
    likelihood.eval()
    
    all_means = []
    all_variances = []
    
    with torch.no_grad():
        for i in range(x_t.shape[0]):
            x_i = x_t[i:i+1, :]
            observed_pred = model(x_i)
            mean_pred_i = observed_pred.mean
            var_pred_i = observed_pred.variance 
            all_means.append(mean_pred_i.cpu().numpy())
            all_variances.append(var_pred_i.cpu().numpy())
    
    all_means_np = np.array(all_means)
    all_variances_np = np.array(all_variances)
    
    mean_pred = torch.tensor(all_means_np).squeeze(1)
    var_pred = torch.tensor(all_variances_np).squeeze(1)
    
    mean_pred = y_normalizer.decode(mean_pred.detach().cpu())
    var_pred = y_normalizer.decode(var_pred.detach().cpu())
    
    mse_loss = torch.nn.MSELoss()(mean_pred.to(device), y_t)
    relative_error = torch.mean(torch.linalg.norm(mean_pred.to(device) - y_t, dim=1) / torch.linalg.norm(y_t, dim=1))
    
    return mean_pred.numpy(), var_pred.numpy(), mse_loss.item(), relative_error.item()

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
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(config.result_dir, current_time)
    os.makedirs(result_dir, exist_ok=True)
    
    x_tr, y_tr, x_t, y_t, x_normalizer, y_normalizer, device = load_and_preprocess(config)
    
    wno = WNO1d(config.width, config.level, x_tr)
    likelihood = MultitaskGaussianLikelihood(num_tasks=y_tr.shape[-1]).to(device)
    model = GaussianProcessRegression(x_tr, y_tr, likelihood, wno).to(device)
    
    losses_nll, losses_mse = train_model(model, likelihood, x_tr, y_tr, config, wandb_run)
    
    mean_pred, var_pred, mse_loss, relative_error = predict(model, likelihood, x_t, y_t, config, y_normalizer, device, wandb_run)
    
    print(f"MSE Testing Error: {mse_loss:.6f}")
    print(f"Mean Relative Error: {100 * relative_error:.2f}%")
    display_final_results(mse_loss, relative_error, result_dir)
    
    os.makedirs(config.model_save_dir, exist_ok=True)
    model_state_path = os.path.join(result_dir, "model_state_dict.pth")
    torch.save(model.state_dict(), model_state_path)
    
    np.save(os.path.join(result_dir, "mean_pred.npy"), mean_pred)
    np.save(os.path.join(result_dir, "var_pred.npy"), var_pred)
    
    if wandb_run:
        wandb_run.log({
            "Mean Testing MSE": mse_loss,
            "Mean Relative Error (%)": 100 * relative_error,
            "Model State Path": model_state_path,
            "Mean Predictions Path": os.path.join(result_dir, "mean_pred.npy"),
            "Variance Predictions Path": os.path.join(result_dir, "var_pred.npy")
        })
        wandb.finish()
    
    print(f"Results saved in {result_dir}")

if __name__ == "__main__":
    main()
