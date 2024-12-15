
import os
import torch
import numpy as np
from datetime import datetime
from prettytable import PrettyTable 
from torchsummary import summary 

def display_config_table(config) -> None:
    table = PrettyTable()
    table.title = "Experiment Configuration"
    table.field_names = ["Parameter", "Value"]
    
    for field_name, field_value in config.__dict__.items():
        table.add_row([field_name, field_value])
    
    print(table)

def display_training_progress(iteration: int, loss: float) -> None:

    table = PrettyTable()
    table.title = f"Training Progress - Iteration {iteration}"
    table.field_names = ["Iteration", "Loss"]
    table.add_row([iteration, f"{loss:.6f}"])
    print(table)

def display_final_results(mse_loss: float, relative_error: float, result_dir: str) -> None:

    table = PrettyTable()
    table.title = "Final Evaluation Metrics"
    table.field_names = ["Metric", "Value"]
    table.add_row(["Mean Testing MSE", f"{mse_loss:.6f}"])
    table.add_row(["Mean Relative Error (%)", f"{100 * relative_error:.2f}"])
    table.add_row(["Results Saved At", result_dir])
    print(table)




def save_results(mean_pred, var_pred, model, save_dir, loss_history=None,loss_mse_history=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    date_str = datetime.now().strftime("%d%m_%H%M")
    mean_pred_file = os.path.join(save_dir, f'mean_predictions_{date_str}.npy')
    var_pred_file = os.path.join(save_dir, f'variance_predictions_{date_str}.npy')
    model_file = os.path.join(save_dir, f'model_{date_str}.pth')
    model_state_dict_file = os.path.join(save_dir, f'model_state_dict_{date_str}.pth')
    if loss_history is not None:
        loss_history_file = os.path.join(save_dir,f'loss_hist_{date_str}.npy')
        np.save(loss_history_file, loss_history.numpy())

    if loss_mse_history is not None:
        loss_mse_history_file = os.path.join(save_dir, f'loss_mse_hist_{date_str}.npy')    
        np.save(loss_mse_history_file, loss_mse_history.numpy())
        
    np.save(mean_pred_file, mean_pred.numpy())
    np.save(var_pred_file, var_pred.numpy())
    torch.save(model, model_file)
    torch.save(model.state_dict(), model_state_dict_file)

def print_model_details(model, input_size):
    summary(model, input_size=input_size)