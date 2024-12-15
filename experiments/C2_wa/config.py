# config.py
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Config:
    # Paths
    train_path: str = '/home/user/Documents/GP_WNO/DATA/train_IC2.npz'
    test_path: str = '/home/user/Documents/GP_WNO/DATA/test_IC2.npz'
    model_save_dir: str = '/home/user/Documents/GPO/C2_wa/models'
    result_dir: str = './results'  
    
    # Data parameters
    ntrain: int = 500
    ntest: int = 200
    batch_size: int = 25
    r: int = 15  # Downsampling rate
    
    # Model parameters
    level: int = 4
    width: int = 64
    
    # Training parameters
    num_iterations: int = 1500
    learning_rate: float = 0.01
    weight_decay: float = 1e-6
    step_size: int = 50
    gamma: float = 0.75
    
    # Device
    device_id: int = 1
    
    # Random seed
    seed: int = 0
    
    # wandb parameters
    use_wandb: bool = False # Flag to enable/disable wandb logging
    wandb_project: str = "GPO_project"
    wandb_entity: str = "sawankr02"  # Replace with your wandb username
    wandb_run_name: str = "GPO_run"
