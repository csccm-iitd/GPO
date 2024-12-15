# config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    # Paths
    data_path: str = "/home/user/Documents/GP_WNO/DATA/burgers_data_R10.mat"
    model_save_dir: str = "./models"
    result_dir: str = "./results"
    
    # Data parameters
    ntrain: int = 1000
    ntest: int = 50
    subsampling_rate: int = 16  # 2**4
    
    # Model parameters
    width: int = 64
    level: int = 8  # Level of wavelet decomposition
    
    # Training parameters
    num_iterations: int = 1200
    learning_rate: float = 0.1
    weight_decay: float = 1e-6
    step_size: int = 50
    gamma: float = 0.75
    
    device_id: Optional[int] = 0  # Set to None for CPU
    
    # Random seed
    seed: int = 0
    
    # wandb parameters
    use_wandb: bool = False  
    wandb_project: str = "GPO_project"
    wandb_entity: str = "sawankr02"  # Replace with your wandb username
    wandb_run_name: str = "gp_burgers_run"
