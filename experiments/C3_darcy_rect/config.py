# config.py
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Config:
    # Paths
    train_path: str = '/home/user/Documents/GP_WNO/DATA/piececonst_r421_N1024_smooth1.mat'
    test_path: str = '/home/user/Documents/GP_WNO/DATA/piececonst_r421_N1024_smooth2.mat'
    model_save_dir: str = '/home/user/Documents/GPO/NS/models'
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
    use_wandb: bool = False 
    wandb_project: str = "GPO_project"
    wandb_entity: str = "sawankr02"  
