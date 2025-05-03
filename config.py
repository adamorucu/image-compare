from dataclasses import dataclass
from typing import Literal

@dataclass
class TrainingConfig:
    # Dataset settings
    dataset: Literal['mnist', 'cifar10'] = 'cifar10'
    batch_size: int = 128
    num_workers: int = 4
    
    # Model settings
    model_type: Literal['mlp', 'cnn', 'resnet', 'mixer', 'mycnn'] = 'mlp'
    model_config: str = '2x64'  # For MLP: '2x64', '4x64', '4x512'; For CNN: 'shallow', 'deep'
    
    # Normalization and regularization settings
    use_batchnorm: bool = True
    use_dropout: bool = True
    dropout_rate: float = 0.5
    weight_decay: float = 0.0001
    
    # Training settings
    epochs: int = 50
    learning_rate: float = 0.001
    momentum: float = 0.9
    early_stopping: int = 1e9
   
    # Device settings
    device: str = 'cuda'  # 'cuda' or 'cpu'
    
    # Weights & Biases settings
    use_wandb: bool = True
    wandb_project: str = 'model-evaluation'
    wandb_entity: str = ''  # Your W&B username
    
    # Checkpoint settings
    save_dir: str = 'checkpoints'
    save_every: int = 10  # Save checkpoint every N epochs 