import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple, Literal

def get_dataset(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare the dataset based on configuration."""
    if config.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # Load full training set
        full_trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                 download=True, transform=transform)
        # Split into train and validation
        train_size = int(0.8 * len(full_trainset))
        val_size = len(full_trainset) - train_size
        trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
        
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    else:  # cifar10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Load full training set
        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                   download=True, transform=transform)
        # Split into train and validation
        train_size = int(0.8 * len(full_trainset))
        val_size = len(full_trainset) - train_size
        trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.num_workers)
    valloader = DataLoader(valset, batch_size=config.batch_size,
                          shuffle=False, num_workers=config.num_workers)
    testloader = DataLoader(testset, batch_size=config.batch_size,
                          shuffle=False, num_workers=config.num_workers)
    
    return trainloader, valloader, testloader

def get_model(config):
    """Create model based on configuration."""
    if config.model_type == 'mlp':
        from models.mlp import create_mlp_2x64, create_mlp_4x64, create_mlp_4x512
        input_size = 28*28 if config.dataset == 'mnist' else 32*32*3
        num_classes = 10
        
        if config.model_config == '2x64':
            return create_mlp_2x64(input_size, num_classes, 
                                 use_batchnorm=config.use_batchnorm,
                                 use_dropout=config.use_dropout,
                                 dropout_rate=config.dropout_rate)
        elif config.model_config == '4x64':
            return create_mlp_4x64(input_size, num_classes,
                                 use_batchnorm=config.use_batchnorm,
                                 use_dropout=config.use_dropout,
                                 dropout_rate=config.dropout_rate)
        elif config.model_config == '4x512':
            return create_mlp_4x512(input_size, num_classes,
                                  use_batchnorm=config.use_batchnorm,
                                  use_dropout=config.use_dropout,
                                  dropout_rate=config.dropout_rate)
            
    elif config.model_type == 'cnn':
        from models.cnn import create_shallow_cnn, create_deep_cnn
        num_classes = 10
        in_channels = 1 if config.dataset == 'mnist' else 3
        
        if config.model_config == 'shallow':
            return create_shallow_cnn(num_classes, in_channels=in_channels)
        elif config.model_config == 'deep':
            return create_deep_cnn(num_classes, in_channels=in_channels)
    
    elif config.model_type == 'mixer':
        from models.mixer import create_mixer
        num_classes = 10
        in_channels = 1 if config.dataset == 'mnist' else 3
        return create_mixer(num_classes, depth=config.model_config, in_channels=in_channels)
    
    raise ValueError(f"Unsupported model type: {config.model_type}")

def save_checkpoint(model, optimizer, epoch, config):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'config': config
    }
    loc = f'{config.save_dir}/{config.model_type}_{config.model_config}_epoch{epoch}.pt'
    torch.save(checkpoint, loc)
    print(f"Checkpoint saved to {loc}")
    return loc

def load_checkpoint(path, model, optimizer):
    """Load model checkpoint."""
    from config import TrainingConfig
    import torch.serialization
    
    # Add TrainingConfig to safe globals
    torch.serialization.add_safe_globals([TrainingConfig])
    
    # Load checkpoint with weights_only=False to allow loading the config
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'] 

class EarlyStopping:
    """Early stopping class."""
    def __init__(self, patience=10, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False