import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import os

from config import TrainingConfig
from utils import get_dataset, get_model, save_checkpoint, EarlyStopping

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(config: TrainingConfig):
    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=vars(config)
        )
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Get dataset and model
    trainloader, valloader, _ = get_dataset(config)
    model = get_model(config)
    print(f"Model {model.__class__.__name__} has {count_parameters(model):,} trainable parameters")
    model = model.to(config.device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                         momentum=config.momentum, weight_decay=config.weight_decay)
    
    # Early stopping setup
    early_stopping = EarlyStopping(patience=config.early_stopping)
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{config.epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with training metrics
            pbar.set_postfix({
                'train_loss': f'{running_loss/total:.3f}',
                'train_acc': f'{100.*correct/total:.1f}%'
            })
        
        # Calculate training metrics
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(valloader, desc='Validating')
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'val_loss': f'{val_loss/val_total:.3f}',
                    'val_acc': f'{100.*val_correct/val_total:.1f}%'
                })
        
        # Calculate validation metrics
        val_loss = val_loss / len(valloader)
        val_acc = 100. * val_correct / val_total
        
        # Print final metrics for the epoch
        print(f'\nEpoch {epoch+1}/{config.epochs}:')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.1f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.1f}%')
        
        # Log metrics to wandb
        if config.use_wandb:
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch + 1
            })
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break                    
    
    loc = save_checkpoint(model, optimizer, epoch + 1, config)

    if config.use_wandb:
        wandb.finish()
    
    return loc

if __name__ == '__main__':
    config = TrainingConfig()
    train(config) 