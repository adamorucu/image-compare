import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import os

from config import TrainingConfig
from utils import get_dataset, get_model, load_checkpoint

def evaluate(config: TrainingConfig, checkpoint_path: str):
    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=vars(config)
        )
    
    # Get dataset and model
    _, _, testloader = get_dataset(config)
    model = get_model(config)
    model = model.to(config.device)
    
    # Load checkpoint
    epoch = load_checkpoint(checkpoint_path, model, None)
    model.eval()
    
    # Evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc='Evaluating'):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Calculate metrics
    test_loss = test_loss / len(testloader)
    test_acc = 100. * correct / total
    
    # Log results
    if config.use_wandb:
        wandb.log({
            'test_loss': test_loss,
            'test_acc': test_acc,
            'epoch': epoch
        })
    
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
    if config.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    config = TrainingConfig()
    checkpoint_path = 'checkpoints/checkpoint_epoch_50.pt'  # Example path
    evaluate(config, checkpoint_path) 