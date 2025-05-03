import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[64, 64], 
                 use_batchnorm=True, use_dropout=True, dropout_rate=0.5):
        super(MLP, self).__init__()
        
        # Create layers based on hidden_sizes
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
                
            layers.append(nn.ReLU())
            
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_size = hidden_size
            
        # Add final layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        return self.network(x)

def create_mlp_2x64(input_size, num_classes, use_batchnorm=True, use_dropout=True, dropout_rate=0.5):
    return MLP(input_size, num_classes, hidden_sizes=[64, 64], 
              use_batchnorm=use_batchnorm, use_dropout=use_dropout, dropout_rate=dropout_rate)

def create_mlp_4x64(input_size, num_classes, use_batchnorm=True, use_dropout=True, dropout_rate=0.5):
    return MLP(input_size, num_classes, hidden_sizes=[64, 64, 64, 64], 
              use_batchnorm=use_batchnorm, use_dropout=use_dropout, dropout_rate=dropout_rate)

def create_mlp_4x512(input_size, num_classes, use_batchnorm=True, use_dropout=True, dropout_rate=0.5):
    return MLP(input_size, num_classes, hidden_sizes=[512, 512, 512, 512], 
              use_batchnorm=use_batchnorm, use_dropout=use_dropout, dropout_rate=dropout_rate) 