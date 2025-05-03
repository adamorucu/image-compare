import torch
import torch.nn as nn
import torch.nn.functional as F

class UnfoldConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(UnfoldConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        # Add padding if needed
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
            
        # Get input dimensions
        batch_size, _, height, width = x.shape
        
        # Calculate output dimensions
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        
        # Unfold the input tensor
        unfolded = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        
        # Reshape weight for matrix multiplication
        weight_reshaped = self.weight.view(self.out_channels, -1)
        
        # Perform convolution using matrix multiplication
        output = torch.matmul(weight_reshaped, unfolded) + self.bias.view(-1, 1)
        
        # Reshape output to match expected shape
        output = output.view(batch_size, self.out_channels, output_height, output_width)
        
        return output

class MyCNN(nn.Module):
    def __init__(self, num_classes, depth='shallow', in_channels=3, input_size=32):
        super(MyCNN, self).__init__()
        self.input_size = input_size
        
        if depth == 'shallow':
            self.features = nn.Sequential(
                UnfoldConv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                UnfoldConv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            # Calculate feature map size after convolutions and pooling
            self.feature_size = self._get_feature_size()
            self.classifier = nn.Sequential(
                nn.Linear(64 * self.feature_size * self.feature_size, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        else:  # deep
            self.features = nn.Sequential(
                UnfoldConv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                UnfoldConv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                UnfoldConv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                UnfoldConv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                UnfoldConv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                UnfoldConv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            # Calculate feature map size after convolutions and pooling
            self.feature_size = self._get_feature_size()
            self.classifier = nn.Sequential(
                nn.Linear(256 * self.feature_size * self.feature_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
    
    def _get_feature_size(self):
        """Calculate the size of feature maps after all convolutions and pooling"""
        size = self.input_size
        if isinstance(self.features, nn.Sequential):
            for module in self.features:
                if isinstance(module, nn.MaxPool2d):
                    size = size // 2
        return size
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_shallow_mycnn(num_classes, in_channels=3, input_size=32):
    return MyCNN(num_classes, depth='shallow', in_channels=in_channels, input_size=input_size)

def create_deep_mycnn(num_classes, in_channels=3, input_size=32):
    return MyCNN(num_classes, depth='deep', in_channels=in_channels, input_size=input_size)
