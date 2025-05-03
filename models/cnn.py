import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes, depth='shallow', in_channels=3, input_size=32):
        super(CNN, self).__init__()
        self.input_size = input_size
        
        if depth == 'shallow':
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
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
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
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

def create_shallow_cnn(num_classes, in_channels=3, input_size=32):
    return CNN(num_classes, depth='shallow', in_channels=in_channels, input_size=input_size)

def create_deep_cnn(num_classes, in_channels=3, input_size=32):
    return CNN(num_classes, depth='deep', in_channels=in_channels, input_size=input_size) 