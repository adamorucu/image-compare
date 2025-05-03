"""
Mixer model implementation
"""
import torch
import torch.nn as nn

class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(-2, -1)

class Mixer(nn.Module):
    def __init__(self, num_classes, depth='shallow', patch_size=4, hidden_dim=512, tokens_mlp_dim=256, channels_mlp_dim=2048, in_channels=3):
        super(Mixer, self).__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * in_channels
        self.num_patches = (32 // patch_size) ** 2  # 32x32 input images
        
        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)
        
        # Number of mixer layers based on depth
        num_layers = 2 if depth == 'shallow' else 8
        
        # Mixer layers
        self.mixer_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.mixer_layers.append(nn.Sequential(
                # Token-mixing MLP
                nn.LayerNorm(hidden_dim),
                Transpose(),
                nn.Linear(self.num_patches, tokens_mlp_dim),
                nn.GELU(),
                nn.Linear(tokens_mlp_dim, self.num_patches),
                Transpose(),
                # Channel-mixing MLP
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, channels_mlp_dim),
                nn.GELU(),
                nn.Linear(channels_mlp_dim, hidden_dim)
            ))
            
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Reshape into patches
        B, C, H, W = x.shape
        # Split into patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Reshape to (B, num_patches, patch_dim)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, self.num_patches, self.patch_dim)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Apply mixer layers
        for mixer_layer in self.mixer_layers:
            x = x + mixer_layer(x)
            
        # Global average pooling
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.head(x)
        return x

def create_mixer(num_classes, depth='shallow', in_channels=3):
    return Mixer(num_classes, depth, in_channels=in_channels)
