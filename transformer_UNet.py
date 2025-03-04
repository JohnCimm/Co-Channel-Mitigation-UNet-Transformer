import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from config_transformer import ModelConfig

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, inner_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, model_dim)
        )
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = cfg.input_channels
        self.encoder_channels = []
        for i in range(cfg.encoder_depth):
            out_channels = min(cfg.hidden_dim * (2 ** i), cfg.max_hidden_dim)
            self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=cfg.kernel_size, stride=cfg.stride, padding=1))
            self.encoder_channels.append(out_channels)
            in_channels = out_channels
        self.output_channels = in_channels  # Track final encoder output channels
    
    def forward(self, x):
        skips = []
        for layer in self.layers:
            x = F.gelu(layer(x))
            skips.append(x)  # Store skip connections for decoder
        return x, skips

class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(cfg.model_dim, cfg.attention_heads, cfg.inner_dim)
            for _ in range(cfg.bottleneck_depth)
        ])
        self.projection = nn.Linear(cfg.output_channels, cfg.model_dim)
        self.revert_projection = nn.Linear(cfg.model_dim, cfg.output_channels)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for transformer (B, C, T) -> (B, T, C)
        x = self.projection(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.revert_projection(x)
        x = x.permute(0, 2, 1)  # Reshape back for decoder (B, T, C) -> (B, C, T)
        return x

class Decoder(nn.Module):
    def __init__(self, cfg, encoder_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        self.conv_align = nn.ModuleList()
        in_channels = cfg.output_channels
        for i in range(cfg.decoder_depth):
            out_channels = encoder_channels[-(i + 1)]
            self.conv_align.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))  # Ensure alignment
            self.layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=cfg.kernel_size, stride=cfg.stride, padding=1, output_padding=1))
            in_channels = out_channels
    
    def forward(self, x, skips):
        for i, (layer, align_layer) in enumerate(zip(self.layers, self.conv_align)):
            x = F.gelu(layer(x))
            skip_connection = skips[-(i + 1)]
            if skip_connection.shape[1] != x.shape[1]:
                skip_connection = F.interpolate(skip_connection, size=x.shape[-1], mode='nearest')
                skip_connection = nn.Conv1d(skip_connection.shape[1], x.shape[1], kernel_size=1).to(x.device)(skip_connection)  # Align channels
            if skip_connection.shape[-1] != x.shape[-1]:
                skip_connection = F.interpolate(skip_connection, size=x.shape[-1], mode='nearest')  # Align time dimension
            x += skip_connection
        return x

class TransformerUNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = Encoder(cfg)
        cfg.output_channels = self.encoder.output_channels  # Track output channels
        self.bottleneck = Bottleneck(cfg)
        self.decoder = Decoder(cfg, self.encoder.encoder_channels)
        self.final_layer = nn.Conv1d(cfg.hidden_dim, cfg.input_channels, kernel_size=1)
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.final_layer(x)
    
    def get_loss(self, model_outputs, ground_truth):
        return self.criterion(model_outputs, ground_truth)
