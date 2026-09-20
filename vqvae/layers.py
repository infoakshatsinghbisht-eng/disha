"""
Neural network building blocks for the Visual Tokenizer (VQ-VAE).
Includes Residual Blocks, Downsampling, Upsampling, and Normalization layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual Block with Group Normalization and SiLU activation.
    Applies Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm + Skip Connection.
    """
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Ensure num_groups divides channels cleanly
        groups_in = min(num_groups, in_channels)
        while in_channels % groups_in != 0 and groups_in > 1:
            groups_in -= 1
            
        groups_out = min(num_groups, out_channels)
        while out_channels % groups_out != 0 and groups_out > 1:
            groups_out -= 1

        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip_conv(x)


class Downsample2d(nn.Module):
    """
    Spatial downsampling block (2x factor) using strided convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2d(nn.Module):
    """
    Spatial upsampling block (2x factor) using nearest-neighbor interpolation + Conv or Transposed Conv.
    Nearest-neighbor + Conv avoids checkerboard artifacts.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
