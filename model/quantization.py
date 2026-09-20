"""
Dynamic Quantization and Memory Optimization Engine for 8GB RAM Laptops.
Quantizes Linear layers dynamically to 8-bit representations, reducing memory bandwidth and RAM usage by up to 50-75%.
"""

from typing import Dict, Any
import torch
import torch.nn as nn


class Int8Linear(nn.Module):
    """
    Dynamically Quantized 8-Bit Linear Layer.
    Stores weights in int8 with per-channel scaling factors, dequantizing on the fly for matrix multiply.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer("qweight", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones((out_features, 1), dtype=torch.float32))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, float_linear: nn.Linear) -> "Int8Linear":
        """Converts standard float32/fp16 nn.Linear into quantized Int8Linear."""
        q_layer = cls(
            in_features=float_linear.in_features,
            out_features=float_linear.out_features,
            bias=float_linear.bias is not None,
        )
        
        # Calculate per-channel maximum absolute scale
        weight_data = float_linear.weight.data.float()
        max_vals = torch.max(torch.abs(weight_data), dim=1, keepdim=True)[0].clamp(min=1e-5)
        scales = max_vals / 127.0
        
        # Quantize to int8: [-127, 127]
        qweight = torch.round(weight_data / scales).clamp(-127, 127).to(torch.int8)
        
        q_layer.qweight.copy_(qweight)
        q_layer.scales.copy_(scales)
        
        if float_linear.bias is not None:
            q_layer.bias.copy_(float_linear.bias.data.float())
            
        return q_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic on-the-fly dequantization
        dequant_weight = self.qweight.float() * self.scales
        out = torch.matmul(x, dequant_weight.t().type_as(x))
        if self.bias is not None:
            out = out + self.bias.type_as(x)
        return out


def quantize_model_for_low_ram(model: nn.Module) -> nn.Module:
    """
    Recursively replaces standard linear projection layers with 8-bit quantized linear layers.
    Significantly cuts RAM consumption on 8GB laptops.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, Int8Linear.from_float(child))
        else:
            quantize_model_for_low_ram(child)
    return model


def get_model_ram_mb(model: nn.Module) -> float:
    """Calculates total model parameter memory footprint in megabytes (MB)."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / (1024 * 1024)
