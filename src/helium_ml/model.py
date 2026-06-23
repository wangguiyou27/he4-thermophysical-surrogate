from __future__ import annotations

import torch
from torch import nn


class MultiPropertyMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for width in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = width
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

