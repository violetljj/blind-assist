"""Untie near/far readout weights while retaining independent count semantics."""
import torch
from torch import nn


class RangeReadout(nn.Module):
    """Two copied linear maps; lateral/body sharing remains unchanged.

    Query order is BODY near/far, HEAD near/far, three lateral cells each.
    No image-dependent routing, evaluator input, or near/far exclusivity.
    """
    def __init__(self, linear):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().repeat(2, 1, 1))
        self.bias = nn.Parameter(linear.bias.detach().repeat(2, 1))

    def forward(self, feature):
        if feature.ndim != 3 or feature.shape[1:] != (12, 32):
            raise ValueError('Expected Bx12x32 query embeddings')
        return torch.cat([
            torch.nn.functional.linear(feature[:, j:j+3], self.weight[(j//3)%2],
                                       self.bias[(j//3)%2])
            for j in (0, 3, 6, 9)
        ], dim=1)
