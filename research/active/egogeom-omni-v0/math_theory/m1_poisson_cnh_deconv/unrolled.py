"""Fixed five multiplicative KL steps on a discrete IRF bank; no data-dependent control flow."""

import numpy as np
import torch
from poisson_cnh import DC, XTALK, kernel
from torch import nn


class FiveStepCNH(nn.Module):
    def __init__(self, bins=48, penalty=0.025):
        super().__init__()
        z = np.linspace(0.15, 4, bins)
        atoms = torch.tensor(np.vstack((kernel(z), XTALK, DC)), dtype=torch.float32)
        cost = torch.zeros(len(atoms))
        cost[:bins] = penalty
        self.register_buffer("atoms", atoms)
        self.register_buffer("norm", atoms.sum(1) + cost)

    def forward(self, counts):
        # NN positive EM update; channels last matmul shares the bank across zones.
        y = counts.movedim(1, -1)
        mass = (y.sum(-1, keepdim=True) + 1e-7) / self.norm / self.norm.numel()
        for _ in range(5):
            mass = mass * ((y / (mass @ self.atoms + 1e-7)) @ self.atoms.T) / self.norm
        return mass.movedim(-1, 1)
