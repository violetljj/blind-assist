"""Separate query parameters while cloning the frozen shared initial function."""
import copy
import torch
from torch import nn
from mz30_select import BranchSelector


class QuerySeparatedSelector(nn.Module):
    def __init__(self, initial_state):
        super().__init__()
        template=BranchSelector()
        template.load_state_dict(initial_state)
        self.heads=nn.ModuleList([copy.deepcopy(template.net) for _ in range(4)])

    def forward(self, features):
        assert features.ndim==2 and features.shape[1]==264
        onehot=torch.eye(4,device=features.device,dtype=features.dtype)
        return torch.cat([head(torch.cat([features,onehot[q:q+1].expand(len(features),-1)],-1))
                          for q,head in enumerate(self.heads)],-1)
