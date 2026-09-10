"""Choose a sensor branch only on unsupported baseline-positive disagreements."""
import torch
from torch import nn


class BranchSelector(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(268,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,features):
        assert features.ndim==2 and features.shape[1]==264
        onehot=torch.eye(4,device=features.device,dtype=features.dtype)[None].expand(len(features),-1,-1)
        x=torch.cat([features[:,None,:].expand(-1,4,-1),onehot],-1)
        return self.net(x).squeeze(-1)


def negative_branch_confidence(rgb_logits,selector_logits):
    probability=selector_logits.sigmoid()
    return torch.where(rgb_logits<0,probability,1-probability)
