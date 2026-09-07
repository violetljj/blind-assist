"""G10 predicted-region gating with exactly the unchanged G8 parameter set."""
import torch
from whisker_model import WhiskerModel


class DiversityModel(WhiskerModel):
    def __init__(self, region=False):
        super().__init__('ordinary_video')
        self.region = region

    def forward(self, rgb):
        features = self.spatial(rgb)
        support = self.support(features)
        if not self.region:
            return self.near(features), support
        # Unnormalized weighted mean retains absence. Each head uses its own
        # predicted support, with gradients through both mask and features.
        gated = features[:, None] * support.sigmoid()[:, :, None]
        b, heads, channels, height, width = gated.shape
        pooled = self.near[0](gated.reshape(b*heads, channels, height, width))
        pooled = pooled.reshape(b, heads, -1)
        logits = (pooled * self.near[2].weight[None]).sum(-1) + self.near[2].bias
        return logits, support
