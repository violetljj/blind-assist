"""G11 sole intervention: detach the predicted region gate from near loss."""
from diversity_model import DiversityModel


class DetachedModel(DiversityModel):
    def __init__(self, detached=True):
        super().__init__(region=True)
        self.detached = detached

    def forward(self, rgb):
        if not self.detached:
            return super().forward(rgb)
        features = self.spatial(rgb)
        support = self.support(features)
        gated = features[:, None] * support.sigmoid().detach()[:, :, None]
        b, heads, channels, height, width = gated.shape
        pooled = self.near[0](gated.reshape(b*heads, channels, height, width))
        pooled = pooled.reshape(b, heads, -1)
        near = (pooled * self.near[2].weight[None]).sum(-1) + self.near[2].bias
        return near, support
