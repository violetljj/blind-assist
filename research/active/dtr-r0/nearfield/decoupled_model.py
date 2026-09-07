"""G13: detail participates in attached support gating, deep contents only."""
import torch.nn.functional as F
from representation_model import RepresentationModel
from whisker_model import IMAGE_SIZE,SUPPORT_SIZE


class DecoupledModel(RepresentationModel):
    def __init__(self,pretrained_root):
        super().__init__(pretrained_root,detail=True)

    def forward(self,rgb):
        if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(3,*IMAGE_SIZE): raise ValueError('Expected current RGB Bx3x144x256')
        deep,shallow=self.extract((rgb-self.image_mean)/self.image_std)
        deep=F.interpolate(self.deep_projection(deep),size=SUPPORT_SIZE,mode='bilinear',align_corners=False)
        detail=self.detail(shallow)
        support=self.support(deep+detail)
        gated=deep[:,None]*support.sigmoid()[:,:,None]
        b,heads,channels,h,w=gated.shape
        pooled=self.near[0](gated.reshape(b*heads,channels,h,w)).reshape(b,heads,-1)
        near=(pooled*self.near[2].weight[None]).sum(-1)+self.near[2].bias
        return near,support
