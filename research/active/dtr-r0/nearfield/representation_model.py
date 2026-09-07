"""G12 official RepViT backbone with common fresh region heads and optional detail."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
from whisker_model import IMAGE_SIZE,SUPPORT_SIZE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pretrained_manifest(root):
    root=root.resolve()
    if (root/'manifest.json').exists():
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
    else:
        provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8-sig'))
        records={r['file']:r for r in provenance['records']}
        manifest=dict(architecture='repvit_m0_9',distillation=True,source='repvit.py',
            checkpoint='repvit_m0_9_distill_300e.pth',source_sha256=records['repvit.py']['sha256'],
            checkpoint_sha256=records['repvit_m0_9_distill_300e.pth']['sha256'],
            mean=[.485,.456,.406],std=[.229,.224,.225],provenance_sha256=sha(root/'provenance.json'),
            normalization_source_sha256=records['data_datasets.py']['sha256'])
        if sha(root/'data_datasets.py')!=manifest['normalization_source_sha256']:
            raise ValueError('Official transform source hash mismatch')
    if manifest['architecture']!='repvit_m0_9' or manifest['distillation'] is not True:
        raise ValueError('Frozen official RepViT m0_9 distillation checkpoint required')
    for key in ('source','checkpoint'):
        path=(root/manifest[key]).resolve()
        if not path.is_relative_to(root) or sha(path)!=manifest[key+'_sha256']:
            raise ValueError('Pretrained artifact path/hash mismatch')
    if manifest['mean']!=[.485,.456,.406] or manifest['std']!=[.229,.224,.225]:
        raise ValueError('Frozen ImageNet normalization required')
    return manifest


def load_official(root):
    root=Path(root).resolve(); manifest=pretrained_manifest(root)
    name='_g12_official_repvit_'+manifest['source_sha256'][:16]
    if name in sys.modules:
        module=sys.modules[name]
    else:
        spec=importlib.util.spec_from_file_location(name,root/manifest['source'])
        module=importlib.util.module_from_spec(spec); sys.modules[name]=module
        spec.loader.exec_module(module)
    backbone=module.repvit_m0_9(pretrained=False,distillation=True,num_classes=1000)
    checkpoint=torch.load(root/manifest['checkpoint'],map_location='cpu',weights_only=True)
    state=checkpoint['model'] if isinstance(checkpoint,dict) and 'model' in checkpoint else checkpoint
    backbone.load_state_dict(state,strict=True)
    return backbone,manifest


class RepresentationModel(nn.Module):
    def __init__(self,pretrained_root,detail=False):
        super().__init__()
        official,self.pretrained=load_official(pretrained_root)
        # Strict load includes the classification heads; only feature modules
        # enter the actual downstream model, so discarded classifier is untrained.
        self.backbone=official.features
        was_training=self.backbone.training; self.backbone.eval()
        with torch.no_grad():
            final,shallow=self.extract(torch.zeros(1,3,*IMAGE_SIZE))
        self.backbone.train(was_training)
        self.deep_channels=final.shape[1]; self.shallow_channels=shallow.shape[1]
        self.deep_projection=nn.Conv2d(self.deep_channels,32,1)
        self.support=nn.Conv2d(32,2,1)
        self.near=nn.Sequential(nn.AvgPool2d((6,8)),nn.Flatten(),nn.Linear(384,2))
        self.detail_enabled=bool(detail)
        # Append after ALL shared random initialization: B/C common parameters
        # are identical when each constructor starts from the same seed.
        if detail:
            self.detail=nn.Sequential(nn.Conv2d(self.shallow_channels,16,1),nn.Conv2d(16,32,3,2,1))
        self.register_buffer('image_mean',torch.tensor(self.pretrained['mean']).reshape(1,3,1,1))
        self.register_buffer('image_std',torch.tensor(self.pretrained['std']).reshape(1,3,1,1))

    def extract(self,rgb):
        shallow=None
        for layer in self.backbone:
            rgb=layer(rgb)
            if tuple(rgb.shape[-2:])==(IMAGE_SIZE[0]//4,IMAGE_SIZE[1]//4): shallow=rgb
        if shallow is None or tuple(rgb.shape[-2:])!=((IMAGE_SIZE[0]+31)//32,(IMAGE_SIZE[1]+31)//32):
            raise ValueError('Official backbone stride4/stride32 feature contract changed')
        return rgb,shallow

    def forward(self,rgb):
        if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(3,*IMAGE_SIZE): raise ValueError('Expected current RGB Bx3x144x256 in [0,1]')
        deep,shallow=self.extract((rgb-self.image_mean)/self.image_std)
        features=F.interpolate(self.deep_projection(deep),size=SUPPORT_SIZE,mode='bilinear',align_corners=False)
        if self.detail_enabled: features=features+self.detail(shallow)
        support=self.support(features)
        gated=features[:,None]*support.sigmoid()[:,:,None]
        b,heads,channels,h,w=gated.shape
        pooled=self.near[0](gated.reshape(b*heads,channels,h,w)).reshape(b,heads,-1)
        near=(pooled*self.near[2].weight[None]).sum(-1)+self.near[2].bias
        return near,support
