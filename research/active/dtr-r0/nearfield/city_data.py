"""City visible-support adapter; independent of frozen G13 data helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def pool_support(mask):
    """Positive wins; otherwise any unknown makes the pooled cell unknown."""
    if mask.ndim != 4 or tuple(mask.shape[1:]) != (2,360,640):
        raise ValueError('Expected Bx2x360x640 native support')
    if not torch.all((mask==-1)|(mask==0)|(mask==1)):
        raise ValueError('Support must be -1/0/1')
    positive = F.max_pool2d((mask==1).float(),20,20).bool()
    unknown = F.max_pool2d((mask==-1).float(),20,20).bool()
    return torch.where(positive,1,torch.where(unknown,-1,0)).to(torch.int8)


def pixel_support_bce(logits, targets):
    """Class-balanced support BCE, with zero gradient at unknown pixels."""
    if logits.shape != targets.shape or not torch.all((targets==-1)|(targets==0)|(targets==1)):
        raise ValueError('Support shape or values invalid')
    loss = F.binary_cross_entropy_with_logits(logits,targets.clamp_min(0).to(logits.dtype),reduction='none')
    positive, negative = targets==1, targets==0
    pc,nc = positive.sum(),negative.sum()
    classes = (pc>0).to(logits.dtype)+(nc>0).to(logits.dtype)
    return ((loss*positive).sum()/pc.clamp_min(1)+(loss*negative).sum()/nc.clamp_min(1))/classes.clamp_min(1)


def _array(root, record, allowed=None):
    path = (root/record['path']).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('Cache path escape')
    if allowed is not None and not path.is_relative_to((root/allowed).resolve()):
        raise ValueError('Cache crosses model/supervision boundary')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=record['sha256']:
            raise ValueError('Cache hash mismatch')
    return np.load(path,mmap_mode='r',allow_pickle=False)


class CityRGBDataset(Dataset):
    """RGB-only access, including test. The model performs normalization once."""
    def __init__(self, root, split):
        self.root=Path(root).resolve(strict=True)
        manifest=json.loads((self.root/'manifest.json').read_text())
        if manifest.get('schema')!='city-training-cache-v1' or manifest.get('status')!='PASS':
            raise ValueError('Completed City cache required')
        self.entry=manifest['partitions'][split]
        self.rgb=_array(self.root,self.entry['rgb'],'model')
        self.ids=self.entry['sample_indices']
        if self.rgb.dtype!=np.uint8 or self.rgb.shape!=(len(self.ids),144,256,3):
            raise ValueError('Invalid RGB cache')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self,index):
        rgb=torch.from_numpy(np.array(self.rgb[index],copy=True)).permute(2,0,1).float()/255.
        return {'rgb':rgb,'sample_index':self.ids[index]}


class CitySupervisedDataset(CityRGBDataset):
    """Fitting access explicitly refuses TEST labels."""
    def __init__(self,root,split='train'):
        if split not in ('train','val'):
            raise ValueError('TEST labels are evaluator-only')
        super().__init__(root,split)
        labels=json.loads((self.root/'supervision'/f'{split}.json').read_text())
        if labels['sample_indices']!=self.ids:
            raise ValueError('Supervision sample order mismatch')
        self.near=_array(self.root,labels['near'],'supervision')
        self.support=_array(self.root,labels['support'],'supervision')
        if self.near.shape!=(len(self),2) or self.support.shape!=(len(self),2,18,32):
            raise ValueError('Supervision dimensions invalid')
        if self.near.dtype!=np.float32 or not np.all((self.near==0)|(self.near==1)):
            raise ValueError('Near supervision must be finite float32 binary targets')
        if self.support.dtype!=np.int8 or not np.all((self.support==-1)|(self.support==0)|(self.support==1)):
            raise ValueError('Support supervision must be int8 -1/0/1')

    def __getitem__(self,index):
        item=super().__getitem__(index)
        item.update(near=torch.from_numpy(np.array(self.near[index],copy=True)).float(),
                    support=torch.from_numpy(np.array(self.support[index],copy=True)))
        return item
