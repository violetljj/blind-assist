"""Frozen original FoundationStereo with bounded recurrent/upsampling adaptation.

No upstream source edits. The whole model stays in evaluation mode, including
BatchNorm/dropout; gradients are enabled only on three explicitly named modules.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from foundation_geometry_infer import load_model, sha

SELECTED = ('update_block', 'spx_2_gru', 'spx_gru')


def selected_name(name):
    return any(name.startswith(prefix+'.') for prefix in SELECTED)


def state_hashes(model):
    """Exact tensor hashes, including every frozen parameter and registered buffer."""
    result={'selected':{},'frozen':{},'buffers':{}}
    for name,value in model.named_parameters():
        key='selected' if selected_name(name) else 'frozen'
        result[key][name]=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    for name,value in model.named_buffers():
        result['buffers'][name]=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return result


def freeze_scope(model):
    model.eval()
    for name,value in model.named_parameters():value.requires_grad_(selected_name(name))
    assert all(not module.training for module in model.modules())
    return {prefix:sum(p.numel() for name,p in model.named_parameters() if name.startswith(prefix+'.'))
            for prefix in SELECTED}


def checkpoint_update_block(block):
    """Checkpoint actual recurrence without capturing a subsequently mutated list."""
    from torch.utils.checkpoint import checkpoint
    original=block.forward
    def wrapped(net,inp,corr,disp,att):
        if not torch.is_grad_enabled():return original(net,inp,corr,disp,att)
        n=len(net);ni=len(inp);na=len(att)
        def step(*flat):
            local_net=list(flat[:n]);local_inp=list(flat[n:n+ni])
            local_corr,local_disp=flat[n+ni:n+ni+2]
            local_att=list(flat[n+ni+2:n+ni+2+na])
            updated,mask,delta=original(local_net,local_inp,local_corr,local_disp,local_att)
            return (*updated,mask,delta)
        values=checkpoint(step,*tuple(net),*tuple(inp),corr,disp,*tuple(att),use_reentrant=False)
        return list(values[:n]),values[n],values[n+1]
    block.forward=wrapped


class StereoAdaptModel:
    def __init__(self,config_path,*,checkpoint_recurrence=False):
        from omegaconf import OmegaConf
        self.config_path=Path(config_path)
        self.plan=json.loads(self.config_path.read_text(encoding='utf-8'))
        self.source=Path(self.plan['source']);checkpoint=Path(self.plan['checkpoint'])
        assert sha(checkpoint)==self.plan['checkpoint_sha256']
        assert sha(checkpoint.parent/'cfg.yaml')==self.plan['cfg_sha256']
        for name,digest in self.plan['source_hashes'].items():assert sha(self.source/name)==digest,name
        assert torch.cuda.is_available(),'CUDA required'
        os.environ['XFORMERS_DISABLED']='1'
        cfg=OmegaConf.load(checkpoint.parent/'cfg.yaml')
        cfg.vit_size='vitl';cfg.valid_iters=32;cfg.low_memory=self.plan['low_memory']
        assert cfg.max_disp==416 and cfg.mixed_precision is True
        self.model,self.checkpoint_info=load_model(self.source,checkpoint,cfg,torch,np)
        self.selected_counts=freeze_scope(self.model)
        self.checkpoint_recurrence=checkpoint_recurrence
        if checkpoint_recurrence:checkpoint_update_block(self.model.update_block)
        from core.utils.utils import InputPadder
        self.Padder=InputPadder

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_parameters(self):
        return self.parameters()

    def predict(self,left,right,*,iters=8,training=True,return_sequence=False):
        """Batch-one RGB uint8/float0..255 arrays -> unpadded axial disparity tensor."""
        if iters<1:raise ValueError('Positive iteration count required')
        if np.shape(left)!=(360,640,3) or np.shape(right)!=(360,640,3):
            raise ValueError('Full640x360 RGB required; no crop or resize')
        assert all(not module.training for module in self.model.modules())
        with torch.set_grad_enabled(training):
            images=[torch.as_tensor(np.ascontiguousarray(im),device='cuda').float()[None].permute(0,3,1,2)
                    for im in (left,right)]
            padder=self.Padder(images[0].shape,divis_by=32,force_square=False)
            a,b=padder.pad(*images)
            assert tuple(a.shape)==(1,3,384,640)
            with torch.autocast('cuda',dtype=torch.float16):
                # test_mode controls output collection, not module gradients.
                prediction=self.model(a,b,iters=iters,test_mode=not return_sequence)
            if return_sequence:
                _,sequence=prediction
                assert len(sequence)==iters and all(p.is_cuda for p in sequence)
                return [padder.unpad(p.float())[0,0] for p in sequence]
            assert prediction.is_cuda
            return padder.unpad(prediction.float())[0,0]

    def save_updates(self,path):
        torch.save(dict(base_checkpoint_sha256=self.plan['checkpoint_sha256'],selected_modules=SELECTED,
            state={name:value.detach().cpu() for name,value in self.model.named_parameters() if selected_name(name)}),path)

    def load_updates(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['base_checkpoint_sha256']==self.plan['checkpoint_sha256']
        assert tuple(saved['selected_modules'])==SELECTED
        expected={name for name,_ in self.model.named_parameters() if selected_name(name)}
        assert set(saved['state'])==expected
        with torch.no_grad():
            for name,value in self.model.named_parameters():
                if name in expected:value.copy_(saved['state'][name])


def load_adapter(config_path,*,checkpoint_recurrence=False):
    return StereoAdaptModel(config_path,checkpoint_recurrence=checkpoint_recurrence)
