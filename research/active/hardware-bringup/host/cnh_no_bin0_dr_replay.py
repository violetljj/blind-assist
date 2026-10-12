"""Stationary engineering replay of sealed design candidates, no calibration.

Mask native bins before every inherited extractor. A one-seed sensitivity arm
may repeat its paths in ensemble slots; repeated weights add no independent seed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import cnh_sim2real_frozen as F

ROOT=F.ROOT

class DesignReplay(F.FrozenReplay):
    def __init__(self, manifest_path, **kwargs):
        self.design_path=Path(manifest_path)
        self.design=json.loads(self.design_path.read_text(encoding='utf8'))
        self.drop_bins=tuple(self.design['drop_bins'])
        if self.drop_bins not in ((0,),(0,1)):
            raise ValueError('Only prewritten bin exclusions accepted')
        super().__init__(**kwargs)
        torch=self.torch
        self.m3.clear();self.ordinary.clear();self.s_models.clear()
        self.design_hashes={}
        def verify(value):
            path=ROOT/value['path'] if isinstance(value,dict) else ROOT/value
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            if isinstance(value,dict) and value['sha256']!=actual:
                raise ValueError('Sealed model hash mismatch: '+str(path))
            self.design_hashes[str(path)]=actual
            return path
        for value in self.design['m3_paths']:
            path=verify(value);model=self.CVR().to(self.device).eval()
            state=torch.load(path,map_location=self.device,weights_only=True)
            model.load_state_dict(state.get('state_dict',state));self.m3.append(model)
        for value in self.design['ordinary_paths']:
            state=torch.load(verify(value),map_location=self.device,weights_only=True)
            model=self.M.BoundaryTokenReadout().to(self.device).eval()
            model.load_state_dict(state['state_dict']);self.ordinary.append(model)
        for pair in self.design['s_paths']:
            models=[]
            for value in pair:
                with verify(value).open('rb') as stream:models.append(pickle.load(stream))
            self.s_models.append(models)
        if len(self.ordinary)!=3 or len(self.s_models)!=3 or not self.m3:
            raise ValueError('Three interface slots required; sensitivity duplicates explicit')
        self.cuts=self.design['cuts']
        self.single=np.asarray(self.cuts['single'],float)
        self.s_tau=float(self.cuts['s_tau'])

    def infer(self,z,seq,ms,check=lambda:None):
        # Copy rather than modifying a caller's comparison observation.
        masked=np.array(z,copy=True);masked[...,self.drop_bins]=0
        old=(F.M3_TAU,F.RAISED_TAU,F.LOCAL_TAU)
        try:
            F.M3_TAU=float(self.cuts['m3_tau'])
            F.RAISED_TAU=float(self.cuts['raised_tau'])
            F.LOCAL_TAU=float(self.cuts['local_tau'])
            return super().infer(masked,seq,ms,check)
        finally:
            F.M3_TAU,F.RAISED_TAU,F.LOCAL_TAU=old
