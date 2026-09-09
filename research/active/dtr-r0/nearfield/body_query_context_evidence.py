"""Two-output research interface: retained alert plus independent range evidence.

This is deliberately not a replacement count-bottleneck model or Android release.
Raw RGB and fixed calibration are the only model inputs. No absence means safe.
"""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from body_query_context_decoder import CountDecoder, BASE_SHA
from body_query_model import BodyQueryModel, near_from_counts
from body_query_range import range_from_counts
from body_query_data import read, sha
from whisker_model import IMAGE_SIZE, SUPPORT_SIZE
import numpy as np


class ContextEvidence(nn.Module):
    def __init__(self, baseline, decoder_run, pretrained, arm='JOINT'):
        super().__init__()
        baseline,decoder_run=Path(baseline),Path(decoder_run)
        if arm not in ('LOCAL','JOINT'):
            raise ValueError('Only frozen visual decoder arms are supported')
        fits=read(decoder_run/'fits.json');receipt=read(decoder_run/'receipt.json')
        if receipt['status']!='PASS' or sha(baseline/'NEW-step2000.pt')!=BASE_SHA:
            raise ValueError('Original admitted10k baseline required')
        if sha(decoder_run/f'{arm}.pt')!=fits[arm]['sha256']:
            raise ValueError('Decoder checkpoint identity mismatch')
        if sha(decoder_run/'selection.json')!=receipt['selection_sha256']:
            raise ValueError('Frozen DEV selection mismatch')
        self.base=BodyQueryModel(pretrained,'B')
        self.base.load_state_dict(torch.load(baseline/'NEW-step2000.pt',map_location='cpu',weights_only=True))
        xyz=(self.base.query_xyz*self.base.query_valid[:,:,None]).sum(1)/self.base.query_valid.sum(1)[:,None].clamp_min(1)
        self.decoder=CountDecoder(arm,xyz)
        self.decoder.load_state_dict(torch.load(decoder_run/f'{arm}.pt',map_location='cpu',weights_only=True))
        norm=np.load(decoder_run/'normalization.npz',allow_pickle=False)
        self.register_buffer('feature_mean',torch.from_numpy(norm['mean']))
        self.register_buffer('feature_std',torch.from_numpy(norm['std']))
        self.register_buffer('alert_thresholds',torch.tensor(read(decoder_run/'selection.json')['BASE']['thresholds']))
        self.eval()
        for parameter in self.parameters():parameter.requires_grad_(False)

    def forward(self,rgb):
        if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(3,*IMAGE_SIZE):
            raise ValueError('Expected RGB Bx3x144x256 in [0,1]')
        m=self.base
        deep,shallow=m.extract((rgb-m.image_mean)/m.image_std)
        deep=F.interpolate(m.deep_projection(deep),size=SUPPORT_SIZE,mode='bilinear',align_corners=False)
        detail=m.detail(shallow);support=m.support(deep+detail)
        sampled=torch.matmul(torch.cat((deep,detail),1).flatten(2),m.query_projection.T)
        sampled=sampled.transpose(1,2).reshape(len(rgb),12,27,64)
        xyz=m.query_xyz[None].expand(len(rgb),-1,-1,-1);mask=m.query_valid[None,:,:,None]
        feature=m.query_point(torch.cat((sampled,xyz),-1))
        feature=(feature*mask).sum(2)/mask.sum(2).clamp_min(1)
        legacy=m.query_readout(feature);alert_logits=near_from_counts(legacy)
        raw=(sampled*mask).sum(2)/mask.sum(2).clamp_min(1)
        geometry=self.decoder((raw-self.feature_mean)/self.feature_std)
        ranges=range_from_counts(geometry).sigmoid()
        alert=alert_logits.sigmoid()>=self.alert_thresholds
        # 0 UNKNOWN/no asserted range; 1 NEAR; 2 FAR; 3 BOTH. Never CLEAR.
        flags=ranges>=.5
        code=flags[:,:,0].long()+2*flags[:,:,1].long()
        return dict(alert_logits=alert_logits,alerts=alert,support_logits=support,
                    legacy_count_logits=legacy,geometry_count_logits=geometry,
                    geometry_alert_logits=near_from_counts(geometry),range_probabilities=ranges,
                    range_code=code,alert_range_disagreement=alert!=flags.any(-1))
