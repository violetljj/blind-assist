"""Export frozen five-seed M3 strong readout and synthetic mobile parity fixtures.

CPU-only engineering conversion. No training, label access, evaluation, or cut fitting.
The online candidate projects all eight observations with the current causal pose;
fixed-pose fixtures verify arithmetic, not equivalence to segment-mean evaluation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'research/active/dtr-r0/nearfield'))
from cnh_cvr_pilot import CVR, rotation
from cnh_cvr_projection import Projector, query_masks, SHAPE, SUB, EDGE, WIDTH
from cnh_temporal_readout_model import prepare_voxels
from real_contract_v5 import adapter_v5
from local_signed_bias_adapter import native_evidence

MANIFEST = ROOT / 'artifacts.local/work/cnh-pose-floor-noise-dr-dev-20261012/real_manifest.json'
CAL = ROOT / 'artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot'
OUT = ROOT / 'artifacts.local/work/mobile-fusion-integration-20261012'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MobileM3(torch.nn.Module):
    """Fixed query masks make non-divisible adaptive pooling an exact constant."""
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        masks = torch.from_numpy(query_masks())[None]
        self.register_buffer('mask', torch.nn.functional.adaptive_avg_pool3d(masks, (12, 9, 17))[:, :, None])

    def forward(self, x):
        scores = []
        for model in self.models:
            f = model.body(x)[:, None]
            mx = f.masked_fill(self.mask == 0, -1e4).flatten(3).max(-1).values
            mean = (f * self.mask).flatten(3).sum(-1) / self.mask.flatten(3).sum(-1).clamp_min(1e-8)
            scores.append(model.head(torch.cat((mx, mean, model.emb.weight[None]), -1)).squeeze(-1))
        return torch.stack(scores).mean(0)


def project(native, pitch, roll):
    p = Projector(device='cpu')
    c, s = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
    t = torch.tensor(rotation(pitch, 'x') @ np.array([[c,-s,0],[s,c,0],[0,0,1]]), dtype=torch.float32)
    xyz = torch.bmm(p.points.float()[None], t[None])[0]
    radius = torch.linalg.vector_norm(xyz, dim=1)
    ij = torch.floor((xyz[:, :2]/xyz[:, 2:3].clamp_min(1e-30)+EDGE)/(2*EDGE)*8).long()
    bins = torch.floor(radius/WIDTH).long()
    valid = (xyz[:, 2]>0)&(ij>=0).all(1)&(ij<8).all(1)&(bins>=0)&(bins<16)
    index = (ij[:,1].clamp(0,7)*8+ij[:,0].clamp(0,7))*16+bins.clamp(0,15)
    weight = valid*p.voxel_volume/(SUB**3)/p.volumes.float()[index]
    coverage = valid.reshape(-1,27).float().mean(1).reshape(SHAPE)
    values = torch.from_numpy(native).reshape(len(native),1024)
    evidence = (values[:,index]*weight).reshape(len(native),-1,27).sum(2).reshape(len(native),*SHAPE)
    count = coverage[None].expand(len(native),-1,-1,-1).sum(0)
    voxels = torch.stack((evidence.sum(0),count,evidence[-1]))[None]
    return prepare_voxels(voxels), index, weight


def main(out):
    import onnxruntime as ort
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['drop_bins'] == [0] and manifest['cuts']['local_tau'] == 1e308
    models = []
    for entry in manifest['m3_paths']:
        path = ROOT / entry['path']
        assert sha(path) == entry['sha256'], path
        m = CVR().eval()
        state = torch.load(path,map_location='cpu',weights_only=True)
        m.load_state_dict(state.get('state_dict',state)); models.append(m)
    model = MobileM3(models).eval()
    bias = np.load(CAL/'instrument_arrays.npz')['bias'].reshape(64,16)
    noise = json.loads((CAL/'instrument_parameters.json').read_text())['noise']
    calibration = dict(bias=bias.reshape(-1).tolist(), coefficients=noise['adapter_coefficients_a_b_c'],
                       varianceFloor=noise['numerical_variance_floor'], raisedTau=manifest['cuts']['raised_tau'],
                       manifestSha256=sha(MANIFEST), source='pose_floor_noise_dr_20261012')
    (out/'cnh_strong_calibration.json').write_text(json.dumps(calibration),encoding='utf8')
    np.array([*calibration['coefficients'],calibration['varianceFloor'],calibration['raisedTau'],
              *calibration['bias']],dtype='<f8').tofile(out/'cnh_strong_calibration.bin')
    generator=np.random.default_rng(20261012)
    hist=(generator.normal(0,2,(8,64,16))+np.maximum(bias,0)).astype(np.float32)
    ambient=generator.uniform(1,80,(8,64)).astype(np.float32)
    native,positive=native_evidence(adapter_v5(hist,ambient,bias,noise,None,True,False),(0,))
    x,_,_=project(native,-13.25,7.5)
    path=out/'cnh_strong_m3.onnx'
    with torch.inference_mode():
        original=torch.stack([m(x) for m in models]).mean(0)
        np.testing.assert_allclose(model(x).numpy(),original.numpy(),rtol=1e-6,atol=1e-6)
        torch.onnx.export(model,x,str(path),input_names=['voxels'],output_names=['logits'],opset_version=17,dynamo=False)
        session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
        errors=[]
        fixture_scores=[]
        for pitch,roll in [(-13.25,7.5),(0.,0.),(-35.,-12.)]:
            x,_,_=project(native,pitch,roll)
            ref=torch.stack([m(x) for m in models]).mean(0).numpy()
            got=session.run(None,{'voxels':x.numpy()})[0]
            np.testing.assert_allclose(got,ref,rtol=2e-5,atol=2e-5)
            errors.append(float(np.abs(got-ref).max()))
            tag=f'pose_{pitch}_{roll}'
            x.numpy().astype('<f4').tofile(out/f'{tag}_voxels.bin')
            raw=[]
            for length in range(1,9):
                if length<4:
                    fixture_scores.extend([np.nan,np.nan]);continue
                partial,_,_=project(native[:length],pitch,roll)
                raw.append(torch.stack([m(partial) for m in models]).mean(0).numpy()[0])
                weights=2.**np.arange(len(raw[-5:]))
                fixture_scores.extend((np.array(raw[-5:],dtype=np.float64)*weights[:,None]).sum(0)/weights.sum())
        np.array(fixture_scores,dtype='<f8').tofile(out/'fixture_scores.bin')
        hist.astype('<f4').tofile(out/'fixture_histogram.bin')
        ambient.astype('<f4').tofile(out/'fixture_ambient.bin')
        native.astype('<f4').tofile(out/'fixture_native.bin')
    report=dict(kind='synthetic_engineering_parity',manifestSha256=sha(MANIFEST),
                modelSha256=sha(path),calibrationSha256=sha(out/'cnh_strong_calibration.json'),
                calibrationBinarySha256=sha(out/'cnh_strong_calibration.bin'),
                onnxMaxAbsoluteError=max(errors),poses=[[-13.25,7.5],[0.,0.],[-35.,-12.]],
                modelBytes=path.stat().st_size,limits='Current-pose causal candidate; no new evaluation data opened.')
    (out/'cnh_strong_export_receipt.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    main(parser.parse_args().out)
