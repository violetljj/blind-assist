"""Independent direct-node and scalar-peak audit of corridor evidence payload."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
OUTPUT = ROOT/'artifacts.local/work/cnh-graded-corridor-dev-20261010/features'
BIAS = ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy'


def audit(output=OUTPUT):
    began=time.monotonic()
    checks=0
    discrepancies=[]
    edge=np.float32(np.tan(np.pi/8)); width=np.float32(8*.0375348)
    # Build the individual node hypotheses directly, independently of einsum.
    nodes=[]
    for y in range(8):
        for x in range(8):
            for b in range(16):
                angular=[]
                for ay in (1/6,.5,5/6):
                    for ax in (1/6,.5,5/6):
                        direction=np.array([-edge+np.float32(x+ax)*(2*edge/8),
                                            -edge+np.float32(y+ay)*(2*edge/8),1],np.float32)
                        direction/=np.linalg.norm(direction)
                        angular.append(direction*np.float32((b+.5)*width))
                nodes.append(angular)
    native=np.asarray(nodes,np.float32)
    bias=np.load(BIAS).astype(np.float32)
    for split in ('cal','validation'):
        directory=SOURCE/'data'/split
        with np.load(directory/'geometry.npz') as a:
            sensor=a['sensor']; query=a['public_query']
        with np.load(directory/'physics.npz') as a:
            ambient=a['ambient'].astype(np.float32)
        with np.load(output/f'{split}_features.npz') as saved:
            stored_features=saved['features']; stored_valid=saved['valid']
        geometry=np.load(output/f'{split}_public_geometry.npz')
        hist=np.load(directory/'hist.npy',mmap_mode='r')
        for i,k in ((0,0),(127,1),(383,3)):
            raw=np.array(hist[i,k],copy=True).astype(np.float32)
            z=((raw-bias)/np.sqrt(np.maximum(16*ambient[...,None]+np.maximum(bias,0),1e-9))).astype(np.float16).astype(np.float32).reshape(16,1024)
            logs=np.sign(z)*np.log1p(np.abs(z))
            for fj,f in enumerate(range(3,16)):
                first=max(0,f-7); length=f-first+1
                current_to_query=query[f]@np.linalg.inv(sensor[f])
                positions=[]
                for t in range(first,f+1):
                    transform=(current_to_query@sensor[t]).astype(np.float32)
                    positions.append(native@transform[:3,:3].T+transform[:3,3])
                positions=np.stack(positions)
                for q,(ylo,yhi) in enumerate(((-.2,.42),(.42,.9))):
                    x=positions[...,0]; y=positions[...,1]; depth=positions[...,2]
                    near=(y>=ylo)&(y<=yhi)&(depth>=.30)&(depth<=3)
                    inner=near&(abs(x)<=.30); expanded=near&(abs(x)<=.40)
                    peak_maps=[]
                    for r,node_mask in enumerate((inner,expanded,expanded&~inner)):
                        w=node_mask.mean(-1,dtype=np.float32)
                        np.testing.assert_array_equal(node_mask,geometry['node_membership'][fj,q,r,8-length:])
                        checks+=node_mask.size
                        conditional_depth=np.divide((node_mask*depth).sum(-1),node_mask.sum(-1),
                                                    out=np.zeros_like(w),where=node_mask.sum(-1)>0)
                        low=np.where(node_mask,depth,np.inf).min(-1)
                        high=np.where(node_mask,depth,-np.inf).max(-1)
                        signed=logs[first:f+1]*w
                        positive_history=np.maximum(signed,0)
                        mode_peaks=[]
                        for mode,data in enumerate((signed[-1],signed.mean(0))):
                            positive=np.maximum(data,0)
                            peaks=np.zeros(1024,np.float32)
                            for zone in range(64):
                                for b in range(16):
                                    ix=zone*16+b
                                    left=-np.inf if b==0 else positive[ix-1]
                                    right=-np.inf if b==15 else positive[ix+1]
                                    if positive[ix]>0 and positive[ix]>left and positive[ix]>=right:
                                        peaks[ix]=positive[ix]
                            mode_peaks.append(peaks)
                            best=int(np.argmax(peaks)); values=sorted(peaks.tolist(),reverse=True)[:3]
                            expected=[float(peaks[best]),sum(values),float(data.sum()),float(positive.sum()),
                                      float(positive.sum()**2/max(float((positive*positive).sum()),1e-20)),
                                      float((positive>0).sum()),float((peaks>0).sum()),
                                      float(w[-1].sum() if mode==0 else w.sum()/length)]
                            if peaks[best]>0:
                                if mode==0:
                                    d=conditional_depth[-1,best]; lo=low[-1,best]; hi=high[-1,best]
                                    native_z=z[f,best]; member=w[-1,best]
                                else:
                                    contribution=positive_history[:,best]
                                    d=(contribution*conditional_depth[:,best]).sum()/contribution.sum()
                                    allow=contribution>0
                                    lo=low[allow,best].min(); hi=high[allow,best].max()
                                    native_z=np.where(w[:,best]>0,z[first:f+1,best],0).max()
                                    member=w[:,best].sum()/length
                                expected += [float(d),float((best%16+.5)*width),float(member),float(native_z),float(lo),float(hi)]
                            else:
                                expected += [np.nan]*6
                            offset=(r*2+mode)*14
                            actual=stored_features[i,k,fj,q,offset:offset+14]
                            np.testing.assert_array_equal(np.isfinite(actual),stored_valid[i,k,fj,q,offset:offset+14])
                            valid=np.isfinite(expected)
                            # Independent scalar accumulation and node-matmul FP32
                            # geometry round differently by <= low-micro units.
                            np.testing.assert_allclose(actual[valid],np.asarray(expected)[valid],atol=3e-5,rtol=3e-6)
                            for metric in (5,6):
                                assert actual[metric]==expected[metric]
                            checks+=28
                            discrepancies.append(float(np.max(np.abs(actual[valid]-np.asarray(expected)[valid]))))
                        peak_maps.append(mode_peaks[0])
                    anchor=int(np.argmax(peak_maps[1]))%16
                    nearby=abs(np.tile(np.arange(16),64)-anchor)<=1
                    a=float(peak_maps[0][nearby].max()); b=float(peak_maps[2][nearby].max())
                    comp=np.array([anchor,a,b,a/(a+b)])
                    np.testing.assert_allclose(stored_features[i,k,fj,q,84:],comp,atol=3e-6,rtol=3e-6)
                    checks+=4
        del hist
    result=dict(status='PASS',checks=checks,representative_observations=6,query_frames=156,
                max_absolute_difference=max(discrepancies),wall_seconds=time.monotonic()-began,
                audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                independent='Direct native ray loops, matrix multiplication, hard node masks, scalar radial peaks and explicit contribution-depth reference',
                scope='Representative cal/validation geometry/features and competition, no labels/boxes read; extraction separately tested exact source histories and past prefixes',
                command=[sys.executable,*sys.argv])
    (output/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':
    audit()
