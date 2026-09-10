"""Two fixed unresolved-return proxies; no label or neural input is consumed."""
import argparse
from pathlib import Path
import numpy as np
from mz5_ensemble_readout import read,write,sha,load_npz


def constrain(ranges,valid,mode):
    if mode not in ('MERGE_CLOSE','DROP_CLOSE'):raise ValueError(mode)
    assert ranges.dtype==np.float32 and valid.dtype==np.bool_ and ranges.shape==valid.shape
    assert ranges.ndim==3 and ranges.shape[1:]==(64,2)
    close=valid.all(2)&((ranges[:,:,1].astype(float)-ranges[:,:,0].astype(float))<.600)
    rr=ranges.copy();vv=valid.copy()
    if mode=='MERGE_CLOSE':
        rr[:,:,0][close]=((ranges[:,:,0].astype(float)+ranges[:,:,1].astype(float))*.5)[close].astype(np.float32)
        rr[:,:,1][close]=0;vv[:,:,1][close]=False
    else:rr[close]=0;vv[close]=False
    np.testing.assert_array_equal(rr[~close],ranges[~close]);np.testing.assert_array_equal(vv[~close],valid[~close])
    assert (rr[~vv]==0).all()
    assert not (vv.all(2)&((rr[:,:,1].astype(float)-rr[:,:,0].astype(float))<.600)).any()
    return dict(ranges=rr,valid=vv),close


def prepare(root,output):
    root=root.resolve();output=output.resolve();assert output.is_relative_to((root/'artifacts.local').resolve()) and not output.exists()
    prior=root/'artifacts.local/work/mz36-new-source-20260910';inputs={}
    def bind(p,h=None):
        digest=sha(p);assert h is None or digest==h,str(p);inputs[str(p)]=digest;return p
    receipt=read(bind(prior/'inference-v1/receipt.json'));assert receipt['status']=='PASS'
    packet=load_npz(bind(prior/'inference-v1/predictions.npz',receipt['outputs']['predictions.npz']))
    manifest=read(bind(prior/'admission-v1/predictor-manifest.json'))
    assert len(manifest['frames'])==380
    np.testing.assert_array_equal(packet['frame_ids'],[r['frame_id'] for r in manifest['frames']])
    bind(Path(__file__).with_name('MZ40_L8CX_CONSTRAINED_PROTOCOL_20260910.md'));bind(Path(__file__))
    audit39=read(bind(root/'artifacts.local/work/mz39-l8cx-readout-audit-20260910/run-v1/result.json'))
    assert audit39['status']=='PASS' and audit39['sub600mm_zones']==4234
    # The predictor manifest contains no native-depth path or evaluator fields.
    manifest=dict(camera=manifest['camera'],frames=[{k:r[k] for k in ('frame_id','rgb_path')} for r in manifest['frames']])
    output.mkdir(parents=True);write(output/'manifest.json',manifest)
    write(output/'parity-manifest.json',dict(camera=manifest['camera'],frames=manifest['frames'][:16]))
    ids=packet['frame_ids'];ranges=packet['ranges'];valid=packet['valid'];stats={}
    np.savez_compressed(output/'PARITY.npz',frame_ids=ids[:16],ranges=ranges[:16],valid=valid[:16])
    for mode,expected in [('MERGE_CLOSE',[16592,7069,659]),('DROP_CLOSE',[20826,2835,659])]:
        changed,close=constrain(ranges,valid,mode)
        assert close.sum()==4234 and close.any(1).sum()==346
        count=changed['valid'].sum(2);hist=[int((count==n).sum()) for n in range(3)];assert hist==expected
        np.savez_compressed(output/(mode+'.npz'),frame_ids=ids,**changed)
        stats[mode]=dict(zone_return_counts=hist,valid_slots=int(changed['valid'].sum()),
            all_tof_missing_frames=int((~changed['valid'].any((1,2))).sum()),changed_zones=int(close.sum()),changed_frames=int(close.any(1).sum()))
    np.savez_compressed(output/'ambiguity.npz',frame_ids=ids,unresolved=close)
    write(output/'result.json',dict(status='PASS',frames=380,arms=stats,limits='Limited sensitivity proxies, not physical sensor emulation'))
    for p,h in inputs.items():assert sha(p)==h,p
    write(output/'receipt.json',dict(status='PASS',inputs=inputs,backend='CPU saved-scalar packet mutation; TASK_NOT_GPU_SUITABLE',
        training_steps=0,model_inference_frames=0,outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(stats)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();prepare(a.root,a.output)
