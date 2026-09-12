"""One frozen consumed replay and one fresh stereo witness comparison."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
import mz101_spatial as base
import run_mz101_spatial as previous
import mz102_stereo_witness as witness

ROOT=previous.ROOT
TASK=ROOT/'artifacts.local/work/mz102-stereo-support-20260912'
OLD=ROOT/'artifacts.local/work/mz101-stereo-tof-spatial-20260912'
NAMES=('tof','stereo','union','surface','surface_interval')


def gates(pred,gt,obs,spec,stats,name):
    tof=pred['tof'];old=pred['union'];candidate=pred[name]
    increment=old&~tof&gt;thin=np.array([f['family'] in ('thin_left','thin_right') for f in spec['frames']])[:,None]
    extra=old&~tof&~gt;extra_after=candidate&~tof&~gt
    retention=lambda mask:float((candidate&mask).sum()/mask.sum()) if mask.any() else None
    paired=list(zip(stats['union']['event_details'],stats[name]['event_details']))
    losses=[x for x,y in paired if x['detected'] and not y['detected']]
    delays=[y['first_correct_delay_s']-x['first_correct_delay_s'] for x,y in paired if x['detected'] and y['detected']]
    overall=retention(increment);thin_ret=retention(increment&thin)
    g=dict(tof_TP_preserved=not bool((tof&gt&~candidate).any()),
        incremental_TP_retention_ge90=overall is not None and overall>=.9,
        incremental_thin_TP_retention_ge90=thin_ret is not None and thin_ret>=.9,
        extra_FP_at_least_halved=bool(extra.any() and extra_after.sum()<=extra.sum()*.5),
        no_new_false_sessions=stats[name]['false_sessions']<=stats['tof']['false_sessions'],
        no_lost_union_events=not losses,delay_increase_le025=not delays or max(delays)<=.25,
        F1_not_lower=stats[name]['F1']>=stats['union']['F1'])
    return dict(pass_all=all(g.values()),gates=g,incremental_TP=int(increment.sum()),
        incremental_TP_retained=int((candidate&increment).sum()),incremental_TP_retention=overall,
        incremental_thin_TP=int((increment&thin).sum()),incremental_thin_TP_retained=int((candidate&increment&thin).sum()),
        incremental_thin_TP_retention=thin_ret,extra_FP_before=int(extra.sum()),extra_FP_after=int(extra_after.sum()),
        max_event_delay_increase_s=max(delays) if delays else None)


def run(mode):
    out=TASK/(mode+'-v1');assert not out.exists();out.mkdir(parents=True)
    capture=OLD/'capture-v1' if mode=='development' else TASK/'capture-v1'
    spec_path=capture/'spec.json';spec=json.loads(spec_path.read_text())
    assert len(spec['frames'])==288 and spec['rig']==base.RIG
    receipt=json.loads((capture/'receipt.json').read_text());assert receipt['status']=='PASS'
    assert previous.sha(spec_path)==receipt['spec_sha256']
    for name,digest in receipt['hashes'].items():assert previous.sha(capture/name)==digest,name
    for name in ('mz102_stereo_witness.py','run_mz102_stereo_witness.py','mz102_spatial_spec.py','MZ102_STEREO_WITNESS_PROTOCOL_20260912.md'):
        shutil.copyfile(Path(__file__).parent/name,out/name)
    obs=previous.observation_contract(spec);previous.write(out/'observations.json',obs)
    counts={n:[] for n in NAMES};support_trace=[];input_hashes={};match_times=[];witness_times=[]
    (out/'depth').mkdir();(out/'accepted').mkdir();started=time.perf_counter()
    for i,o in enumerate(obs):
        folder=capture/'frame'/o['id']
        for name in ('left.png','right.png','tof-range.npy','tof-valid.npy'):
            input_hashes[str((folder/name).relative_to(capture))]=previous.sha(folder/name)
        if mode=='development':
            path=OLD/'comparison-v1/stereo-depth'/(o['id']+'.npy')
            depth=np.load(path);input_hashes[str(path.relative_to(ROOT/'artifacts.local'))]=previous.sha(path)
        else:
            depth,timing=base.stereo_depth(cv2.imread(str(folder/'left.png')),cv2.imread(str(folder/'right.png')))
            match_times.append(timing['seconds'])
        np.save(out/'depth'/(o['id']+'.npy'),depth)
        tof=base.readout(base.tof_points(np.load(folder/'tof-range.npy'),np.load(folder/'tof-valid.npy')),o['pose'])[0]
        stereo=base.readout(base.depth_points(depth),o['pose'])[0]
        t=time.perf_counter();surface,diag,mask=witness.qualify(depth,o['pose'])
        stable,stable_diag,_=witness.qualify(depth,o['pose'],perturbation=True)
        witness_times.append(time.perf_counter()-t)
        np.savez_compressed(out/'accepted'/(o['id']+'.npz'),mask=mask)
        for n,value in dict(tof=tof,stereo=stereo,union=tof+stereo,surface=tof+surface,surface_interval=tof+stable).items():counts[n].append(value)
        support_trace.append(dict(id=o['id'],surface=diag,perturbation=stable_diag))
    counts={n:np.array(c) for n,c in counts.items()};ids=[o['episode'] for o in obs]
    pred={n:base.hysteresis(c,ids) for n,c in counts.items()}
    assert not (pred['tof']&~pred['surface']).any()
    assert not (pred['surface']&~pred['union']).any()
    assert not (pred['surface_interval']&~pred['surface']).any()
    parity=None
    if mode=='development':
        old=np.load(OLD/'comparison-v1/predictions.npz')
        for n in ('tof','stereo','union'):np.testing.assert_array_equal(pred[n],old['common_'+n])
        parity=True
    np.savez_compressed(out/'predictions.npz',**pred,**{n+'_support':c for n,c in counts.items()})
    previous.write(out/'support-trace.json',support_trace)
    previous.write(out/'prediction-seal.json',dict(status='PREDICTIONS_SEALED_BEFORE_GT_SCORING',mode=mode,
        prediction_sha256=previous.sha(out/'predictions.npz'),observation_sha256=previous.sha(out/'observations.json'),
        spec_sha256=previous.sha(spec_path),input_hashes=input_hashes,
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    # No prediction mutation below this point. Actor geometry is evaluator-only.
    manifest=json.loads((capture/'manifest.json').read_text());assert len(manifest['frames'])==288
    for f,row in zip(spec['frames'],manifest['frames']):
        assert f['id']==row['id'] and len(f['objects'])==len(row['native_bounds'])
        for obj,bounds in zip(f['objects'],row['native_bounds']):
            np.testing.assert_allclose(obj['center_m'],bounds['center_m'],atol=1e-5)
            np.testing.assert_allclose(np.array(obj['size_m'])/2,bounds['extent_m'],atol=1e-5)
        for name in ('floor','background'):
            obj=receipt[name];a=np.array(obj['center_m'])-np.array(obj['size_m'])/2-f['body_origin_m']
            b=np.array(obj['center_m'])+np.array(obj['size_m'])/2-f['body_origin_m']
            assert not any(((b>=low)&(a<=high)).all() for low,high in base.BOXES)
    gt=previous.ground_truth(spec);np.save(out/'truth.npy',gt)
    stats={n:previous.metrics(p,gt,obs) for n,p in pred.items()}
    slices={family:{n:previous.metrics(p,gt,obs,[f['family']==family for f in spec['frames']]) for n,p in pred.items()}
        for family in dict.fromkeys(f['family'] for f in spec['frames'])}
    results=dict(mode=mode,frames=288,episodes=24,primary='surface',baseline_parity=parity,
        results=stats,slices=slices,primary_gates=gates(pred,gt,obs,spec,stats,'surface'),
        diagnostic_gates=gates(pred,gt,obs,spec,stats,'surface_interval'),
        direct_no_support_queries={n:int((c==0).sum()) for n,c in counts.items()},
        backend=dict(stereo='OpenCV4.10 CPU GPU_BACKEND_UNAVAILABLE',witness='SciPy sparse graph CPU TASK_NOT_GPU_SUITABLE',
            stereo_seconds=sum(match_times),two_witness_contrasts_seconds=sum(witness_times)))
    video=cv2.VideoWriter(str(out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),4.,(1280,460))
    assert video.isOpened();tiles=[]
    try:
        for i,o in enumerate(obs):
            left=cv2.imread(str(capture/'frame'/o['id']/'left.png'))
            depth=np.load(out/'depth'/(o['id']+'.npy'));mask=np.load(out/'accepted'/(o['id']+'.npz'))['mask']
            canvas=np.zeros((460,1280,3),np.uint8);canvas[:360,:640]=left
            depth_color=cv2.applyColorMap(np.uint8(np.clip(np.nan_to_num(depth)/4*255,0,255)),cv2.COLORMAP_TURBO)
            depth_color[~np.isfinite(depth)]=0;canvas[:360,640:]=depth_color
            canvas[:360,:640][mask[0]]=(255,230,0);canvas[:360,:640][mask[1]]=(255,0,230)
            cv2.putText(canvas,'Qualified BODY cyan / HEAD pink',(10,25),0,.60,(255,255,255),1)
            cv2.putText(canvas,'Original stereo depth (black=UNKNOWN)',(650,25),0,.55,(255,255,255),1)
            cv2.putText(canvas,o['id']+f" t={o['time_s']:.2f}s GT {gt[i].astype(int).tolist()}",(10,390),0,.55,(255,255,255),1)
            text='  '.join(f'{n}: {pred[n][i].astype(int).tolist()}' for n in ('tof','union','surface','surface_interval'))
            cv2.putText(canvas,text,(10,430),0,.57,(0,230,230),1);video.write(canvas)
            if i%12==7:
                cv2.imwrite(str(out/(o['episode']+'.jpg')),canvas);tiles.append(cv2.resize(canvas,(640,230)))
    finally:video.release()
    for app,indices in [('textured',range(0,24,2)),('flat',range(1,24,2))]:
        t=[tiles[i] for i in indices];cv2.imwrite(str(out/('contact-sheet-'+app+'.jpg')),np.vstack([np.hstack(t[j:j+2]) for j in range(0,12,2)]))
    results['elapsed_s']=time.perf_counter()-started
    previous.write(out/'summary.json',results)
    previous.write(out/'receipt.json',dict(status='PASS',hashes={p.name:previous.sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(mode=mode,primary=results['primary_gates'],diagnostic=results['diagnostic_gates'],
        results={n:{k:v for k,v in s.items() if k!='event_details'} for n,s in stats.items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('development','fresh'),required=True)
    run(p.parse_args().mode)
