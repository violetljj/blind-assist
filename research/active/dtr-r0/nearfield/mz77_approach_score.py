"""Post-seal native evaluator, temporal scalar score and diagnostic timeline."""
import argparse
import gc
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from mz5_ensemble_readout import read,write,sha
from multizone64_observation import native_events
from verify_temporal_structure import rays,intersect
from mz77_approach_evaluate import PROFILES,METHODS,TASK
from mz77_sequence_metrics import analyze


def score(root,capture,sensor,inference,out):
    out=out.resolve();assert out.is_relative_to((root/'artifacts.local/work'/TASK).resolve())
    out.mkdir(parents=True,exist_ok=False);inputs={};started=time.perf_counter()
    def bind(p,digest=None):
        p=Path(p).resolve();value=sha(p);assert digest is None or digest==value,str(p)
        inputs[str(p)]=value;return p
    for name in ('mz77_approach_score.py','mz77_sequence_metrics.py','multizone64_observation.py','verify_temporal_structure.py'):
        shutil.copyfile(Path(__file__).with_name(name),out/name);bind(out/name)
    run=read(bind(inference/'receipt.json'));assert run['status']=='PASS' and run['frames']==160
    sr=read(bind(sensor/'receipt.json'));assert sr['status']=='PASS'
    for path,digest in run['inputs'].items():bind(path,digest)
    with np.load(bind(sensor/'observations.npz',sr['outputs']['observations.npz'])) as z: obs=dict(z)
    with np.load(bind(inference/'predictions.npz',run['outputs']['predictions.npz'])) as z: pred=dict(z)
    for key in ('clip','index','time_s'):np.testing.assert_array_equal(obs[key],pred[key])
    spec=read(bind(capture/'evaluator/spec.json'));assert len(spec['cases'])==160
    # Labels become available only after sealed predictor output above.
    full_counts=[];target_counts=[];known=[];invalid=[];front=[]
    torch.set_num_threads(1)
    with torch.inference_mode():
        for begin in range(0,160,16):
            native=[];target=[]
            for i in range(begin,begin+16):
                case=spec['cases'][i]
                assert case['clip_id']==obs['clip'][i] and case['frame_in_clip']==obs['index'][i]
                d=np.load(bind(capture/f'evaluator/native/{i:04d}.npy'),allow_pickle=False)
                native.append(d); invalid.append(int((~np.isfinite(d)|(d<=0)).sum()))
                origin=np.array([case['camera'][k] for k in ('x','y','z')])
                objects=case['objects'];assert len(objects)<=1
                if objects:
                    obj=objects[0];expected=intersect(origin,rays(case['camera']),obj)
                    visible=np.isfinite(expected)&np.isfinite(d)&(d>0)&(np.abs(d-expected)<.03)
                    assert visible.any(),f'No intended target native support frame{i}'
                    target.append(np.where(visible,d,0))
                    front.append(obj['center_m'][0]-obj['size_m'][0]/2-origin[0])
                else:target.append(np.zeros_like(d));front.append(np.nan)
            full=native_events(torch.from_numpy(np.stack(native)).cuda(),crop=False)
            own=native_events(torch.from_numpy(np.stack(target)).cuda(),crop=False)
            full_counts.append(full['counts'].cpu().numpy());target_counts.append(own['counts'].cpu().numpy())
            known.append(np.repeat(full['observation_valid'].cpu().numpy()[:,None],4,axis=1))
    evaluator=dict(truth=np.concatenate(full_counts)>=3,known=np.concatenate(known),
        target_truth=np.concatenate(target_counts)>=3,front_distance_m=np.array(front),
        native_counts=np.concatenate(full_counts),target_counts=np.concatenate(target_counts),invalid_native_pixels=np.array(invalid))
    np.savez_compressed(out/'evaluator.npz',**evaluator)
    arrays={**obs,**evaluator}
    arrays['packets']={p:{k:obs[p+'/'+k] for k in ('ranges','valid')} for p in PROFILES}
    predictions={p:{m:pred[p+'/'+m] for m in METHODS} for p in PROFILES}
    scalar_started=time.perf_counter();result=analyze(arrays,predictions)
    # Recount all four native query bits without the metric module's masks.
    audited=0
    for clip in dict.fromkeys(obs['clip'].tolist()):
        ids=np.flatnonzero(obs['clip']==clip)
        for profile in PROFILES:
            for method in METHODS:
                for q,name in enumerate(('BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR')):
                    counts=dict(TP=0,FP=0,FN=0,TN=0);unknown=0
                    for i in ids:
                        if not evaluator['known'][i,q]: unknown+=1;continue
                        y=bool(evaluator['truth'][i,q]);a=bool(pred[profile+'/'+method][i,q]>=0)
                        counts['TP' if y and a else 'FN' if y else 'FP' if a else 'TN']+=1;audited+=1
                    metric=result['clips'][clip]['conditions'][profile]['methods'][method]['full_scene'][name]
                    assert counts==metric['confusion'] and unknown==metric['unknown_samples']
    scalar_seconds=time.perf_counter()-scalar_started
    write(out/'result.json',result)
    # Compact visual evidence: exact sample flags, no smoothing/interpolation.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    clips=list(dict.fromkeys(obs['clip'].tolist()))
    fig,axes=plt.subplots(2,4,figsize=(16,6.5),sharex=True,layout='constrained')
    cmap=ListedColormap(['#e5e7eb','#d95f02'])
    for col,clip in enumerate(clips):
        idx=np.flatnonzero(obs['clip']==clip)
        for row,profile in enumerate(PROFILES):
            truth=evaluator['truth'][idx];old=pred[profile+'/OLD_NEG'][idx]>=0;new=pred[profile+'/DIVERSE'][idx]>=0
            flags=np.stack([x[:,sl].any(1) for sl in (slice(0,2),slice(2,4)) for x in (truth,old,new)])
            ax=axes[row,col];ax.imshow(flags,aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=1,extent=(-.05,3.95,5.5,-.5))
            ax.set_yticks(range(6),['Body truth','Body OLD','Body D','Head truth','Head OLD','Head D'])
            ax.set_title(clip+' / '+profile);ax.set_xlabel('Nominal time (s)')
            if profile=='CENTER_GAP':
                for left,right in ((1.95,2.25),(2.95,3.25)):
                    ax.axvspan(left,right,facecolor='none',edgecolor='#2563eb',linewidth=1.5)
    fig.suptitle('MZ77 fixed-model approach: orange = positive support / alert; blue = partial ToF outage\nPosed 10 Hz simulation, not measured real-time latency',fontsize=12)
    fig.savefig(out/'timeline.png',dpi=140);plt.close(fig)
    for path,digest in inputs.items():assert sha(path)==digest,path
    write(out/'receipt.json',dict(status='PASS',inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        frames=160,profiles=PROFILES,methods=METHODS,training_steps=0,new_cutoffs=0,
        evaluator_backend='CUDA native pixel geometry',scoring_backend='CPU TASK_NOT_GPU_SUITABLE',
        scalar_seconds=scalar_seconds,seconds=time.perf_counter()-started,
        native_query_bits_independently_recounted=audited,
        target_interpretation='Alert coincides with target-visible query support; no predicted winner target-localization claim',
        source_scope='Posed simulated samples, nominal timing, no actual contact or safety evidence'))
    gc.collect();print('SCORE PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('E:/linnan/linnan'))
    for key in ('capture','sensor','inference','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();score(a.root.resolve(),a.capture.resolve(),a.sensor.resolve(),a.inference.resolve(),a.output)
