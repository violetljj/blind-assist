"""Frozen positive-branch restoration with one oldDEV calibration; no model call."""
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
from mz5_ensemble_readout import read, write, sha, load_npz
from mz36_evaluate import metrics

NORMAL=('DEV','clean','stress','relation10000','distance5000')


def calibrate_positive(confidence, eligible, truth):
    cut=[];details=[]
    for q in range(4):
        values=confidence[eligible[:,q],q].astype(float)
        thresholds=np.unique(np.r_[.5,values[values>=.5],np.nextafter(1.,np.inf)])
        best=None
        for threshold in thresholds:
            add=eligible[:,q]&(confidence[:,q].astype(float)>=threshold)
            if (add&~truth[:,q]).any():continue
            count=int((add&truth[:,q]).sum());key=(count,float(threshold))
            if best is None or key>best[0]:best=(key,float(threshold))
        assert best is not None
        threshold=best[1] if best[0][0]>0 else float(np.nextafter(1.,np.inf))
        cut.append(threshold);details.append(dict(threshold=threshold,candidates=len(thresholds),
            eligible_tp=int((eligible[:,q]&truth[:,q]).sum()),eligible_fp=int((eligible[:,q]&~truth[:,q]).sum()),
            recovered_tp=best[0][0],added_fp=0,disabled=threshold>1))
    return np.array(cut),details


def compose(data, cutoff=None):
    rgb,tof,base,old,support,known=(data[k] for k in ('rgb','tof','baseline','oldscore','support','known'))
    confidence=(np.float32(1)-data['negative_confidence'].astype(np.float32)).astype(np.float32)
    eligible=(base<0)&(old<0)&support&((rgb>=0)!=(tof>=0))&known
    positive=np.where(rgb>=0,rgb,tof)
    data.update(eligible=eligible,positive_confidence=confidence,positive_branch=positive)
    if cutoff is not None:
        added=eligible&(confidence.astype(float)>=cutoff)
        candidate=np.where(added,positive,old)
        assert (positive[added]>=0).all() and not ((old>=0)&(candidate<0)&known).any()
        data.update(added=added,candidate=candidate)
    return data


def main(root, run):
    root=root.resolve();artifact=(root/'artifacts.local').resolve()
    if run.exists() or not run.resolve().is_relative_to(artifact):raise ValueError('Fresh artifacts.local run required')
    run.mkdir(parents=True);started=time.perf_counter();work=root/'artifacts.local/work';inputs={}
    def bind(p,h=None):
        digest=sha(p);assert h is None or h==digest,str(p);inputs[str(p)]=digest;return p
    try:
        protocol=Path(__file__).with_name('MZ37_POSITIVE_RESTORATION_PROTOCOL_20260910.md');bind(protocol)
        prior=work/'mz35-responsibility-convergence-20260910/run-v1'
        receipt=read(bind(prior/'receipt.json'));assert receipt['status']=='PASS'
        for name in ('predictions.npz','selector.pt','cutoff.npy'):
            bind(prior/name,receipt['outputs'][name])
        assert read(bind(prior/'audit.json'))['status']=='PASS'
        source=load_npz(prior/'predictions.npz');datasets={}
        for name in NORMAL+tuple(n+'_wrong' for n in NORMAL if n!='DEV'):
            fields=dict(oldscore='candidate',truth='truth',baseline='baseline',support='support',rgb='rgb',tof='tof',negative_confidence='confidence',frame_ids='global_ids')
            data={key:source[name+'/'+value].copy() for key,value in fields.items()}
            data['known']=np.ones_like(data['truth'],bool);datasets[name]=compose(data)
        olddev=datasets['DEV'];cut,calibration=calibrate_positive(olddev['positive_confidence'],olddev['eligible'],olddev['truth'])
        np.save(run/'cutoff.npy',cut);write(run/'calibration.json',calibration)
        write(run/'calibration-freeze.json',dict(status='FROZEN_BEFORE_NONCALIBRATION_REPLAY',cutoff=cut.tolist(),
            cutoff_sha256=sha(run/'cutoff.npy'),source='oldDEV1000 only',inputs=inputs,training_steps=0,model_inference_frames=0))
        new=work/'mz36-new-source-20260910'
        for folder in ('admission-v1','inference-v1','score-v1'):
            r=read(bind(new/folder/'receipt.json'));assert r['status']=='PASS'
            name='result.json' if folder=='admission-v1' else 'predictions.npz' if folder=='inference-v1' else 'scored.npz'
            bind(new/folder/name,r['outputs'][name])
        assert read(bind(new/'audit.json'))['status']=='PASS'
        frozen=read(bind(new/'frozen-models.json'))
        for p,h in frozen['frozen'].items():bind(Path(p),h)
        raw=load_npz(new/'inference-v1/predictions.npz');scored=load_npz(new/'score-v1/scored.npz')
        data={k:scored[v].copy() for k,v in dict(oldscore='MZ35',baseline='MZ5',truth='truth',known='known',frame_ids='frame_ids').items()}
        eligible_rows=np.flatnonzero(data['known'].all(1));assert len(data['frame_ids'])==400 and len(eligible_rows)==380
        np.testing.assert_array_equal(data['frame_ids'][eligible_rows],raw['frame_ids'])
        for target,source_name in dict(rgb='rgb',tof='tof',support='original_support',negative_confidence='MZ35/confidence').items():
            value=raw[source_name];array=np.zeros((400,4),bool) if value.dtype==bool else np.full((400,4),np.nan,dtype=value.dtype)
            array[eligible_rows]=value;data[target]=array
        datasets['MZ36']=compose(data)
        result=dict(metrics={},cutoff=cut.tolist(),calibration=calibration,training_steps=0,model_inference_frames=0,
            backend='CPU saved-scalar replay; TASK_NOT_GPU_SUITABLE',scope='All consumed Development including MZ36; no independent confirmation')
        arrays={}
        for name,data in datasets.items():
            compose(data,cut);truth=data['truth'];known=data['known'];before=data['oldscore'];candidate=data['candidate'];add=data['added']
            result['metrics'][name]=dict(before=metrics(before,truth,known),after=metrics(candidate,truth,known),
                added_tp=(add&truth&known).sum(0).tolist(),added_fp=(add&~truth&known).sum(0).tolist(),
                lost_tp=((candidate<0)&(before>=0)&truth&known).sum(0).tolist(),
                prior_positives_preserved=bool(not ((before>=0)&(candidate<0)&known).any()),
                eligible_tp=(data['eligible']&truth&known).sum(0).tolist(),eligible_fp=(data['eligible']&~truth&known).sum(0).tolist())
            for key,value in data.items():arrays[name+'/'+key]=value
            print(name,'TP+',result['metrics'][name]['added_tp'],'FP+',result['metrics'][name]['added_fp'],flush=True)
        normal=NORMAL+('MZ36',)
        gates=dict(restored_noncalibration_tp=sum(sum(result['metrics'][n]['added_tp']) for n in normal if n!='DEV')>0,
            no_added_fp=all(sum(result['metrics'][n]['added_fp'])==0 for n in normal),
            all_mz35_positives_retained=all(result['metrics'][n]['prior_positives_preserved'] for n in datasets))
        # MZ28 additions have nonnegative MZ35 scores by the inherited baseline contract.
        gates['prior_additions_preserved']=all(np.array_equal(d['candidate'][(d['oldscore']>=0)&(d['baseline']<0)&d['known']],d['oldscore'][(d['oldscore']>=0)&(d['baseline']<0)&d['known']]) for d in datasets.values())
        gates['useful_effect']=all(gates.values());result['gates']=gates
        result['wrong_controls']={n:dict(added_tp=result['metrics'][n+'_wrong']['added_tp'],added_fp=result['metrics'][n+'_wrong']['added_fp']) for n in NORMAL if n!='DEV'}
        np.savez_compressed(run/'predictions.npz',**arrays);write(run/'result.json',result)
        for p,h in inputs.items():assert sha(p)==h,p
        write(run/'receipt.json',dict(status='PASS',inputs=inputs,training_steps=0,model_inference_frames=0,
            seconds=time.perf_counter()-started,backend=result['backend'],code_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('mz36_evaluate.py'))},
            outputs={p.name:sha(p) for p in run.iterdir() if p.is_file()}))
        print('PASS',gates,flush=True)
    except Exception:
        write(run/'failure.json',dict(status='FAIL',inputs=inputs,error=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.root,a.run)
