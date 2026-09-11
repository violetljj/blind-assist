"""Bounded engineering replay of 2048 consumed HELD frames, original MZ70 only.

Verify explicit historical encoder layout without reopening MZ76 or its scorer.
Also export frozen RGB/ToF/MZ5 branches on identical simulated packets for a
separate scalar baseline comparison. No training, threshold selection or labels.
"""
import argparse
import gc
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from mz5_ensemble_readout import read, write, sha
from mz74_return_survival import SOURCES, setup, load_models, DesignBindings, RGBStore, DESIGN_SHA
from mz76_observed_reliability import open_images, R70_SHA
from mz76_replay_diagnostic import nchw_features, compare
from mz56_global_anchor import saved_outputs
from mz54_full_rgb import parameters_sha
from mz63_geometry_transfer import packet
from mz62_profile_coverage import candidate_values


def run(root, out):
    out=out.resolve(); assert out.is_relative_to((root/'artifacts.local').resolve())
    out.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    work=root/'artifacts.local/work'
    dp=work/'mz69-topology-transfer-20260911/design-inputs.json'
    assert sha(dp)==DESIGN_SHA
    design=read(dp); bind=DesignBindings(design); bind(dp,DESIGN_SHA)
    for name in ('mz76_replay_verify.py','mz76_replay_diagnostic.py'):
        shutil.copyfile(Path(__file__).with_name(name),out/name); bind(out/name)
    old=work/'mz70-diverse-learning-20260911/run-v1'
    receipt=read(bind(old/'receipt.json',R70_SHA))
    previous=bind(old/'predictions.npz',receipt['outputs']['predictions.npz'])
    schedule_path=bind(old/'schedule.npz',receipt['outputs']['schedule.npz'])
    selection=read(bind(work/'mz76-observed-reliability-20260911/selection.json'))['sources']
    store=models=spatial=heads=None
    try:
        setup(); models,spatial,heads,cuts,checkpoints=load_models(root,bind,design,out)
        head=heads['DIVERSE']; before=parameters_sha(head); store=RGBStore()
        predictions={}; checks=[]; timing=dict(rgb_decode=0.,encoders=0.,readouts=0.)
        with np.load(previous) as prior,np.load(schedule_path) as schedule,torch.inference_mode():
            for source,(name,digest,loader) in SOURCES.items():
                data=loader(work/name,bind,digest)[0]
                ids=np.array(selection[source]['held_ids'],np.int64)
                assert len(ids)==1024 and not np.isin(ids,schedule[source+'_fit_ids']).any()
                np.testing.assert_array_equal(data['frame_ids'][ids],selection[source]['held_frame_ids'])
                # Every selected frame was encoded in a short, padded historical
                # batch. Thus a fixed explicit NCHW encoder is the correct replay.
                counts=[]
                for i in ids:
                    block=np.arange(i//16*16,i//16*16+16)
                    counts.append(int((~np.isin(block,schedule[source+'_fit_ids'])).sum()))
                assert min(counts)>0 and max(counts)<16
                predictions[source+'/frame_ids']=data['frame_ids'][ids]
                predictions[source+'/original_indices']=ids
                chunks={}
                for begin in range(0,len(ids),16):
                    batch=ids[begin:begin+16]; tick=time.perf_counter()
                    images=open_images([data['rgb_refs'][int(i)] for i in batch],store)
                    timing['rgb_decode']+=time.perf_counter()-tick
                    try:
                        tick=time.perf_counter()
                        visual,detail=models.visual(images)
                        dense=nchw_features(images,models)
                        torch.cuda.synchronize(); timing['encoders']+=time.perf_counter()-tick
                        for profile in ('IDEAL','MERGE_CLOSE','DROP_CLOSE'):
                            obs=packet(data['ranges'][batch],data['valid'][batch],profile)
                            prefix=source+'/'+profile+'/'
                            for key in ('ranges','valid'):
                                np.testing.assert_array_equal(obs[key],prior[prefix+key][batch])
                            tick=time.perf_counter()
                            branches=models.predict(visual,detail,obs['ranges'],obs['valid'])
                            values={k:branches[k] for k in ('rgb','tof','MZ5','MZ37')}
                            actual=saved_outputs(head,(dense,torch.from_numpy(obs['ranges']).cuda(),torch.from_numpy(obs['valid']).cuda()))
                            values.update({'MZ70/DIVERSE/'+k:v for k,v in actual.items()})
                            values['MZ70/DIVERSE/candidate']=candidate_values(actual['raw'],actual['support'],branches['MZ37'],cuts['MZ70/DIVERSE'])
                            torch.cuda.synchronize(); timing['readouts']+=time.perf_counter()-tick
                            for k,v in values.items(): chunks.setdefault(prefix+k,[]).append(v)
                    finally:
                        for image in images: image.close()
                    if begin%256==0:
                        print(source,begin+16,flush=True)
                        write(out/'progress.json',dict(source=source,frames=begin+16))
                for key,parts in chunks.items(): predictions[key]=np.concatenate(parts)
                for profile in ('IDEAL','MERGE_CLOSE','DROP_CLOSE'):
                    prefix=source+'/'+profile+'/'
                    for key in ('MZ37','MZ70/DIVERSE/raw','MZ70/DIVERSE/support','MZ70/DIVERSE/winner',
                                'MZ70/DIVERSE/anchor_available','MZ70/DIVERSE/anchor_vector','MZ70/DIVERSE/candidate'):
                        actual=predictions[prefix+key]; expected=prior[prefix+key][ids]
                        row=dict(source=source,profile=profile,key=key,**compare(actual,expected))
                        if key=='MZ37' or key.endswith('/candidate'):
                            row['sign_changes']=int(((actual>=0)!=(expected>=0)).sum())
                        checks.append(row)
        assert parameters_sha(head)==before
        np.savez_compressed(out/'predictions.npz',**predictions)
        numeric=[r for r in checks if not r['key'].endswith('/winner')]
        passed=all(r['violations']==0 and r.get('sign_changes',0)==0 for r in numeric)
        bind.check()
        write(out/'result.json',dict(status='PASS_NUMERIC_REPLAY' if passed else 'FAIL_NUMERIC_REPLAY',
            scope='Consumed synthetic engineering replay; MZ76 strict failure remains unchanged',
            frames=2048,profiles=['IDEAL','MERGE_CLOSE','DROP_CLOSE'],checks=checks,
            training_steps=0,new_cutoffs=0,evaluator_labels_read=False,parameters_unchanged=True,
            rgb_loads=store.loads,backend='CUDA',device=torch.cuda.get_device_name(),
            torch_version=torch.__version__,cudnn_version=torch.backends.cudnn.version(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
            timing_seconds=timing,seconds=time.perf_counter()-started,inputs=bind.inputs,
            outputs={'predictions.npz':sha(out/'predictions.npz')}))
        print('REPLAY',passed,'winner changes',sum(not r['exact'] for r in checks if r['key'].endswith('/winner')),flush=True)
    except BaseException:
        write(out/'failure.json',dict(error=traceback.format_exc(),inputs=bind.inputs)); raise
    finally:
        if store is not None: store.close()
        models=spatial=heads=None; gc.collect()
        write(out/'release.json',dict(rgb_handles_closed=True,persistent_feature_cache=False,process_exit_required=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('E:/linnan/linnan'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(args.root.resolve(),args.output)
