"""MZ77 sensor simulation and frozen inference, with distinct input boundaries."""
import argparse
import gc
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mz5_ensemble_readout import read, write, sha
from mz74_return_survival import SOURCES, setup, load_models, DesignBindings, RGBStore, DESIGN_SHA
from mz76_observed_reliability import open_images, R70_SHA
from mz76_replay_diagnostic import nchw_features
from mz56_global_anchor import saved_outputs
from mz54_full_rgb import parameters_sha
from mz50_train import infer as infer_spatial
from mz62_profile_coverage import candidate_values
from multizone64_observation import observe

TASK='mz77-approach-sequence-20260911'
PROFILES=('CLEAN','CENTER_GAP')
METHODS=('rgb','tof','MZ5','MZ37','OLD_NEG','DIVERSE')


def gap_packet(ranges,valid,indices):
    """Fixed current-index outage; no labels, hidden geometry or history."""
    assert ranges.shape==valid.shape and ranges.shape[1:]==(64,2)
    assert len(indices)==len(ranges) and valid.dtype==bool
    mask=np.isin(indices,(20,21,22,30,31,32))[:,None,None] & np.isin(np.arange(64)%8,(2,3,4,5))[None,:,None]
    return np.where(mask,0,ranges), valid & ~mask


def output_dir(root,path):
    path=path.resolve(); assert path.is_relative_to((root/'artifacts.local/work'/TASK).resolve())
    path.mkdir(parents=True,exist_ok=False); return path


def sensor(root,capture,out):
    """Only this stage reads native depth to generate synthetic observations."""
    out=output_dir(root,out); started=time.perf_counter(); inputs={}
    def bind(path):
        path=Path(path).resolve(); inputs[str(path)]=sha(path); return path
    shutil.copyfile(__file__,out/'sensor-source.py'); bind(out/'sensor-source.py')
    manifest=read(bind(capture/'model/sensor_manifest.json'))
    assert set(manifest)=={'calibration','frames'}
    assert read(bind(capture/'source-admission.json'))['status']=='PASS'
    assert read(bind(capture/'receipt.json'))['status']=='PASS'
    frames=manifest['frames']; assert len(frames)==160
    assert len({f['clip_id'] for f in frames})==4
    clip=np.array([r['clip_id'] for r in frames]); index=np.tile(np.arange(40),4)
    times=np.array([r['nominal_time_s'] for r in frames])
    np.testing.assert_allclose(times,index/10,atol=1e-12,rtol=0)
    assert [r['sample_index'] for r in frames]==list(range(160))
    rr=[];vv=[];setup()
    with torch.inference_mode():
        for begin in range(0,160,16):
            depths=[]
            for i in range(begin,begin+16):
                d=np.load(bind(capture/f'evaluator/native/{i:04d}.npy'),allow_pickle=False)
                assert d.shape==(360,640);depths.append(d)
            packet=observe(torch.from_numpy(np.stack(depths)).cuda(),8,'multi_surface')
            rr.append(torch.nan_to_num(packet['range_m'],nan=0).float().cpu().numpy())
            vv.append(packet['valid'].cpu().numpy())
    ranges=np.concatenate(rr);valid=np.concatenate(vv)
    gr,gv=gap_packet(ranges,valid,index)
    arrays=dict(clip=clip,index=index,time_s=times)
    for mode,r,v in [('CLEAN',ranges,valid),('CENTER_GAP',gr,gv)]:
        arrays[mode+'/ranges']=r;arrays[mode+'/valid']=v
    np.savez_compressed(out/'observations.npz',**arrays)
    # Model manifest contains observed frame order and RGB references only.
    model_rows=[]
    for i,row in enumerate(frames):
        rgb=(capture/'model'/row['rgb_path']).resolve()
        assert rgb.is_relative_to((capture/'model').resolve())
        model_rows.append(dict(sample_index=i,clip_id=row['clip_id'],index=int(index[i]),
                               nominal_time_s=float(times[i]),rgb_path=str(rgb),rgb_sha256=sha(rgb)))
    write(out/'predictor.json',dict(calibration=manifest['calibration'],frames=model_rows))
    assert not np.any(gv&~valid) and np.array_equal(gr[gv],ranges[gv])
    for path,digest in inputs.items(): assert sha(path)==digest
    write(out/'receipt.json',dict(status='PASS',stage='SIMULATED_SENSOR_ONLY',frames=160,profiles=PROFILES,
        deleted_valid_slots=int((valid&~gv).sum()),changed_frames=int(np.any(valid!=gv,axis=(1,2)).sum()),
        inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        backend='CUDA',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-started,
        claim='Generic geometry packets and fixed partial outage, not measured sensor physics',model_inference_frames=0))
    print('SENSOR PASS',int((valid&~gv).sum()),'removed slots',flush=True)


def batch_predictions(images,packets,models,spatial,head,cuts):
    visual,detail=models.visual(images);dense=nchw_features(images,models)
    detail_cpu=detail.cpu().numpy(); outputs={}
    for profile,(rr,vv) in packets.items():
        branches=models.predict(visual,detail,rr,vv)
        row={k:branches[k] for k in ('rgb','tof','MZ5','MZ37')}
        local=infer_spatial(spatial,detail_cpu,rr,vv,models)
        for mode in ('OPEN','GATED'):
            row['OLD_NEG/'+mode]=candidate_values(local[mode+'/raw'],local[mode+'/support'],row['MZ37'],cuts['OLD_NEG/'+mode])
        row['OLD_NEG']=np.maximum(row['OLD_NEG/OPEN'],row['OLD_NEG/GATED'])
        current=saved_outputs(head,(dense,torch.from_numpy(rr).cuda(),torch.from_numpy(vv).cuda()))
        row['DIVERSE/raw']=current['raw']; row['DIVERSE/support']=current['support'];row['DIVERSE/winner']=current['winner']
        row['DIVERSE/candidate']=candidate_values(current['raw'],current['support'],row['MZ37'],cuts['MZ70/DIVERSE'])
        row['DIVERSE']=np.maximum(row['OLD_NEG'],row['DIVERSE/candidate'])
        outputs[profile]=row
    return outputs


def infer(root,sensor_dir,out):
    out=output_dir(root,out); started=time.perf_counter()
    work=root/'artifacts.local/work';dp=work/'mz69-topology-transfer-20260911/design-inputs.json'
    assert sha(dp)==DESIGN_SHA
    design=read(dp);bind=DesignBindings(design);bind(dp,DESIGN_SHA)
    for name in ('mz77_approach_evaluate.py','mz76_replay_diagnostic.py'):
        shutil.copyfile(Path(__file__).with_name(name),out/name);bind(out/name)
    sensor_receipt=read(bind(sensor_dir/'receipt.json'));assert sensor_receipt['status']=='PASS'
    observed_path=bind(sensor_dir/'observations.npz',sensor_receipt['outputs']['observations.npz'])
    manifest=read(bind(sensor_dir/'predictor.json',sensor_receipt['outputs']['predictor.json']))
    with np.load(observed_path) as z: observations=dict(z)
    assert set(observations)=={'clip','index','time_s'}|{p+'/'+k for p in PROFILES for k in ('ranges','valid')}
    models=spatial=heads=store=None
    try:
        setup();models,spatial,heads,cuts,checkpoints=load_models(root,bind,design,out)
        head=heads['DIVERSE'];before=parameters_sha(head)
        # One already consumed prior block verifies the full retained pipeline.
        old=work/'mz70-diverse-learning-20260911/run-v1'
        r70=read(bind(old/'receipt.json',R70_SHA))
        data=SOURCES['mz61'][2](work/SOURCES['mz61'][0],bind,SOURCES['mz61'][1])[0]
        selection=read(bind(work/'mz76-observed-reliability-20260911/selection.json'))
        ids=np.array(selection['sources']['mz61']['held_ids'][:16]);store=RGBStore()
        images=open_images([data['rgb_refs'][int(i)] for i in ids],store)
        with torch.inference_mode():
            try: actual=batch_predictions(images,{'CLEAN':(data['ranges'][ids],data['valid'][ids])},models,spatial,head,cuts)['CLEAN']
            finally:
                for image in images:image.close()
        checks={}
        with np.load(bind(old/'predictions.npz',r70['outputs']['predictions.npz'])) as prior:
            for key,old_key in [('MZ37','MZ37'),('OLD_NEG','OLD_NEG/UNION'),('DIVERSE','MZ70/DIVERSE/UNION'),('DIVERSE/raw','MZ70/DIVERSE/raw')]:
                expected=prior['mz61/IDEAL/'+old_key][ids]
                np.testing.assert_allclose(actual[key],expected,atol=2e-5,rtol=1e-6)
                if key!='DIVERSE/raw':np.testing.assert_array_equal(actual[key]>=0,expected>=0)
                checks[key]=float(np.abs(actual[key]-expected).max())
        write(out/'parity.json',dict(status='PASS',frames=16,ids=ids.tolist(),max_abs=checks))
        arrays={k:observations[k] for k in ('clip','index','time_s')};chunks={}
        with torch.inference_mode():
            for begin in range(0,160,16):
                images=[]
                try:
                    for row in manifest['frames'][begin:begin+16]:
                        path=bind(row['rgb_path'],row['rgb_sha256'])
                        with Image.open(path) as im:images.append(im.convert('RGB'))
                    packets={p:(observations[p+'/ranges'][begin:begin+16],observations[p+'/valid'][begin:begin+16]) for p in PROFILES}
                    rows=batch_predictions(images,packets,models,spatial,head,cuts)
                    for p,row in rows.items():
                        for key,value in row.items():chunks.setdefault(p+'/'+key,[]).append(value)
                finally:
                    for image in images:image.close()
                print('INFER',begin+16,flush=True)
        for key,parts in chunks.items():arrays[key]=np.concatenate(parts)
        assert parameters_sha(head)==before
        np.savez_compressed(out/'predictions.npz',**arrays);bind.check()
        write(out/'receipt.json',dict(status='PASS',frames=160,parity_frames=16,profiles=PROFILES,methods=METHODS,
            inputs=bind.inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},
            training_steps=0,new_cutoffs=0,native_depth_reads=0,evaluator_label_reads=0,parameters_unchanged=True,
            backend='CUDA',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-started,
            claim='Fixed framewise predictors on posed temporal samples; no added temporal model or real-time latency'))
        print('INFERENCE PASS',flush=True)
    except BaseException:
        write(out/'failure.json',dict(error=traceback.format_exc(),inputs=bind.inputs));raise
    finally:
        if store is not None:store.close()
        models=spatial=heads=None;gc.collect()
        write(out/'release.json',dict(image_handles_closed=True,persistent_feature_cache=False,process_exit_required=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('sensor','infer'))
    p.add_argument('--root',type=Path,default=Path('E:/linnan/linnan'));p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    (sensor if a.phase=='sensor' else infer)(a.root.resolve(),a.input.resolve(),a.output)
