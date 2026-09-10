"""Compact-source preparation and fixed-model richer-object transfer."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
from contextlib import ExitStack
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
import torch
from body_query_context_evidence import ContextEvidence
from body_query_fresh_size_eval import FROZEN
from body_query_range import range_from_counts
from data_lightweight import CompactSource
from multizone64_observation import observe
from mz1_tiny_fusion import TinyFusion
from mz5_ensemble_readout import CompactEnsemble, read, write, sha, load_npz
from mz15_shared_support import LocalSupportReadout
from mz16_detail_cache import CROP
from mz23_availability import restrict_candidates
from mz28_packet_availability import PacketAvailability
from mz30_select import BranchSelector, negative_branch_confidence
from mz36_frozen_inference import fixed_batch_dense, CAMERA
from mz37_restore import compose
from mz40_packets import constrain
from mz40_evaluate import metrics, paired

CONDITIONS=('IDEAL','MERGE_CLOSE','DROP_CLOSE')
METHODS=('rgb','tof','MZ5','MZ28','MZ35','MZ37','MZ43_TOF','MZ43_ENSEMBLE','IDEAL_FUSION','MZ43_FUSION')


def setup():
    assert torch.cuda.is_available()
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True; torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False; torch.use_deterministic_algorithms(False)


class Bindings:
    def __init__(self): self.inputs={}
    def __call__(self,path,expected=None):
        path=Path(path); digest=sha(path)
        assert expected is None or digest==expected,str(path)
        self.inputs[str(path)]=digest
        return path
    def check(self):
        for path,digest in self.inputs.items(): assert sha(path)==digest,path
    def checkpoint(self,folder,name):
        r=read(self(folder/'receipt.json')); assert r['status']=='PASS'
        return self(folder/name,r['outputs'][name])


def prepare(root,task):
    out=task/'prepared-v1'; out.mkdir(parents=True,exist_ok=False)
    bind=Bindings(); started=time.perf_counter(); setup()
    frames=[]; metadata=[]; truth=[]; known=[]; rr=[]; vv=[]
    for block,eid in [('near','mz42-rich-objects-20260910'),('far','mz44-rich-far-objects-20260911')]:
        source=root/'artifacts.local/work'/eid
        terminal=read(bind(source/'receipt.json')); assert terminal['status']=='PASS'
        verification=read(bind(source/'return-verification.json'))
        package=bind(source/'returned-v1/capture.source.zip',verification['archive_sha256'])
        folder=source/'returned-v1/dataset-v1'
        ds=read(bind(folder/('result.json' if block=='near' else 'range-result.json')))
        labels=load_npz(bind(folder/'labels.npz'))
        records=ds['records']; assert len(records)==20
        np.testing.assert_array_equal(labels['frame_ids'],[r['frame_id'] for r in records])
        with CompactSource(package) as compact:
            spec=compact.read_json('source/spec.json')
            for case,row in zip(spec['cases'],records):
                assert case['name']==row['frame_id']
                assert case['camera']['pitch']==case['camera']['roll']==case['camera']['yaw']==0
                assert abs(case['camera']['z']-row['floor']['declared_floor_z_m']-1.7)<1e-6
                for key,hkey in [('rgb','rgb_sha256'),('native','native_sha256')]:
                    assert compact.entries[row[key]]['sha256']==row[hkey]
                frames.append(dict(frame_id=row['frame_id'],archive=str(package),rgb=row['rgb']))
                metadata.append(dict(frame_id=row['frame_id'],block=block,family=row['family'],variant=row['variant']))
            for begin in range(0,20,16):
                depth=np.stack([compact.load_array(row['native']) for row in records[begin:begin+16]])
                assert depth.dtype==np.float32 and depth.shape[1:]==(360,640)
                with torch.inference_mode(): packet=observe(torch.from_numpy(depth).cuda(),readout='multi_surface')
                rr.append(torch.nan_to_num(packet['range_m']).float().cpu().numpy())
                vv.append(packet['valid'].cpu().numpy())
        truth.append(labels['truth']); known.append(labels['known'])
    ranges=np.concatenate(rr); valid=np.concatenate(vv)
    assert ranges.shape==valid.shape==(40,64,2)
    packets=dict(frame_ids=np.array([r['frame_id'] for r in frames]),ranges=ranges,valid=valid)
    stats={}
    for condition in CONDITIONS[1:]:
        changed,close=constrain(ranges,valid,condition)
        for key,value in changed.items():packets[condition+'/'+key]=value
        stats[condition]=dict(changed_zones=int(close.sum()),changed_frames=int(close.any(1).sum()))
    np.savez_compressed(out/'packets.npz',**packets)
    np.savez_compressed(out/'evaluator.npz',frame_ids=packets['frame_ids'],truth=np.concatenate(truth),known=np.concatenate(known))
    write(out/'predictor.json',dict(camera=CAMERA,frames=frames))
    write(out/'groups.json',dict(records=metadata))
    bind(Path(__file__).with_name('MZ45_OBJECT_TRANSFER_20260911.md'));bind(Path(__file__))
    bind.check()
    write(out/'receipt.json',dict(status='PASS',inputs=bind.inputs,attempts=40,training_steps=0,
        extracted_files=0,permanent_dense_cache=False,backend='CUDA packet synthesis; CPU compact I/O',
        device=torch.cuda.get_device_name(),seconds=time.perf_counter()-started,restrictions=stats,
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('PREPARED',stats,flush=True)


class FrozenModels:
    def __init__(self,root,bind):
        work=root/'artifacts.local/work'
        baseline=work/'body-query-10000-b-20260909/run-v1'; context=work/'body-query-context-decoder-20260909/run-v1'
        reference=read(bind(work/'mz36-new-source-20260910/inference-v1/receipt.json'))
        for name,digest in reference['code_sha256'].items():
            if name!='mz36_frozen_inference.py':bind(Path(__file__).with_name(name),digest)
        for name,digest in FROZEN.items():bind((baseline if name=='NEW-step2000.pt' else context)/name,digest)
        self.context=ContextEvidence(baseline,context,work/'body-query-v1-20260908/model-inputs/pretrained').cuda().eval()
        compact=work/'mz5-fixed-ensemble-20260910/compact-v1'
        cr=read(bind(compact/'export-receipt.json'))
        self.original=CompactEnsemble.from_checkpoint(bind(compact/'compact.pt',cr['compact_sha256'])).cuda()
        mixed=work/'mz43-restricted-training-20260911'
        mr=read(bind(mixed/'compact-v1/receipt.json'))
        self.mixed=CompactEnsemble.from_checkpoint(bind(mixed/'compact-v1/compact.pt',mr['compact']['sha256'])).cuda()
        tr=read(bind(mixed/'run-v1/receipt.json')); self.joint={}
        for label,name,arm in [('IDEAL_FUSION','IDEAL-FUSION','FUSION'),('MZ43_FUSION','MIXED-FUSION','FUSION'),('DENSE_MIXED_TOF','MIXED-TOF_ONLY','TOF_ONLY')]:
            model=TinyFusion(arm)
            model.load_state_dict(torch.load(bind(mixed/'run-v1'/(name+'.pt'),tr['outputs'][name+'.pt']),map_location='cpu',weights_only=True))
            self.joint[label]=model.cuda().eval().requires_grad_(False)
        rank=work/'mz20-rank-objective-20260910/run-v1'
        self.rank=LocalSupportReadout(shared=False)
        self.rank.load_state_dict(torch.load(bind.checkpoint(rank,'BODY_RANK.pt'),map_location='cpu',weights_only=True))
        self.rank=self.rank.cuda().eval().requires_grad_(False)
        self.rank_cut=np.load(bind.checkpoint(rank,'BODY_RANK-cutoff.npy'))
        bank=load_npz(bind.checkpoint(work/'mz28-packet-availability-20260910/run-v1','bank.npz'))
        self.bank=PacketAvailability(*(torch.from_numpy(bank[k]) for k in ('ids','packet','known'))).cuda().eval()
        norm=load_npz(bind.checkpoint(work/'mz9-source-supervision-20260910/run-v1','normalization.npz'))
        self.detail_mean,self.detail_std=(torch.from_numpy(norm[k]).cuda() for k in ('mean','std'))
        norm=load_npz(bind.checkpoint(work/'mz30-branch-responsibility-20260910/run-v1','normalization.npz'))
        self.selector_mean,self.selector_std=(torch.from_numpy(norm[k]).cuda() for k in ('mean','std'))
        selector=work/'mz35-responsibility-convergence-20260910/run-v1'
        self.selector=BranchSelector()
        self.selector.load_state_dict(torch.load(bind.checkpoint(selector,'selector.pt'),map_location='cpu',weights_only=True))
        self.selector=self.selector.cuda().eval().requires_grad_(False)
        self.selector_cut=np.load(bind.checkpoint(selector,'cutoff.npy'))
        self.restore_cut=np.load(bind.checkpoint(work/'mz37-positive-restoration-20260910/run-v1','cutoff.npy'))

    def visual(self,images):
        low=np.stack([np.array(im.convert('RGB').resize((256,144),Image.Resampling.BOX)) for im in images])
        high=np.stack([np.array(im.convert('RGB').crop(CROP)) for im in images])
        upload=lambda a:torch.from_numpy(a).permute(0,3,1,2).cuda().float()/255
        full=fixed_batch_dense(self.context.base,upload(low),legacy=True)
        detail=fixed_batch_dense(self.context.base,upload(high))
        m=self.context.base
        sampled=(full.flatten(2)@m.query_projection.T).transpose(1,2).reshape(len(images),12,27,64)
        qmask=m.query_valid[None,:,:,None]
        pooled=(sampled*qmask).sum(2)/qmask.sum(2).clamp_min(1)
        normalized=(pooled-self.context.feature_mean)/self.context.feature_std
        visual=torch.cat([normalized.flatten(1),range_from_counts(self.context.decoder(normalized)).sigmoid().flatten(1)],1)
        return visual,detail

    def predict(self,visual,detail,ranges,valid):
        ranges=torch.from_numpy(ranges).cuda();valid=torch.from_numpy(valid).cuda()
        tof=torch.cat([ranges.flatten(1)/4,valid.flatten(1).float()],1)
        rh=self.original.rgb[:2](visual);th=self.original.tof[:2](tof)
        rgb=self.original.rgb[2](rh);tl=self.original.tof[2](th)
        features=torch.cat([rh,th,rgb,tl],1)
        availability=self.bank(ranges,valid)
        original=self.rank.inspect((detail-self.detail_mean)/self.detail_std,ranges,valid)
        restricted=restrict_candidates(original,availability['availability'])
        selector=self.selector((features-self.selector_mean)/self.selector_std)
        mixed_rgb,mixed_tof=self.mixed.branches(visual,tof)
        torch.testing.assert_close(rgb,mixed_rgb,rtol=0,atol=0)
        dense_mixed=self.joint['DENSE_MIXED_TOF'](visual,tof)
        assert torch.equal(dense_mixed>=0,mixed_tof>=0)
        values=dict(visual=visual,ranges=ranges,valid=valid,rgb=rgb,tof=tl,MZ5=(rgb+tl)*.5,
            original_support=original['support'],original_raw=original['logits'],
            restricted_support=restricted['support'],restricted_raw=restricted['logits'],
            MZ35_confidence=negative_branch_confidence(rgb,selector),MZ43_TOF=mixed_tof,
            MZ43_ENSEMBLE=(rgb+mixed_tof)*.5,
            IDEAL_FUSION=self.joint['IDEAL_FUSION'](visual,tof),MZ43_FUSION=self.joint['MZ43_FUSION'](visual,tof))
        a={key:value.cpu().numpy() for key,value in values.items()}
        margin=np.where(a['restricted_support'],a['restricted_raw'].astype(float)-self.rank_cut,-1e6)
        accepted=(a['MZ5']<0)&a['restricted_support']&(margin>=0)
        a['MZ28']=np.where(accepted,margin,a['MZ5'])
        eligible=(a['MZ5']>=0)&~a['original_support']&((a['rgb']>=0)!=(a['tof']>=0))
        remove=eligible&(a['MZ35_confidence'].astype(float)>=self.selector_cut)
        a['MZ35']=np.where(remove,np.where(a['rgb']<0,a['rgb'],a['tof']),a['MZ28'])
        restore=compose(dict(rgb=a['rgb'],tof=a['tof'],baseline=a['MZ5'],oldscore=a['MZ35'],
            support=a['original_support'],known=np.ones_like(a['MZ35'],bool),negative_confidence=a['MZ35_confidence']),self.restore_cut)
        a['MZ37']=restore['candidate']
        return a


def infer(root,task):
    prepared=task/'prepared-v1'; out=task/'inference-v1'; out.mkdir(exist_ok=False)
    started=time.perf_counter(); setup(); bind=Bindings()
    r=read(bind(prepared/'receipt.json')); assert r['status']=='PASS'
    manifest=read(bind(prepared/'predictor.json',r['outputs']['predictor.json']))
    packets=load_npz(bind(prepared/'packets.npz',r['outputs']['packets.npz']))
    assert manifest['camera']==CAMERA
    np.testing.assert_array_equal(packets['frame_ids'],[r['frame_id'] for r in manifest['frames']])
    bind(Path(__file__));bind(Path(__file__).with_name('MZ45_OBJECT_TRANSFER_20260911.md'))
    models=None
    try:
        models=FrozenModels(root,bind)
        write(out/'start.json',dict(status='STARTED',inputs=bind.inputs.copy(),training_steps=0,
            frames=40,parity_frames=16,device=torch.cuda.get_device_name(),cudnn_tf32=True,matmul_tf32=False,
            input_contract='RGB + ranges/validity + fixed calibration; predictor never reads evaluator.npz/groups.json',
            permanent_dense_cache=False))
        with torch.inference_mode():
            prior=root/'artifacts.local/work/mz40-l8cx-constrained-20260910/parity-v1'
            pr=read(bind(prior/'receipt.json'))
            reference=load_npz(bind(prior/'predictions.npz',pr['outputs']['predictions.npz']))
            source=root/'artifacts.local/work/mz36-new-source-20260910/admission-v1/predictor-manifest.json'
            old_frames=read(bind(source))['frames'][:16]
            images=[]
            for row in old_frames:
                path=bind(row['rgb_path'])
                with Image.open(path) as im:images.append(im.copy())
            v,d=models.visual(images); actual=models.predict(v,d,reference['ranges'],reference['valid'])
            parity={}
            for key in ('visual','ranges','valid','rgb','tof','MZ5','MZ28','MZ35','MZ37','original_support','original_raw','restricted_support','restricted_raw'):
                np.testing.assert_array_equal(actual[key],reference[key],err_msg=key)
                parity[key]=int(actual[key].size)
            write(out/'parity.json',dict(status='PASS',frames=16,exact_arrays=parity))
            pieces={condition:[] for condition in CONDITIONS}; timings=dict(visual=0.,readout=0.)
            with ExitStack() as stack:
                sources={p:stack.enter_context(CompactSource(bind(p))) for p in sorted({r['archive'] for r in manifest['frames']})}
                for begin in range(0,40,16):
                    rows=manifest['frames'][begin:begin+16]
                    images=[sources[row['archive']].load_image(row['rgb']) for row in rows]
                    assert all(im.size==(640,360) for im in images)
                    torch.cuda.synchronize();tick=time.perf_counter();v,d=models.visual(images)
                    torch.cuda.synchronize();timings['visual']+=time.perf_counter()-tick
                    for condition in CONDITIONS:
                        prefix='' if condition=='IDEAL' else condition+'/'
                        torch.cuda.synchronize();tick=time.perf_counter()
                        a=models.predict(v,d,packets[prefix+'ranges'][begin:begin+len(rows)],packets[prefix+'valid'][begin:begin+len(rows)])
                        torch.cuda.synchronize();timings['readout']+=time.perf_counter()-tick
                        pieces[condition].append(a)
                    print('INFERENCE',begin+len(rows),'/40',flush=True)
            arrays={'frame_ids':packets['frame_ids']}
            for condition,parts in pieces.items():
                for key in parts[0]:arrays[condition+'/'+key]=np.concatenate([p[key] for p in parts])
            np.savez_compressed(out/'predictions.npz',**arrays)
        bind.check()
        write(out/'receipt.json',dict(status='PASS',inputs=bind.inputs,training_steps=0,frames=40,parity_frames=16,
            backend='CUDA',device=torch.cuda.get_device_name(),timing_seconds=timings,total_seconds=time.perf_counter()-started,
            extracted_files=0,evaluator_labels_read=False,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        print('INFERENCE PASS',timings,flush=True)
    except Exception:
        write(out/'failure.json',dict(status='FAILED',error=traceback.format_exc()));raise
    finally:
        del models;torch.cuda.empty_cache()


def score(root,task):
    out=task/'score-v1';out.mkdir(exist_ok=False);bind=Bindings();started=time.perf_counter()
    prepared=task/'prepared-v1';inference=task/'inference-v1'
    pr=read(bind(prepared/'receipt.json'));ir=read(bind(inference/'receipt.json'))
    data=load_npz(bind(inference/'predictions.npz',ir['outputs']['predictions.npz']))
    labels=load_npz(bind(prepared/'evaluator.npz',pr['outputs']['evaluator.npz']))
    rows=read(bind(prepared/'groups.json',pr['outputs']['groups.json']))['records']
    np.testing.assert_array_equal(data['frame_ids'],labels['frame_ids'])
    truth,known=labels['truth'],labels['known']
    assert len(truth)==40 and known.all()
    masks=dict(all=np.ones(40,bool),new_forms=np.array([r['family']!='oblique_rod' for r in rows]),control=np.array([r['family']=='oblique_rod' for r in rows]))
    for field in ('family','block'):
        for value in dict.fromkeys(r[field] for r in rows):masks[field+'/'+value]=np.array([r[field]==value for r in rows])
    summary={};failures=[]
    for condition in CONDITIONS:
        summary[condition]={group:{method:metrics(data[condition+'/'+method][mask],truth[mask],known[mask]) for method in METHODS} for group,mask in masks.items()}
        for i,row in enumerate(rows):
            for method in METHODS:
                pred=data[condition+'/'+method][i]>=0
                for q in range(4):
                    if known[i,q] and pred[q]!=truth[i,q]:
                        failures.append(dict(**row,condition=condition,method=method,query=q,error='FP' if pred[q] else 'FN',
                            original_support=bool(data[condition+'/original_support'][i,q]),
                            rgb_positive=bool(data[condition+'/rgb'][i,q]>=0),tof_positive=bool(data[condition+'/tof'][i,q]>=0),
                            valid_return_slots=int(data[condition+'/valid'][i].sum())))
    comparisons={}
    for candidate,baseline in [('MZ43_ENSEMBLE','MZ5'),('MZ43_FUSION','IDEAL_FUSION')]:
        change={key:sum(sum(summary[c]['new_forms'][candidate][key])-sum(summary[c]['new_forms'][baseline][key]) for c in CONDITIONS[1:]) for key in ('fp','fn')}
        comparisons[candidate]=dict(baseline=baseline,restricted_new_form_change=change,
            transfer_supported=change['fn']<0 and change['fp']<=0,
            paired={c:paired(data[c+'/'+candidate],data[c+'/'+baseline],truth,known) for c in CONDITIONS})
    # Independent scalar event recount, including the fixed new/control split.
    checked=0
    for condition in CONDITIONS:
        for group,mask in masks.items():
            for method in METHODS:
                expected=summary[condition][group][method];counts={k:[0]*4 for k in ('tp','fp','fn','tn')};exact=0
                for i in np.flatnonzero(mask):
                    correct=True
                    for q in range(4):
                        p=bool(data[condition+'/'+method][i,q]>=0);y=bool(truth[i,q])
                        key=('tp' if y else 'fp') if p else ('fn' if y else 'tn')
                        counts[key][q]+=1;correct &= p==y;checked+=1
                    exact+=int(correct)
                assert all(counts[k]==expected[k] for k in counts) and exact==expected['exact_frames']
    write(out/'result.json',dict(status='PASS',attempted_frames=40,known_event_bits=int(known.sum()),unknown_event_bits=int((~known).sum()),
        new_form_frames=32,control_frames=8,conditions=summary,comparisons=comparisons,training_steps=0,
        scope='Same-site controlled object replacement; no physical sensor/natural transfer claim'))
    write(out/'failures.json',dict(records=failures))
    write(out/'audit.json',dict(status='PASS',scalar_scored_bits=checked,source_id_alignment=40,zero_fitting=True,
        dense_feature_files_written=0,all_attempts_retained=True))
    bind.check();write(out/'receipt.json',dict(status='PASS',inputs=bind.inputs,source_sha256=sha(__file__),
        backend='CPU scalar scoring; TASK_NOT_GPU_SUITABLE',seconds=time.perf_counter()-started,
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('RESULT',comparisons,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','infer','score'],required=True)
    a=parser.parse_args();root=a.root.resolve();task=a.task.resolve()
    assert task.is_relative_to((root/'artifacts.local').resolve())
    globals()[a.phase](root,task)
