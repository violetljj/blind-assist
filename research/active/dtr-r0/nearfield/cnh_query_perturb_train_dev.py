"""Single-seed, matched-continuation query-yaw robustness screen on Development.

Only current query direction changes; original M3 supervision stays in travel
coordinates. Both continuations start from the same weights, not optimizer resume.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import cnh_active_scan_dev as A
import cnh_near_range as NR
import cnh_temporal_readout_train as V
import cnh_temporal_readout_model as M
import cnh_query_direction_dev as Q
import cnh_tristate_dev as R
from cnh_query_perturb_project_dev import QueryPerturbProjector, perturb_matrices

ROOT=A.ROOT
OUT=ROOT/'artifacts.local/work/cnh-query-perturb-train-dev-20261008'
PARENT=ROOT/'artifacts.local/work/cnh-temporal-readout-20261004/inputs/train'
LABELS=ROOT/'artifacts.local/work/cnh-margin-labels-20261002/train_labels.npz'
CAL=list(range(98000,98008));EVAL=list(range(99000,99024))
MODELS=('frozen_M3_5','frozen_M3_seed0','Control','Aug')
EPOCHS=2;BATCH=256;LR=3e-4;WD=1e-4;SEED=0;PERTURB_SEED=2026100811


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(f'Preserve {path}')
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def progress(value):
    p=OUT/'progress.json';tmp=p.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(p)


def checkpoint(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.pt');torch.save(value,temporary);temporary.replace(path)


def delta_for(unit,config):
    rng=np.random.default_rng([PERTURB_SEED,int(unit),int(config)])
    return 0. if rng.random()<.25 else float(np.clip(rng.normal(0.,10.),-20.,20.))


def freeze():
    sources=[Path(__file__),Path(__file__).with_name('cnh_query_perturb_project_dev.py'),
        Path(__file__).with_name('cnh_query_perturb_evaluate_dev.py'),Path(M.__file__),Path(V.__file__),Path(Q.__file__)]
    paths=sources+[LABELS,OUT/'bench/benchmark.json',OUT/'bench/benchmark_batch4.json',*A.M3_MODELS]
    paths += sorted((NR.OUT/'data/train').glob('features_c*.npy'))+sorted((NR.OUT/'data/train').glob('metadata_c*.npz'))
    paths += [PARENT/name for name in ('rows.npz','histories.npy','transforms.npy','length.npy','receipt.json')]
    paths += [Path(p) for p in (A.__file__,R.__file__,NR.__file__)]
    paths += [Path(__file__).with_name(name) for name in ('cnh_cvr_pilot.py','cnh_cvr_projection.py','cnh_cvr_v2_materialize.py')]
    paths += [(A.AUG if u<99000 else A.CONT)/'observations'/('calibration' if u<99000 else 'evaluation')/f'unit{u}.npz' for u in CAL+EVAL]
    plan=dict(task='CNH_QUERY_PERTURB_TRAIN_DEV_20261008',lane='EXPLORE consumed simulation Development',
        authority='User supplied corrected roundtable explicitly authorizes single-seed perturbation-training screen; local scale/budget chosen as bounded implementation, not quoted as user-confirmed96/384 or2-3h',
        goal='At matched alarm-time cost, test whether query perturbation continuation improves absolute E1 timely count over identical unperturbed continuation and frozen five-seed M3 without excessive exact loss',
        training=dict(seed=SEED,units='Original M3 train93000..93095 only',rows=27456,epochs=EPOCHS,batch=BATCH,lr=LR,weight_decay=WD,
            optimizer='Fresh AdamW, CosineAnnealingLR(T_max=2), final epoch only; weights continuation, no original optimizer state',
            initialization='Both load original model_seed0.pt; frozen seed0 paired reference plus historical frozen five-seed ensemble',
            loss='Original M3 hard-label unweighted BCEWithLogits mean over rows and two queries',
            labels='Original train_labels.M3 held fixed in actual travel corridor; +/-0.33m margin supervision, input masks +/-0.30m. Not strict zero-margin collision labels.',
            distribution='One deterministic delta per unit/config for both epochs:25%zero else clipped Normal(0,10deg) at+/-20deg; only current query left rotated Ry(-delta), estimated historical sensor poses/photons unchanged'),
        evaluation=dict(calibration_units=CAL,evaluation_units=EVAL,
            cal_contacts=23,eval_contacts=47,cal_controls=65,eval_controls=224,
            conditions=['exact','E1'],configs=['single','dual'],main_timely='First any main alarm outputs2..deadline inclusive; startup any-before separately',
            cost='Primary same integer floor(.025*10*eval_control_clips) alarm intervals in outputs[2:12], >=whole ties, lowest feasible threshold selected from eval clear only, no contact objective.10*.2=2.0 proxy seconds/clear clip. Not deployment calibration or equal episode/audio cost.',
            segments='Also report maximal consecutive-positive main-output runs reset per2s clip, and merged clip episode(any alarm); neither equals interval count or real session rate',
            calibration='Secondary floor(.025*10*cal_control_clips) whole-tie intervals; freeze theta then report actualevalFA, same calibration target only',
            guard='Per configuration exact Aug vs frozen_M3_5 paired timely losses<=2 and net>=-2 at same actual interval cost. Absolute2event investment tolerance, not safety/noninferiority.',
            rule='Per single/dual independently: all applicable workpoints residual0; Aug E1 timely strictly>Control E1 and>frozen_M3_5 E1, exact guard passes. Minimum strict gain1event. Otherwise STOP_THIS_RECIPE; no second seed, epoch/distribution tuning or automatic broader experiment.'),
        budget_wall_seconds=dict(preflight_benchmark=120,gpu_pipeline=3600,cpu_summary_verification=300),
        decision_check='24 paired eval source units/47contacts/224controls; cal8units/23contacts/65controls from inherited consumed Development ledger before new prediction, never native-motion229 nor480batch1002. One event2.128pp is minimum strict increment; exact retention may lose2(4.255pp) intentionally. Historical direction loss supplies nonzero headroom; initial new working-point E1 count not read. Ensemble comparison practical unequal averaging, onlyAug-v-Control isolates training perturbation inputs; first screen makes no mechanism claim. Benchmark144unit projection9665s exceeded proposed3600s; pre-result scale reduced to8cal+24eval, predicted projection2148s plus~443s nonzero training projection, matched fits/inference/I/O have remaining~1009s. Small cohort limits precision.',
        observation_contract='Zero delta directly reuses original cached FP16 voxel bytes; nonzero original FP64 projector geometry/FP32 sequential history sums/FP16 cache, representative zero recomputation parity. No alternative FP32fused operator, new photons, label rotation or truth-selected augmentation.',
        adjustable='Implementation and execution corrections within cumulative budget only; frozen seed, distribution, data, epochs, thresholds and decision remain fixed',
        stop='Complete this one screen or cumulative phase budget/unresolved source/parity failure. Preserve failures, partial voxels and epoch checkpoints; no algorithm ranking for incomplete run, no added seeds or retuning on failure.',
        boundaries='No BlindWays or HEADS-UP in this fit/selection;480confirmation retains prior identity. Frozen M3/phone retained; no device/user claims, mechanism inference, historical-exact-pose ablation, direction-label tuning orL3 work.',
        delivery='Matched models/row orders, saved raw logits beforetruth join, event/cost/paired ledger, focused checks, report/currents/RUNS/master push',
        hashes={p.relative_to(ROOT).as_posix():A.sha(p) for p in paths})
    if (OUT/'PLAN.json').exists():
        prior=A.read(OUT/'PLAN.json')
        # A repair may change source, but never silently rewrite frozen recipe.
        if {k:v for k,v in prior.items() if k!='hashes'}!={k:v for k,v in plan.items() if k!='hashes'}:
            raise ValueError('Frozen recipe changed')
        return prior
    save(OUT/'PLAN.json',plan)
    (OUT/'PLAN.sha256').write_text(A.sha(OUT/'PLAN.json')+'\n',encoding='utf8')
    for source in sources:
        dest=OUT/'source'/source.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(source.read_bytes())
    return plan


def prepare(projector,check):
    old_arrays,meta=NR.load_split('train');n=sum(len(x) for x in old_arrays)
    assert n==27456
    with np.load(LABELS) as z:labels=z['M3'].astype(np.float32)
    with np.load(PARENT/'rows.npz') as z:md={k:z[k] for k in ('unit','config','frame','domain')}
    lookup={(int(u),int(c),int(f)):i for i,(u,c,f,d) in enumerate(zip(md['unit'],md['config'],md['frame'],md['domain'])) if d==0}
    ids=np.array([lookup[int(u),int(c),int(f)] for u,c,f in zip(meta['unit'],meta['config'],meta['frame'])])
    histories=np.load(PARENT/'histories.npy',mmap_mode='r');matrices=np.load(PARENT/'transforms.npy',mmap_mode='r');length=np.load(PARENT/'length.npy',mmap_mode='r')
    deltas=np.array([delta_for(u,c) for u,c in zip(meta['unit'],meta['config'])]);lens=np.asarray(length[ids])
    folder=OUT/'inputs';folder.mkdir(exist_ok=True)
    oldpath=folder/'Control.npy';newpath=folder/'Aug.npy';shape=(n,3,24,17,33)
    if (folder/'complete.json').exists():
        return oldpath,newpath,labels
    old=np.lib.format.open_memmap(oldpath,mode='w+',dtype=np.float16,shape=shape)
    new=np.lib.format.open_memmap(newpath,mode='w+',dtype=np.float16,shape=shape)
    offset=0
    for values in old_arrays:old[offset:offset+len(values)]=values;offset+=len(values)
    zero=deltas==0;new[zero]=old[zero]
    completed=int(zero.sum());start=time.monotonic()
    for L in np.unique(lens):
        selected=np.flatnonzero((lens==L)&~zero)
        for begin in range(0,len(selected),4):
            check();row=selected[begin:begin+4];parent=ids[row]
            z=np.array(histories[parent,-int(L):],copy=True);transforms=np.array(matrices[parent,-int(L):],copy=True)
            moved=perturb_matrices(transforms,deltas[row]);values=projector.project(z,moved)
            new[row]=values.cpu().numpy();completed+=len(row)
            if completed%256<4:
                new.flush();progress(dict(stage='train_projection',completed=completed,total=n,seconds=time.monotonic()-start))
        print('TRAIN_PROJECTION history',L,'completed',completed,'/',n,'seconds',round(time.monotonic()-start,2),flush=True)
    old.flush();new.flush();np.testing.assert_array_equal(new[zero],old[zero])
    save(folder/'complete.json',dict(status='COMPLETE',rows=n,zero_rows=int(zero.sum()),seconds=time.monotonic()-start,
        Control_sha256=A.sha(oldpath),Aug_sha256=A.sha(newpath),labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
        order_sha256=hashlib.sha256(np.stack([meta[k] for k in ('unit','config','frame')],-1).tobytes()).hexdigest(),
        histories_source='Original cached temporal natural histories; only originalNR train rows',zero_input_bitwise_equal=True))
    Bsave=dict(unit=meta['unit'],config=meta['config'],frame=meta['frame'],delta=deltas,labels=labels,parent_row=ids)
    np.savez_compressed(folder/'rows.npz',**Bsave)
    del old,new,histories,matrices;gc.collect();torch.cuda.empty_cache()
    return oldpath,newpath,labels


def fit(arm,path,labels,check):
    dest=OUT/'models'/arm;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    data=np.load(path,mmap_mode='r');x=torch.from_numpy(np.array(data)).cuda();y=torch.from_numpy(labels).cuda();del data
    net,_=V.network('V',SEED,'cuda');opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS);rng=np.random.default_rng(SEED);history=[];start=time.monotonic()
    state=dest/'resume.pt';begin=0
    if state.exists():
        previous=torch.load(state,map_location='cuda',weights_only=False);net.load_state_dict(previous['net']);opt.load_state_dict(previous['opt']);scheduler.load_state_dict(previous['scheduler']);rng.bit_generator.state=previous['rng'];history=previous['history'];begin=previous['epoch']
    try:
        for epoch in range(begin,EPOCHS):
            check();tick=time.monotonic();order=rng.permutation(len(x));total=0.;updates=0;net.train()
            for i in range(0,len(order),BATCH):
                check();rows=np.sort(order[i:i+BATCH]);xb=x[torch.as_tensor(rows,device='cuda')];target=y[torch.as_tensor(rows,device='cuda')]
                opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(net(M.prepare_voxels(xb)),target)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();opt.step();total+=float(loss.detach())*len(rows);updates+=1
            scheduler.step();history.append(dict(epoch=epoch+1,updates=updates,rows=len(x),loss=total/len(x),seconds=time.monotonic()-tick,order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
            checkpoint(state,dict(net=net.state_dict(),opt=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng.bit_generator.state,epoch=epoch+1,history=history))
            progress(dict(stage='fit_'+arm,completed=epoch+1,total=EPOCHS,history=history));print('FIT',arm,history[-1],flush=True)
        final=dest/'model.pt';checkpoint(final,{k:v.detach().cpu() for k,v in net.state_dict().items()})
        save(dest/'complete.json',dict(status='COMPLETE',arm=arm,seed=SEED,history=history,seconds=time.monotonic()-start,model_sha256=A.sha(final),baseline_sha256=A.sha(A.M3_MODELS[0]),input_sha256=A.sha(path)))
    finally:
        del x,y,net,opt,scheduler;gc.collect();torch.cuda.empty_cache()


def predict(projector,check):
    nets=[]
    for path in [*A.M3_MODELS,OUT/'models/Control/model.pt',OUT/'models/Aug/model.pt']:
        n=M.CVR().cuda().eval();n.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));nets.append(n)
    costs=[]
    try:
        for u in CAL+EVAL:
            check();dest=OUT/'units'/f'unit{u}.npz'
            if dest.exists():continue
            tick=time.monotonic();obs,stored=A.stored(u)
            try:
                noisy=obs['noisy_center'];C=len(noisy);z=obs['z1'];public=np.repeat(obs['public_query'][None],C,0)
                e1,_=Q.queries(obs,'est');raw=np.empty((4,2,3,C,13,2),np.float32)
                for qi,query in enumerate((public,e1)):
                    for branch,angle in enumerate(A.ANGLES):
                        ex=R.extrinsic(angle);nn=noisy@ex;qq=query@ex
                        for output,f in enumerate(range(3,16)):
                            ix=np.arange(max(0,f-7),f+1)
                            for b in range(0,C,4):
                                check();pose=nn[b:b+4];q=qq[b:b+4,f]
                                transforms=(q@np.linalg.inv(pose[:,f]))[:,None]@pose[:,ix]
                                vox=projector.project(z[branch,b:b+4][:,ix],transforms)
                                with torch.inference_mode():
                                    x=M.prepare_voxels(vox);pred=[n(x).float() for n in nets]
                                    values=torch.stack([torch.stack(pred[:5]).mean(0),pred[0],pred[5],pred[6]])
                                raw[:,qi,branch,b:b+4,output]=values.cpu().numpy()
                check();dest.parent.mkdir(exist_ok=True);tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,unit=u,raw=raw);tmp.replace(dest)
                costs.append(time.monotonic()-tick);progress(dict(stage='predict',completed=sum(1 for _ in (OUT/'units').glob('unit*.npz')),total=len(CAL+EVAL),unit=u,seconds_per_last_unit=costs[-1]))
                print('PREDICT',u,round(costs[-1],2),'s',flush=True)
            finally:obs.close();stored.close()
    finally:
        del nets;gc.collect();torch.cuda.empty_cache()


def analyze():
    from cnh_query_perturb_evaluate_dev import evaluate,load_metadata
    rows=A.read(A.R3/'rows.json');index=[i for i,r in enumerate(rows) if r['unit'] in CAL+EVAL];selected=[rows[i] for i in index]
    assert len(selected)==1280
    raw=np.empty((4,2,3,1280,13,2),np.float32)
    for u in CAL+EVAL:
        at=np.array([i for i,r in enumerate(selected) if r['unit']==u]);cfg=np.array([selected[i]['config'] for i in at])
        with np.load(OUT/'units'/f'unit{u}.npz') as z:raw[:,:,:,at]=z['raw'][:,:,:,cfg]
    # Persist model outputs before evaluator-only metadata is joined.
    np.savez_compressed(OUT/'raw.npz',raw=raw,unit=np.array([r['unit'] for r in selected]),config=np.array([r['config'] for r in selected]))
    metadata=load_metadata(selected)
    result=evaluate(raw,metadata,MODELS);save(OUT/'result.json',result)
    np.savez_compressed(OUT/'metadata.npz',**metadata)
    print('SCREEN_RESULT',{k:v['decision'] for k,v in result['sensors'].items()},flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=True);plan=freeze();V.configure('cuda')
    started=time.monotonic();receipts=OUT/'attempts';receipts.mkdir(exist_ok=True)
    spent=sum(A.read(p)['seconds'] for p in receipts.glob('gpu*.json'));status='FAILED';error=None
    def check():
        if spent+time.monotonic()-started>=3600:raise TimeoutError('Cumulative3600s GPU pipeline cap')
    try:
        for p,h in plan['hashes'].items():
            if A.sha(ROOT/p)!=h:raise ValueError('Frozen source/input changed:'+p)
        if not (OUT/'backend.json').exists():
            save(OUT/'backend.json',dict(device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),pid=os.getpid(),dtype='FP64 geometry,FP32 accumulation/model/loss,FP16 rawcache;TF32off'))
        projector=QueryPerturbProjector();old,new,labels=prepare(projector,check);del projector;torch.cuda.empty_cache()
        fit('Control',old,labels,check);fit('Aug',new,labels,check)
        a=A.read(OUT/'models/Control/complete.json');b=A.read(OUT/'models/Aug/complete.json')
        assert [(h['order_sha256'],h['updates'],h['rows']) for h in a['history']]==[(h['order_sha256'],h['updates'],h['rows']) for h in b['history']]
        projector=QueryPerturbProjector();predict(projector,check);del projector;check();status='COMPLETE'
    except BaseException as exc:error=repr(exc);raise
    finally:
        gc.collect();torch.cuda.empty_cache();save(receipts/f'gpu{time.time_ns()}.json',dict(status=status,error=error,seconds=time.monotonic()-started,previous_seconds=spent,cap_seconds=3600))
    cpu_start=time.monotonic();cpu_status='FAILED'
    try:
        analyze();cpu_status='COMPLETE' if time.monotonic()-cpu_start<=300 else 'BUDGET_EXCEEDED'
    finally:
        save(receipts/f'cpu{time.time_ns()}.json',dict(status=cpu_status,seconds=time.monotonic()-cpu_start,cap_seconds=300))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:OUT.mkdir(parents=True,exist_ok=True);freeze()
    else:run()
