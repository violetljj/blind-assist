"""New backbone/S task-cost evaluation: observation-only scores, sealed negative cuts.

The consumed dorm recording is deliberately absent from this module. All public
histories are masked before CVR, local patches, native tokens and S spatial features.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import json
import pickle
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
import cnh_counterfactual_common_dev as C
import cnh_counterfactual_data_dev as D
import cnh_counterfactual_eval_dev as E
import cnh_bar_fusion_probe_dev as B
import cnh_frozen_e2e_tof_20261010 as F
import cnh_task_cost_train_20261010 as T
import cnh_task_cost_metrics_20261010 as TM
import cnh_s_ensemble_tradeoff_metrics_20261010 as V
import cnh_cost_v2_metrics_20261010 as CM
import cnh_frozen_e2e_metrics_20261010 as FM
import cnh_graded_corridor_eval_dev as Corr
import cnh_no_bin0_dr_source_20261011 as SOURCE

OUT=C.ROOT/'artifacts.local/work/cnh-no-bin0-dr-retrain-dev-20261011'
POOL=('cnh-cost-v2-holdout-dev-20261010','cnh-task-cost-retrain-dev-20261010','cnh-s-ensemble-confirm-dev-20261010')
ARMS=('no_bin0','no_bin01')
CHECK=lambda:None

def paths(arm):
    folder=OUT/'models'/arm
    m3=sorted((folder/'M3').glob('model_seed*.pt'))
    ordinary=sorted(folder.glob('ordinary_seed*.pt'))
    if not m3 or not ordinary:raise FileNotFoundError('Missing completed backbone '+arm)
    return m3,ordinary

def infer(z,sensor,query,arms=ARMS,include_old=False,check=lambda:None):
    """z[scene,K,16,8,8,16]; return original float16 feature-cache contracts."""
    import cnh_aligned_boundary_dev as A
    import cnh_bar_local_readout_dev as L
    import cnh_counterfactual_train_dev as CT
    import cnh_boundary_token_model as M
    import cnh_graded_peak_tracks_dev as P
    import cnh_graded_peak_joint_features_dev as PF
    active,_,_,_=A.imports();active.OUT=OUT/'runtime'
    engine=active.Engine();torch=engine.torch
    n,k=z.shape[:2];count=n*k;flat=np.asarray(z).reshape(count,16,8,8,16)
    models={};results={};names=list(arms)+(['old'] if include_old else [])
    began=time.monotonic()
    try:
        for arm in names:
            if arm=='old':mp=[C.ROOT/f'artifacts.local/work/cnh-margin-labels-20261002/models/M3/model_seed{s}.pt' for s in range(5)];op=[F.OLD/f'models/ordinary_seed{s}.pt' for s in T.SEEDS]
            else:mp,op=paths(arm)
            cvrs=[];ordinary=[]
            for path in mp:
                from cnh_cvr_pilot import CVR
                model=CVR().cuda().eval();state=torch.load(path,map_location='cpu',weights_only=True)
                model.load_state_dict(state.get('state_dict',state));cvrs.append(model)
            for path in op:
                model=M.BoundaryTokenReadout().cuda().eval();state=torch.load(path,map_location='cpu',weights_only=True)
                model.load_state_dict(state['state_dict']);ordinary.append(model)
            while len(ordinary)<3:ordinary.append(ordinary[0])
            models[arm]=(cvrs,ordinary)
            results[arm]=dict(m3_raw=np.empty((count,13,2),np.float32),local_raw=np.empty((count,13,2),float),ordinary_raw=np.empty((3,count,13,2),np.float32))
        qs,locations=L.patches()
        with torch.inference_mode():
            for fj,f in enumerate(range(3,16)):
                check();start=max(0,f-7);tr=(query[f]@np.linalg.inv(sensor[f]))[None]@sensor[start:f+1]
                projection=L.Projection(engine,tr);norms=[]
                for patch in qs:
                    mapped=patch@projection.p;variance=np.asarray(mapped.multiply(mapped).sum(1)).ravel()
                    norms.append(np.where(variance>0,np.sqrt(variance),np.inf))
                length=len(tr);padded=np.repeat(np.eye(4)[None],8,0);padded[-length:]=tr
                for offset in range(0,count,64):
                    check();stop=min(count,offset+64);b=stop-offset
                    transforms=torch.as_tensor(np.broadcast_to(padded,(b,8,4,4)).copy(),device='cuda',dtype=torch.float32)
                    lengths=torch.full((b,),length,device='cuda',dtype=torch.int64)
                    geometry=M.feature_geometry(transforms,lengths,'center')
                    for arm in names:
                        drop=[] if arm=='old' else ([0] if arm=='no_bin0' else [0,1])
                        native=flat[offset:stop,start:f+1].copy();native[...,drop]=0
                        voxels=projection(native).cpu().numpy().astype(np.float16)
                        cvrs,ordinary=models[arm];engine.nets=cvrs
                        results[arm]['m3_raw'][offset:stop,fj]=engine.predict(voxels)
                        results[arm]['local_raw'][offset:stop,fj]=L.scan(voxels,qs,norms,locations)[0]
                        history=np.zeros((b,8,8,8,16),np.float16);history[:,-length:]=native
                        h=torch.as_tensor(history,device='cuda')
                        compact=M.build_features(h,geometry,lengths)[:,:,:,list(CT.KEPT_CHANNELS)].half()
                        features=CT.restore_features(compact)
                        for si,model in enumerate(ordinary):results[arm]['ordinary_raw'][si,offset:stop,fj]=model(features,lengths).cpu().numpy()
                del projection
                print('NO_BIN_INFER',','.join(names),f,round(time.monotonic()-began,2),flush=True)
        for arm in names:
            check();native=flat.copy();native[...,([] if arm=='old' else [0] if arm=='no_bin0' else [0,1])]=0
            candidates=P.current_candidates(native,P.native_geometry(sensor,query))
            current,valid=PF.build(dict(candidate_amplitude=candidates['amplitude'],candidate_valid=candidates['valid'],candidate_inner_share=candidates['inner_share'],candidate_xyz=candidates['xyz'],candidate_support_low_xyz=candidates['support_low_xyz'],candidate_support_high_xyz=candidates['support_high_xyz']))
            result=results[arm]
            for key in ('m3_raw','local_raw'):result[key]=result[key].reshape(n,k,13,2)
            result['ordinary_raw']=result['ordinary_raw'].reshape(3,n,k,13,2)
            result['current']=current.reshape(n,k,13,2,22)
            result['current_valid']=valid.reshape(n,k,13,2,22)
        return results
    finally:
        models.clear();engine.nets.clear()
        if hasattr(engine,'_dll'):engine._dll.close()
        torch.cuda.empty_cache()

def normalized(folder,ids=None):
    import cnh_aligned_boundary_dev as A
    *_,normal=A.imports()
    hist=np.load(Path(folder)/'hist.npy',mmap_mode='r')
    if ids is not None:hist=hist[ids]
    ambient_path=Path(folder)/'ambient.npy'
    if ambient_path.exists():ambient=np.load(ambient_path)
    else:
        with np.load(Path(folder)/'physics.npz') as physics:ambient=physics['ambient']
    n,k=hist.shape[:2]
    ambient=np.asarray(ambient)
    if ambient.shape==(n,k,16,8,8):ambient=ambient.reshape(n*k,16,8,8)
    elif ambient.shape!=(n*k,16,8,8):ambient=np.broadcast_to(ambient,(n*k,16,8,8))
    return normal(hist.reshape(n*k,16,8,8,16),ambient).reshape(n,k,16,8,8,16)

def score_folder(folder,output,arms=ARMS,ids=None,include_old=False,check=lambda:None):
    folder=Path(folder);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with np.load(folder/'geometry.npz') as g:sensor,query=g['sensor'],g['public_query']
    result=infer(normalized(folder,ids),sensor,query,arms,include_old,check)
    for arm,scores in result.items():np.savez_compressed(output/(arm+'_scores.npz'),**scores)
    return result

def strong(scores,cuts):
    return (2*((E.smooth(scores['m3_raw'])>=cuts['raised_tau'])|(E.smooth(scores['local_raw'])>=cuts['local_tau']))).astype(np.int8)

def s_features(scores,cuts):
    current=scores['current'].astype(float);valid=scores['current_valid']&np.isfinite(current)
    spatial=np.concatenate((np.where(valid,current,np.nan),(~valid).astype(float)),-1)
    raw=scores['ordinary_raw']
    return np.array([np.concatenate((Corr.build_score_features(raw[si],E.smooth(raw[si]),cuts['single'][si]),spatial),-1) for si in range(3)],np.float32)

def calibrate_backbones(old,new,category):
    clear=(category=='clear').all(1);oldm=E.smooth(old['m3_raw']);oldl=E.smooth(old['local_raw'])
    target_m=int((oldm[clear]>=E.M3_THETA).any(-1).sum())
    target_raise=int((oldm[clear]>=E.OLD_RAISED).any(-1).sum())
    target_fusion=int(E.old_fusion(oldm,oldl)[clear].any(-1).sum());cuts={}
    for arm,scores in new.items():
        m,l=E.smooth(scores['m3_raw']),E.smooth(scores['local_raw'])
        mt,mc=B.nearest(m[clear].max(-1),target_m)
        rt,rc=B.nearest(m[clear].max(-1),target_raise,floor=mt)
        already=(m[clear]>=rt).any(-1)
        lt=E.OLD_LOCAL
        lc=int(((l[clear]>=lt).any(-1)&~already).sum())
        single=[E.threshold(E.calibrate(oldm,oldl,E.smooth(scores['ordinary_raw'][si]),category)['standalone']) for si in range(3)]
        cuts[arm]=dict(m3_tau=mt,raised_tau=rt,local_tau=lt,single=single,
            same_clear_slot_targets=dict(m3=target_m,raised=target_raise,old5=target_fusion),
            actual_clear_slots=dict(m3=mc,raised=rc,old5=rc+lc),rule='M3/raised whole-score-tie same-clear-cost rule, higher threshold on equal residual; analytic local cut retained; no contact or real selection')
    return cuts

def fit_s(train_scores,cuts,rows,category,support):
    for arm in ARMS:
        output=OUT/arm;output.mkdir(exist_ok=True)
        grades=strong(train_scores[arm],cuts[arm])
        labels,mask,weights,counts=T.supervision(category,support,grades>0,[r['lateral_gap_m'] for r in rows])
        feats=s_features(train_scores[arm],cuts[arm])
        np.save(output/'S_train_features.npy',feats)
        np.savez_compressed(output/'S_train_labels.npz',labels=labels,mask=mask,weights=weights)
        T.train_s(output,feats,labels,weights,CHECK)
        C.save(output/'supervision_receipt.json',counts)

def seal(pool_scores,cuts,rows,category,old5):
    masks,_,_,_=CM.negative_masks(category,rows);assert masks['all_negative'].all()
    den=int(masks['far_pass_plus_clear'].sum()*old5.shape[1])
    oldnotice=FM.replay_gap1(old5);base=int((oldnotice[masks['far_pass_plus_clear']]>0).any((2,3)).sum())
    for arm in ARMS:
        scores=T.predict_s(OUT/arm,s_features(pool_scores[arm],cuts[arm])).mean(0)
        newstrong=strong(pool_scores[arm],cuts[arm])
        taus,stats,ties=V.sweep(newstrong,scores,category,rows,CHECK,True)
        # The cap uses the same original old5 negatives as original S tradeoff.
        feasible=np.flatnonzero(stats[:,8]<=base+.03*den)
        if len(feasible):chosen=int(feasible[0]);status='SEALED'
        else:chosen=len(taus)-1;status='STRONG_ALREADY_EXCEEDS_POOLED_CAP'
        cuts[arm]['s_tau']=float(taus[chosen]) if np.isfinite(taus[chosen]) else (1e308 if taus[chosen]>0 else -1e308)
        cuts[arm]['s_tau_record']=TM.threshold_record(taus[chosen])
        cuts[arm]['s_calibration']=dict(status=status,old5_far_clear_clips=base,far_clear_denominator=den,
            cap_clips=base+.03*den,selected_far_clear_clips=int(stats[chosen,8]),ties=ties,negative_only=True)
        np.savez_compressed(OUT/arm/'pool_light_scores.npz',score=scores,strong=newstrong)
    C.save(OUT/'sealed_calibration.json',dict(cuts=cuts,pool_scenes=len(rows),pool_clip_denominator=len(rows)*old5.shape[1],old5_far_clear_clips=base,far_clear_denominator=den,rule='Lowest whole-tie pooled-negative tau <= originalold5+3pp; no contact utility',sealed_before_fresh_hold=True))
    return cuts

def describe(grade,baseline,category,rows):
    notice=FM.replay_gap1(grade);base=FM.replay_gap1(baseline)
    masks,near,_,_=CM.negative_masks(category,rows)
    first=CM.first_notice(notice);bfirst=CM.first_notice(base)
    hit=(first>=0)&(first<11);bhit=(bfirst>=0)&(bfirst<11)
    output={}
    for q,name in enumerate(('HEAD','BODY')):
        contact=category[:,q]=='contact';a,b=bhit[contact,:,q],hit[contact,:,q]
        output[name]=dict(denominator=int(a.size),timely=int(b.sum()),baseline_timely=int(a.sum()),rescue=int((~a&b).sum()),loss=int((a&~b).sum()),net=int(b.sum()-a.sum()))
    joint=notice.max(-1);fc=masks['far_pass_plus_clear'];neg=masks['all_negative']
    weight=np.where(near[:,None,None],.25,1.)
    output['far_clear']=dict(notified_clips=int((joint[fc]>0).any(-1).sum()),denominator=int(fc.sum()*grade.shape[1]))
    output['far_clear']['rate']=output['far_clear']['notified_clips']/output['far_clear']['denominator']
    output['weighted_cost']=float((((joint==1)*weight+(joint==2))*neg[:,None,None]).sum())
    output['groups']={}
    for name,mask in [('dark_thin',np.array([r['dark_thin'] for r in rows])),('sign_edge',np.array([r['shape_family']=='sign_edge' for r in rows]))]:
        output['groups'][name]=[dict(denominator=int((mask&(category[:,q]=='contact')).sum()*grade.shape[1]),timely=int(hit[mask&(category[:,q]=='contact'),:,q].sum()),baseline=int(bhit[mask&(category[:,q]=='contact'),:,q].sum())) for q in range(2)]
    return output,notice

def read_scores(folder):
    return {arm:{key:value.copy() for key,value in np.load(Path(folder)/(arm+'_scores.npz')).items()} for arm in ARMS}

def backbone_cal():
    src=F.OLD/'data/cal';result=score_folder(src,OUT/'backbone_cal_scores',check=CHECK)
    with np.load(F.OLD/'baselines/cal_ideal.npz') as a:old={key:a[key].copy() for key in ('m3_raw','local_raw')}
    with np.load(src/'geometry.npz') as g:category=g['category']
    cuts=calibrate_backbones(old,result,category)
    for cut in cuts.values():
        for key in ('m3_tau','raised_tau','local_tau'):cut[key]=float(cut[key]) if np.isfinite(cut[key]) else 1e308
        cut['single']=[float(t) if np.isfinite(t) else 1e308 for t in cut['single']]
    C.save(OUT/'backbone_calibration.json',cuts)

def s_train_scores():
    score_folder(OUT.parent/'cnh-task-cost-retrain-dev-20261010/data/train',OUT/'S_train_scores',check=CHECK)

def pool_scores():
    rows=[];blocks={arm:[] for arm in ARMS};oldblocks=[];categories=[]
    for dataset in POOL:
        source=OUT.parent/dataset;allrows=C.read(source/'scene_rows.json')
        for split in ('cal','hold'):
            folder=source/'data'/split
            ids=np.array([i for i,row in enumerate(allrows[split]) if row['placement'] in ('pass','clear')])
            target=OUT/'pool_scores'/dataset/split
            if (target/'no_bin0_scores.npz').exists() and (target/'no_bin01_scores.npz').exists():result=read_scores(target)
            else:result=score_folder(folder,target,ids=ids,check=CHECK)
            with np.load(folder/'geometry.npz') as g:categories.append(g['category'][ids])
            with np.load(folder/'scores.npz') as old:oldblocks.append((2*E.old_fusion(E.smooth(old['m3_raw'][ids]),E.smooth(old['local_raw'][ids]))).astype(np.int8))
            for arm in ARMS:blocks[arm].append(result[arm])
            rows.extend(dict(allrows[split][i],source_dataset=dataset,source_split=split) for i in ids)
    for arm in ARMS:
        joined={key:np.concatenate([b[key] for b in blocks[arm]],axis=1 if key=='ordinary_raw' else 0) for key in blocks[arm][0]}
        np.savez_compressed(OUT/(arm+'_pool_scores.npz'),**joined)
    np.savez_compressed(OUT/'pool_metadata.npz',category=np.concatenate(categories),old5=np.concatenate(oldblocks))
    C.save(OUT/'pool_rows.json',rows)

def fit_s_stage():
    source=OUT.parent/'cnh-task-cost-retrain-dev-20261010'
    with np.load(source/'data/train/geometry.npz') as g:category,support=g['category'],g['target_support']
    fit_s(read_scores(OUT/'S_train_scores'),C.read(OUT/'backbone_calibration.json'),C.read(source/'scene_rows.json')['train'],category,support)

def seal_stage():
    pool={arm:{key:value.copy() for key,value in np.load(OUT/(arm+'_pool_scores.npz')).items()} for arm in ARMS}
    with np.load(OUT/'pool_metadata.npz') as a:category,old5=a['category'],a['old5']
    cuts=seal(pool,C.read(OUT/'backbone_calibration.json'),C.read(OUT/'pool_rows.json'),category,old5)
    manifests={};hashes={};allmodels={}
    def entry(path):
        rel=C.logical_path(path) if hasattr(C,'logical_path') else str(path.relative_to(C.ROOT)).replace('\\','/')
        digest=C.sha(path);allmodels[rel]=digest;return dict(path=rel,sha256=digest)
    for arm,key in zip(ARMS,('main','drop01')):
        m3,ordinary=paths(arm)
        while len(ordinary)<3:ordinary.append(ordinary[0])
        manifest=dict(candidate=arm,drop_bins=[0] if arm=='no_bin0' else [0,1],
            m3_paths=[entry(p) for p in m3],ordinary_paths=[entry(p) for p in ordinary],
            s_paths=[[entry(OUT/arm/'models'/f'S_{seed}_{height}.pickle') for height in ('HEAD','BODY')] for seed in T.SEEDS],cuts=cuts[arm],
            development_only=True,real_selection=False,hold_selection=False)
        path=OUT/(key+'_real_manifest.json');C.save(path,manifest);manifests[key]=str(path);hashes[key]=C.sha(path)
    here=Path(__file__).parent;host=C.ROOT/'research/active/hardware-bringup/host'
    science=[Path(__file__),Path(SOURCE.__file__),here/'cnh_no_bin0_dr_train_20261011.py',
             host/'cnh_no_bin0_dr_replay.py',host/'cnh_no_bin0_dr_real.py']
    science += [Path(module.__file__) for name,module in sys.modules.items()
                if name.startswith('cnh_') and getattr(module,'__file__',None)]
    def logical(path):
        path=Path(path)
        if path.is_relative_to(C.ROOT):return path.relative_to(C.ROOT).as_posix()
        return 'artifacts.local/'+path.resolve().relative_to((C.ROOT/'artifacts.local').resolve()).as_posix()
    code_hashes={logical(path):C.sha(path) for path in sorted(set(science),key=str)}
    oldmanifest=C.read(OUT.parent/'cnh-s-ensemble-tradeoff-dev-20261010/candidate_manifest.json')
    oldbindings=oldmanifest['models']+oldmanifest['features']['implementation']+oldmanifest['strong_source']['frozen_dependencies']
    oldhashes={row['path']:row['sha256'] for row in oldbindings}
    for path,digest in oldhashes.items():assert C.sha(C.ROOT/path)==digest,path
    C.save(OUT/'sealed_candidates.json',dict(real_manifest_hashes=hashes,real_manifests=manifests,models_sha256=allmodels,
        scientific_code_sha256=code_hashes,old_frozen_dependencies_sha256=oldhashes,
        calibration_sha256=C.sha(OUT/'sealed_calibration.json'),plan_sha256=C.sha(OUT/'PLAN.json'),
        plan_markdown_sha256=C.sha(OUT/'PLAN.md'),sealed_before_fresh_hold=True))

def verify_seal():
    sealdata=C.read(OUT/'sealed_candidates.json')
    for mapping in ('models_sha256','scientific_code_sha256','old_frozen_dependencies_sha256'):
        for path,digest in sealdata[mapping].items():assert C.sha(C.ROOT/path)==digest,path
    for arm,path in sealdata['real_manifests'].items():assert C.sha(path)==sealdata['real_manifest_hashes'][arm],arm
    for filename,key in (('sealed_calibration.json','calibration_sha256'),('PLAN.json','plan_sha256'),('PLAN.md','plan_markdown_sha256')):
        assert C.sha(OUT/filename)==sealdata[key],filename
    return sealdata

def hold_render():
    verify_seal()
    SOURCE.prepare(OUT)
    F.S=SimpleNamespace(SPLITS=('hold',),K=SOURCE.K,sampling_seed=SOURCE.sampling_seed)
    result=F.render(OUT,{'hold':C.read(OUT/'scene_rows.json')['hold']},CHECK)
    C.save(OUT/'hold_render_receipt.json',result)

def hold_scores():
    verify_seal()
    score_folder(OUT/'data/hold',OUT/'hold_scores',include_old=True,check=CHECK)

def evaluate():
    verify_seal()
    rows=C.read(OUT/'scene_rows.json')['hold'];cuts=C.read(OUT/'sealed_calibration.json')['cuts']
    with np.load(OUT/'data/hold/geometry.npz') as a:category=a['category']
    with np.load(OUT/'hold_scores/old_scores.npz') as a:old={key:a[key].copy() for key in a.files}
    oldstrong=(2*E.old_fusion(E.smooth(old['m3_raw']),E.smooth(old['local_raw']))).astype(np.int8)
    singlecuts=C.read(F.THRESHOLDS)
    oldcuts=dict(single=[singlecuts[str(seed)]['single'] for seed in T.SEEDS])
    olds=T.predict_s(OUT.parent/'cnh-task-cost-retrain-dev-20261010',s_features(old,oldcuts)).mean(0)
    oldmanifest=C.read(OUT.parent/'cnh-s-ensemble-tradeoff-dev-20261010/candidate_manifest.json')
    grades={'original_old5':oldstrong,'old_default_S':TM.apply_gate(oldstrong,olds,oldmanifest['threshold']['tau'])}
    for arm,scores in read_scores(OUT/'hold_scores').items():
        stronggrade=strong(scores,cuts[arm]);light=T.predict_s(OUT/arm,s_features(scores,cuts[arm])).mean(0)
        grades[arm+'_old5']=stronggrade;grades[arm+'_S']=TM.apply_gate(stronggrade,light,cuts[arm]['s_tau'])
        grades[arm+'_M3']=(2*(E.smooth(scores['m3_raw'])>=cuts[arm]['m3_tau'])).astype(np.int8)
    stats={};notices={}
    for arm,grade in grades.items():stats[arm],notices[arm]=describe(grade,oldstrong,category,rows)
    oldgain=sum(stats['old_default_S'][q]['net'] for q in ('HEAD','BODY'))
    newgain=sum(stats['no_bin0_S'][q]['net'] for q in ('HEAD','BODY'))
    rateincrement=stats['no_bin0_S']['far_clear']['rate']-stats['old_default_S']['far_clear']['rate']
    acceptance=dict(old_gain_vs_common_old5=oldgain,new_gain_vs_common_old5=newgain,
        timely_retention_rule='newnet>=0.8*oldnet ifoldnet>0; else newtimely>=oldtimely',
        timely_pass=bool(newgain>=.8*oldgain if oldgain>0 else newgain>=oldgain),
        far_clear_increment_pp=100*rateincrement,far_clear_pass=bool(rateincrement<=.01+1e-12))
    stats['acceptance']=acceptance
    stats['new_vs_old_paired']=describe(grades['no_bin0_S'],grades['old_default_S'],category,rows)[0]
    stats['new_vs_old_clustered_bootstrap']=V.clustered_bootstrap(notices['no_bin0_S'],notices['old_default_S'],category,rows,replicates=2000)
    np.savez_compressed(OUT/'hold_grades_notifications.npz',keys=np.array(list(grades)),grades=np.array(list(grades.values())),notifications=np.array([notices[key] for key in grades]),category=category)
    C.save(OUT/'simulation_evaluation.json',stats)
    table=[]
    for arm in grades:
        for q in ('HEAD','BODY'):
            table.append(dict(arm=arm,height=q,**stats[arm][q],far_clear_notified_clips=stats[arm]['far_clear']['notified_clips'],
                far_clear_clip_denominator=stats[arm]['far_clear']['denominator'],far_clear_clip_rate=stats[arm]['far_clear']['rate'],weighted_cost=stats[arm]['weighted_cost']))
    with (OUT/'simulation_comparison.csv').open('x',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    groups=[]
    for arm in grades:
        for group,values in stats[arm]['groups'].items():
            for q,value in zip(('HEAD','BODY'),values):groups.append(dict(arm=arm,group=group,height=q,**value))
    with (OUT/'simulation_groups.csv').open('x',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    print(json.dumps(stats['acceptance']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('backbone_cal','s_train_scores','pool_scores','fit_s','seal','hold_render','hold_scores','evaluate'));parser.add_argument('--seconds',type=float,default=1200);args=parser.parse_args()
    began=time.monotonic()
    gpu=args.stage in ('backbone_cal','s_train_scores','pool_scores','hold_render','hold_scores')
    receipt_folder=OUT/('gpu_receipts' if gpu else 'cpu_receipts');receipt_folder.mkdir(exist_ok=True)
    previous=sum(C.read(p)['seconds'] for p in receipt_folder.glob('*.json'))
    cap=5400 if gpu else 1800
    def deadline():
        elapsed=time.monotonic()-began
        if elapsed>=args.seconds or previous+elapsed>=cap-2:raise TimeoutError('Reserved/shared stage command-wall budget reached')
    CHECK=deadline
    try:
        with threadpool_limits(limits=2):globals()['fit_s_stage' if args.stage=='fit_s' else 'seal_stage' if args.stage=='seal' else args.stage]()
        row=dict(stage=args.stage,status='COMPLETE',seconds=time.monotonic()-began,previous_seconds=previous)
        C.save(OUT/(args.stage+'_command_receipt.json'),row)
        C.save(receipt_folder/(args.stage+'_'+str(time.time_ns())+'.json'),row)
    except BaseException as exc:
        row=dict(stage=args.stage,status='FAILED',seconds=time.monotonic()-began,error=repr(exc),previous_seconds=previous)
        C.save(OUT/(args.stage+'_failure_'+str(time.time_ns())+'.json'),row)
        C.save(receipt_folder/(args.stage+'_failure_'+str(time.time_ns())+'.json'),row)
        raise
