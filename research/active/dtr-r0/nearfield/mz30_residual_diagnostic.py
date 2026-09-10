"""Explain frozen MZ30 residuals and TRAIN responsibility errors; no inference."""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
from mz5_ensemble_readout import read,write,sha,load_npz

ORDER=['BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR']


def main(root,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();work=root/'artifacts.local/work';run=work/'mz30-branch-responsibility-20260910/run-v1';inputs={}
    def bind(path,expected=None):
        h=sha(path);assert expected is None or h==expected,str(path);inputs[str(path)]=h
    receipt=read(run/'receipt.json');assert receipt['status']=='PASS';bind(run/'receipt.json')
    for name in ['predictions.npz','branches.npz','selector-scores.npz','cutoff.npy','result.json','train-start.json']:
        bind(run/name,receipt['outputs'][name])
    bind(run/'audit.json');assert read(run/'audit.json')['status']=='PASS'
    oldpath=work/'mz8-attribution-20260910/cache-v5';newpath=work/'mz15-shared-support-20260910/cache-v1';cache=work/'mz16-visual-detail-20260910/cache-v2'
    for folder,name in [(oldpath,'observations.npz'),(newpath,'selected.json'),(cache,'selected.json')]:
        bind(folder/'receipt.json');bind(folder/name,read(folder/'receipt.json')['files'][name])
    indexpath=work/'body-query-5000-20260909/dataset-v1/index.json';identityreceipt=work/'mz1-tiny-fusion-20260910/cache-v1/features-receipt.json'
    bind(identityreceipt);bind(indexpath,read(identityreceipt)['source_index_sha256'])
    data=load_npz(run/'branches.npz');pred=load_npz(run/'predictions.npz');scores=load_npz(run/'selector-scores.npz');cut=np.load(run/'cutoff.npy');result=read(run/'result.json')
    old=load_npz(oldpath/'observations.npz');sourceindex=read(indexpath)['frames'];newrows=read(newpath/'selected.json');cachemeta={r['global_id']:r for r in read(cache/'selected.json')}
    ids=data['global_ids'];lookup=np.full(14200,-1,int);lookup[ids]=np.arange(len(ids));train=data['train_ids'];assert len(train)==7562
    def identity(gid):
        meta=cachemeta[gid]
        if gid<3500:
            row=sourceindex[int(old['old_index'][gid])];assert row['rgb_sha256']==meta['rgb_sha']
            return dict(site=row['site_id'],group=row['group_id'],family=row['family'],source='old5000',source_index=int(old['old_index'][gid]),identity_status='EXPLICIT')
        if gid>=3700:
            row=newrows[gid-3700];assert row['cache_index']==gid-3700 and row['rgb_sha']==meta['rgb_sha']
            return dict(site=row['site'],group=row['group'],family=row['family'],source=row['dataset'],source_index=row['index'],identity_status='EXPLICIT')
        return dict(site=None,group=None,family=None,source='MZ6_sequence',clip=str(old['clip'][gid-3500]),identity_status='UNKNOWN_SITE_GROUP')
    trainrows=[]
    for local,q in np.argwhere(data['normal/eligible'][lookup[train]]):
        gid=int(train[local]);i=int(lookup[gid]);q=int(q);target=bool(data['normal/target'][i,q]);logit=float(scores['normal/logits'][i,q]);confidence=float(scores['normal/confidence'][i,q]);truth=bool(data['normal/truth'][i,q])
        rgb=float(data['normal/rgb'][i,q]);tof=float(data['normal/tof'][i,q]);assert (rgb>=0)!=(tof>=0)
        choose_rgb=logit>=0;removed=confidence>=cut[q]
        trainrows.append(dict(global_id=gid,query=q,query_name=ORDER[q],**identity(gid),truth=truth,rgb=rgb,tof=tof,
            target_rgb_correct=target,predicted_rgb_correct=choose_rgb,responsibility_error=choose_rgb!=target,
            selector_logit=logit,negative_branch_confidence=confidence,fixed_cutoff=float(cut[q]),removed_at_fixed_cutoff=bool(removed),
            baseline_tp_lost_at_fixed_cutoff=bool(removed and truth),baseline_fp_removed_at_fixed_cutoff=bool(removed and not truth)))
    assert len(trainrows)==168 and sum(r['target_rgb_correct'] for r in trainrows)==120
    train_summary={}
    for q in range(4):
        rr=[r for r in trainrows if r['query']==q];classes={}
        for target,label in [(True,'RGB_correct'),(False,'ToF_correct')]:
            group=[r for r in rr if r['target_rgb_correct']==target]
            classes[label]=dict(query_examples=len(group),frames=len({r['global_id'] for r in group}),explicit_sites=len({r['site'] for r in group if r['site'] is not None}),
                unknown_site_examples=sum(r['site'] is None for r in group),source_counts=dict(Counter(r['source'] for r in group)),site_counts=dict(sorted(Counter(r['site'] if r['site'] is not None else 'UNKNOWN_SITE' for r in group).items())),
                responsibility_errors=sum(r['responsibility_error'] for r in group),fixed_cutoff_tp_lost=sum(r['baseline_tp_lost_at_fixed_cutoff'] for r in group),fixed_cutoff_fp_removed=sum(r['baseline_fp_removed_at_fixed_cutoff'] for r in group))
        train_summary[ORDER[q]]=classes
    write(output/'start.json',dict(status='STARTED',training_steps=0,new_inference_frames=0,inputs=inputs,code_sha256=sha(Path(__file__)),
        scope='Frozen MZ30 residual explanation and in-sample TRAIN replay of saved scores; no cutoff changes.'))
    def record(name,i,q):
        gid=int(pred[name+'/global_ids'][i]);rgb=float(pred[name+'/rgb'][i,q]);tof=float(pred[name+'/tof'][i,q]);confidence=float(pred[name+'/confidence'][i,q]);support=bool(pred[name+'/support'][i,q]);eligible=bool(pred[name+'/eligible'][i,q])
        baseline=float(pred[name+'/baseline'][i,q]);truth=bool(pred[name+'/truth'][i,q]);agreement=(rgb>=0)==(tof>=0)
        assert eligible==bool(baseline>=0 and not support and not agreement)
        return dict(cohort=name,frame=int(i),global_id=gid,query=int(q),query_name=ORDER[q],**identity(gid),truth=truth,baseline=baseline,
            mz28=float(pred[name+'/mz28'][i,q]),candidate=float(pred[name+'/candidate'][i,q]),rgb=rgb,tof=tof,rgb_positive=rgb>=0,tof_positive=tof>=0,
            selector_logit=float(pred[name+'/selector_logits'][i,q]),negative_branch_confidence=confidence,cutoff=float(cut[q]),confidence_minus_cutoff=confidence-float(cut[q]),
            original_geometry_support=support,branch_agreement=agreement,eligible=eligible,removed=bool(pred[name+'/remove'][i,q]))
    lost=[];remaining=[]
    for name in ['DEV','clean','stress','relation10000','distance5000']:
        for i,q in np.argwhere(pred[name+'/remove']&pred[name+'/truth']):lost.append(record(name,int(i),int(q)))
        if name not in ['relation10000','distance5000']:continue
        for i,q in np.argwhere((pred[name+'/candidate']>=0)&~pred[name+'/truth']):
            r=record(name,int(i),int(q))
            if r['baseline']<0:
                reason='INHERITED_MZ28_ADDITION';assert r['mz28']>=0 and not r['eligible']
            elif r['original_geometry_support']:reason='BASELINE_GEOMETRY_SUPPORTED'
            elif r['branch_agreement']:reason='BASELINE_BRANCH_AGREEMENT'
            else:
                reason='ELIGIBLE_BELOW_FIXED_CUTOFF';assert r['eligible'] and r['negative_branch_confidence']<r['cutoff']
            r['residual_reason']=reason;remaining.append(r)
    assert len(lost)==1 and lost[0]['cohort']=='relation10000' and lost[0]['query']==0
    assert len(remaining)==20 and sum(r['baseline']>=0 for r in remaining)==18 and sum(r['baseline']<0 for r in remaining)==2
    lost[0]['same_query_eligible_train']=train_summary[ORDER[lost[0]['query']]]
    counts=dict(Counter(r['residual_reason'] for r in remaining))
    assert result['placement_baseline_fp_removed']==63 and 81-result['placement_baseline_fp_removed']==18
    for name in ['relation10000','distance5000']:
        for q in range(4):assert sum(r['cohort']==name and r['query']==q for r in remaining)==result['metrics'][name]['candidate']['fp'][q]
    outcome=dict(status='PASS',training_steps=0,new_inference_frames=0,inputs=inputs,code_sha256=sha(Path(__file__)),lost_true_positives=lost,
        remaining_placement_false_positives=remaining,residual_counts=counts,train_eligible_query_examples=len(trainrows),train_summary=train_summary,train_eligible_rows=trainrows,
        train_responsibility_errors=sum(r['responsibility_error'] for r in trainrows),train_fixed_cutoff_tp_lost=sum(r['baseline_tp_lost_at_fixed_cutoff'] for r in trainrows),
        train_fixed_cutoff_fp_removed=sum(r['baseline_fp_removed_at_fixed_cutoff'] for r in trainrows),
        definitions=dict(raw_train_error='selector_logit>=0 choosesRGB, compare to RGB-sign-equals-event-truth on eligible TRAIN queries',
            applied_train_error='use frozen oldDEV confidence cutoff on saved eligible TRAIN scores; no recalibration',
            identities='explicit source-index/cache sites; sequence sites remain UNKNOWN'),
        limits=['Residual identities and event truth are diagnostic only, not new inference inputs.','TRAIN base predictions and selector scores are in-sample; accuracy is not independent generalization.',
                'No new model, threshold, score or source is introduced.'])
    write(output/'result.json',outcome)
    write(output/'receipt.json',dict(status='PASS',training_steps=0,new_inference_frames=0,backend='NumPy CPU saved scalar/identity reductions',seconds=time.perf_counter()-started,
        inputs=inputs,code_sha256=sha(Path(__file__)),outputs={name:sha(output/name) for name in ['start.json','result.json']}))
    print('PASS',counts,'lost',[(r['cohort'],r['frame'],r['query_name']) for r in lost],'TRAINerrors',outcome['train_responsibility_errors'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
