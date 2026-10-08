"""Predeclared grouped two-fold clear calibration of one fixed cache fusion mechanism."""
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S

OUT=B.ROOT/'artifacts.local/work/cnh-bar-transfer-dev-20261009/crosscal'
LOCAL=B.ROOT/'artifacts.local/work/cnh-bar-local-readout-dev-20261009'
FUSION=B.ROOT/'artifacts.local/work/cnh-bar-fusion-vertical-dev-20261009/fusion'
REP=B.ROOT/'artifacts.local/work/cnh-bar-representation-dev-20261008'
SEEDS=(2026100901,2026100902,2026100903,2026100904,2026100905)


def save(name,value):B.save(OUT/name,value)


def smooth(raw):
    raw=np.asarray(raw,float);out=np.empty_like(raw)
    for f in range(13):
        start=max(0,f-4);w=2.**np.arange(f-start+1)
        out[...,f,:]=(raw[...,start:f+1,:]*w[:,None]).sum(-2)/w.sum()
    return out


def nearest(values,target,floor=None):
    values=np.asarray(values,float).reshape(-1);values=np.sort(values[np.isfinite(values)])
    candidates=np.r_[np.unique(values),np.inf]
    if floor is not None:candidates=candidates[candidates>=floor]
    costs=len(values)-np.searchsorted(values,candidates,side='left')
    i=np.lexsort((-candidates,np.abs(costs-target)))[0]
    return float(candidates[i]),int(costs[i])


def metrics(flags,cat):
    joint=flags.any(-1);contact=cat=='contact';clear=(cat=='clear').all(1)
    timely=flags[:,:,:11].any(2)&contact[:,None,:]
    late=flags[:,:,11:].any(2)&contact[:,None,:]&~timely
    cc=joint[clear]
    out=dict(contact_denominators=[int(contact[:,q].sum()*flags.shape[1]) for q in (0,1)],
        timely=timely.sum((0,1)).tolist(),late=late.sum((0,1)).tolist(),clear_scenes=int(clear.sum()),
        clear_slots=int(cc.sum()),clear_denominator=int(cc.size),
        clear_segments=int(cc[:,:,0].sum()+((~cc[:,:,:-1])&cc[:,:,1:]).sum()),clear_clips=int(cc.any(2).sum()),
        physical_contact_any_height=int(joint[:,:,:11].any(2)[contact.any(1)].sum()),
        pass_clips=int(joint[(cat=='pass').any(1)&~contact.any(1)].any(2).sum()))
    return out,timely


def compare(a,b,cat,ids):
    out=[]
    for q in (0,1):
        gain=np.argwhere((~a[:,:,q])&b[:,:,q]);loss=np.argwhere(a[:,:,q]&(~b[:,:,q]))
        out.append(dict(height=('HEAD','BODY')[q],denominator=int((cat[:,q]=='contact').sum()*a.shape[1]),
            baseline=int(a[:,:,q].sum()),candidate=int(b[:,:,q].sum()),gain=len(gain),loss=len(loss),net=int(b[:,:,q].sum()-a[:,:,q].sum()),
            gain_scene_keys=[[int(ids[i]),int(k)] for i,k in gain],loss_scene_keys=[[int(ids[i]),int(k)] for i,k in loss]))
    return out


def prepare():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PLAN.json').exists():raise FileExistsError('Preserve crosscal frozen PLAN')
    rows=json.loads((S.OUT/'PLAN.json').read_text())['scene_rows']
    with np.load(S.OUT/'evaluated.npz') as z:cat=z['category']
    physical=np.where((cat=='contact').any(1),'contact',np.where((cat=='pass').any(1),'pass','clear'))
    assert ((cat=='clear').all(1)==(physical=='clear')).all()
    keys=sorted({(row['family'],physical[i]) for i,row in enumerate(rows)})
    splits=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed);halves=[[],[]];strata=[]
        for index,(family,category) in enumerate(keys):
            members=np.array([i for i,row in enumerate(rows) if row['family']==family and physical[i]==category])
            shuffled=rng.permutation(members);cut=len(shuffled)//2
            # If odd, alternate which half gets the surplus using only declared seed/stratum index.
            if len(shuffled)%2 and (index+seed)%2==0:cut+=1
            left=shuffled[:cut].tolist();right=shuffled[cut:].tolist();halves[0].extend(left);halves[1].extend(right)
            strata.append(dict(family=family,physical_category=category,total=len(members),half0=sorted(left),half1=sorted(right)))
        halves=[sorted(h) for h in halves]
        assert set(halves[0]).isdisjoint(halves[1]) and sorted(halves[0]+halves[1])==list(range(492))
        for fold in (0,1):
            cal=halves[fold];hold=halves[1-fold]
            splits.append(dict(seed=seed,cal_half=fold,name=f'seed{seed}_cal{fold}',cal_scene_ids=cal,holdout_scene_ids=hold,
                cal_clear_scenes=int((physical[cal]=='clear').sum()),holdout_clear_scenes=int((physical[hold]=='clear').sum()),strata=strata))
    old=json.loads((LOCAL/'result_analysis.json').read_text());reference=json.loads((FUSION/'k5_unbounded_below.json').read_text())
    inputs=[S.OUT/f for f in ('PLAN.json','physical.npz','evaluated.npz')]+[LOCAL/f for f in ('cached_scores.npz','result_analysis.json')]+[
        FUSION/'PLAN.json',FUSION/'k5_unbounded_below.json',REP/'fp16/PLAN.json']
    sources=[Path(__file__),Path(B.__file__),Path(S.__file__)]
    plan=dict(task='CNH_BAR_FUSION_CROSSCAL_DEV_20261009',lane='EXPLORE consumed Development transfer diagnostic',
        authorization='User advanced cross-calibration, pose nuisances and backgrounds; this branch executes cache-only cross-calibration',
        budget_CPU_wall_seconds=180,adjustable_scope='Implementation repairs and grouped cache statistics only; no extra seeds, candidate selection, GPU, noise samples, training or inference',
        mechanism='Fixed retained k5 unbounded OR: (smoothM3>=raisedtheta) OR (smoothLocal>=localthreshold). Original smooth last5 and full13 outputs; timely f3..13.',
        seeds=list(SEEDS),folds=2,splits=splits,grouping='Scenario is indivisible: allK4, both queries, all13 outputs stay together; stratify by family x physical contact/pass/jointclear. Related shape/background configurations may still occur on both sides.',
        original_M3_threshold=old['baseline']['threshold'],full_batch_reference=dict(raised_M3_threshold=reference['raised_M3_threshold'],local_threshold=reference['new_local_threshold']),
        calibration='For each cal half, C0 is original M3 clear cost at frozen originaltheta; Nclear=cal_joint_clear_scenes*K4*13, k_scaled=5*Nclear/4576. Raisedtheta>=original calibrates nearest C0-k_scaled whole-score ties; higher threshold on equal absolute residual. Strong-local threshold adds on raised-unoccupied cal clear slots nearest C0. No fixed23/5 half-sample assumption.',
        holdout='No holdout scores/category outcomes used for either threshold; apply cal-frozen thresholds, compare original M3 and full-batch frozen k5 reference. Reference thresholds were calibrated on all492 clear and include the held-out clear; reference is contextual, not independent holdout validation.',
        decision_check='All10folds reported. One scene appears in holdout once per seed; folds/seeds overlap and do not establish10 independent replications or a confidence interval. Reusing already-selected Development does not estimate prior four-point selection bias or new-scene transfer.',
        stop='Complete grouped calibration/report and focused independent recomputation within cumulative180s including prepare and failures; no outcome-driven retries or seed expansion',
        deliverables='Frozen split/stratum keys, all10cal/holdout thresholds and costs, paired family/dark event ledger, reference comparison, independent focusedcheck, summary; no Git/current/report edits by branch',
        input_sha256={B.logical_path(p):B.sha(p) for p in inputs},source_sha256={B.logical_path(p):B.sha(p) for p in sources})
    plan['prepare_seconds']=time.monotonic()-start;save('PLAN.json',plan);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('PREPARED',plan['prepare_seconds'],'splits',len(splits),flush=True)


def csv_write(name,records):
    with (OUT/name).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def run():
    start=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text())
    previous=plan['prepare_seconds']+sum(json.loads(f.read_text())['seconds'] for f in OUT.glob('failure_*.json'))
    def check():
        if previous+time.monotonic()-start>180:raise TimeoutError('crosscal cumulative CPU wall180s')
    folds=[]
    try:
        for name,digest in {**plan['input_sha256'],**plan['source_sha256']}.items():assert B.sha(B.ROOT/name)==digest,('frozen input/source drift',name)
        rows=json.loads((S.OUT/'PLAN.json').read_text())['scene_rows']
        with np.load(S.OUT/'evaluated.npz') as z:cat=z['category'];m3=z['scores']
        with np.load(S.OUT/'physical.npz') as z:origraw=z['raw']
        with np.load(LOCAL/'cached_scores.npz') as z:local=smooth(z['raw'])
        np.testing.assert_allclose(smooth(origraw),m3,rtol=1e-13,atol=1e-13)
        darkids=set(json.loads((REP/'fp16/PLAN.json').read_text())['contact_scene_ids'])
        theta0=plan['original_M3_threshold'];fulltheta=plan['full_batch_reference']['raised_M3_threshold'];fulllocal=plan['full_batch_reference']['local_threshold']
        oldflags=m3>=theta0;fullflags=(m3>=fulltheta)|(local>=fulllocal)
        ledger=[];summaries=[];flags=[]
        for split in plan['splits']:
            check();cal=np.array(split['cal_scene_ids']);hold=np.array(split['holdout_scene_ids']);cc=cat[cal];hc=cat[hold]
            cm=m3[cal];cl=local[cal];hm=m3[hold];hl=local[hold];c_clear=(cc=='clear').all(1)
            cal_old,_=metrics(oldflags[cal],cc);Nclear=cal_old['clear_denominator'];k=5*Nclear/4576.;C0=cal_old['clear_slots']
            raised,raisedcost=nearest(cm[c_clear].max(-1),C0-k,floor=theta0)
            retained=(cm>=raised)[c_clear].any(-1)
            eligible=cl[c_clear].max(-1)[~retained]
            localtheta,added=nearest(eligible,C0-raisedcost)
            candidate=(m3>=raised)|(local>=localtheta)
            cal_new,_=metrics(candidate[cal],cc);assert cal_new['clear_slots']==raisedcost+added
            hold_old,ht0=metrics(oldflags[hold],hc);hold_new,ht1=metrics(candidate[hold],hc);reference,htr=metrics(fullflags[hold],hc)
            paired=compare(ht0,ht1,hc,hold);refpaired=compare(ht0,htr,hc,hold)
            basehold=oldflags[hold].any(-1);raisedhold=(hm>=raised).any(-1);localhold=(hl>=localtheta).any(-1);holdclear=(hc=='clear').all(1)
            removed=(basehold&~raisedhold)[holdclear];localadd=(~raisedhold&localhold)[holdclear]
            family=[]
            for f in S.FAMILIES:
                mask=np.array([rows[i]['family']==f for i in hold]);family.append(dict(family=f,paired=compare(ht0[mask],ht1[mask],hc[mask],hold[mask])))
            dark=np.array([i in darkids for i in hold]);darkpaired=compare(ht0[dark],ht1[dark],hc[dark],hold[dark])
            point=dict(name=split['name'],seed=split['seed'],cal_half=split['cal_half'],cal_scenes=len(cal),holdout_scenes=len(hold),
                cal_original=cal_old,cal_candidate=cal_new,k_scaled=k,raised_M3_target=C0-k,raised_M3_threshold=raised,
                local_threshold=localtheta,cal_raised_M3_cost=raisedcost,cal_local_added_cost=added,
                cal_raised_residual=raisedcost-(C0-k),cal_total_residual=cal_new['clear_slots']-C0,
                holdout_original=hold_old,holdout_candidate=hold_new,holdout_fixed_fullbatch_reference=reference,
                holdout_clear_M3_removed=int(removed.sum()),holdout_clear_local_added=int(localadd.sum()),
                holdout_clear_net_difference=hold_new['clear_slots']-hold_old['clear_slots'],paired=paired,families=family,dark4cm=darkpaired,
                fixed_fullbatch_reference_paired=refpaired,holdout_calibration_used=False)
            assert point['holdout_clear_net_difference']==point['holdout_clear_local_added']-point['holdout_clear_M3_removed']
            folds.append(point);flags.append(candidate[hold]);save(split['name']+'.json',point)
            summary=dict(name=point['name'],seed=point['seed'],cal_half=point['cal_half'],cal_scenes=len(cal),hold_scenes=len(hold),
                cal_clear_scenes=cal_old['clear_scenes'],hold_clear_scenes=hold_old['clear_scenes'],cal_clear_denominator=Nclear,hold_clear_denominator=hold_old['clear_denominator'],
                cal_original_clear=C0,k_scaled=k,raised_target=C0-k,raised_threshold=raised,local_threshold=localtheta,
                cal_raised_clear=raisedcost,cal_local_add=added,cal_clear_residual=point['cal_total_residual'],
                hold_baseline_clear=hold_old['clear_slots'],hold_candidate_clear=hold_new['clear_slots'],hold_clear_net=point['holdout_clear_net_difference'],
                hold_M3_removed=point['holdout_clear_M3_removed'],hold_local_added=point['holdout_clear_local_added'],
                hold_baseline_segments=hold_old['clear_segments'],hold_candidate_segments=hold_new['clear_segments'],
                hold_baseline_clips=hold_old['clear_clips'],hold_candidate_clips=hold_new['clear_clips'],
                reference_hold_clear=reference['clear_slots'],reference_HEAD=refpaired[0]['candidate'],reference_BODY=refpaired[1]['candidate'],
                HEAD_denominator=paired[0]['denominator'],HEAD_gain=paired[0]['gain'],HEAD_loss=paired[0]['loss'],HEAD_net=paired[0]['net'],
                BODY_denominator=paired[1]['denominator'],BODY_gain=paired[1]['gain'],BODY_loss=paired[1]['loss'],BODY_net=paired[1]['net'],
                dark_BODY_denominator=darkpaired[1]['denominator'],dark_BODY_gain=darkpaired[1]['gain'],dark_BODY_loss=darkpaired[1]['loss'])
            summaries.append(summary)
            for li,rep,q in np.argwhere(np.broadcast_to((hc=='contact')[:,None,:],ht0.shape)):
                scene=int(hold[li]);a=np.flatnonzero(oldflags[scene,rep,:11,q]);b=np.flatnonzero(candidate[scene,rep,:11,q])
                ledger.append(dict(split=split['name'],seed=split['seed'],cal_half=split['cal_half'],scene=scene,replica=int(rep),height=('HEAD','BODY')[q],
                    family=rows[scene]['family'],variant=rows[scene]['variant'],dark4cm=int(scene in darkids),baseline=int(a.size>0),candidate=int(b.size>0),
                    gain=int(a.size==0 and b.size>0),loss=int(a.size>0 and b.size==0),baseline_first_timely_frame=int(a[0]+3) if a.size else '',candidate_first_timely_frame=int(b[0]+3) if b.size else ''))
            print('FOLD',point['name'],'calC0/k',C0,k,'hold clear',hold_old['clear_slots'],hold_new['clear_slots'],'H/B net',paired[0]['net'],paired[1]['net'],flush=True)
        assert len(ledger)==5*1376
        csv_write('summary.csv',summaries);csv_write('holdout_event_ledger.csv',ledger)
        np.savez_compressed(OUT/'scores.npz',M3_smooth=m3,local_smooth=local,original_alarm=oldflags,fixed_fullbatch_reference_alarm=fullflags)
        finite_raised=[f['raised_M3_threshold'] for f in folds if np.isfinite(f['raised_M3_threshold'])]
        finite_local=[f['local_threshold'] for f in folds if np.isfinite(f['local_threshold'])]
        aggregate=dict(folds=10,distinct_seeds=5,holdout_event_appearances=len(ledger),holdout_clear_denominator=sum(f['holdout_original']['clear_denominator'] for f in folds),
            original_clear=sum(f['holdout_original']['clear_slots'] for f in folds),candidate_clear=sum(f['holdout_candidate']['clear_slots'] for f in folds),
            clear_net=sum(f['holdout_clear_net_difference'] for f in folds),M3_removed=sum(f['holdout_clear_M3_removed'] for f in folds),local_added=sum(f['holdout_clear_local_added'] for f in folds),
            HEAD_gain=sum(f['paired'][0]['gain'] for f in folds),HEAD_loss=sum(f['paired'][0]['loss'] for f in folds),
            BODY_gain=sum(f['paired'][1]['gain'] for f in folds),BODY_loss=sum(f['paired'][1]['loss'] for f in folds),
            folds_both_height_net_positive=sum(all(p['net']>0 for p in f['paired']) for f in folds),
            folds_any_height_net_negative=sum(any(p['net']<0 for p in f['paired']) for f in folds),
            folds_any_paired_loss=sum(any(p['loss']>0 for p in f['paired']) for f in folds),
            raised_threshold_finite_range=[min(finite_raised),max(finite_raised)],raised_infinite_count=10-len(finite_raised),
            local_threshold_finite_range=[min(finite_local),max(finite_local)],local_infinite_count=10-len(finite_local),
            interpretation='Sums count overlapping held-out appearances across5seeds, not independent samples or a new cohort.')
        check();save('result.json',dict(status='COMPLETE',seconds=time.monotonic()-start,cumulative_seconds=previous+time.monotonic()-start,
            folds=folds,aggregate=aggregate,original_M3_threshold=theta0,full_batch_reference=plan['full_batch_reference'],decision='REPORT_ALL_TEN_FOLDS; no candidate reselection, promotion, independentCI or fresh-geometry claim',
            GPU=False,new_samples=0,training=0,model_inference_examples=0))
        print('COMPLETE',json.dumps(aggregate),flush=True)
    except BaseException as error:
        save('failure_'+str(time.time_ns())+'.json',dict(error=repr(error),seconds=time.monotonic()-start,completed_folds=len(folds)));raise


if __name__=='__main__':globals()[sys.argv[1]]()
