"""Additive light scores jointly using intrusion and public native-peak support.

Fixed per-height HGB, complete grade-zero fit population, separated cal cuts;
consumed simulated Development, no backbone fit or target-truth descriptors.
"""
import json
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import cnh_graded_corridor_eval_dev as C
import cnh_graded_corridor_interaction_dev as I
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

OUT = C.ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'
BANKS = ('score_only','score_current','score_temporal')
BUDGETS = ((5,32),(15,64),(30,128))


def descriptors(archive,bank):
    x=archive[f'{bank}_features'].astype(float)
    v=archive[f'{bank}_valid']&np.isfinite(x)
    names=archive[f'{bank}_names'].tolist()
    return np.concatenate((np.where(v,x,np.nan),(~v).astype(float)),-1), names+[n+'_missing' for n in names]


def fit_height(x,category,baseline,mask,q):
    """Mean-one equal eligible-frame weight per scene/K/query, grade0 only."""
    eligible=(baseline[mask,...,q]==0)
    count=eligible.sum(-1,keepdims=True)
    weight=np.broadcast_to(np.divide(1.,count,out=np.zeros_like(count,dtype=float),where=count>0),eligible.shape)[eligible]
    weight*=len(weight)/weight.sum()
    y=np.broadcast_to((category[mask,q]=='contact')[:,None,None],eligible.shape)[eligible].astype(int)
    if len(np.unique(y))!=2: raise ValueError('Two cal-fit classes required')
    model=HistGradientBoostingClassifier(**I.PARAMS).fit(x[mask,...,q,:][eligible],y,sample_weight=weight)
    return model,dict(rows=len(y),contact_rows=int(y.sum()),weight_sum=float(weight.sum()),
        eligible_scene_replica_groups=int((count>0).sum()),iterations=int(model.n_iter_),
        baseline_logit=float(model._baseline_prediction[0,0]))


def cutoff(score,eligible,category,mask,clear_cap,pass_cap):
    s,e,c=score[mask],eligible[mask],category[mask]
    clear=(c=='clear').all(1); passed=(c=='pass').any(1)&~(c=='contact').any(1)
    candidate=np.where(e&np.isfinite(s),s,-np.inf)
    theta=max(C.at_most(candidate[clear].max(-1),clear_cap),C.at_most(candidate[passed].max(-1),pass_cap))
    flags=e&np.isfinite(s)&(s>=theta)
    ac,ap=int(flags[clear].any(-1).sum()),int(flags[passed].any(-1).sum())
    assert ac<=clear_cap and ap<=pass_cap
    # HGB logits can be negative; nonbinding lower cutoff is -infinity, not0.
    return dict(theta=None if theta==-np.inf else float(theta),nonbinding=bool(theta==-np.inf),
        clear_cap=clear_cap,pass_cap=pass_cap,actual_extra_candidate_clear_slots=ac,
        actual_extra_candidate_pass_slots=ap,clear_unused=clear_cap-ac,pass_unused=pass_cap-ap,
        clear_slot_denominator=int(clear.sum())*4*13,pass_slot_denominator=int(passed.sum())*4*13)


def run():
    began=time.monotonic()
    if (OUT/'PLAN.json').exists(): raise FileExistsError('Preserve declared joint-score run')
    C.save(OUT/'PLAN.json',dict(task_record_sha256=C.sha(OUT/'TASK.json'),params=I.PARAMS,banks=BANKS,budgets=BUDGETS,
        fit='cal8/10 only, complete original grade0, mean1 equal sceneKquery weights; perheight separately',
        cutoff='cal9/11 only, shared query-union additional slot caps, complete ties; logit maynegative/None=-inf',
        dimensions=dict(score_only=3,score_current=47,score_temporal=117),
        missing='HGB native nan with separate explicit indicator, public descriptor invalid not treated as free',
        inputs_sha256={str(p.relative_to(C.ROOT)):C.sha(p) for p in (
            OUT/'features/cal_features.npz',OUT/'features/validation_features.npz',OUT/'features/schema.json',
            C.OUT/'input_manifest.json',C.PARENT/'thresholds.json')},source_sha256=C.sha(Path(__file__))))
    try:
        data=C.load();fit_mask,part=C.split_cal(data['cal']['rows']);C.save(OUT/'cal_partition.json',part)
        thresholds=C.read(C.PARENT/'thresholds.json');spatial={};desc_names={}
        for split,d in data.items():
            with np.load(OUT/'features'/f'{split}_features.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['scene_ids'],d['scene_ids'])
                spatial[split]={}
                for bank in ('current','temporal'):
                    spatial[split][bank],desc_names[bank]=descriptors(a,bank)
        C.save(OUT/'feature_names.json',dict(score_only=['ordinary_smooth_margin','ordinary_raw_slope5','ordinary_raw_detrended_fluctuation5'],
            score_current=desc_names['current'],score_temporal=desc_names['temporal']))
        metrics,cuts,models,ledger={},{},{},[]
        summary=[];scores_saved={s:[] for s in data};grades_saved={s:[] for s in data}
        for si,seed in enumerate(G.SEEDS):
            if time.monotonic()-began>=300: raise TimeoutError('300s main phase cap')
            th=thresholds[str(seed)];refs={};tensor={}
            for split,d in data.items():
                ordinary=d['candidates'][0,si]
                raw=E.read_job(C.SOURCE/f'scores/ordinary_seed{seed}_{split}.npz','ordinary',seed,split,'ideal')
                base=C.build_score_features(raw,ordinary,th['single'])
                strong=E.old_fusion(d['m3'],d['local'])|(ordinary>=th['addition'])
                baseline=np.where(strong,2,np.where(ordinary>=th['single'],1,0)).astype(np.int8)
                refs[split]=dict(baseline=baseline,strong=strong,prior=baseline>0)
                tensor[split]=dict(score_only=base,score_current=np.concatenate((base,spatial[split]['current']),-1),
                    score_temporal=np.concatenate((base,spatial[split]['temporal']),-1))
            for bank in BANKS:
                score={split:np.empty(r['baseline'].shape,dtype=float) for split,r in refs.items()}
                for q,height in enumerate(E.HEIGHTS):
                    model,record=fit_height(tensor['cal'][bank],data['cal']['category'],refs['cal']['baseline'],fit_mask,q)
                    path=OUT/f'models/{seed}_{bank}_{height}.pickle';path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(pickle.dumps(model,protocol=5))
                    models[f'{seed}/{bank}/{height}']=dict(**record,path=str(path.relative_to(OUT)),sha256=C.sha(path),dimensions=tensor['cal'][bank].shape[-1])
                    for split in data:
                        xx=tensor[split][bank][...,q,:]
                        score[split][...,q]=model.decision_function(xx.reshape(-1,xx.shape[-1])).reshape(xx.shape[:-1])
                for split in data:scores_saved[split].append((f'{seed}/{bank}',score[split]))
                eligible={s:(r['baseline']==0)&np.isfinite(score[s]) for s,r in refs.items()}
                for cc,pc in BUDGETS:
                    key=f'{seed}/{bank}/c{cc}_p{pc}'
                    cut=cutoff(score['cal'],eligible['cal'],data['cal']['category'],~fit_mask,cc,pc);cuts[key]=cut
                    theta=-np.inf if cut['nonbinding'] else cut['theta']
                    for split,d in data.items():
                        r=refs[split];added=eligible[split]&(score[split]>=theta)
                        grade=np.where(added,1,r['baseline']).astype(np.int8)
                        assert np.all(grade>=r['baseline']);np.testing.assert_array_equal(grade==2,r['strong'])
                        comp=dict(prior_light=r['prior'],ordinary_OR=r['strong'],M3=d['m3']>=E.M3_THETA,old_fusion=E.old_fusion(d['m3'],d['local']))
                        result=G.describe(grade,d,comp);base_report=G.describe(r['baseline'],d,comp);metrics[f'{split}/{key}']=result
                        p=result['paired_any']['prior_light'];cat=d['category'];clear=(cat=='clear').all(1);passed=(cat=='pass').any(1)&~(cat=='contact').any(1)
                        summary.append(dict(split=split,seed=seed,bank=bank,clear_cap=cc,pass_cap=pc,
                            HEAD=result['any']['counts'][0],BODY=result['any']['counts'][1],HEAD_rescue=p[0]['rescue'],BODY_rescue=p[1]['rescue'],
                            HEAD_loss=p[0]['loss'],BODY_loss=p[1]['loss'],HEAD_earlier=p[0]['earlier'],BODY_earlier=p[1]['earlier'],HEAD_later=p[0]['later'],BODY_later=p[1]['later'],
                            clear_slots=result['any']['clear_slots'],pass_clips=result['any']['pass_clips'],
                            extra_total_clear_slots=result['any']['clear_slots']-base_report['any']['clear_slots'],extra_total_pass_clips=result['any']['pass_clips']-base_report['any']['pass_clips'],
                            extra_candidate_clear_slots=int(added[clear].any(-1).sum()),extra_candidate_pass_slots=int(added[passed].any(-1).sum()),
                            light_pass_slots=result['joint_costs']['pass']['light']['slots'],light_pass_longest_frames=result['joint_costs']['pass']['light']['longest_run_frames']))
                        grades_saved[split].append((key,grade))
                        if split=='validation':
                            before,after=G.first(r['prior']),G.first(grade>0)
                            for n,row in enumerate(d['rows']):
                                for k in range(4):
                                    for q,height in enumerate(E.HEIGHTS):
                                        ledger.append(dict(key=key,scene=int(d['scene_ids'][n]),replica=k,height=height,category=cat[n,q],family=row['shape_family'],
                                            background_family=row['background_family'],before_first=int(before[n,k,q]),after_first=int(after[n,k,q])))
        for split in data:
            np.savez_compressed(OUT/f'{split}_scores.npz',keys=np.array([k for k,v in scores_saved[split]]),scores=np.array([v for k,v in scores_saved[split]]))
            np.savez_compressed(OUT/f'{split}_grades.npz',keys=np.array([k for k,v in grades_saved[split]]),grades=np.array([v for k,v in grades_saved[split]]))
        G.write_csv(OUT/'summary.csv',summary);G.write_csv(OUT/'event_ledger.csv',ledger)
        C.save(OUT/'metrics.json',metrics);C.save(OUT/'models.json',models);C.save(OUT/'calibrations.json',cuts)
        C.save(OUT/'receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,cells=len(summary),fits=len(models),event_rows=len(ledger),GPU_seconds=0,source_sha256=C.sha(Path(__file__))))
        print(json.dumps(dict(status='COMPLETE',seconds=time.monotonic()-began,cells=len(summary),fits=len(models))))
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
