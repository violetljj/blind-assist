"""Factorial height-specific light scoring/cutoffs on consumed Development.

Runtime descriptors and strong masks are unchanged. Truth joins are confined to
cal fitting/cost cutoff and evaluation. Allocation caps candidate-light unions,
whereas reported joint light cost excludes a slot with any strong query.
"""
import json
from pathlib import Path
import pickle
import time

import numpy as np
from threadpoolctl import threadpool_limits
import cnh_graded_corridor_eval_dev as C
import cnh_graded_corridor_interaction_dev as I
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

OUT = C.ROOT/'artifacts.local/work/cnh-graded-height-cal-dev-20261010'
MODES = ('shared_shared','shared_height','separate_shared','separate_height')
BANKS = ('score_only','score_spatial')


def allocate(total, weights):
    """Integer largest remainder; equal residual ties go to HEAD, not tuned."""
    weights = np.asarray(weights,dtype=float)
    if not weights.sum():
        if total: raise ValueError('Nonzero budget with no baseline cost')
        return np.zeros(2,dtype=int)
    real = total*weights/weights.sum()
    target = np.floor(real).astype(int)
    remainder = total-int(target.sum())
    order = np.argsort(-(real-target),kind='stable')
    target[order[:remainder]] += 1
    assert target.sum()==total
    return target


def height_cut(score,eligible,category,mask,original_light,fraction):
    score,eligible,original_light = score[mask],eligible[mask],original_light[mask]
    cat = category[mask]
    passed = (cat=='pass').any(1)&~(cat=='contact').any(1)
    clear = (cat=='clear').all(1)
    bp = int(original_light[passed].any((-1,-2)).sum())
    bc = int(original_light[clear].any(-1).sum())
    total_p,total_c = int(np.floor(fraction*bp)),int(np.floor(fraction*bc))
    wp = original_light[passed].any(-2).sum((0,1)).astype(int)
    wc = original_light[clear].sum((0,1,2)).astype(int)
    tp,tc = allocate(total_p,wp),allocate(total_c,wc)
    candidate = np.where(eligible,score,-np.inf)
    theta = [max(C.at_most(candidate[passed,...,q].max(-1),int(tp[q])),
                 C.at_most(candidate[clear,...,q],int(tc[q]))) for q in range(2)]
    flags = eligible&(score>=np.array(theta))
    ap = flags[passed].any(-2).sum((0,1)).astype(int)
    ac = flags[clear].sum((0,1,2)).astype(int)
    joint_p = int(flags[passed].any((-1,-2)).sum())
    joint_c = int(flags[clear].any(-1).sum())
    assert np.all(ap<=tp) and np.all(ac<=tc)
    assert joint_p<=total_p and joint_c<=total_c
    return dict(theta=theta,fraction=fraction,baseline_light_pass_clips=bp,baseline_light_clear_slots=bc,
        target_light_pass_clips=total_p,target_light_clear_slots=total_c,
        actual_light_pass_clips=joint_p,actual_light_clear_slots=joint_c,
        pass_residual=joint_p-total_p,clear_residual=joint_c-total_c,
        baseline_height_pass_clips=wp.tolist(),baseline_height_clear_slots=wc.tolist(),
        height_target_pass_clips=tp.tolist(),height_target_clear_slots=tc.tolist(),
        height_actual_pass_clips=ap.tolist(),height_actual_clear_slots=ac.tolist(),
        height_sum_actual_pass_clips=int(ap.sum()),height_sum_actual_clear_slots=int(ac.sum()),
        pass_union_overlap=int(ap.sum())-joint_p,clear_union_overlap=int(ac.sum())-joint_c)


def run():
    began = time.monotonic()
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Preserve completed or declared height run')
    C.save(OUT/'PLAN.json',dict(task='CNH_GRADED_HEIGHT_CAL_DEV_20261010',base_commit='fb61588d',
        authorization='User continues height-specific scoring/calibration/budgets',lane='EXPLORE consumed simulated Development',
        goal='Determine whether shared H/B light scoring/cutoffs contribute BODY loss, with fixed strong and matched score-only controls',
        budget=dict(CPU_command_wall_seconds_cap=600,GPU_seconds=0,main_command_cap_seconds=300),
        backend='TASK_NOT_GPU_SUITABLE: small scalar cache fits; prior CPU/CUDA evidence unchanged, no native re-extraction',
        adjustable_scope='Fixed factorial, implementation repairs, focused metric/cutoff checks and task documentation',
        stop='Finish all banks/modes/policies/fractions and deliver or600s command CPU cap; no old run retuning or new draws/backbone/protected480',
        decision_check='Each height384 contact events, minimum change1event; report full paired rescue/loss/timing and purepass256clip/3328slot, clear512clip/6656slot. No allseed/zero-loss gate; utility and weakBODY losses matter.',
        modes=list(MODES),banks=list(BANKS),policies=list(C.POLICIES),fractions=list(C.FRACTIONS),params=I.PARAMS,
        fit='Old shared score caches exact-reused. Separate H/B each fit old fixed HGB on samecal8/10 with equal eligible episodeweight normalizedmean1. Separate fitting changes model count/sample support, not only score scale.',
        calibration='Cal9/11 only. Shared cutoff exactlyold union budgets. Height cutoff allocates that same total budget proportional to old light perheightcost, largest remainder HEAD-firsttie; complete scoreties; no validation choices.',
        cost_semantics='Budget counts candidate-light query unions, including light query when another height is strong. Evaluation joint light only if max grade1; actual highest-grade costs separately saved.',
        features='Oldfixed3 score/34 spatial descriptors, no truth in runtime inputs; features alreadyconsumed and audited',
        strong='Every prior ordinaryOR strong grade/time retained',
        evidence='Prior validation motivated height split; both prior and this run are consumed Development, not independent confirmation',
        inputs_sha256={str(p.relative_to(C.ROOT)):C.sha(p) for p in (
            C.OUT/'input_manifest.json',C.OUT/'features/cal_features.npz',C.OUT/'features/validation_features.npz',
            I.OUT/'cal_scores.npz',I.OUT/'validation_scores.npz',I.OUT/'models.json',C.PARENT/'thresholds.json')},
        source_sha256=C.sha(Path(__file__))))
    try:
        data = C.load();fit_mask,partitions = C.split_cal(data['cal']['rows'])
        C.save(OUT/'cal_partition.json',partitions)
        thresholds = C.read(C.PARENT/'thresholds.json')
        spatial,shared_scores = {},{}
        for split,ds in data.items():
            with np.load(C.OUT/'features'/f'{split}_features.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['scene_ids'],ds['scene_ids'])
                spatial[split],_ = C.spatial_features(a)
            with np.load(I.OUT/f'{split}_scores.npz',allow_pickle=False) as a:
                shared_scores[split] = dict(zip(a['keys'].tolist(),a['scores']))
        summary,metrics,models,calibrations,ledger = [],{},{},{},[]
        grades_saved = {split:[] for split in data};scores_saved = {split:[] for split in data}
        for si,seed in enumerate(G.SEEDS):
            if time.monotonic()-began>300: raise TimeoutError('Main CPU300s cap')
            th = thresholds[str(seed)];tensor,refs = {},{}
            for split,ds in data.items():
                ordinary = ds['candidates'][0,si]
                raw = E.read_job(C.SOURCE/f'scores/ordinary_seed{seed}_{split}.npz','ordinary',seed,split,'ideal')
                base = C.build_score_features(raw,ordinary,th['single'])
                tensor[split] = dict(score_only=base,score_spatial=np.concatenate((base,spatial[split]),-1))
                strong = E.old_fusion(ds['m3'],ds['local'])|(ordinary>=th['addition'])
                light = ~strong&(ordinary>=th['single'])
                refs[split] = dict(strong=strong,light=light,prior=strong|light)
            for bank in BANKS:
                separated = {split:np.empty_like(refs[split]['strong'],dtype=float) for split in data}
                for q,height in enumerate(E.HEIGHTS):
                    model,record = I.fit(tensor['cal'][bank][...,q:q+1,:],data['cal']['category'][:,q:q+1],refs['cal']['strong'][...,q:q+1],fit_mask)
                    path = OUT/f'models/seed{seed}_{bank}_{height}.pickle';path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(pickle.dumps(model,protocol=5))
                    models[f'{seed}/{bank}/{height}'] = dict(**record,path=str(path.relative_to(OUT)),sha256=C.sha(path))
                    for split in data:
                        x = tensor[split][bank][...,q,:]
                        separated[split][...,q] = model.decision_function(x.reshape(-1,x.shape[-1])).reshape(x.shape[:-1])
                for split in data:
                    scores_saved[split].append((f'{seed}/{bank}/separate',separated[split]))
                for mode in MODES:
                    score = {split:shared_scores[split][f'{seed}/{bank}'] if mode.startswith('shared_') else separated[split] for split in data}
                    for policy in C.POLICIES:
                        eligible = {s:r['light'] if policy=='filter' else ~r['strong'] for s,r in refs.items()}
                        for fraction in C.FRACTIONS:
                            key = f'{seed}/{bank}/{mode}/{policy}/{fraction}'
                            calibrator = height_cut if mode.endswith('_height') else C.calibrate
                            record = calibrator(score['cal'],eligible['cal'],data['cal']['category'],~fit_mask,refs['cal']['light'],fraction)
                            calibrations[key] = record
                            for split,ds in data.items():
                                r = refs[split];flag = eligible[split]&(score[split]>=np.array(record['theta']))
                                grade = np.where(r['strong'],2,np.where(flag,1,0)).astype(np.int8)
                                np.testing.assert_array_equal(grade==2,r['strong'])
                                if policy=='filter': assert np.all(~(grade>0)|r['prior'])
                                comparison = dict(ordinary_OR=r['strong'],prior_light=r['prior'],M3=ds['m3']>=E.M3_THETA,old_fusion=E.old_fusion(ds['m3'],ds['local']))
                                report = G.describe(grade,ds,comparison);metrics[f'{split}/{key}'] = report
                                p = report['paired_any']['prior_light']
                                summary.append(dict(split=split,seed=seed,bank=bank,mode=mode,policy=policy,fraction=fraction,
                                    HEAD=report['any']['counts'][0],BODY=report['any']['counts'][1],HEAD_gain_prior=p[0]['rescue'],HEAD_loss_prior=p[0]['loss'],
                                    BODY_gain_prior=p[1]['rescue'],BODY_loss_prior=p[1]['loss'],HEAD_delay=p[0]['later'],BODY_delay=p[1]['later'],
                                    clear_slots=report['any']['clear_slots'],pass_clips=report['any']['pass_clips'],
                                    light_clear_slots=report['joint_costs']['clear']['light']['slots'],light_pass_slots=report['joint_costs']['pass']['light']['slots'],
                                    light_pass_clips=report['joint_costs']['pass']['light']['clips']))
                                grades_saved[split].append((key,grade))
                                if split=='validation':
                                    before,after = G.first(r['prior']),G.first(grade>0)
                                    for n,row in enumerate(ds['rows']):
                                        for k in range(4):
                                            for q,h in enumerate(E.HEIGHTS):
                                                ledger.append(dict(key=key,scene=int(ds['scene_ids'][n]),replica=k,height=h,category=ds['category'][n,q],
                                                    family=row['shape_family'],background_family=row['background_family'],before_first=int(before[n,k,q]),after_first=int(after[n,k,q])))
        for split in data:
            np.savez_compressed(OUT/f'{split}_grades.npz',keys=np.array([k for k,v in grades_saved[split]]),grades=np.array([v for k,v in grades_saved[split]]))
            np.savez_compressed(OUT/f'{split}_separate_scores.npz',keys=np.array([k for k,v in scores_saved[split]]),scores=np.array([v for k,v in scores_saved[split]]))
        G.write_csv(OUT/'summary.csv',summary);G.write_csv(OUT/'event_ledger.csv',ledger)
        C.save(OUT/'metrics.json',metrics);C.save(OUT/'models.json',models);C.save(OUT/'calibrations.json',calibrations)
        C.save(OUT/'receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,cells=len(summary),model_fits=len(models),event_rows=len(ledger),GPU_seconds=0,source_sha256=C.sha(Path(__file__))))
        print(json.dumps(dict(status='COMPLETE',seconds=time.monotonic()-began,cells=len(summary),models=len(models))))
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began));raise


if __name__=='__main__':
    with threadpool_limits(limits=2): run()
