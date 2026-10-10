"""Matched shallow readouts retaining weak all-bin mass before history averaging.

Runtime descriptors are constructed separately. Truth joins are only labels,
calibration costs and evaluation; the frozen ordinary/M3/light base is retained.
"""
from pathlib import Path
import json
import pickle
import time

import numpy as np
from threadpoolctl import threadpool_limits
import cnh_graded_corridor_eval_dev as C
import cnh_graded_corridor_interaction_dev as I
import cnh_graded_peak_joint_dev as J
import cnh_graded_peak_head_cal_dev as H
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

ROOT = C.ROOT
OUT = ROOT/'artifacts.local/work/cnh-graded-weak-mass-dev-20261010'
ARMS = ('current_control','weak_mass')


def declare():
    if (OUT/'TASK.json').exists(): raise FileExistsError('Preserve existing weak-mass task')
    C.save(OUT/'TASK.json',dict(task='CNH_GRADED_WEAK_MASS_DEV_20261010',base_commit='1c6fc341',
        authorization='User 推进 after target-support diagnosis',lane='EXPLORE consumed ideal simulated Development',
        goal='Retain short weak positive mass before signed temporal cancellation, distinguish current/history support, matched-cost paired timely detection',
        budget=dict(feature_CPU_command_wall_seconds=240,feature_GPU_stage_wall_seconds=120,
                    fit_evaluate_CPU_command_wall_seconds=360,verification_CPU_command_wall_seconds=180,
                    integration_CPU_command_wall_seconds=180,contract_CPU_command_wall_seconds=90,
                    total_CPU_command_wall_seconds=1050),
        frozen='Original baseline strong and ordinary_single light retained slotwise; only joint-added light replaced. M3/5slot/L2/480/weak_pass and old stop recipes unchanged.',
        arms=ARMS,feature_contract='current22 plus original3score descriptors; weak_mass adds28 runtime all-bin positive/negative current/past8 descriptors + explicit missing',
        fit='Both arms same fixed HGB params/init, each ordinary seed/height separately; original baseline grade0 complete cal8/10 fit; equal eligible-frame group weight mean1',
        calibration='cal9/11 only; each ordinary seed single shared HEAD/BODY cut. Lowest whole-tie cut with total union clear slots and purepass clips <= frozen head50 same-subset costs. Deduct baseline-covered units, not candidate slot counts.',
        decision_check='Contact timely unit1/384 perheight seed, family96/rho48. Report paired rescue/loss/earlier/later, both arms versus head50 and each other, full clear/pass clips/slots. Cal may underuse budget because ties; val cost may drift and must be reported. No zero-loss gate.',
        runtime_boundary='Feature builder loads hist/ambient/bias/public geometry identities only; no target truth/rho/shape/boxes/support diagnosis. Categories only labels/cal costs/evaluation, never model features.',
        scope='Two specified banks and inherited fixed HGB100tree recipe; source/schema/numerical repair allowed inside budget; no parameter/window scan, new scene generation, hold recalibration, Android or hardware',
        stop='Deliver fixed paired readout comparison, focused checks and decision; preserve negative results. New frozen E2E hold is not read or retuned.',
        backend='Torch CUDA vectorized raw feature reductions; sklearn HGB shallow CPU implementation suitable for small structured table. No backbone retraining.',
        source_sha256=C.sha(Path(__file__))))


def costs(flags,category,mask=None):
    if mask is not None: flags,category = flags[mask],category[mask]
    clear = (category=='clear').all(1)
    passed = (category=='pass').any(1)&~(category=='contact').any(1)
    union = flags.any(-1)
    return dict(clear_slots=int(union[clear].sum()),clear_clips=int(union[clear].any(-1).sum()),
                pass_slots=int(union[passed].sum()),pass_clips=int(union[passed].any(-1).sum()),
                clear_clip_denominator=int(clear.sum())*4,pass_clip_denominator=int(passed.sum())*4,
                clear_slot_denominator=int(clear.sum())*4*13,pass_slot_denominator=int(passed.sum())*4*13)


def calibrate(score,base,reference,category,mask):
    score,base,reference,cat = score[mask],base[mask],reference[mask],category[mask]
    fixed = base>0
    parent_cost,base_cost = costs(reference>0,cat),costs(fixed,cat)
    caps = {name:parent_cost[name]-base_cost[name] for name in ('clear_slots','pass_clips')}
    if min(caps.values())<0: raise ValueError('Frozen baseline exceeds reference budget: NOT_FEASIBLE')
    clear = (cat=='clear').all(1)
    passed = (cat=='pass').any(1)&~(cat=='contact').any(1)
    candidate = np.where((base==0)&np.isfinite(score),score,-np.inf)
    # Exclude fixed-base-covered cost units before taking order statistics.
    clear_values = np.where(fixed.any(-1),-np.inf,candidate.max(-1))[clear]
    pass_values = np.where(fixed.any((-1,-2)),-np.inf,candidate.max((-1,-2)))[passed]
    theta = max(C.at_most(clear_values,caps['clear_slots']),C.at_most(pass_values,caps['pass_clips']))
    grade = np.where((base==0)&np.isfinite(score)&(score>=theta),1,base)
    actual = costs(grade>0,cat)
    assert actual['clear_slots']<=parent_cost['clear_slots'] and actual['pass_clips']<=parent_cost['pass_clips']
    return dict(theta=None if theta==-np.inf else float(theta),nonbinding=bool(theta==-np.inf),
                reference_cost=parent_cost,fixed_base_cost=base_cost,additional_caps=caps,actual_cost=actual,
                clear_unused=parent_cost['clear_slots']-actual['clear_slots'],
                pass_unused=parent_cost['pass_clips']-actual['pass_clips'])


def outcomes(grade,category,rows):
    first = G.first(grade>0,11)
    full = G.first(grade>0,13)
    result = {}
    for q,height in enumerate(E.HEIGHTS):
        for family in ('all',*sorted({r['shape_family'] for r in rows})):
            for rho in ('all',.25,.65):
                selected = (category[:,q]=='contact') & np.array([
                    (family=='all' or r['shape_family']==family) and (rho=='all' or r['rho']==rho) for r in rows])
                t,f = first[selected,:,q],full[selected,:,q]
                result[f'{height}/{family}/{rho}'] = dict(events=int(t.size),timely=int((t>=0).sum()),
                    late=int(((t<0)&(f>=0)).sum()),silent=int((f<0).sum()))
    return result


def paired_cohorts(before,after,category,rows):
    bf,af = G.first(before>0,11),G.first(after>0,11)
    result = {}
    for q,height in enumerate(E.HEIGHTS):
        for family in ('all',*sorted({r['shape_family'] for r in rows})):
            for rho in ('all',.25,.65):
                selected = (category[:,q]=='contact') & np.array([
                    (family=='all' or r['shape_family']==family) and (rho=='all' or r['rho']==rho) for r in rows])
                b,a=bf[selected,:,q],af[selected,:,q]
                both=(a>=0)&(b>=0)
                result[f'{height}/{family}/{rho}'] = dict(events=int(a.size),
                    rescue=int(((b<0)&(a>=0)).sum()),loss=int(((b>=0)&(a<0)).sum()),
                    earlier=int((both&(a<b)).sum()),later=int((both&(a>b)).sum()))
    return result


def run():
    started = time.monotonic()
    if (OUT/'PLAN.json').exists(): raise FileExistsError('Preserve fitted weak-mass attempt')
    paths = [OUT/'TASK.json',OUT/'features/schema.json',C.PARENT/'thresholds.json',H.OUT/'cal_grades.npz',H.OUT/'validation_grades.npz',
             C.SOURCE/'eval_manifest.json',Path(__file__),Path(J.__file__),Path(I.__file__)]
    for split in ('cal','validation'):
        paths += [OUT/'features'/f'{split}_features.npz',J.OUT/'features'/f'{split}_features.npz',J.OUT/f'{split}_scores.npz']
    inputs = {p.relative_to(ROOT).as_posix():C.sha(p) for p in paths}
    C.save(OUT/'PLAN.json',dict(task_sha256=C.sha(OUT/'TASK.json'),params=I.PARAMS,
        dimensions=dict(current_control=47,weak_mass=103),inputs_sha256=inputs,source_sha256=C.sha(Path(__file__))))
    try:
        data = C.load()
        fit_mask,partition = C.split_cal(data['cal']['rows'])
        C.save(OUT/'cal_partition.json',partition)
        th = C.read(C.PARENT/'thresholds.json')
        current,weak,parents,old_scores = {},{},{},{}
        for split,d in data.items():
            with np.load(J.OUT/'features'/f'{split}_features.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['scene_ids'],d['scene_ids'])
                current[split],current_names = J.descriptors(a,'current')
            with np.load(OUT/'features'/f'{split}_features.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['scene_ids'],d['scene_ids'])
                x,v,names=a['features'].astype(float),a['valid'],a['names'].tolist()
                weak[split]=np.concatenate((np.where(v,x,np.nan),(~v).astype(float)),-1)
            with np.load(H.OUT/f'{split}_grades.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['scene_ids'],d['scene_ids'])
                parents[split]=dict(zip(a['keys'].tolist(),a['grades']))
            with np.load(J.OUT/f'{split}_scores.npz',allow_pickle=False) as a:
                old_scores[split]=dict(zip(a['keys'].tolist(),a['scores']))
        C.save(OUT/'feature_names.json',dict(current_control=['ordinary_smooth_margin','ordinary_raw_slope5','ordinary_raw_detrended_fluctuation5']+current_names,
            weak_mass_extra=names+[n+'_missing' for n in names]))
        metrics,cuts,models,cohorts,summary = {},{},{},{},[]
        grades_saved={s:[] for s in data};scores_saved={s:[] for s in data}
        parity={}
        for si,seed in enumerate(G.SEEDS):
            base,tensors,head50={}, {}, {}
            for split,d in data.items():
                base[split] = H.baseline(d,si,th[str(seed)])
                head50[split]=parents[split][f'{seed}/head50']
                ordinary=d['candidates'][0,si]
                raw=E.read_job(C.SOURCE/f'scores/ordinary_seed{seed}_{split}.npz','ordinary',seed,split,'ideal')
                score_features=C.build_score_features(raw,ordinary,th[str(seed)]['single'])
                control=np.concatenate((score_features,current[split]),-1)
                tensors[split]=dict(current_control=control,weak_mass=np.concatenate((control,weak[split]),-1))
            arm_grades={}
            for arm in ARMS:
                score={s:np.empty(b.shape,dtype=float) for s,b in base.items()}
                for q,height in enumerate(E.HEIGHTS):
                    if time.monotonic()-started>=360: raise TimeoutError('Fit/evaluate360 CPU command-wall s cap')
                    model,record=J.fit_height(tensors['cal'][arm],data['cal']['category'],base['cal'],fit_mask,q)
                    path=OUT/f'models/{seed}_{arm}_{height}.pickle';path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(pickle.dumps(model,protocol=5))
                    models[f'{seed}/{arm}/{height}']=dict(**record,path=path.relative_to(OUT).as_posix(),sha256=C.sha(path),dimensions=tensors['cal'][arm].shape[-1])
                    for split in data:
                        x=tensors[split][arm][...,q,:]
                        score[split][...,q]=model.decision_function(x.reshape(-1,x.shape[-1])).reshape(x.shape[:-1])
                if arm=='current_control':
                    for split in data:
                        delta=float(abs(score[split]-old_scores[split][f'{seed}/score_current']).max())
                        np.testing.assert_allclose(score[split],old_scores[split][f'{seed}/score_current'],rtol=1e-10,atol=1e-10)
                        parity[f'{split}/{seed}']=delta
                cut=calibrate(score['cal'],base['cal'],head50['cal'],data['cal']['category'],~fit_mask)
                cuts[f'{seed}/{arm}']=cut;theta=-np.inf if cut['nonbinding'] else cut['theta']
                arm_grades[arm]={}
                for split,d in data.items():
                    grade=np.where((base[split]==0)&np.isfinite(score[split])&(score[split]>=theta),1,base[split]).astype(np.int8)
                    np.testing.assert_array_equal(grade==2,base[split]==2)
                    assert np.all(grade>=base[split])
                    arm_grades[arm][split]=grade
                    refs=dict(prior_light=head50[split]>0,ordinary_OR=base[split]==2,M3=d['m3']>=E.M3_THETA,old_fusion=E.old_fusion(d['m3'],d['local']))
                    report=G.describe(grade,d,refs)
                    report.update(cost=costs(grade>0,d['category']),head50_cost=costs(head50[split]>0,d['category']),
                                  paired_vs_head50=G.paired_timing(head50[split]>0,grade>0,d['category']),
                                  physical_vs_head50=H.B.physical_pair(head50[split]>0,grade>0,d['category']))
                    key=f'{split}/{seed}/{arm}';metrics[key]=report
                    cohorts[key]=dict(outcomes=outcomes(grade,d['category'],d['rows']),paired_vs_head50=paired_cohorts(head50[split],grade,d['category'],d['rows']))
                    summary.append(dict(split=split,seed=seed,arm=arm,HEAD=report['any']['counts'][0],BODY=report['any']['counts'][1],
                        HEAD_rescue=report['paired_vs_head50'][0]['rescue'],HEAD_loss=report['paired_vs_head50'][0]['loss'],
                        BODY_rescue=report['paired_vs_head50'][1]['rescue'],BODY_loss=report['paired_vs_head50'][1]['loss'],**report['cost']))
                    grades_saved[split].append((f'{seed}/{arm}',grade));scores_saved[split].append((f'{seed}/{arm}',score[split]))
            for split,d in data.items():
                key=f'{split}/{seed}/weak_mass'
                metrics[key]['paired_vs_current_control']=G.paired_timing(arm_grades['current_control'][split]>0,arm_grades['weak_mass'][split]>0,d['category'])
                cohorts[key]['paired_vs_current_control']=paired_cohorts(arm_grades['current_control'][split],arm_grades['weak_mass'][split],d['category'],d['rows'])
            print(f'seed {seed} COMPLETE {time.monotonic()-started:.3f}s',flush=True)
        for split,d in data.items():
            np.savez_compressed(OUT/f'{split}_grades.npz',keys=np.array([k for k,v in grades_saved[split]]),grades=np.array([v for k,v in grades_saved[split]]),scene_ids=d['scene_ids'])
            np.savez_compressed(OUT/f'{split}_scores.npz',keys=np.array([k for k,v in scores_saved[split]]),scores=np.array([v for k,v in scores_saved[split]]),scene_ids=d['scene_ids'])
        G.write_csv(OUT/'summary.csv',summary)
        for name,value in [('metrics',metrics),('calibrations',cuts),('models',models),('cohorts',cohorts),('control_checkpoint_parity',parity)]: C.save(OUT/f'{name}.json',value)
        C.save(OUT/'receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-started,cells=len(summary),fits=len(models),
            params=I.PARAMS,backend='CPU sklearn HGB fixed small table; feature GPU phase recorded separately',source_sha256=C.sha(Path(__file__)),
            persistent_resources=0,new_backbone_forward=0,new_data=0))
        print('COMPLETE',round(time.monotonic()-started,3),flush=True)
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-started));raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--declare',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        declare() if args.declare else run()
