"""Optional +/-22.5deg arm, selected from geometry before alarm inspection.

Shares the main run's wall-clock deadline, input cohort, frozen models and RNG.
No new threshold is fitted. The main +/-15 calibration threshold is diagnostic.
"""
from datetime import datetime, timezone
import argparse
import shutil
import time

import numpy as np

import cnh_dual_sensor_alarm as A

MAIN = A.OUT
OUT = MAIN/'secondary22p5'
ANGLE = 22.5
ORIGINAL_EXTRINSIC = A.extrinsic


def configure():
    A.OUT = OUT
    # A's fixed selectors -15/+15 are left/right labels in this reused path.
    # Only their physical extrinsic angles change; photon sensor indices stay.
    A.extrinsic = lambda selector: ORIGINAL_EXTRINSIC(float(np.sign(selector))*ANGLE)


def freeze():
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Frozen secondary PLAN already exists')
    main = A.read(MAIN/'PLAN.json')
    plan = dict(main)
    plan.update(run='CNH_DUAL_SENSOR_SECONDARY22P5_20261005',
        created_utc=datetime.now(timezone.utc).isoformat(), splay_deg=[-ANGLE,ANGLE],
        parent_plan_sha256=A.sha(MAIN/'PLAN.json'),
        selection='Parent delegation selected22.5 from geometry before this agent inspected any main alarm outcome. Highest eligible mean, >=15deg+5pp and center3/3 at all3 heights:22.5deg91.79%, vs15deg79.64%. Main result already existed when supplementary PLAN file was written; do not claim PLAN preceded result production. No alarm-based selection.',
        main_result_existed_at_supplement_freeze=(MAIN/'result.json').exists(),
        splay_result_sha256=A.sha(MAIN/'splay/result.json'),
        selector_mapping='Reused A left/right selectors -15/+15 map to -22.5/+22.5 in extrinsic only. Same original center poses, pitch-10, same photon prefix and shared estimated head trajectory.',
        matching='No new calibration. Only original frozen threshold and the already calibrated main +/-15 threshold as a transferred diagnostic.',
        interpretation='Secondary diagnostic only; excluded from primary +/-15 decision. Report all/evaluation OUT/IN shallow/deep and joint clear; no arm promotion.',
        source_sha256={A.__file__:A.sha(A.__file__), __file__:A.sha(__file__)})
    OUT.mkdir(parents=True, exist_ok=True)
    A.save(OUT/'PLAN.json',plan)
    (OUT/'PLAN.sha256').write_text(A.sha(OUT/'PLAN.json')+'\n',encoding='utf8')
    source=OUT/'source'; source.mkdir()
    for path in (A.__file__,__file__): shutil.copy2(path,source)
    A.save(OUT/'request.json',dict(plan_sha256=A.sha(OUT/'PLAN.json'),deadline_unix=main['deadline_unix'],budget='inherited; never reset'))
    print('SECONDARY FROZEN',A.sha(OUT/'PLAN.json'),flush=True)


def gpu():
    # Explicit handoff receipt ensures the priority main job completed and freed GPU.
    assert A.read(MAIN/'inference_result.json')['status']=='COMPLETE'
    release=A.read(MAIN/'inference_release.json')
    process_release=A.read(OUT/'main_process_release.json')
    assert process_release['main_python_process_count']==0
    # Function-local tensors may survive finally; confirmed process exit is the
    # resource-release authority. The original finally counters are preserved.
    A.save(OUT/'engineering/gpu_handoff.json',dict(parent_release=release,
        process_release=process_release,source_sha256=A.sha(__file__),
        change='Engineering handoff correction: require main process absence, not local finally tensor counters. No model/data/threshold/geometry change.'))
    configure(); A.deadline(); A.engineering(); A.render(); A.infer()


def evaluate():
    import cnh_dual_sensor_evaluate as E
    configure(); A.deadline()
    if (OUT/'result.json').exists(): raise FileExistsError('Secondary result immutable')
    plan=A.read(OUT/'PLAN.json'); main=A.read(MAIN/'result.json')
    assert main['status']=='COMPLETE'
    data=E.load_baseline(); units=data['units']; scenes=data['scenes']
    assert units.tolist()==plan['units']
    splits=E.split_masks(plan,units)
    full=[]; alt=[]; hashes={}
    for u in units:
        path=OUT/'scores'/f'unit{u}.npz'
        h=A.sha(path); receipt=A.read(path.with_suffix('.json'))
        assert h==receipt['score_sha256'] and receipt['plan_sha256']==A.sha(OUT/'PLAN.json')
        with np.load(path,allow_pickle=False) as z:
            assert int(z['unit'])==u and np.array_equal(z['frames'],E.FRAMES)
            full.append(E.smooth_full(z['raw_full']))
            alt.append(E.smooth_alternating(z['raw_alternating']))
        hashes[str(path)]=h
    full,alt=np.stack(full),np.stack(alt)
    thresholds={'single':E.THRESHOLD,'secondary_frozen':E.THRESHOLD,
        'secondary_transferred_main_threshold':main['thresholds']['dual_matched']['threshold'],
        'secondary_alternating_frozen':E.THRESHOLD}
    both={'single':data['both'],'secondary_frozen':full.max(1),
        'secondary_transferred_main_threshold':full.max(1),'secondary_alternating_frozen':alt.max(1)}
    ranges=np.asarray([s['front_range_m'] for s in scenes])
    ledgers={a:E.L.events(E.select_target(b,scenes),{'threshold':thresholds[a],'operator':'>='},ranges,data['covered']) for a,b in both.items()}
    clear_den=E.joint_clear_den(data)
    stops={a:b[:,[5,6]].max(axis=(-2,-1))>=thresholds[a] for a,b in both.items()}
    sequence={}
    for split_index,(split,split_keep) in enumerate(splits.items()):
        boot=E.boot_weights(split_keep,E.BOOT_SEED+split_index); sequence[split]={}
        for group,gkeep in E.group_masks(scenes).items():
            keep=split_keep&gkeep; cells=sequence[split][group]={}
            for arm,event in ledgers.items():
                cell=cells[arm]=dict(scenes=int(keep.sum()),branches={})
                for name,ids in E.BRANCHES.items():
                    den=np.broadcast_to(data['covered'][:,ids,None],(48,len(ids),4))
                    metric,_=E.pooled(event['timely'][:,ids]&den,den,keep,boot)
                    cell['branches'][name]=dict(timely=metric,
                        paired_minus_single=E.paired(event['timely'][:,ids],ledgers['single']['timely'][:,ids],den,keep,boot))
                metric,_=E.pooled(stops[arm]&clear_den,clear_den,keep,boot)
                cell['joint_clear']=dict(**metric,paired_minus_single=E.paired(stops[arm],stops['single'],clear_den,keep,boot))
    comparison={}
    for split in ('all','evaluation'):
        comparison[split]={}
        for group in ('all','FOV_OUT','FOV_IN'):
            comparison[split][group]={'secondary':sequence[split][group],
                'main15':main['sequence'][split][group]}
    result=dict(status='COMPLETE',scope='Consumed synthetic Development; optional geometry-selected angle; excluded primary decision',
        angle=ANGLE,thresholds=thresholds,threshold_fit='NONE; transferred diagnostic is main15 threshold unchanged',
        sequence=sequence,comparison_to_main15=comparison,
        provenance=dict(plan_sha256=A.sha(OUT/'PLAN.json'),parent_result_sha256=A.sha(MAIN/'result.json'),scores_sha256=hashes,source_sha256=A.sha(__file__)),
        completed_unix=time.time(),deadline_unix=plan['deadline_unix'])
    A.save(OUT/'result.json',result)
    lines=['# +/-22.5deg secondary alarm diagnostic','','Parent selected22.5 from geometry before inspecting main alarm outcomes. Supplementary PLAN was written after main result already existed; the selection did not use alarm scores. No new threshold calibration. Synthetic Development only.','',
        '| split | FOV | arm | shallow timely | deep timely | joint clear |', '|---|---|---|---|---|---|']
    def frac(x): return f"{x['stops']}/{x['n']}"
    for split in ('all','evaluation'):
        for group in ('all','FOV_OUT','FOV_IN'):
            for arm,cell in sequence[split][group].items():
                lines.append(f"| {split} | {group} | {arm} | {frac(cell['branches']['shallow']['timely'])} | {frac(cell['branches']['deep']['timely'])} | {frac(cell['joint_clear'])} |")
            for arm in ('dual_frozen','dual_matched','alternating_frozen'):
                cell=main['sequence'][split][group][arm]
                lines.append(f"| {split} | {group} | main15/{arm} | {frac(cell['branches']['shallow']['timely'])} | {frac(cell['branches']['deep']['timely'])} | {frac(cell['joint_clear'])} |")
    lines+=['','All branch denominators, group/mode tables and paired scene bootstrap intervals are retained in result.json. FOV follows original single labels. Joint-clear counts each physical outer15/20cm episode once when both queries are clear throughout retained frames. Mode2 has no samples.','',
        f"PLAN SHA256: {A.sha(OUT/'PLAN.json')}; source and score hashes in result.json. Mother deadline inherited: {plan['deadline_unix']}. No measured hardware SNR/crosstalk/power evidence."]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('SECONDARY EVALUATED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['freeze','gpu','evaluate']); args=parser.parse_args()
    try: {'freeze':freeze,'gpu':gpu,'evaluate':evaluate}[args.stage]()
    except Exception as exc:
        A.save(OUT/'failures'/f'{args.stage}-{time.time_ns()}.json',dict(error=repr(exc),stage=args.stage))
        raise
