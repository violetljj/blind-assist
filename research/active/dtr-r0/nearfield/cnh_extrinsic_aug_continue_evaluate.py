"""Authorized continuation routing; original scientific functions remain frozen.

The original pilot FAIL stays in its parent root. This process accepts only the
explicit child root and verifies its original evaluation block byte-equivalent
as canonical JSON. V is a separately calibrated descriptive comparator.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import cnh_extrinsic_aug_evaluate as E

PARENT=E.OUT
OUT=PARENT/'continuation-r1'
SEEDS=[0,1,2]

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)

def revision_guard(out=OUT):
    out=Path(out)
    if out.resolve()!=OUT.resolve():raise ValueError('Explicit continuation-r1 root required; never route to original FAIL root')
    plan=E.read(out/'PLAN.json');original=E.read(PARENT/'PLAN.json')
    revision_path=out/'PLAN_REVISION.json';revision=E.read(revision_path)
    if revision['status']!='AUTHORIZED_POSTHOC_REVISION' or revision['new_predicate']!='net>=0':raise ValueError('Explicit authorized revision required')
    if revision['new_plan_sha256']!=E.sha(out/'PLAN.json') or revision['parent_plan_sha256']!=E.sha(PARENT/'PLAN.json'):raise ValueError('Revision PLAN identities differ')
    copies=list((out/'source').glob('PARENT_PLAN_*.json'))
    if len(copies)!=1 or E.sha(copies[0])!=revision['parent_plan_sha256']:raise ValueError('Complete parent PLAN bytecopy required')
    if canonical(plan['evaluation'])!=canonical(original['evaluation']):raise ValueError('Original evaluation block must remain completely identical')
    if time.time()>=plan['deadline_unix']:raise TimeoutError('Authorized continuation deadline reached')
    gate_path=out/plan['calibration']['pilot_gate_path'];gate=E.read(gate_path)
    if gate.get('judgment')!='PASS':raise PermissionError('Authorized revised pilot PASS required')
    if gate.get('original_judgment')!='FAIL' or gate['revision_sha256']!=E.sha(revision_path) or gate['plan_sha256']!=E.sha(out/'PLAN.json'):raise ValueError('Revised gate identity mismatch')
    if E.read(PARENT/'evaluator/pilot_gate.json').get('judgment')!='FAIL':raise ValueError('Preserve original pilot FAIL')
    if gate['original_pilot_sha256']!=E.sha(PARENT/'evaluator/pilot_gate.json'):raise ValueError('Original sealed FAIL identity mismatch')
    return gate_path

def configure(out=OUT):
    revision_guard(out)
    E.require_pilot_pass=revision_guard
    return Path(out)

def require_final_calibration(out=OUT):
    out=configure(out);gate=revision_guard(out);p=out/'evaluator/final_calibration.json';c=E.read(p)
    if c.get('status')!='COMPLETE' or c.get('seeds')!=SEEDS:raise PermissionError('Augmented ensemble3 final calibration must seal before evaluation')
    if c['plan_sha256']!=E.sha(out/'PLAN.json') or c['revision_gate_sha256']!=E.sha(gate):raise ValueError('Final calibration revision identity mismatch')
    return c

def final_calibration(out=OUT):
    out=configure(out);gate=revision_guard(out);g,m3,aug,seeds,hashes=E.load_scores(out,'calibration/ensemble3')
    if seeds!=SEEDS:raise ValueError('Fixed augmented seeds0/1/2 required')
    augmented=E.cost_threshold(aug[:,1:].max(1),m3[:,0],g['clear_all'])
    if augmented['status']!='COMPLETE':raise ValueError('Augmented physical-clear calibration not evaluable')
    gv,mv,v,vs,vhashes=E.load_scores(out,'calibration/V3')
    if vs!=SEEDS:raise ValueError('Fixed V comparator seeds0/1/2 required')
    np.testing.assert_array_equal(m3,mv)
    for key in ('unit','config','query','clear_all','covered','ref_category','frame_ranges'):np.testing.assert_array_equal(g[key],gv[key])
    comparator=E.cost_threshold(v[:,1:].max(1),m3[:,0],g['clear_all'])
    if comparator['status']!='COMPLETE':raise ValueError('V physical-clear calibration not evaluable')
    common=dict(plan_sha256=E.sha(out/'PLAN.json'),revision_gate_sha256=E.sha(gate),original_plan_sha256=E.sha(PARENT/'PLAN.json'),
        original_pilot_fail_sha256=E.sha(PARENT/'evaluator/pilot_gate.json'),script_sha256=E.sha(__file__),
        threshold_function_sha256=E.read(PARENT/'evaluator/THRESHOLD_CONTRACT.json')['threshold_function_sha256'])
    v_result=dict(**comparator,**common,seeds=vs,phase='calibration/V3',model='V',descriptive_only=True,input_sha256=vhashes,
        sharing='One scalar for V single/dual/HEAD/BODY, same original physicalcost rule, no V outcome enters main gates')
    E.save(out/'evaluator/V_calibration.json',v_result)
    result=dict(**augmented,**common,seeds=seeds,phase='calibration/ensemble3',model='M3_aug',input_sha256=hashes,
        V_calibration_sha256=E.sha(out/'evaluator/V_calibration.json'),selection_bias='Same original calibration, revised pilot gate observed outcomes; exploratory continuation')
    E.save(out/'evaluator/final_calibration.json',result)
    print('FINAL CALIBRATIONS SEALED',json.dumps(dict(augmented_threshold=result['threshold'],V_threshold=v_result['threshold'])),flush=True)
    return result

def natural(out=OUT):
    out=configure(out);require_final_calibration(out);main=E.evaluate_natural(out,'evaluation/ensemble3')
    g,m3,v,seeds,hashes=E.load_scores(out,'evaluation/V3');vc=E.read(out/'evaluator/V_calibration.json')
    if seeds!=SEEDS:raise ValueError('V ensemble mismatch')
    with np.load(out/'evaluator/natural_evaluation_ledger.npz') as z:
        np.testing.assert_array_equal(z['S_M3'],m3[:,0]);np.testing.assert_array_equal(z['DUAL_M3'],m3[:,1:].max(1))
        ledger={k:z[k].copy() for k in z.files}
    groups=dict(all=np.ones(len(g['unit']),bool),mode0=g['mode']==0,mode1=g['mode']==1,mode2=g['mode']==2,
        turn_left=(g['mode']==2)&(g['turn']=='left'),turn_right=(g['mode']==2)&(g['turn']=='right'))
    cells=dict(main['cells']);v_arms=dict(S_V=v[:,0],DUAL_V=v[:,1:].max(1))
    for arm,score in v_arms.items():cells[arm]={name:E.metric_cell(score,vc['threshold'],g,mask,baseline=m3[:,0]) for name,mask in groups.items()}
    ledger.update(**v_arms);E.save_npz(out/'evaluator/natural_six_arm_ledger.npz',**ledger)
    paired={arm:{name:E.metric_cell(score,vc['threshold'],g,mask,baseline=ledger[{'S_V':'S_NEW','DUAL_V':'DUAL_NEW'}[arm]],baseline_threshold=main['threshold']) for name,mask in groups.items()} for arm,score in v_arms.items()}
    result=dict(main);result.update(cells=cells,V_threshold=vc['threshold'],V_descriptive_only=True,paired_V_vs_augmented=paired,
        paired_V_vs_augmented_sign='Positive net means V timely count exceeds augmented at each independently calibrated threshold',
        original_four_arm_result_sha256=E.sha(out/'evaluator/natural_result.json'),V_input_sha256=hashes,
        six_arm_ledger_sha256=E.sha(out/'evaluator/natural_six_arm_ledger.npz'))
    E.save(out/'evaluator/natural_six_arm_result.json',result);return result

def angular(out=OUT):
    out=configure(out);require_final_calibration(out);main=E.evaluate_angular(out,'angular/ensemble3')
    vc=E.read(out/'evaluator/V_calibration.json');receipt=out/'scores/angular/V3/receipt.json'
    if E.read(receipt).get('status')!='COMPLETE':raise ValueError('Whole V angularbatch must seal first')
    with np.load(out/'evaluator/angular_evaluation_ledger.npz') as z:ledger={k:z[k].copy() for k in z.files}
    import cnh_dual_sensor_evaluate as D
    old=D.load_baseline();indices=[old['units'].tolist().index(int(u)) for u in ledger['units']];scenes=[old['scenes'][i] for i in indices]
    axes=[-15,-5,0,5,10,15,20,25,30,35,40,45];lookup={a:i for i,a in enumerate(axes)};refs=[];values=[];hashes={str(receipt):E.sha(receipt)}
    for u in ledger['units']:
        path=out/f'scores/angular/V3/unit{int(u)}.npz'
        with np.load(path) as z:
            if z['seeds'].tolist()!=SEEDS or z['axes_degrees'].tolist()!=axes:raise ValueError('V angular optical axes/ensemble mismatch')
            if z['reference'].shape!=(12,28,13,2) or z['augmented_mean'].shape!=(12,28,13,2):raise ValueError('V angular identity/shape mismatch')
            refs.append(E.smooth(z['reference']).reshape(12,7,4,13,2));values.append(E.smooth(z['augmented_mean']).reshape(12,7,4,13,2))
        hashes[str(path)]=E.sha(path)
    refs=np.stack(refs);values=np.stack(values);nscene=len(scenes);masks={};covered=ledger['covered'];ranges=ledger['ranges'];clear=ledger['clear_query']
    for group in ('all','opposite','same'):
        shallow=np.zeros((nscene,7,4,2),bool);deep=shallow.copy();eligible=shallow.copy()
        for i,s in enumerate(scenes):
            if group=='all' or (group=='same')==bool(s['fov_in']):
                shallow[i,:2,:,s['group']]=covered[i,:2,None];deep[i,2,:,s['group']]=covered[i,2];eligible[i,:3,:,s['group']]=True
        masks[group]=(shallow.reshape(-1),deep.reshape(-1),eligible.reshape(-1))
    sequence=dict(main['sequence']);paired_sequence={}
    for psi in main['psi_deg']:
        single_ref=refs[:,lookup[psi]].transpose(0,1,2,4,3).reshape(-1,13)
        dual_ref=refs[:,[lookup[psi-15],lookup[psi+15]]].max(1).transpose(0,1,2,4,3).reshape(-1,13)
        np.testing.assert_array_equal(single_ref,ledger[f'psi{psi}_S_M3']);np.testing.assert_array_equal(dual_ref,ledger[f'psi{psi}_DUAL_M3'])
        arms=dict(S_V=values[:,lookup[psi]].transpose(0,1,2,4,3).reshape(-1,13),DUAL_V=values[:,[lookup[psi-15],lookup[psi+15]]].max(1).transpose(0,1,2,4,3).reshape(-1,13))
        base=E.events(single_ref,E.ORIGINAL_THRESHOLD,ranges);cells=dict(sequence[str(psi)]);paired_cells={}
        for arm,score in arms.items():
            ev=E.events(score,vc['threshold'],ranges);cell={}
            for group,(shallow,deep,eligible) in masks.items():
                item={}
                for label,den in (('shallow',shallow),('deep',deep)):
                    n=int(den.sum());num=int((den&ev['timely']).sum());rescue=int((den&ev['timely']&~base['timely']).sum());loss=int((den&~ev['timely']&base['timely']).sum())
                    item[label]=dict(timely=num,n=n,rate=num/n if n else None,rescue=rescue,loss=loss,net=rescue-loss,baseline_timely=int((den&base['timely']).sum()))
                missing=eligible&~np.broadcast_to(covered[:,:,None,None],(nscene,7,4,2)).reshape(-1)
                item['censored']=dict(n=int(missing.sum()),stopped=int((missing&ev['stopped']).sum()));cell[group]=item
            qn=int(clear.sum());qs=int((clear&ev['stopped']).sum());jc=E.joint_clear(clear);pn=int(jc.sum());ps=int((jc&ev['stopped'].reshape(-1,2).any(1)).sum())
            cell['query_clear']=dict(stops=qs,n=qn,proxy_minutes=qn*2.6/60,rate_per_proxy_minute=qs/(qn*2.6/60) if qn else None)
            cell['physical_clear']=dict(stops=ps,n=pn,proxy_minutes=pn*2.6/60,rate_per_proxy_minute=ps/(pn*2.6/60) if pn else None)
            augmented_arm={'S_V':'S_NEW','DUAL_V':'DUAL_NEW'}[arm];aug_ev=E.events(ledger[f'psi{psi}_{augmented_arm}'],main['threshold'],ranges);paired={}
            for group,(shallow,deep,eligible) in masks.items():
                paired[group]={}
                for label,den in (('shallow',shallow),('deep',deep)):
                    rescue=int((den&ev['timely']&~aug_ev['timely']).sum());loss=int((den&~ev['timely']&aug_ev['timely']).sum())
                    paired[group][label]=dict(rescue=rescue,loss=loss,net=rescue-loss,n=int(den.sum()),V_timely=int((den&ev['timely']).sum()),augmented_timely=int((den&aug_ev['timely']).sum()))
            paired_cells[arm]=paired;cells[arm]=cell;ledger[f'psi{psi}_{arm}']=score
            for key,val in ev.items():ledger[f'psi{psi}_{arm}_{key}']=val
        sequence[str(psi)]=cells;paired_sequence[str(psi)]=paired_cells
    E.save_npz(out/'evaluator/angular_six_arm_ledger.npz',**ledger)
    result=dict(main);result.update(sequence=sequence,V_threshold=vc['threshold'],V_descriptive_only=True,V_input_sha256=hashes,paired_V_vs_augmented=paired_sequence,
        paired_V_vs_augmented_sign='Positive net means V timely count exceeds augmented at each independently calibrated threshold',
        original_four_arm_result_sha256=E.sha(out/'evaluator/angular_result.json'),six_arm_ledger_sha256=E.sha(out/'evaluator/angular_six_arm_ledger.npz'))
    E.save(out/'evaluator/angular_six_arm_result.json',result);return result

def finish(out=OUT):
    out=configure(out);require_final_calibration(out)
    paths=[out/'evaluator/natural_six_arm_result.json',out/'evaluator/angular_six_arm_result.json']
    if any(E.read(p).get('status')!='COMPLETE' for p in paths):raise ValueError('Six-arm descriptive deliverables must complete')
    main=E.finalize(out);result=dict(main)
    result.update(original_pilot_judgment='FAIL',revised_pilot_judgment='PASS',
        revision_sha256=E.sha(out/'PLAN_REVISION.json'),wrapper_sha256=E.sha(__file__),
        original_scientific_source_sha256=E.sha(E.__file__),V_descriptive_only=True,
        six_arm_result_sha256={str(p):E.sha(p) for p in paths})
    E.save(out/'evaluator/continuation_result.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('calibrate-final','geometry','evaluate-natural','evaluate-angular','finalize'))
    p.add_argument('--root',type=Path,required=True);args=p.parse_args();out=configure(args.root)
    if args.stage=='calibrate-final':final_calibration(out)
    elif args.stage=='geometry':require_final_calibration(out);E.geometry_evaluation(out)
    elif args.stage=='evaluate-natural':natural(out)
    elif args.stage=='evaluate-angular':angular(out)
    else:finish(out)
