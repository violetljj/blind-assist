"""Evaluator-only mirrored natural geometry and frozen M3-aug cost protocol.

Calibration/gate use fresh98000 only. Evaluation access is denied until the
seed0 pilot gate passes. Models, observations, labels and old verdicts are not
modified; geometry receives the actual mirrored travel rather than default yaw.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-extrinsic-aug-20261006'
ORIGINAL_THRESHOLD=.8557642486787612
FRAMES=np.arange(3,16)
CAL_UNITS=list(range(98000,98048))
EVAL_UNITS=list(range(99000,99096))
CONTACTS=('contact0-2cm','contact2-5cm','contact>5cm')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def save_npz(path,**data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve existing payload: '+str(path))
    np.savez_compressed(path,**data)

def deadline(out):
    if time.time()>=read(Path(out)/'PLAN.json')['deadline_unix']:raise TimeoutError('Four-hour task deadline reached')

def require_pilot_pass(out):
    path=Path(out)/'evaluator/pilot_gate.json'
    if not path.exists() or read(path).get('judgment')!='PASS':raise PermissionError('Evaluation is unavailable before pilot PASS')
    return path

def smooth(raw):
    raw=np.asarray(raw,np.float64)
    if raw.shape[-2:]!=(13,2) or not np.isfinite(raw).all():raise ValueError('Expected finite original13frame2query raw logits')
    result=np.empty_like(raw)
    for t in range(13):
        begin=max(0,t-4);weights=2.**np.arange(t-begin+1)
        result[...,t,:]=sum(raw[...,f,:]*w for f,w in zip(range(begin,t+1),weights))/weights.sum()
    return result

def geometry_scene(boxes,travel):
    """Two original surface-query rows, using explicit true mirrored travel."""
    import cnh_sequence_observed_geometry as G
    travel=np.asarray(travel,float)
    if travel.shape!=(16,4,4):raise ValueError('Actual travel16 poses required')
    poses=travel[FRAMES];target=G.corners(boxes[0]);triangles=np.concatenate([G.S.box_mesh(b['lo'],b['hi']) for b in boxes])
    ranges,ref=G.deadline_reference(target,poses)
    local=[G.transform(triangles,p) for p in poses]
    ref_tri=G.transform(triangles,ref['reference_pose']) if ref['covered'] else None
    rows=[]
    for q in (0,1):
        cats=np.array([G.surface_category(t,q) for t in local])
        rows.append(dict(query=q,frame_ranges=ranges,frame_category=cats,clear_all=bool(np.all(cats=='clear')),
            covered=bool(ref['covered']),ref_category=G.surface_category(ref_tri,q) if ref['covered'] else 'censored',
            reference_time_s=ref['reference_time_s'],reference_fraction=ref['reference_fraction'],
            censor_reason=ref['censor_reason'],last_category=str(cats[-1])))
    return rows

def geometry(out=OUT,split='calibration'):
    out=Path(out);deadline(out)
    if split=='evaluation':return geometry_evaluation(out)
    if split not in ('calibration','evaluation'):raise ValueError('Unknown split')
    units=CAL_UNITS if split=='calibration' else EVAL_UNITS;inputs={};allrows=[];tick=time.monotonic()
    snapshot=out/f'source/cnh_extrinsic_aug_evaluate_geometry_{split}.py';snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists():raise FileExistsError('Inspect previous geometry attempt before rerunning: '+str(snapshot))
    snapshot.write_bytes(Path(__file__).read_bytes());execution_source_sha=sha(snapshot)
    for unit in units:
        source=out/f'truth/{split}/unit{unit}.json';dest=out/f'geometry/{split}/unit{unit}.npz'
        while not source.exists():
            deadline(out);time.sleep(1.)
        truth=read(source);inputs[str(source)]=sha(source);assert int(truth['unit'])==unit
        travel=np.asarray(truth['travel']);scenes=truth['scenes'];assert len(scenes)==40
        if [int(s['config']) for s in scenes]!=list(range(40)):raise ValueError('All40 frozen configurations required')
        yaw=np.unwrap(np.arctan2(travel[:,0,2],travel[:,2,2]));direction='right' if yaw[-1]>yaw[0]+1e-10 else 'left' if yaw[-1]<yaw[0]-1e-10 else 'none'
        if dest.exists():
            receipt=read(dest.with_suffix('.json'));assert receipt['truth_sha256']==inputs[str(source)] and receipt['geometry_sha256']==sha(dest)
            with np.load(dest) as z:rows=[{k:z[k][i] for k in z.files} for i in range(len(z['unit']))]
        else:
            rows=[]
            for scene in scenes:
                for row in geometry_scene(scene['boxes'],travel):
                    row.update(unit=unit,config=int(scene['config']),mode=int(truth['mode']),mirror=bool(truth['mirror']),turn=direction)
                    rows.append(row)
            arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]};save_npz(dest,**arrays)
            save(dest.with_suffix('.json'),dict(status='COMPLETE',unit=unit,truth_sha256=inputs[str(source)],geometry_sha256=sha(dest),execution_source_sha256=execution_source_sha))
        allrows.extend(rows);print('geometry',split,unit,len(allrows),'rows',round(time.monotonic()-tick,1),'seconds',flush=True)
    data={k:np.asarray([r[k] for r in allrows]) for k in allrows[0]}
    merged=out/f'geometry/{split}.npz';save_npz(merged,**data)
    save(merged.with_suffix('.json'),dict(status='COMPLETE',split=split,query_rows=len(allrows),units=units,configs=40,inputs_sha256=inputs,
        geometry_sha256=sha(merged),seconds=time.monotonic()-tick,execution_source_sha256=execution_source_sha,
        semantics='Unchanged original all-object surface_category, observed0.9m reference, clear acrossall13 frames; actual mirrored travel explicitly supplied'))
    return data

def geometry_evaluation(out=OUT):
    """Post-PASS only, support runtime-chosen uniform config subsets."""
    out=Path(out);require_pilot_pass(out);deadline(out);tick=time.monotonic()
    snapshot=out/'source/cnh_extrinsic_aug_evaluate_geometry_evaluation.py';snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists():raise FileExistsError('Evaluation geometry attempt immutable')
    snapshot.write_bytes(Path(__file__).read_bytes());source_hash=sha(snapshot)
    inputs={};allrows=[];configs_by_unit={};expected_count=None
    for u in EVAL_UNITS:
        path=out/f'truth/evaluation/unit{u}.json'
        while not path.exists():deadline(out);time.sleep(1.)
        truth=read(path);assert int(truth['unit'])==u;travel=np.asarray(truth['travel']);scenes=truth['scenes']
        configs=[int(s['config']) for s in scenes]
        if not configs or len(set(configs))!=len(configs) or not set(configs)<=set(range(40)):raise ValueError('Original global evaluation config IDs required')
        if expected_count is None:expected_count=len(configs)
        if len(configs)!=expected_count:raise ValueError('Uniform per-unit config count required')
        configs_by_unit[str(u)]=configs;inputs[str(path)]=sha(path)
        yaw=np.unwrap(np.arctan2(travel[:,0,2],travel[:,2,2]));direction='left' if yaw[-1]<yaw[0]-1e-10 else 'right' if yaw[-1]>yaw[0]+1e-10 else 'none'
        rows=[]
        for scene in scenes:
            for row in geometry_scene(scene['boxes'],travel):
                row.update(unit=u,config=int(scene['config']),mode=int(truth['mode']),mirror=bool(truth['mirror']),turn=direction);rows.append(row)
        dest=out/f'geometry/evaluation/unit{u}.npz';save_npz(dest,**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
        save(dest.with_suffix('.json'),dict(status='COMPLETE',unit=u,configs=configs,truth_sha256=inputs[str(path)],geometry_sha256=sha(dest),execution_source_sha256=source_hash))
        allrows.extend(rows);print('geometry evaluation',u,'configs',len(configs),'seconds',round(time.monotonic()-tick,1),flush=True)
    data={k:np.asarray([r[k] for r in allrows]) for k in allrows[0]};dest=out/'geometry/evaluation.npz';save_npz(dest,**data)
    save(dest.with_suffix('.json'),dict(status='COMPLETE',units=EVAL_UNITS,configs_by_unit=configs_by_unit,config_count=expected_count,
        query_rows=len(allrows),geometry_sha256=sha(dest),inputs_sha256=inputs,execution_source_sha256=source_hash,seconds=time.monotonic()-tick))
    return data

def events(scores,threshold,ranges):
    score=np.asarray(scores,float);alarm=score>=threshold;stopped=alarm.any(1);first=alarm.argmax(1)
    distance=np.take_along_axis(ranges,first[:,None],1)[:,0]
    return dict(stopped=stopped,timely=stopped&(distance>=.9),first_index=first,first_range=distance,alarm=alarm)

def joint_clear(clear):return np.asarray(clear,bool).reshape(-1,2).all(1)

def cost_threshold(new_dual,baseline_single,clear,reference_threshold=ORIGINAL_THRESHOLD):
    """One minimum conservative threshold matching physical joint-clear stops.

    Ties cannot be split. No threshold/label group or branch-specific fit.
    Lowest finite available score is used when the entire clear cohort fits.
    """
    clear=np.asarray(clear,bool);jc=joint_clear(clear)
    if not jc.any():return dict(status='NOT_EVALUABLE',reason='No physical jointly-clear calibration episodes')
    baseline=(np.asarray(baseline_single)>=reference_threshold).any(1).reshape(-1,2).any(1)
    budget=int((baseline&jc).sum());new=np.asarray(new_dual,float)
    if new.shape!=baseline_single.shape or not np.isfinite(new).all():raise ValueError('Finite paired query logit arrays required')
    maxima=new.reshape(-1,2,new.shape[-1]).max((1,2))[jc]
    values,counts=np.unique(maxima,return_counts=True);values=values[::-1];counts=counts[::-1]
    allowed=int((np.cumsum(counts)<=budget).sum())
    threshold=float(np.nextafter(values[allowed],np.inf)) if allowed<len(values) else float(np.nextafter(new.min(),-np.inf))
    observed=int((jc&(new>=threshold).any(1).reshape(-1,2).any(1)).sum())
    assert observed<=budget
    return dict(status='COMPLETE',threshold=threshold,operator='>=',calibration_physical_clear_n=int(jc.sum()),
        original_single_stops=budget,new_dual_stops=observed,quantization_unused_budget=budget-observed,
        reference_threshold=reference_threshold,rule='Lowest attainable finite logit threshold with wholephysical clear stops<=S_M3; no split of tied episodes')

def metric_cell(score,threshold,g,mask=None,baseline=None,baseline_threshold=ORIGINAL_THRESHOLD):
    mask=np.ones(len(score),bool) if mask is None else np.asarray(mask,bool);ev=events(score,threshold,g['frame_ranges'])
    base=events(baseline,baseline_threshold,g['frame_ranges']) if baseline is not None else None
    clear=mask&g['clear_all'];n=int(clear.sum());stops=int((clear&ev['stopped']).sum());minutes=n*2.6/60
    both=joint_clear(g['clear_all'])&mask.reshape(-1,2).all(1);pn=int(both.sum());ps=int((both&ev['stopped'].reshape(-1,2).any(1)).sum());pm=pn*2.6/60
    result=dict(query_clear=dict(stops=stops,n=n,proxy_minutes=minutes,rate_per_proxy_minute=stops/minutes if minutes else None),
        physical_clear=dict(stops=ps,n=pn,proxy_minutes=pm,rate_per_proxy_minute=ps/pm if pm else None),
        censored=dict(n=int((mask&~g['covered']).sum()),stopped=int((mask&~g['covered']&ev['stopped']).sum())))
    for label,cats in [('shallow0_2',[CONTACTS[0]]),('mid2_5',[CONTACTS[1]]),('shallow0_5',CONTACTS[:2]),('deep',CONTACTS[2:])]:
        den=mask&g['covered']&np.isin(g['ref_category'],cats);count=int(den.sum());num=int((den&ev['timely']).sum())
        item=dict(timely=num,n=count,rate=num/count if count else None,stopped=int((den&ev['stopped']).sum()))
        if base is not None:
            rescue=int((den&ev['timely']&~base['timely']).sum());loss=int((den&~ev['timely']&base['timely']).sum())
            item.update(rescue=rescue,loss=loss,net=rescue-loss,baseline_timely=int((den&base['timely']).sum()))
        result[label]=item
    return result

def load_scores(out,phase):
    out=Path(out);split=phase.split('/')[0];evaluation=split=='evaluation'
    if split not in ('calibration','evaluation'):raise ValueError('Scorephase must explicitly begin calibration/ or evaluation/')
    if evaluation:require_pilot_pass(out)
    units=EVAL_UNITS if evaluation else CAL_UNITS
    geo=out/f'geometry/{"evaluation" if evaluation else "calibration"}.npz'
    with np.load(geo) as z:g={k:z[k].copy() for k in z.files}
    reference=[];aug=[];hashes={str(geo):sha(geo)};seeds=None;keys=[];common_configs=None
    score_receipt=out/f'scores/{phase}/receipt.json'
    if read(score_receipt).get('status')!='COMPLETE':raise ValueError('Whole frozen scorebatch must be sealed before calibration/evaluation')
    hashes[str(score_receipt)]=sha(score_receipt)
    for u in units:
        p=out/f'scores/{phase}/unit{u}.npz'
        with np.load(p) as z:
            ref=z['reference'];new=z['augmented_mean'];ss=z['seeds'].tolist()
            if seeds is None:seeds=ss
            if ss!=seeds:raise ValueError('One consistent ensemble required')
            configs=z['configs'].tolist();count=len(configs)
            if ref.shape!=new.shape or ref.shape!=(3,count,13,2):raise ValueError('S/L/R×actualconfig×13frame×HEAD/BODY required')
            if not evaluation and configs!=list(range(40)):raise ValueError('Calibration is fixed all40 configs')
            if common_configs is None:common_configs=count
            if count!=common_configs:raise ValueError('Uniform config count required')
            if len(set(configs))!=count or not set(configs)<=set(range(40)):raise ValueError('Actual global config IDs required')
            if 'unit' in z:assert int(z['unit'])==u
            if 'frames' in z:np.testing.assert_array_equal(z['frames'],FRAMES)
            if 'sensors' in z:assert z['sensors'].tolist()==['S','L','R']
            reference.append(smooth(ref).transpose(1,3,0,2).reshape(count*2,3,13))
            aug.append(smooth(new).transpose(1,3,0,2).reshape(count*2,3,13))
            keys.extend((u,int(c),q) for c in configs for q in (0,1))
        hashes[str(p)]=sha(p)
    reference=np.concatenate(reference);aug=np.concatenate(aug)
    if list(zip(g['unit'].tolist(),g['config'].tolist(),g['query'].tolist()))!=keys:raise ValueError('Scores/geometry actualunit/config/query identities differ')
    return g,reference,aug,seeds,hashes

def calibrate(out=OUT,phase='calibration/seed0',pilot=True):
    out=Path(out);deadline(out)
    if not pilot:
        require_pilot_pass(out)
        if phase=='calibration/seed0':
            previous=out/'evaluator/seed0_calibration.json';selection=read(previous)
            assert selection['seeds']==[0] and selection['status']=='COMPLETE'
            selection.update(derived_from_frozen_seed0_calibration_sha256=sha(previous),budget_fallback='User-authorized seed0 only; no new thresholdselection or calibration rescoring')
            save(out/'evaluator/final_calibration.json',selection);return selection
    g,m3,aug,seeds,hashes=load_scores(out,phase)
    if pilot:assert seeds==[0],'Pilot uses seed0 only'
    arms=dict(S_M3=m3[:,0],DUAL_M3=m3[:,1:].max(1),S_NEW=aug[:,0],DUAL_NEW=aug[:,1:].max(1))
    fit=cost_threshold(arms['DUAL_NEW'],arms['S_M3'],g['clear_all'])
    name='seed0_calibration' if pilot else 'final_calibration'
    if fit['status']!='COMPLETE':
        save(out/f'evaluator/{name}.json',dict(**fit,seeds=seeds,phase=phase,input_sha256=hashes))
        if pilot:save(out/'evaluator/pilot_gate.json',dict(status='COMPLETE',judgment='NOT_EVALUABLE',reason=fit['reason']))
        return fit
    threshold=fit['threshold'];turn=g['mode']==2;cells={}
    for arm,scores in arms.items():
        thr=threshold if arm.endswith('NEW') else ORIGINAL_THRESHOLD
        cells[arm]=dict(all=metric_cell(scores,thr,g,baseline=arms['S_M3']),turn=metric_cell(scores,thr,g,turn,baseline=arms['DUAL_M3']))
    selection=dict(**fit,seeds=seeds,phase=phase,cells=cells,input_sha256=hashes,script_sha256=sha(__file__),
        limits=['Samefresh calibration batch selects threshold and seed0 gate; optimistic EXPLORE, not fresh validation.',
            'Single shared newthreshold for S/dual and HEAD/BODY, selected only by pooled physicalclear cost, no performance threshold fitting.'])
    save(out/f'evaluator/{name}.json',selection)
    if pilot:
        base=cells['DUAL_M3']['turn'];new=cells['DUAL_NEW']['turn']
        nclear=base['query_clear']['n'];nc=base['query_clear']['stops'];ns=base['shallow0_2']['n']
        evaluable=nclear>0 and nc>0 and ns>0
        reduction=1-new['query_clear']['stops']/nc if nc else None
        improvement=new['shallow0_2']['timely']-base['shallow0_2']['timely']
        passed=evaluable and reduction>=.3 and improvement>0
        gate=dict(status='COMPLETE',judgment='PASS' if passed else 'FAIL' if evaluable else 'NOT_EVALUABLE',
            turn_query_clear_reduction=reduction,turn_shallow0_2_timely_net=improvement,turn_shallow0_2_n=ns,
            original_turn_query_clear=base['query_clear'],new_turn_query_clear=new['query_clear'],threshold=threshold,
            calibration_sha256=sha(out/f'evaluator/{name}.json'),phase=phase,seeds=seeds,
            plan_sha256=sha(out/'PLAN.json'),threshold_contract_sha256=sha(out/'evaluator/THRESHOLD_CONTRACT.json'),score_inputs_sha256=hashes,
            caveat='Seed0 versus five-seed M3, threshold and gate both consume this calibration batch; no evaluation data opened')
        save(out/'evaluator/pilot_gate.json',gate);print(json.dumps(gate),flush=True)
        return gate
    print('FINAL CALIBRATION',threshold,seeds,flush=True);return selection

def evaluate_natural(out=OUT,score_phase=None):
    out=Path(out);deadline(out);require_pilot_pass(out);selection=read(out/'evaluator/final_calibration.json')
    phase=score_phase or 'evaluation/'+selection['phase'].split('/')[1]
    g,m3,aug,seeds,hashes=load_scores(out,phase);assert seeds==selection['seeds']
    threshold=selection['threshold'];arms=dict(S_M3=m3[:,0],DUAL_M3=m3[:,1:].max(1),S_NEW=aug[:,0],DUAL_NEW=aug[:,1:].max(1))
    groups=dict(all=np.ones(len(g['unit']),bool),mode0=g['mode']==0,mode1=g['mode']==1,mode2=g['mode']==2,
        turn_left=(g['mode']==2)&(g['turn']=='left'),turn_right=(g['mode']==2)&(g['turn']=='right'))
    cells={arm:{name:metric_cell(score,threshold if arm.endswith('NEW') else ORIGINAL_THRESHOLD,g,mask,baseline=arms['S_M3']) for name,mask in groups.items()} for arm,score in arms.items()}
    baseline=cells['S_M3'];dual=cells['DUAL_NEW'];single=cells['S_NEW']
    def cost_pass(a,b,factor):return None if not a['n'] or not b['n'] else a['stops']<=b['stops']*factor
    checks=dict(dual_physicalclear_1p15=cost_pass(dual['all']['physical_clear'],baseline['all']['physical_clear'],1.15),
        dual_turn_queryclear_1p5=cost_pass(dual['mode2']['query_clear'],baseline['mode2']['query_clear'],1.5),
        dual_shallow0_5_net_minus2=dual['all']['shallow0_5']['net']>=-2 if dual['all']['shallow0_5']['n'] else None,
        dual_turn_shallow0_2_net_minus1=dual['mode2']['shallow0_2']['net']>=-1 if dual['mode2']['shallow0_2']['n'] else None,
        single_shallow0_2_net_minus2=single['all']['shallow0_2']['net']>=-2 if single['all']['shallow0_2']['n'] else None,
        single_deep_net_minus2=single['all']['deep']['net']>=-2 if single['all']['deep']['n'] else None,
        single_physicalclear_1p1=cost_pass(single['all']['physical_clear'],baseline['all']['physical_clear'],1.1))
    result=dict(status='COMPLETE',cells=cells,checks=checks,seeds=seeds,threshold=threshold,input_sha256=hashes,
        calibration_sha256=sha(out/'evaluator/final_calibration.json'),script_sha256=sha(__file__),
        units=EVAL_UNITS,configs_by_unit={str(u):g['config'][g['unit']==u][::2].tolist() for u in EVAL_UNITS},config_count=int((g['unit']==EVAL_UNITS[0]).sum()//2),
        query_episodes=len(g['unit']),turn_direction_counts={d:int(((g['mode']==2)&(g['turn']==d)).sum()//2) for d in ('left','right')})
    ledger=dict(g);ledger.update(**arms)
    save_npz(out/'evaluator/natural_evaluation_ledger.npz',**ledger)
    result['ledger_sha256']=sha(out/'evaluator/natural_evaluation_ledger.npz')
    save(out/'evaluator/natural_result.json',result);print(json.dumps(checks),flush=True);return result

def calibration_directions(out=OUT):
    """Post-gate descriptive directions only, using the identical seed0 fit."""
    out=Path(out);deadline(out);gate_path=out/'evaluator/pilot_gate.json';gate=read(gate_path)
    if gate.get('status')!='COMPLETE':raise ValueError('Pilot must finish before descriptive calibration output')
    selection=read(out/'evaluator/seed0_calibration.json')
    if selection.get('status')!='COMPLETE':
        result=dict(status='NOT_EVALUABLE',reason='No calibrated threshold was available',pilot_judgment=gate['judgment'],pilot_gate_sha256=sha(gate_path))
        save(out/'evaluator/calibration_directions.json',result);return result
    g,m3,aug,seeds,hashes=load_scores(out,selection['phase']);assert seeds==[0]
    threshold=selection['threshold'];arms=dict(S_M3=m3[:,0],DUAL_M3=m3[:,1:].max(1),S_NEW=aug[:,0],DUAL_NEW=aug[:,1:].max(1))
    groups={f'turn_{d}':(g['mode']==2)&(g['turn']==d) for d in ('left','right')}
    cells={arm:{direction:metric_cell(score,threshold if arm.endswith('NEW') else ORIGINAL_THRESHOLD,g,mask,baseline=arms['S_M3']) for direction,mask in groups.items()} for arm,score in arms.items()}
    vs_dual={arm:{direction:metric_cell(score,threshold if arm.endswith('NEW') else ORIGINAL_THRESHOLD,g,mask,baseline=arms['DUAL_M3']) for direction,mask in groups.items()} for arm,score in arms.items()}
    result=dict(status='COMPLETE',split='calibration',units=CAL_UNITS,configs=40,seeds=seeds,threshold=threshold,cells=cells,
        paired_baseline='S_M3',versus_dual_M3=vs_dual,pilot_judgment=gate['judgment'],pilot_gate_sha256=sha(gate_path),
        calibration_sha256=sha(out/'evaluator/seed0_calibration.json'),input_sha256=hashes,script_sha256=sha(__file__),
        limits=['Descriptive same48-scene calibration only; no fresh99000 evaluation claim.',
            'No new gate, threshold selection, direction-specific fitting or model selection.'])
    save(out/'evaluator/calibration_directions.json',result);return result

def engineering(out=OUT):
    out=Path(out);a=np.array([[2.,0.],[2.,0.],[1.,0.],[1.,0.],[0.,0.],[0.,0.]]);clear=np.ones(6,bool)
    baseline=np.zeros_like(a);baseline[:2]=2
    fit=cost_threshold(a,baseline,clear,1.);assert fit['original_single_stops']==fit['new_dual_stops']==1
    assert fit['threshold']==float(np.nextafter(1.,np.inf))
    zero=cost_threshold(a,np.zeros_like(a),clear,1.);assert zero['new_dual_stops']==0
    assert cost_threshold(a,baseline,np.zeros(6,bool))['status']=='NOT_EVALUABLE'
    raw=np.arange(26.).reshape(13,2);np.testing.assert_array_equal(smooth(raw)[0],raw[0]);assert np.isfinite(smooth(raw)).all()
    save(out/'evaluator/engineering.json',dict(status='PASS',cases=5,script_sha256=sha(__file__)))

def evaluate_angular(out=OUT,score_phase=None):
    """Reused pilot2 evaluation observations only; never threshold calibration."""
    out=Path(out);require_pilot_pass(out);deadline(out)
    selection=read(out/'evaluator/final_calibration.json');threshold=selection['threshold']
    phase=score_phase or 'angular/'+selection['phase'].split('/')[1]
    if phase.split('/')[0]!='angular':raise ValueError('Explicit angular scorephase required')
    import cnh_dual_sensor_evaluate as D
    pilot=ROOT/'artifacts.local/work/cnh-dual-sensor-alarm-20261005'
    envelope=ROOT/'artifacts.local/work/cnh-dual-sensor-envelope-20261005'
    baseline=D.load_baseline();parent=read(pilot/'PLAN.json');prior=read(envelope/'PLAN.json')
    units=sorted(set(parent['evaluation_units'])&set(prior['A']['units']))
    if len(units)!=12:raise ValueError('Frozen angular evaluation is exactly12 retained mode0 scenes')
    indices=[baseline['units'].tolist().index(u) for u in units]
    scenes=[baseline['scenes'][i] for i in indices]
    if any(s['mode']!=0 for s in scenes):raise ValueError('Angular retained split must be mode0')
    receipt=out/f'scores/{phase}/receipt.json'
    if read(receipt).get('status')!='COMPLETE':raise ValueError('Angular scorebatch not sealed')
    hashes={str(receipt):sha(receipt),str(pilot/'PLAN.json'):sha(pilot/'PLAN.json'),str(envelope/'PLAN.json'):sha(envelope/'PLAN.json')}
    for u in units:
        old_truth=D.L.OLD/'truth/evaluation'/f'unit{u}.json'
        hashes[str(old_truth)]=sha(old_truth)
    expected_axes=[-15,-5,0,5,10,15,20,25,30,35,40,45];refs=[];news=[]
    for u in units:
        p=out/f'scores/{phase}/unit{u}.npz'
        with np.load(p) as z:
            if z['axes_degrees'].tolist()!=expected_axes:raise ValueError('Angular12 unique physical axes required')
            if z['seeds'].tolist()!=selection['seeds']:raise ValueError('Angular ensemble differs from frozen calibration')
            for key in ('reference','augmented_mean'):
                if z[key].shape!=(12,28,13,2):raise ValueError('Angular axes×variant-replica×frame×query shape mismatch')
            if 'variants' in z:np.testing.assert_array_equal(z['variants'],np.arange(28)//4)
            if 'replicas' in z:np.testing.assert_array_equal(z['replicas'],np.arange(28)%4)
            refs.append(smooth(z['reference']).reshape(12,7,4,13,2))
            news.append(smooth(z['augmented_mean']).reshape(12,7,4,13,2))
        hashes[str(p)]=sha(p)
    ref=np.stack(refs);new=np.stack(news);lookup={a:i for i,a in enumerate(expected_axes)}
    covered=baseline['covered'][indices];categories=baseline['all_categories'][indices]
    clear=np.broadcast_to((categories=='clear').all(2)[:,:,None,:],(12,7,4,2)).reshape(-1)
    ranges=np.broadcast_to(np.array([s['front_range_m'] for s in scenes])[:,None,None,None,:],(12,7,4,2,13)).reshape(-1,13)
    masks={}
    for group in ('all','opposite','same'):
        shallow=np.zeros((12,7,4,2),bool);deep=shallow.copy();eligible=shallow.copy()
        for i,s in enumerate(scenes):
            if group=='all' or (group=='same')==bool(s['fov_in']):
                shallow[i,:2,:,s['group']]=covered[i,:2,None];deep[i,2,:,s['group']]=covered[i,2]
                eligible[i,:3,:,s['group']]=True
        masks[group]=(shallow.reshape(-1),deep.reshape(-1),eligible.reshape(-1))
    sequence={};checks={};ledger=dict(units=np.array(units),covered=covered,clear_query=clear,ranges=ranges,opposite=np.array([not s['fov_in'] for s in scenes]))
    for psi in (0,10,15,20,25,30):
        arms={}
        for suffix,scores in (('M3',ref),('NEW',new)):
            arms['S_'+suffix]=scores[:,lookup[psi]].transpose(0,1,2,4,3).reshape(-1,13)
            arms['DUAL_'+suffix]=scores[:,[lookup[psi-15],lookup[psi+15]]].max(1).transpose(0,1,2,4,3).reshape(-1,13)
        base=events(arms['S_M3'],ORIGINAL_THRESHOLD,ranges);cells={}
        for name,score in arms.items():
            ev=events(score,threshold if name.endswith('NEW') else ORIGINAL_THRESHOLD,ranges);cell={}
            for group,(shallow,deep,eligible) in masks.items():
                item={}
                for label,den in (('shallow',shallow),('deep',deep)):
                    n=int(den.sum());num=int((den&ev['timely']).sum());rescue=int((den&ev['timely']&~base['timely']).sum());loss=int((den&~ev['timely']&base['timely']).sum())
                    item[label]=dict(timely=num,n=n,rate=num/n if n else None,rescue=rescue,loss=loss,net=rescue-loss,baseline_timely=int((den&base['timely']).sum()))
                missing=eligible&~np.broadcast_to(covered[:,:,None,None],(12,7,4,2)).reshape(-1)
                item['censored']=dict(n=int(missing.sum()),stopped=int((missing&ev['stopped']).sum()));cell[group]=item
            qn=int(clear.sum());qs=int((clear&ev['stopped']).sum());jc=joint_clear(clear);pn=int(jc.sum());ps=int((jc&ev['stopped'].reshape(-1,2).any(1)).sum())
            cell['query_clear']=dict(stops=qs,n=qn,proxy_minutes=qn*2.6/60,rate_per_proxy_minute=qs/(qn*2.6/60) if qn else None)
            cell['physical_clear']=dict(stops=ps,n=pn,proxy_minutes=pn*2.6/60,rate_per_proxy_minute=ps/(pn*2.6/60) if pn else None)
            cells[name]=cell;ledger[f'psi{psi}_{name}']=score
            for key,val in ev.items():ledger[f'psi{psi}_{name}_{key}']=val
        sequence[str(psi)]=cells
        if psi<=25:
            c=cells['DUAL_NEW']['opposite']['shallow'];checks[str(psi)]=None if not c['n'] else c['timely']>=.9*c['n']
    result=dict(status='COMPLETE',seeds=selection['seeds'],threshold=threshold,phase=phase,units=units,opposite_scenes=sum(not s['fov_in'] for s in scenes),
        psi_deg=[0,10,15,20,25,30],sequence=sequence,tolerance_checks=checks,tolerance_psi_le25_pass=None if any(v is None for v in checks.values()) else all(checks.values()),
        input_sha256=hashes,calibration_sha256=sha(out/'evaluator/final_calibration.json'),script_sha256=sha(__file__),
        limits=['Retained consumed pilot2 evaluation split, no fresh independent angular validation.',
            'Opposite membership is frozen original psi15 scene construction, also at psi0; no selection by new scores.'])
    save_npz(out/'evaluator/angular_evaluation_ledger.npz',**ledger);result['ledger_sha256']=sha(out/'evaluator/angular_evaluation_ledger.npz')
    save(out/'evaluator/angular_result.json',result);print(json.dumps(checks),flush=True);return result

def finalize(out=OUT):
    out=Path(out);require_pilot_pass(out);deadline(out)
    natural=read(out/'evaluator/natural_result.json');angular=read(out/'evaluator/angular_result.json')
    if natural['calibration_sha256']!=angular['calibration_sha256'] or natural['seeds']!=angular['seeds']:raise ValueError('Frozen model/calibration versions must match')
    checks=dict(natural['checks']);checks['angular_opposite_shallow_psi_le25']=angular['tolerance_psi_le25_pass']
    turn=[checks['dual_turn_queryclear_1p5'],checks['dual_turn_shallow0_2_net_minus1']]
    judgment='NOT_EVALUABLE' if any(v is None for v in checks.values()) else 'EXTRINSIC_AUG_SUPPORTED_SIM' if all(checks.values()) else 'TRADEOFF' if all(turn) else 'NOT_SUPPORTED'
    result=dict(status='COMPLETE',judgment=judgment,checks=checks,seeds=natural['seeds'],threshold=natural['threshold'],
        natural_sha256=sha(out/'evaluator/natural_result.json'),angular_sha256=sha(out/'evaluator/angular_result.json'),plan_sha256=sha(out/'PLAN.json'),
        calibration_sha256=natural['calibration_sha256'],script_sha256=sha(__file__),
        limits=['Simulation Development evidence only, no hardware or real-world safety claim.',
            'Pilot selection and threshold share calibration; angular evaluation reuses consumed observations.',
            'Physical/query clear exposure proxies are not real walking incidence; net timely criteria are event counts.'])
    save(out/'evaluator/result.json',result);print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('geometry','pilot','calibrate-final','calibration-directions','evaluate-natural','evaluate-angular','finalize','engineering'))
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--split',choices=('calibration','evaluation'),default='calibration')
    p.add_argument('--score-phase',default=None);args=p.parse_args()
    if args.stage=='geometry':geometry(args.out,args.split)
    elif args.stage=='pilot':calibrate(args.out,args.score_phase or 'calibration/seed0',True)
    elif args.stage=='calibrate-final':calibrate(args.out,args.score_phase or 'calibration/ensemble3',False)
    elif args.stage=='calibration-directions':calibration_directions(args.out)
    elif args.stage=='evaluate-natural':evaluate_natural(args.out,args.score_phase)
    elif args.stage=='evaluate-angular':evaluate_angular(args.out,args.score_phase)
    elif args.stage=='finalize':finalize(args.out)
    else:engineering(args.out)
