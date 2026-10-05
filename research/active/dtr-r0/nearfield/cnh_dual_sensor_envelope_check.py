"""Independent CPU reconstruction of envelope alarm numerators and provenance.

Uses raw retained logits, scalar causal history, physical scene keys, and sealed
truth. Does not call the new evaluators' event, smoothing, or metric helpers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-dual-sensor-envelope-20261005'
OLD=ROOT/'artifacts.local/work/cnh-dual-sensor-alarm-20261005'
PILOT=ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def smooth(raw,acquired=None):
    raw=np.asarray(raw,dtype=np.float64)
    result=np.full(raw.shape,-np.inf)
    history=[]
    for f in range(13):
        if acquired is None or acquired[f]:
            history.append(f)
            h=history[-5:]; weights=np.asarray([2**j for j in range(len(h))],float)
            value=np.sum(raw[...,h,:]*weights.reshape(-1,1),axis=-2)/sum(weights)
        if history:result[...,f,:]=value
    return result


def event(scores,threshold,ranges):
    flags=scores>=threshold
    first=np.argmax(flags,axis=-1)
    stopped=np.any(flags,axis=-1)
    timely=np.zeros(stopped.shape,bool)
    for key in np.ndindex(stopped.shape):
        timely[key]=bool(stopped[key] and ranges[key[0],first[key]]>=.9)
    return stopped,timely,first


def record(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def preflight():
    plan=read(OUT/'PLAN.json');contract=read(OUT/'ENGINEERING_INPUT_CONTRACT.json')
    assert contract['parent_plan_sha256']==sha(OUT/'PLAN.json')
    assert set(plan['C']['calibration_units']).isdisjoint(plan['C']['evaluation_units'])
    assert plan['thresholds']==dict(single=.8557642486787612,dual15=.6549558985617854,
        dual22p5_frozen=.8557642486787612,dual22p5_transferred=.6549558985617854)
    for key,h in plan['source_inputs_sha256'].items():assert sha(ROOT/key)==h,key
    for key,h in plan['old_model_sha256'].items():assert sha(key)==h,key
    previous=read(OLD/'result.json');ids={u:i for i,u in enumerate(previous['units'])}
    sides=[]
    for u in plan['A']['units']:
        scene=previous['scenes'][ids[u]];assert scene['mode']==0
        truth=read(PILOT/'truth/evaluation'/f'unit{u}.json')
        box=truth['boxes'][0][0];side=np.sign(box['lo'][0]+box['hi'][0])
        assert bool(side>0)==scene['fov_in'];sides.append(int(side))
    assert sides.count(-1)==11 and sides.count(1)==13
    return dict(status='PASS',parent_plan_sha256=sha(OUT/'PLAN.json'),source_input_hashes=len(plan['source_inputs_sha256']),
        frozen_model_hashes=len(plan['old_model_sha256']),physical_target_sides=dict(opposite=sides.count(-1),same=sides.count(1)),
        threshold_scope='dual22p5 frozen/transferred only; no22p5 cost calibration')


def angular():
    plan=read(OUT/'PLAN.json');result=read(OUT/'angular/result.json');previous=read(OLD/'result.json')
    index={u:i for i,u in enumerate(previous['units'])};units=plan['A']['units'];ids=[index[u] for u in units]
    scenes=[previous['scenes'][i] for i in ids];ranges=np.asarray([s['front_range_m'] for s in scenes])
    with np.load(OLD/'evaluation/ledger.npz') as d:covered=d['covered'][ids];clear=d['joint_clear_den'][ids]
    raw=[];rotation_checks=0;known_checks=0;max_rotation_error=0.;old_plan=read(OLD/'PLAN.json')
    import cnh_dual_sensor_alarm as A
    import cnh_temporal_readout_evaluate as T
    rows,_=T.rows_for(PILOT,'fresh_evaluation');single,_=T.prediction(PILOT,'fresh_evaluation','M3')
    for u in units:
        for p0 in (PILOT/'observations/evaluation'/f'unit{u}.npz',PILOT/'truth/evaluation'/f'unit{u}.json'):
            assert sha(p0)==old_plan['input_sha256'][str(p0.relative_to(ROOT))]
        p=OUT/'angular/scores'/f'unit{u}.npz'
        assert sha(p)==result['provenance']['scores_sha256'][str(p)]
        with np.load(p) as d:angles=d['angles_deg'];value=d['raw'];raw.append(value)
        with np.load(OLD/'scores'/f'unit{u}.npz') as d:l,r=d['raw_full']
        with np.load(OLD/'secondary22p5/scores'/f'unit{u}.npz') as d:sl,sr=d['raw_full']
        for a,ref in ((0,l),(15,single[rows['unit']==u].reshape(7,4,13,2)),(30,r),(-7.5,sl),(37.5,sr)):
            np.testing.assert_array_equal(value[np.flatnonzero(angles==a)[0]],ref);known_checks+=1
        with np.load(PILOT/'observations/evaluation'/f'unit{u}.npz') as d:
            oldsensor,travel,oldnoisy=d['sensor'],d['travel'],d['noisy']
        with np.load(OUT/'angular/observations'/f'unit{u}.npz') as d:
            for j,a in enumerate(d['angles_deg']):
                ex=A.extrinsic(a-15);q=np.linalg.inv(travel)@oldsensor@ex
                n=oldnoisy@ex
                for actual,expected in ((d['query'][j],q),(d['noisy'][j],n)):
                    err=float(np.max(abs(actual-expected)));assert err<1e-12
                    max_rotation_error=max(max_rotation_error,err);rotation_checks+=1
                yaw=np.degrees(np.arctan2(q[:,0,2],q[:,2,2]));np.testing.assert_allclose(yaw,a,atol=1e-10,rtol=0)
                np.testing.assert_allclose(q[:,:3,3],0,atol=1e-12)
    scores=smooth(np.asarray(raw));lookup={float(a):i for i,a in enumerate(angles)}
    comparisons=0;source_comparisons=0
    masks=dict(all=np.ones(len(units),bool),opposite=np.asarray([not s['fov_in'] for s in scenes]),same=np.asarray([s['fov_in'] for s in scenes]))
    for psi in plan['A']['psi_deg']:
        for arm in plan['A']['arms']:
            axes=[psi] if arm=='single' else [psi-(15 if arm=='dual15' else 22.5),psi+(15 if arm=='dual15' else 22.5)]
            if any(a not in lookup for a in axes):continue
            branch=scores[:,[lookup[a] for a in axes]];both=branch.max(1)
            target=np.stack([both[i,...,s['group']] for i,s in enumerate(scenes)])
            sensor=np.stack([branch[i,...,s['group']] for i,s in enumerate(scenes)])
            stopped,timely,first=event(target,plan['thresholds'][arm],ranges)
            joint=both[:,[5,6]].max(axis=(-1,-2))>=plan['thresholds'][arm]
            for group,keep in masks.items():
                cell=result['sequence'][str(psi)][group][arm]
                for name,ds in [('shallow',[0,1]),('deep',[2])]:
                    num=den=0;source=dict(L=0,R=0,tie=0,timely_L=0,timely_R=0,timely_tie=0)
                    for i in np.flatnonzero(keep):
                        for d in ds:
                            for k in range(4):
                                den+=int(covered[i,d]);num+=int(covered[i,d] and timely[i,d,k])
                                if arm!='single' and stopped[i,d,k]:
                                    l,r=sensor[i,:,d,k,first[i,d,k]]
                                    label='L' if l>r else 'R' if r>l else 'tie'
                                    source[label]+=1;source['timely_'+label]+=int(timely[i,d,k] and covered[i,d])
                    met=cell['branches'][name]['timely'];assert (num,den)==(met['stops'],met['n']);comparisons+=2
                    if arm!='single':assert source==cell['branches'][name]['first_report_source'];source_comparisons+=6
                num=int((joint[keep]&clear[keep]).sum());den=int(clear[keep].sum())
                assert (num,den)==(cell['joint_clear']['stops'],cell['joint_clear']['n']);comparisons+=2
    return dict(status='PASS',integer_comparisons=comparisons,first_report_source_comparisons=source_comparisons,
        reused_raw_stream_array_equal_checks=known_checks,rotation_query_checks=rotation_checks,max_rotation_error=max_rotation_error,
        result_sha256=sha(OUT/'angular/result.json'))


def alternating():
    result=read(OUT/'alternating/result.json');previous=read(OLD/'result.json');units=previous['units'];scenes=previous['scenes']
    raws=[]
    for u in units:
        with np.load(OLD/'scores'/f'unit{u}.npz') as d:
            raw=d['raw_alternating'];raws.append([smooth(raw[s],np.arange(3,16)%2==s) for s in range(2)])
    alt=np.asarray(raws).max(1)
    with np.load(OLD/'evaluation/ledger.npz') as d:single=d['single_both_query_scores'];clear=d['joint_clear_den'];covered=d['covered']
    cal=np.isin(units,result['split_units']['calibration']);threshold=result['thresholds']['alternating_matched']
    calvalues=alt[:,[5,6]].max(axis=(-1,-2))[clear&cal[:,None,None]]
    assert (calvalues>=threshold).sum()==17
    below=calvalues[calvalues<threshold].max();assert threshold==np.nextafter(below,np.inf)
    assert (calvalues>=below).sum()>17
    ranges=np.asarray([s['front_range_m'] for s in scenes]);comparisons=0
    masks=dict(all=np.ones(len(units),bool),calibration=cal,evaluation=np.isin(units,result['split_units']['evaluation']))
    groups=dict(all=np.ones(len(units),bool),FOV_OUT=np.asarray([not s['fov_in'] for s in scenes]),FOV_IN=np.asarray([s['fov_in'] for s in scenes]))
    groups.update({f'mode{m}':np.asarray([s['mode']==m for s in scenes]) for m in range(3)})
    for arm in result['thresholds']:
        both=single if arm=='single' else alt;thr=result['thresholds'][arm]
        target=np.stack([both[i,...,s['group']] for i,s in enumerate(scenes)]);_,timely,_=event(target,thr,ranges)
        joint=both[:,[5,6]].max(axis=(-1,-2))>=thr
        for split,mask in masks.items():
            for group,gm in groups.items():
                keep=mask&gm;cell=result['sequence'][split][group][arm]
                for name,ds in [('shallow',[0,1]),('deep',[2])]:
                    den=np.broadcast_to(covered[:,ds,None],(len(units),len(ds),4));num=int((timely[:,ds][keep]&den[keep]).sum());n=int(den[keep].sum())
                    met=cell['branches'][name]['timely'];assert (num,n)==(met['stops'],met['n']);comparisons+=2
                assert (int((joint[keep]&clear[keep]).sum()),int(clear[keep].sum()))==(cell['joint_clear']['stops'],cell['joint_clear']['n']);comparisons+=2
    return dict(status='PASS',integer_comparisons=comparisons,calibration_cost=17,cutoff_boundary=True,result_sha256=sha(OUT/'alternating/result.json'))


def natural():
    result=read(OUT/'natural/result.json');units=result['units'];configs=result['configs'];raw=[]
    for p,h in read(OUT/'natural/PLAN.json')['source_sha256'].items():assert sha(p)==h
    contract=read(OUT/'natural/ENGINEERING_BATCH_CONTRACT.json')
    engineering=read(OUT/'natural/engineering/batched_projection.json')
    assert engineering['status']=='PASS'
    assert contract['current_script_sha256']==sha(Path(__file__).with_name('cnh_dual_sensor_envelope_natural.py'))
    assert contract['original_plan_script_sha256']==read(OUT/'natural/PLAN.json')['script_sha256']
    assert contract['voxel_bitwise_receipt_sha256']==sha(OUT/'natural/engineering/batched_projection.json')
    for p,h in contract['dependency_sha256'].items():assert sha(p)==h
    source=ROOT/'artifacts.local/work/cnh-margin-confirm-20261002'
    for u in units:
        p=OUT/'natural/scores'/f'unit{u}.npz'
        assert sha(p)==result['provenance']['scores_sha256'][str(p)]
        receipt=read(p.with_suffix('.json'))
        assert receipt['unit']==u and receipt['parent_plan_sha256']==sha(OUT/'PLAN.json')
        assert receipt['score_sha256']==sha(p)
        assert receipt['observation_sha256']==sha(OUT/'natural/observations'/f'unit{u}.npz')
        with np.load(p) as d:
            assert int(d['unit'])==u and np.array_equal(d['frames'],np.arange(3,16))
            assert list(d['sensors'])==['L','R'] and len(set(d['configs']))==len(d['configs'])
            assert d['raw'].shape==(2,len(d['configs']),13,2) and np.isfinite(d['raw']).all()
            raw.append(d['raw'][:,[list(d['configs']).index(c) for c in configs]])
    sensor=smooth(np.asarray(raw));scores=sensor.max(1).transpose(0,1,3,2).reshape(-1,13)
    picked=sensor.transpose(0,2,4,1,3).reshape(-1,2,13)
    with np.load(OUT/'natural/ledger.npz') as d:
        ledger={k:d[k] for k in d.files}
    baseline=[]
    with np.load(source/'frame_scores_M3_early.npz') as early,np.load(source/'frame_scores_M3.npz') as late:
        for u in units:baseline.append(smooth(np.concatenate([early[str(u)][configs],late[str(u)][configs]],axis=1)))
    baseline=np.asarray(baseline).transpose(0,1,3,2).reshape(-1,13)
    np.testing.assert_allclose(baseline,ledger['single'],atol=1e-12,rtol=0)
    geometry=ROOT/'artifacts.local/work/cnh-observed-sequence-20261002/geometry.npz'
    with np.load(geometry) as g:
        keep=(g['split']=='evaluation')&np.isin(g['unit'],units)&np.isin(g['config'],configs)
        for original,retained in [('unit','unit'),('config','config'),('query','query'),('clear_all','clear'),('covered','covered'),('ref_category','ref_category'),('frame_ranges','frame_ranges')]:
            np.testing.assert_array_equal(g[original][keep],ledger[retained])
    np.testing.assert_allclose(scores,ledger['dual'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(picked,ledger['sensor_scores'],atol=1e-12,rtol=0)
    keys=list(zip(ledger['unit'],ledger['config'],ledger['query']))
    assert keys==[(u,c,q) for u in units for c in configs for q in (0,1)]
    comparisons=0;events={}
    for arm,thr in result['thresholds'].items():
        value=ledger['single'] if arm=='single' else scores
        stopped,timely,first=event(value,thr,ledger['frame_ranges']);events[arm]=(stopped,timely)
        for group in ('all','mode0','mode1','mode2'):
            keep=np.ones(len(keys),bool) if group=='all' else ledger['unit']%3==int(group[-1]);cell=result['cells'][arm][group]
            for category in ('contact0-2cm','contact2-5cm','contact>5cm','clear'):
                den=keep&(ledger['clear'] if category=='clear' else ledger['covered']&(ledger['ref_category']==category));flag=stopped if category=='clear' else timely
                assert (int((den&flag).sum()),int(den.sum()))==(cell[category]['stops'],cell[category]['n']);comparisons+=2
            censored=keep&~ledger['covered']
            assert (int(censored.sum()),int((censored&stopped).sum()))==(cell['censored']['n'],cell['censored']['already_alarm']);comparisons+=2
            if arm!='single':
                source=dict(L=0,R=0,tie=0)
                for i in np.flatnonzero(keep&stopped):
                    l,r=picked[i,:,first[i]];label='L' if l>r else 'R' if r>l else 'tie';source[label]+=1
                assert source==cell['first_report_source'];comparisons+=3
        den=ledger['clear'].reshape(-1,2).all(1);joint=result['joint_clear'][arm]
        assert (int((den&stopped.reshape(-1,2).any(1)).sum()),int(den.sum()))==(joint['stops'],joint['n']);comparisons+=2
        if arm!='single':
            physical=picked.reshape(-1,2,2,13).max(1);peaks=physical.max(1)
            first=np.argmax(peaks>=thr,axis=1);source=dict(L=0,R=0,tie=0)
            for i in np.flatnonzero(den&stopped.reshape(-1,2).any(1)):
                l,r=physical[i,:,first[i]];label='L' if l>r else 'R' if r>l else 'tie';source[label]+=1
            assert source==joint['first_report_source'];comparisons+=3
    for arm in ('dual15','dual15_original_threshold'):
        for category in ('contact0-2cm','contact>5cm','clear'):
            den=ledger['clear'] if category=='clear' else ledger['covered']&(ledger['ref_category']==category);j=0 if category=='clear' else 1
            a,b=events[arm][j],events['single'][j];cell=result['comparisons'][arm][category]
            assert (int((den&a&~b).sum()),int((den&b&~a).sum()))==(cell['added'],cell['removed']);comparisons+=2
    return dict(status='PASS',integer_comparisons=comparisons,paired_key_rows=len(keys),all_config_ids_uniform=configs in [list(range(40)),list(range(0,40,2)),list(range(0,40,4))],result_sha256=sha(OUT/'natural/result.json'))


def main(stage):
    started=time.monotonic();check=preflight()
    if stage=='final':
        cached={name:read(OUT/'checks'/f'{name}.json') for name in ('angular','alternating')}
        for name,receipt in cached.items():
            assert receipt['status']=='PASS' and receipt['result_sha256']==sha(OUT/name/'result.json')
        check.update(**cached,natural=natural(),prior_passes_reused_without_reexecution=True)
        terminal=OUT/'worker/terminal.json'
        if terminal.exists():
            worker=read(terminal)
            assert worker['status']=='NOT_USED_PROVISION_ABORTED' and worker['returned_units']==[]
            assert worker['package_sha256']==sha(OUT/'worker/package.zip')
            check['worker']=dict(status='NOT_USED',returned_scientific_units=0,terminal_sha256=sha(terminal),package_sha256=worker['package_sha256'])
    check.update(seconds=time.monotonic()-started,source_sha256=sha(__file__),stage=stage)
    record(OUT/'checks'/f'{stage}.json',check);print(json.dumps(check,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','final']);main(parser.parse_args().stage)
