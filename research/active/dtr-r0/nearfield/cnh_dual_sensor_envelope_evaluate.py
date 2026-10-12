"""Frozen-threshold alarm envelopes from absolute optical-yaw score streams.

No training or threshold fitting. Evaluator truth selects target query and
geometric deadline only after public-query M3 inference has finished.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

import cnh_dual_sensor_envelope_cpu as C

ROOT=C.ROOT
OUT=C.OUT
OLD=C.PREVIOUS
FRAMES=C.FRAMES
BOOT_SEED=2026100530
BOOT_N=2000


def smooth(raw):
    raw=np.asarray(raw,float)
    assert raw.shape[-2:]==(13,2) and np.isfinite(raw).all()
    output=np.empty_like(raw)
    for t in range(13):
        ids=range(max(0,t-4),t+1)
        weights=2.**np.arange(len(ids))
        output[...,t,:]=sum(raw[...,i,:]*w for i,w in zip(ids,weights))/weights.sum()
    return output


def source(sensor, event, keep):
    if sensor is None:
        return dict(single=int(event['stopped'][keep].sum()))
    counts=dict(L=0,R=0,tie=0,timely_L=0,timely_R=0,timely_tie=0)
    for i in np.flatnonzero(keep):
        for d in range(event['stopped'].shape[1]):
            for k in range(4):
                if not event['stopped'][i,d,k]:continue
                t=event['first_index'][i,d,k]
                left,right=sensor[i,:,d,k,t]
                label='L' if left>right else 'R' if right>left else 'tie'
                counts[label]+=1
                counts['timely_'+label]+=int(event['timely'][i,d,k])
    return counts


def evaluate(out=OUT):
    tick=time.monotonic();out=Path(out);target=out/'angular'
    if (target/'result.json').exists():raise FileExistsError('Completed angular result immutable')
    plan=C.read(out/'PLAN.json');plan_sha=C.sha(out/'PLAN.json')
    contract=out/'ENGINEERING_INPUT_CONTRACT.json'
    assert contract.exists(),'Engineering reuse contract required before evaluation'
    assert C.read(contract)['parent_plan_sha256']==plan_sha
    previous=C.read(OLD/'result.json');secondary=C.read(OLD/'secondary22p5/result.json')
    units=np.asarray(plan['A']['units']);original=np.asarray(previous['units'])
    ids=np.asarray([original.tolist().index(u) for u in units])
    scenes=[previous['scenes'][i] for i in ids]
    assert len(scenes)==24 and all(s['mode']==0 for s in scenes)
    with np.load(OLD/'evaluation/ledger.npz',allow_pickle=False) as z:
        original_both=z['single_both_query_scores'][ids].copy()
        categories=z['both_categories'][ids].copy()
        covered=z['covered'][ids].copy()
        inherited_full=z['full_sensor_smoothed'][ids].copy()
        inherited_clear=z['joint_clear_den'][ids].copy()
    raw,score_hashes=[],{}
    shared_angles=None
    for unit in units:
        path=target/'scores'/f'unit{unit}.npz';h=C.sha(path)
        receipt=C.read(path.with_suffix('.json'))
        assert receipt['status']=='COMPLETE' and receipt['score_sha256']==h and receipt['plan_sha256']==plan_sha
        with np.load(path,allow_pickle=False) as z:
            assert int(z['unit'])==unit and np.array_equal(z['frames'],FRAMES)
            angles=z['angles_deg'].copy();value=z['raw'].copy()
        assert value.shape==(len(angles),7,4,13,2) and np.isfinite(value).all()
        assert len(np.unique(angles))==len(angles)
        if shared_angles is None:shared_angles=angles
        np.testing.assert_array_equal(angles,shared_angles)
        assert C.sha(path)==h
        score_hashes[str(path)]=h;raw.append(value)
    raw=np.asarray(raw);scores=smooth(raw)
    angle_index={float(a):i for i,a in enumerate(shared_angles)}
    np.testing.assert_allclose(scores[:,angle_index[15.]],original_both,atol=1e-12,rtol=0)
    np.testing.assert_allclose(scores[:,angle_index[0.]],inherited_full[:,0],atol=1e-12,rtol=0)
    np.testing.assert_allclose(scores[:,angle_index[30.]],inherited_full[:,1],atol=1e-12,rtol=0)
    inherited22=[]
    for unit in units:
        with np.load(OLD/'secondary22p5/scores'/f'unit{unit}.npz',allow_pickle=False) as z:
            inherited22.append(smooth(z['raw_full']))
    inherited22=np.asarray(inherited22)
    np.testing.assert_allclose(scores[:,angle_index[-7.5]],inherited22[:,0],atol=1e-12,rtol=0)
    np.testing.assert_allclose(scores[:,angle_index[37.5]],inherited22[:,1],atol=1e-12,rtol=0)
    clear=np.broadcast_to(np.all(categories[:,[5,6]]=='clear',axis=(-2,-1))[...,None],(24,2,4))
    np.testing.assert_array_equal(clear,inherited_clear)
    ranges=np.asarray([s['front_range_m'] for s in scenes])
    groups=dict(all=np.ones(24,bool),opposite=np.asarray([not s['fov_in'] for s in scenes]),same=np.asarray([s['fov_in'] for s in scenes]))
    assert (groups['opposite'].sum(),groups['same'].sum())==(11,13)
    rng=np.random.default_rng(BOOT_SEED)
    draws=np.asarray([np.bincount(rng.choice(24,24,replace=True),minlength=24) for _ in range(BOOT_N)])
    sequence={};ledger={};parity={};missing=[]
    for psi in plan['A']['psi_deg']:
        streams={};sensor_streams={}
        for arm in plan['A']['arms']:
            angles=[psi] if arm=='single' else [psi-(15 if arm=='dual15' else 22.5),psi+(15 if arm=='dual15' else 22.5)]
            absent=[a for a in angles if float(a) not in angle_index]
            if absent:
                assert arm.startswith('dual22p5') and psi in (0,10),'Priority primary point missing'
                missing.append(dict(psi_deg=psi,arm=arm,missing_angles=absent,status='NOT_RUN_BUDGET'))
                continue
            per_sensor=scores[:,[angle_index[float(a)] for a in angles]]
            streams[arm]=per_sensor.max(1)
            sensor_streams[arm]=None if arm=='single' else np.stack([per_sensor[i,...,s['group']] for i,s in enumerate(scenes)])
        selected={a:np.stack([v[i,...,s['group']] for i,s in enumerate(scenes)]) for a,v in streams.items()}
        events={a:C.event(v,plan['thresholds'][a],ranges,covered) for a,v in selected.items()}
        joint={a:v[:,[5,6]].max(axis=(-2,-1))>=plan['thresholds'][a] for a,v in streams.items()}
        sequence[str(psi)]={}
        for group,keep in groups.items():
            cells=sequence[str(psi)][group]={}
            for arm,ev in events.items():
                cell=cells[arm]=dict(status='COMPLETE',scenes=int(keep.sum()),threshold=plan['thresholds'][arm],branches={})
                for name,ds in (('shallow',[0,1]),('deep',[2])):
                    den=np.broadcast_to(covered[:,ds,None],(24,len(ds),4))
                    part={k:v[:,ds] for k,v in ev.items() if k!='covered'}
                    sensor=sensor_streams[arm]
                    met=C.metric(ev['timely'][:,ds],events['single']['timely'][:,ds],den,keep,draws)
                    cell['branches'][name]=dict(timely=met,first_report_source=source(None if sensor is None else sensor[:,:,ds],part,keep),
                        first_stops=int(ev['stopped'][:,ds][keep].sum()),censored_n=int((~den)[keep].sum()))
                met=C.metric(joint[arm],joint['single'],clear,keep,draws)
                minutes=met['n']*2.6/60
                cell['joint_clear']=dict(**met,proxy_minutes=minutes,first_stops_per_proxy_minute=met['stops']/minutes if minutes else None)
        for arm,ev in events.items():
            for field in ('alarm','stopped','timely','first_index','first_range'):
                ledger[f'psi{psi}_{arm}_{field}']=ev[field]
            ledger[f'psi{psi}_{arm}_both']=streams[arm]
            ledger[f'psi{psi}_{arm}_joint']=joint[arm]
        if psi==15:
            mapping={'single':(previous,'single'),'dual15':(previous,'dual_matched'),
                     'dual22p5_frozen':(secondary,'secondary_frozen'),'dual22p5_transferred':(secondary,'secondary_transferred_main_threshold')}
            for arm,(old,oldarm) in mapping.items():
                for name in ('shallow','deep'):
                    now=sequence['15']['all'][arm]['branches'][name]['timely']
                    oldcell=old['sequence']['all']['mode0'][oldarm]['branches'][name]['timely']
                    assert (now['stops'],now['n'])==(oldcell['stops'],oldcell['n'])
                now=sequence['15']['all'][arm]['joint_clear'];oldcell=old['sequence']['all']['mode0'][oldarm]['joint_clear']
                assert (now['stops'],now['n'])==(oldcell['stops'],oldcell['n'])
                parity[arm]=dict(status='PASS',previous_arm=oldarm)
    checks=0
    for psi,groups_result in sequence.items():
        for group,cells in groups_result.items():
            keep=groups[group]
            for arm,cell in cells.items():
                for name,ds in (('shallow',[0,1]),('deep',[2])):
                    count=sum(int(ledger[f'psi{psi}_{arm}_timely'][i,d,k] and covered[i,d]) for i in np.flatnonzero(keep) for d in ds for k in range(4))
                    n=sum(int(covered[i,d]) for i in np.flatnonzero(keep) for d in ds for k in range(4))
                    met=cell['branches'][name]['timely'];assert (count,n)==(met['stops'],met['n']);checks+=2
                count=sum(int(ledger[f'psi{psi}_{arm}_joint'][i,d,k] and clear[i,d,k]) for i in np.flatnonzero(keep) for d in range(2) for k in range(4))
                assert count==cell['joint_clear']['stops'];checks+=1
    assert C.sha(out/'PLAN.json')==plan_sha
    ledger_path=target/'ledger.npz'
    if ledger_path.exists():raise FileExistsError('Angular ledger immutable')
    np.savez_compressed(ledger_path,units=units,angles_deg=shared_angles,frames=FRAMES,covered=covered,clear_den=clear,raw=raw,smoothed=scores,**ledger)
    result=dict(status='COMPLETE',scope='Mode0 constant positive head yaw; consumed synthetic Development; frozen thresholds',
        units=units.tolist(),psi_deg=plan['A']['psi_deg'],thresholds=plan['thresholds'],sequence=sequence,missing=missing,
        geometry_predictions=plan['A']['geometry_predictions'],bootstrap=dict(n=BOOT_N,seed=BOOT_SEED,unit='Whole paired scenes; same draws across angles/arms'),
        checks=dict(psi15_parent_parity=parity,reused_optical_streams_max_abs_tolerance=1e-12,independent_integer_comparisons=checks),
        provenance=dict(plan_sha256=plan_sha,contract_sha256=C.sha(contract),scores_sha256=score_hashes,ledger_sha256=C.sha(ledger_path),source_sha256=C.sha(__file__)),
        limits=['Opposite/same retain original FOV_OUT11/FOV_IN13 labels atpsi15; labels not recomputed atnew angles.',
            'dual22p5 uses frozen single threshold and transferred dual15 threshold; neither is equal-cost calibration for22p5.',
            'Zero sensor translation; shared inherited pose error; finite ray geometry and synthetic photons.',
            'Geometric center-point distance is exposure heuristic, not a sufficient alarm criterion. abs formula is conservative whenpsi<splay.',
            'Natural96000 is synthetic clutter, not real-world incidence. No hardware throughput/SNR/crosstalk/power evidence.'],seconds=time.monotonic()-tick)
    C.write_new(target/'result.json',result)
    print('ANGULAR EVALUATED',result['seconds'],'s',checks,'integer checks')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    evaluate(parser.parse_args().out)
