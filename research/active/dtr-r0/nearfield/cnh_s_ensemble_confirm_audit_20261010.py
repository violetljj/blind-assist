"""Independent frozen-S confirmation audit; saved outputs only, no science.

Prior independent scalar audit primitives are reused. No confirmation or task
metric producer is imported. Calibration is replayed in ascending complete ties,
without a monotonicity assumption; K stays grouped within its physical scene.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import audit_cnh_cost_v2_holdout_dev_20261010 as A
import cnh_task_cost_audit_20261010 as I

ROOT=A.ROOT
HEIGHTS=A.HEIGHTS
ARMS=('E','S955','S956','S957','Sensemble')
PRIOR=ROOT/'artifacts.local/work/cnh-task-cost-retrain-dev-20261010'
read,sha=A.read,A.sha


def write_new(path, value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def identities(out):
    manifest=read(out/'execution_manifest.json')
    frozen=manifest['frozen_spec']
    for p,h in {**frozen['inherited'],**frozen['feature_sources'],**frozen['models'],
                **manifest['new_sources']}.items():
        assert sha(p)==h,p
    prior_models=read(PRIOR/'trained_models_manifest.json')
    expected={p:h for p,h in prior_models['models'].items() if Path(p).name.startswith('S_')}
    assert expected==frozen['models'] and len(expected)==6
    assert sha(PRIOR/'trained_models_manifest.json')==frozen['prior_training_manifest_sha256']
    previous=read(PRIOR/'execution_manifest.json')
    for p,h in frozen['feature_sources'].items():
        assert previous['new_sources'][p]==h
    assert len(frozen['feature_sources'])==3
    assert manifest['training']==0 and manifest['protected_access']==0
    protocol=ROOT/'research/active/dtr-r0/nearfield/CNH_S_ENSEMBLE_CONFIRM_PROTOCOL_20261010.md'
    committed=subprocess.check_output(['git','show',manifest['protocol_commit']+':'+protocol.relative_to(ROOT).as_posix()],cwd=ROOT)
    assert committed==protocol.read_bytes()
    freeze=read(out/'freeze_receipt.json')
    assert freeze['status']=='FROZEN_COMMITTED_BEFORE_RENDER'
    assert freeze['manifest_sha256']==sha(out/'execution_manifest.json')
    assert (out/'freeze_receipt.json').stat().st_mtime_ns<(out/'data/cal/hist.npy').stat().st_mtime_ns
    assert (out/'PLAN.json').stat().st_mtime_ns<(out/'data/cal/hist.npy').stat().st_mtime_ns
    seal=read(out/'sealed_calibration.json')
    for p,h in seal['bindings'].items():assert sha(out/p)==h,p
    assert (out/'sealed_calibration.json').stat().st_mtime_ns<(out/'data/hold/hist.npy').stat().st_mtime_ns
    hold=read(out/'data_hold_receipt.json')
    assert hold['training']==0 and hold['cal_seal_sha256']==sha(out/'sealed_calibration.json')
    old_seal=read(PRIOR/'sealed_calibration.json')
    exact=old_seal['calibrations']['Sensemble']['secondary']['tau']
    assert exact==2.4142577648162846
    assert seal['transfer']['tau']==exact and seal['transfer']['tau_kind']=='finite'
    assert seal['transfer']['source_sha256']==sha(PRIOR/'sealed_calibration.json')
    assert sha(frozen['transfer_calibration_path'])==frozen['transfer_calibration_sha256']
    return dict(frozen_models=len(expected),feature_sources=len(frozen['feature_sources']),
                protocol_commit=manifest['protocol_commit'],transfer_tau=exact,training=0)


def source(out,rows):
    keys,families,backgrounds,info={},{},{},{}
    import cnh_counterfactual_data_dev as D
    for split,values in rows.items():
        with np.load(out/'data'/split/'geometry.npz') as g:
            category,sensor=g['category'],g['sensor']
        keys[split],counts=A.geometry(values,sensor,category)
        n=768 if split=='cal' else 1536
        assert len(values)==n and counts=={'contact':n//3,'pass':n//3,'clear':n//3}
        assert (category=='contact').sum(0).tolist()==[n//6]*2
        families[split]={r['background_family'] for r in values}
        backgrounds[split]={D.physical_key(r['background_boxes']) for r in values}
        for layer in ('0-5cm','5-10cm','10-20cm','20-35cm'):
            selected=[r for r in values if r['pass_layer']==layer]
            assert len(selected)==n//12 and {r['group'] for r in selected}=={0,1}
        assert all(r['lateral_gap_m']>=.35 for r in values if r['placement']=='clear')
        info[split]=dict(physical_scenes=n,counts=counts,background_families=len(families[split]))
    assert not keys['cal']&keys['hold']
    assert not families['cal']&families['hold']
    assert not backgrounds['cal']&backgrounds['hold']
    inventory=A.inventory_check(out,keys,rows)
    entries=read(out/'inventory.json')['files']
    assert any('cnh-task-cost-retrain-dev-20261010/scene_rows.json' in e['path'].replace('\\','/') for e in entries)
    assert inventory['prior_keys']>=12304+4608
    assert len(families['cal'])>=2 and len(families['hold'])>=2
    return dict(sources=info,inventory=inventory)


def scores(out,split):
    with np.load(out/'data'/split/'task_scores.npz') as a:
        values={k:a[k].copy() for k in a.files}
    assert tuple(values['arm_keys'])==ARMS
    with np.load(out/'data'/split/'scores.npz') as a:
        ordinary_raw=a['ordinary_raw'].copy()
        ordinary=A.smooth(ordinary_raw)
        mr,lr=A.smooth(a['m3_raw']),A.smooth(a['local_raw'])
        current=a['current'].astype(float)
        valid=a['current_valid'] & np.isfinite(current)
    with np.load(out/'data'/split/'frozen_readout.npz') as a:
        joint=a['joint_scores'].copy()
    np.testing.assert_array_equal(values['old5'],2*((mr>=.9404184587540165)|(lr>=4.625390338985158)))
    np.testing.assert_array_equal(values['m3'],2*(mr>=.8557642486787612))
    th=read(ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
    jc=read(ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010/calibrations.json')
    om,jm=[],[]
    for i,seed in enumerate((2026100955,2026100956,2026100957)):
        cut=jc[f'{seed}/score_current/c15_p64']['theta']
        om.append(ordinary[i]-th[str(seed)]['single']);jm.append(joint[i]-cut)
        expected=np.where((values['old5']>0)|(ordinary[i]>=th[str(seed)]['addition']),2,
                          np.where(ordinary[i]>=th[str(seed)]['single'],1,0)).astype(np.int8)
        expected[(expected==0)&np.isfinite(joint[i])&(joint[i]>=cut)]=1
        np.testing.assert_array_equal(values['both'][i],expected)
    raw=np.maximum(np.array(om).mean(0),np.array(jm).mean(0))
    np.testing.assert_allclose(values['diagnostic_scores'][0],raw,rtol=0,atol=1e-12)
    np.testing.assert_allclose(values['light_scores'][0],np.where(raw>=0,raw,-np.inf),rtol=0,atol=1e-12)
    np.testing.assert_array_equal(values['light_scores'][4],values['light_scores'][1:4].astype(np.float32).mean(0))
    np.testing.assert_array_equal(values['diagnostic_scores'][1:],values['light_scores'][1:])
    features=np.load(out/'data'/split/'S_features.npy',mmap_mode='r')
    assert features.shape[-1]==47 and not np.isinf(features).any()
    slope,residual=np.zeros_like(ordinary_raw,dtype=float),np.zeros_like(ordinary_raw,dtype=float)
    for frame in range(13):
        y=ordinary_raw[...,max(0,frame-4):frame+1,:]
        x=np.arange(y.shape[-2],dtype=float);x-=x.mean()
        if len(x)>1:
            b=(y*x[:,None]).sum(-2)/(x*x).sum()
            slope[...,frame,:]=b
            residual[...,frame,:]=np.sqrt(((y-y.mean(-2)[...,None,:]-b[...,None,:]*x[:,None])**2).mean(-2))
    spatial=np.concatenate((np.where(valid,current,np.nan),(~valid).astype(float)),-1)
    for i,seed in enumerate((2026100955,2026100956,2026100957)):
        first3=np.stack((ordinary[i]-th[str(seed)]['single'],slope[i],residual[i]),-1)
        expected=np.concatenate((first3,spatial),-1).astype(np.float32)
        np.testing.assert_array_equal(features[i],expected)
    receipt=read(out/f'data_{split}_receipt.json')
    for name,digest in receipt['outputs_sha256'].items():assert sha(out/'data'/split/name)==digest
    return values


def target_parameters(out,rows):
    old_dimensions,old_rho=set(),set()
    def collect(value):
        if isinstance(value,dict):
            target=value.get('target_box')
            if isinstance(target,dict) and all(key in target for key in ('lo','hi','rho')):
                old_dimensions.update(round(float(b)-float(a),10) for a,b in zip(target['lo'],target['hi']))
                old_rho.add(round(float(target['rho']),10))
            if isinstance(value.get('target_thickness_m'),(int,float)):old_dimensions.add(round(float(value['target_thickness_m']),10))
            for item in value.values():collect(item)
        elif isinstance(value,list):
            for item in value:collect(item)
    for entry in read(out/'inventory.json')['files']:collect(read(ROOT/entry['path']))
    fresh={}
    for split,values in rows.items():
        dimensions={round(float(b)-float(a),10) for r in values for a,b in zip(r['target_box']['lo'],r['target_box']['hi'])}
        dimensions|={round(float(r['target_thickness_m']),10) for r in values}
        rho={round(float(r['target_box']['rho']),10) for r in values}
        assert not dimensions&old_dimensions and not rho&old_rho
        fresh[split]=(dimensions,rho)
    assert not fresh['cal'][0]&fresh['hold'][0] and not fresh['cal'][1]&fresh['hold'][1]
    return dict(prior_dimension_values=len(old_dimensions),prior_rho_values=len(old_rho),overlaps=0,
                new={s:dict(dimensions=sorted(d),rho=sorted(r)) for s,(d,r) in fresh.items()})


def gates(metrics):
    r=metrics['reports'];base=r['hold/fixed/old5'];s=r['hold/Sensemble/main']
    contact=s['contacts'];assert contact['HEAD_BODY']['denominator']==1024
    assert all(contact[h]['denominator']==512 for h in HEIGHTS)
    seednets={arm:r[f'hold/{arm}/main']['contacts']['HEAD_BODY']['net'] for arm in ARMS[1:4]}
    net=contact['HEAD_BODY']['net'];height={h:contact[h]['net'] for h in HEIGHTS}
    fc=s['costs']['far_pass_plus_clear']['notifications'];bfc=base['costs']['far_pass_plus_clear']['notifications']
    cost=s['costs']['all_negative']['weighted_cost'];bcost=base['costs']['all_negative']['weighted_cost']
    independent=dict(a=net>=80 and all(v>=11 for v in height.values()),b=fc<=1.10*bfc,
                     c=cost<=1.3125*bcost,d=all(v>=40 for v in seednets.values()))
    actual=metrics['strong_signal']
    for key,v in independent.items():assert actual['gates'][key]['pass']==v,(key,v)
    assert actual['gates']['a']['net']==net and actual['gates']['a']['required_net']==80
    assert actual['gates']['a']['height_nets']==height
    assert actual['gates']['a']['height_required']==dict(HEAD=11,BODY=11)
    assert actual['gates']['b']['far_pass_plus_clear_notifications']==fc
    assert actual['gates']['b']['cap']==1.10*bfc
    assert actual['gates']['c']['weighted_cost']==cost and actual['gates']['c']['cap']==1.3125*bcost
    assert actual['gates']['d']['seed_nets']==seednets and actual['gates']['d']['required_each_seed_net']==40
    assert actual['all_pass']==all(independent.values())
    assert actual['failed_gates']==[k for k,v in independent.items() if not v]
    return dict(gates=independent,net=net,height_nets=height,seed_nets=seednets,farclear=fc,
                farclear_cap=1.10*bfc,cost=cost,cost_cap=1.3125*bcost)


def run(out):
    began=time.monotonic();prior=sum(read(p)['seconds'] for p in out.glob('independent_audit_failure_*.json'))
    cached=out/'independent_source_recompute.json'
    if cached.exists():prior+=read(cached)['seconds']
    deadline=began+220-prior
    assert prior<220 and not (out/'independent_audit.json').exists()
    identities_result=identities(out)
    rows=read(out/'scene_rows.json')
    parameters=target_parameters(out,rows)
    if cached.exists():
        cached_result=read(cached)
        assert cached_result['execution_manifest_sha256']==sha(out/'execution_manifest.json')
        assert cached_result['scene_rows_sha256']==sha(out/'scene_rows.json')
        source_result=cached_result['result']
    else:
        source_result=source(out,rows)
    print('AUDIT_SOURCE',round(time.monotonic()-began,3),flush=True)
    prod=read(out/'metrics.json');seal=read(out/'sealed_calibration.json')
    cal_csv=list(csv.DictReader((out/'calibration_curve.csv').open(encoding='utf-8-sig',newline='')))
    archives,curve_details={},{}
    for split in ('cal','hold'):
        check=scores(out,split)
        with np.load(out/f'{split}_grades_notifications.npz') as a:
            keys=a['keys'].tolist();grade=dict(zip(keys,a['grades']));saved=dict(zip(keys,a['notifications']))
            category=a['category'];raw=a['rawlight_scores']
        np.testing.assert_array_equal(raw,check['light_scores'])
        np.testing.assert_array_equal(category,check['category'])
        np.testing.assert_array_equal(grade['fixed/old5'],check['old5'])
        np.testing.assert_array_equal(grade['fixed/m3'],check['m3'])
        np.testing.assert_array_equal(grade['both955'],check['both'][0])
        emission={key:A.scalar_gap1(g) for key,g in grade.items()}
        for key,e in emission.items():
            assert time.monotonic()<deadline,'Audit budget reached'
            np.testing.assert_array_equal(e,saved[key])
            if key not in ('fixed/m3','fixed/old5','both955'):
                np.testing.assert_array_equal(grade[key]==2,grade['fixed/old5']==2)
            independent=dict(costs=I.costs(e,rows[split]),contacts={
                ref:{group:A.paired(e,emission[base],category,rows[split],group)
                     for group in ('all','dark_thin','sign_edge')}
                for ref,base in (('old5','fixed/old5'),('both','both955'))})
            record=prod['reports'][f'{split}/{key}'];I.compare_metric(independent,record)
            fields=('denominator','scene_denominator','timely','late','silent','baseline_timely','rescue','loss','net')
            for reference in ('contacts','contacts_vs_both955'):
                for f in fields:assert record[reference]['HEAD_BODY'][f]==sum(record[reference][h][f] for h in HEIGHTS)
        for ai,arm in enumerate(ARMS):
            tau=I.curve_tau(seal['calibrations'][arm]['main'])
            np.testing.assert_array_equal(I.grade(raw[ai],grade['fixed/old5'],np.inf if tau is None else tau),grade[f'{arm}/main'])
            if split=='cal':
                curve,chosen,cap=I.sweep(raw[ai],grade['fixed/old5'],rows[split],category,-np.inf,deadline)
                theirs=[r for r in cal_csv if r['arm']==arm]
                assert len(curve)==len(theirs)
                for ours,other in zip(curve,theirs):
                    assert ours['tau']==I.curve_tau(other)
                    assert ours['quarters']==int(other['weighted_cost_quarters'])
                    assert ours['farclear']==int(other['far_pass_plus_clear_notifications'])
                    assert ours['secondary']==(other['feasible'].lower()=='true')
                    assert all(int(other[h+'_'+f])==0 for h in HEIGHTS for f in ('timely','rescue','loss'))
                assert chosen['secondary']==tau
                record=seal['calibrations'][arm]['main']
                assert record['weighted_cap_quarters']==cap*1.25 and record['contact_utility_access'] is False
                assert record['candidate_thresholds']==len(curve) and record['negative_margin_ties']==len(curve)-2
                feasible=[j for j,c in enumerate(curve) if c['secondary']]
                assert record['candidate_index']==feasible[0]
                assert record['cal_weighted_cost']==curve[feasible[0]]['quarters']/4
                curve_details[arm]=dict(ties=len(curve)-2,tau=tau)
        np.testing.assert_array_equal(I.grade(raw[4],grade['fixed/old5'],seal['transfer']['tau']),grade['Sensemble/transfer'])
        archives[split]=dict(emissions=emission,category=category)
        print('AUDIT_'+split.upper(),round(time.monotonic()-began,3),flush=True)
    ledger=I.verify_ledgers(out,archives,rows)
    diagnostic=I.verify_diagnostic(out,rows)
    criterion=gates(prod)
    receipt=read(out/'evaluation_receipt.json')
    for p,h in receipt['outputs_sha256'].items():assert sha(out/p)==h
    elapsed=time.monotonic()-began
    assert elapsed+prior<220
    result=dict(status='PASS',seconds=elapsed,audit_command_wall_seconds=elapsed+prior,
        source_sha256=sha(__file__),identities=identities_result,**source_result,target_parameters=parameters,calibration=curve_details,
        ledger_counts=ledger,criteria=criterion,cal_separability=diagnostic,
        checks=['Six frozen HGB and three feature/prediction source byte hashes equal prior run',
          'Committed protocol/PLAN before cal photons; sealed cal before hold photons',
          'All new physical keys and background geometries disjoint all listed inventories including retrain',
          'Independent exact scalar gap1 replay, strong-slot lock and frozen control grades',
          'Every negative score tie and lowest feasible 1.25 gate; no contact utility used',
          'Complete event/notification ledger, per-scene cost/paired subgroup aggregates',
          'Exact prior threshold transfer and all four predeclared gates'],
        rendering=0,inference=0,training=0,protected_access=0)
    write_new(out/'independent_audit.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-only',action='store_true')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010')
    args=p.parse_args();began=time.monotonic()
    try:
        if args.source_only:
            manifest=read(args.out/'execution_manifest.json');frozen=manifest['frozen_spec']
            for name,digest in {**frozen['inherited'],**frozen['feature_sources'],**frozen['models'],**manifest['new_sources']}.items():assert sha(name)==digest
            result=source(args.out,read(args.out/'scene_rows.json'))
            receipt=dict(status='SOURCE_PASS',seconds=time.monotonic()-began,result=result,
                execution_manifest_sha256=sha(args.out/'execution_manifest.json'),scene_rows_sha256=sha(args.out/'scene_rows.json'),
                source_sha256=sha(__file__),rendering=0,inference=0,training=0)
            assert receipt['seconds']<220
            write_new(args.out/'independent_source_recompute.json',receipt)
            print(json.dumps(receipt,ensure_ascii=False),flush=True)
        else:run(args.out)
    except BaseException as error:
        args.out.mkdir(parents=True,exist_ok=True)
        write_new(args.out/f'independent_audit_failure_{time.time_ns()}.json',dict(status='FAILED',
            seconds=time.monotonic()-began,error=repr(error),source_sha256=sha(__file__)))
        raise
