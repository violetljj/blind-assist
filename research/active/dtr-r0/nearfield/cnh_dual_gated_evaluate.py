"""Calibrate public-coverage fusion, then evaluate sealed natural/angular scores.

No training. Geometry and fusion never read evaluator objects/categories.
Calibration uses only pilot2 calibration24 and natural95000. Fresh97000 and
angular evaluation scenes cannot select rule, threshold, estimator or tie breaks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import cnh_dual_gated_geometry as G

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-dual-gated-fusion-20261005'
PILOT=ROOT/'artifacts.local/work/cnh-dual-sensor-alarm-20261005'
ENVELOPE=ROOT/'artifacts.local/work/cnh-dual-sensor-envelope-20261005'
SINGLE=.8557642486787612
RULES=('G1_0.3','G1_0.5','G1_0.7','G2','G3')

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def save_npz(p,**arrays):
    p=Path(p)
    if p.exists():raise FileExistsError('Preserve sealed payload: '+str(p))
    np.savez_compressed(p,**arrays)

def freeze(out=OUT):
    out=Path(out);p=out/'fusion/PLAN.json'
    if p.exists():return read(p)
    plan=dict(parent_plan_sha256=sha(out/'PLAN.json'),frozen_unix=time.time(),
        rules=list(RULES),single_OR_threshold=SINGLE,operator='>=',
        coverage=dict(lateral_m=[-.29,.29],forward_m=[.9,2.5],height_m=G.HEIGHTS,
            geometry='Exact clipped convex-polyhedron volume, rectangular 45 degree optical FOV',
            direction='yaw of estimated translation p[t]-p[t-3], causal 0.6s displacement tangent',
            transform='Ry(-estimated_travel_yaw) @ estimated_head_pose.R @ fixed_installation_extrinsic.R; current estimated head origin',
            zero_horizontal_displacement='Below1um => UNKNOWN coverage0; no clear evidence',
            initial_anchor='Noisy integrated poses retain exact initial synthetic pose anchor; world gravity is assumed public reference',
            angular='Reuse sealed original noisy head poses; apply additional fixed yaw extrinsic(absolute_angle-15), preserving same noise trajectory',
            current_only=True,no_truth=True,heading_lookback_frames=3),
        cost='Query episodes exposed for13 frames2.6s; fully clear across all frames; pilot all variants/query streams, natural all config/query streams; pool query denominators',
        calibration='Pilot2 calibration24 plus existing natural95000 single and newly rendered dual only',
        threshold='Per rule lowest finite >= threshold with pooled clear stops <= floor(1.1*single_clear_stops), excluding absent -inf branches; no evaluation access',
        selection='Highest pooled covered contact timely count/rate (all shallow+deep); ties highest shallow timely, then deep timely, then fixed candidate order',
        evaluator='Natural97000 point gates unchanged; angular evaluation split only, opposite shallow eachpsi<=25 >=.90',
        bootstrap=dict(unit='Paired whole scenes/units',n=1000,seed=2026100543),
        source_sha256={Path(__file__).name:sha(__file__),Path(G.__file__).name:sha(G.__file__)})
    write(p,plan);return plan

def events(scores,threshold,ranges):
    alarm=np.asarray(scores)>=threshold;stopped=alarm.any(-1);first=alarm.argmax(-1)
    distance=np.take_along_axis(ranges,first[...,None],-1)[...,0]
    return dict(alarm=alarm,stopped=stopped,first_index=first,first_range=distance,timely=stopped&(distance>=.9))

def cutoff(values,budget,all_scores):
    values=np.asarray(values,float);finite=values[np.isfinite(values)]
    full=np.asarray(all_scores,float);ff=full[np.isfinite(full)]
    if not len(finite):return float(np.nextafter(ff.min(),-np.inf)) if len(ff) else 0.
    unique,size=np.unique(finite,return_counts=True);unique=unique[::-1];size=size[::-1]
    n_groups=int((np.cumsum(size)<=budget).sum())
    return float(np.nextafter(unique[n_groups],np.inf)) if n_groups<len(unique) else float(np.nextafter(ff.min(),-np.inf))

def load_pilot(which):
    import cnh_dual_sensor_evaluate as D
    baseline=D.load_baseline();parent=read(PILOT/'PLAN.json')
    units=parent[which+'_units'];idx=[baseline['units'].tolist().index(u) for u in units]
    raw=[];cov=[];hashes={}
    for u in units:
        p=PILOT/'scores'/f'unit{u}.npz'
        with np.load(p,allow_pickle=False) as z:raw.append(z['raw_full'].copy())
        op=PILOT/'observations'/f'unit{u}.npz'
        with np.load(op,allow_pickle=False) as z:noisy=z['noisy'][0]@np.linalg.inv(G.extrinsic(-15))
        # [K,sensor,frame,query], computed before consulting labels below.
        cov.append(np.stack([G.query_coverage(G.estimated_query_poses(n)) for n in noisy]))
        hashes[str(p)]=sha(p);hashes[str(op)]=sha(op)
    raw=np.stack(raw);sensor=D.smooth_full(raw).transpose(0,2,3,5,1,4).reshape(-1,2,13)
    coverage=np.stack(cov).transpose(0,1,4,2,3)[:,None]
    coverage=np.broadcast_to(coverage,(len(units),7,4,2,2,13)).reshape(-1,2,13)
    single=baseline['both'][idx].transpose(0,1,2,4,3).reshape(-1,13)
    # ordered_episodes already applied the original causal smoothing once.
    scenes=[baseline['scenes'][i] for i in idx]
    cats=baseline['all_categories'][idx]
    clear=(cats=='clear').all(2)[:,:,None,:]
    clear=np.broadcast_to(clear,(len(units),7,4,2)).reshape(-1)
    covered=np.broadcast_to(baseline['covered'][idx][:,:,None,None],(len(units),7,4,2)).reshape(-1)
    shallow=np.zeros((len(units),7,4,2),bool);deep=shallow.copy()
    for i,s in enumerate(scenes):shallow[i,:2,:,s['group']]=True;deep[i,2,:,s['group']]=True
    ranges=np.asarray([s['front_range_m'] for s in scenes])
    ranges=np.broadcast_to(ranges[:,None,None,None,:],(len(units),7,4,2,13)).reshape(-1,13)
    return dict(single=single,sensor=sensor,coverage=coverage,ranges=ranges,clear=clear,
        shallow=shallow.reshape(-1)&covered,mid=np.zeros(len(single),bool),deep=deep.reshape(-1)&covered,
        unit=np.repeat(units,7*4*2),config=np.tile(np.arange(7*4).repeat(2),len(units)),query=np.tile([0,1],len(single)//2),
        covered=covered,mode=np.repeat([s['mode'] for s in scenes],7*4*2),hashes=hashes,domain='pilot2_'+which)

def load_natural(out,tag):
    import cnh_dual_sensor_evaluate as D
    out=Path(out);work=out/('natural'+str(tag));paths=sorted((work/'scores').glob('unit*.npz'))
    if not paths:raise FileNotFoundError('Natural scores required: '+str(work))
    units=[];configs=None;raw=[];cov=[];hashes={}
    cp=out/'fusion'/f'public_coverage{tag}.npz';cached=None
    if cp.exists():
        with np.load(cp,allow_pickle=False) as z:cached={k:z[k].copy() for k in z.files}
        receipt=read(cp.with_suffix('.json'))
        assert receipt['coverage_sha256']==sha(cp) and receipt['geometry_source_sha256']==sha(G.__file__)
        hashes[str(cp)]=sha(cp);hashes[str(cp.with_suffix('.json'))]=sha(cp.with_suffix('.json'))
    for p in paths:
        with np.load(p,allow_pickle=False) as z:
            u=int(z['unit']);cc=z['configs'].tolist();rr=z['raw'].copy()
            assert np.array_equal(z['frames'],G.FRAMES)
        receipt=p.with_suffix('.json')
        if receipt.exists():assert read(receipt)['score_sha256']==sha(p)
        if configs is None:configs=cc
        assert configs==cc,'Only uniform config selection allowed'
        assert rr.shape==(2 if tag==95000 else 3,len(configs),13,2) and np.isfinite(rr).all()
        units.append(u);raw.append(rr)
        if cached is None:cov.append(np.stack([G.natural_coverage(u,c) for c in configs]))
        else:
            i=cached['units'].tolist().index(u);ids=[cached['configs'].tolist().index(c) for c in configs]
            cov.append(cached['coverage'][i,ids])
        hashes[str(p)]=sha(p)
    raw=np.stack(raw);smoothed=D.smooth_full(raw)
    parent=read(out/'PLAN.json')
    assert units==parent['natural_calibration_units' if tag==95000 else 'natural_evaluation_units'],'All frozen units required before calibration/evaluation'
    assert configs==list(range(0,40,1 if tag==95000 else configs[1]-configs[0])) and configs[0]==0,'Uniform predefined config subset only'
    cov=np.stack(cov).transpose(0,1,4,2,3).reshape(-1,2,13)
    if tag==95000:
        import cnh_sequence_observed_evaluate as OE
        geometry,baseline,provenance=OE.load_inputs()
        keep=(geometry['split']=='calib')&np.isin(geometry['unit'],units)&np.isin(geometry['config'],configs)
        gg={k:v[keep] for k,v in geometry.items() if v.shape[0]==len(keep)};single=baseline['M3'][keep]
        sensor=smoothed.transpose(0,2,4,1,3).reshape(-1,2,13)
        hashes.update(provenance['input_sha256'])
    else:
        with np.load(work/'geometry.npz',allow_pickle=False) as z:gg={k:z[k].copy() for k in z.files}
        keep=np.isin(gg['unit'],units)&np.isin(gg['config'],configs)
        gg={k:v[keep] for k,v in gg.items() if v.shape[0]==len(keep)}
        single=smoothed[:,0].transpose(0,1,3,2).reshape(-1,13)
        sensor=smoothed[:,1:].transpose(0,2,4,1,3).reshape(-1,2,13)
        hashes[str(work/'geometry.npz')]=sha(work/'geometry.npz')
    expected=[(u,c,q) for u in units for c in configs for q in (0,1)]
    assert list(zip(gg['unit'].tolist(),gg['config'].tolist(),gg['query'].tolist()))==expected
    assert sensor.shape==(len(single),2,13) and cov.shape==sensor.shape
    return dict(single=single,sensor=sensor,coverage=cov,ranges=gg['frame_ranges'],clear=gg['clear_all'],
        shallow=gg['covered']&(gg['ref_category']=='contact0-2cm'),
        mid=gg['covered']&(gg['ref_category']=='contact2-5cm'),
        deep=gg['covered']&(gg['ref_category']=='contact>5cm'),
        contact0_2=gg['covered']&(gg['ref_category']=='contact0-2cm'),contact2_5=gg['covered']&(gg['ref_category']=='contact2-5cm'),
        unit=gg['unit'],config=gg['config'],query=gg['query'],covered=gg['covered'],mode=gg['unit']%3,
        hashes=hashes,domain='natural'+str(tag),units=units,configs=configs)

def precompute_coverage(out,tag):
    """Public CPU geometry can run while GPU produces unopened scientific scores."""
    out=Path(out);path=out/'fusion'/f'public_coverage{tag}.npz'
    if path.exists():return
    parent=read(out/'PLAN.json');units=parent['natural_calibration_units' if tag==95000 else 'natural_evaluation_units'];configs=list(range(40));tick=time.monotonic();cov=[]
    for i,u in enumerate(units):
        cov.append(np.stack([G.natural_coverage(u,c) for c in configs]))
        if (i+1)%12==0:print('public coverage',tag,i+1,'/',len(units),round(time.monotonic()-tick,1),flush=True)
    save_npz(path,units=units,configs=configs,coverage=np.stack(cov))
    write(path.with_suffix('.json'),dict(status='COMPLETE',only_public_inputs=True,units=units,configs=configs,
        geometry_source_sha256=sha(G.__file__),coverage_sha256=sha(path),seconds=time.monotonic()-tick))

def simple_metrics(ev,d):
    result={}
    for key,flag in [('clear',ev['stopped']),('shallow',ev['timely']),('mid',ev['timely']),('deep',ev['timely'])]:
        den=d[key];n=int(den.sum());num=int((den&flag).sum())
        result[key]=dict(stops=num,n=n,rate=num/n if n else None)
        if key=='clear':result[key].update(proxy_minutes=n*2.6/60,stops_per_proxy_minute=num/(n*2.6/60) if n else None)
    result['all_contact']=dict(stops=sum(result[k]['stops'] for k in ('shallow','mid','deep')),n=sum(result[k]['n'] for k in ('shallow','mid','deep')))
    return result

def calibrate(out=OUT):
    out=Path(out);tick=time.monotonic();plan=freeze(out)
    target=out/'fusion/selection.json'
    if target.exists():return read(target)
    ds=[load_pilot('calibration'),load_natural(out,95000)]
    single_metrics=[simple_metrics(events(d['single'],SINGLE,d['ranges']),d) for d in ds]
    single_stops=sum(m['clear']['stops'] for m in single_metrics);budget=int(np.floor(single_stops*1.1+1e-12))
    total_n=sum(m['clear']['n'] for m in single_metrics)
    candidates=[];ledger={};provenance={}
    for d,m in zip(ds,single_metrics):
        domain=d['domain'];provenance.update(d['hashes'])
        for k in ('single','sensor','coverage','ranges','clear','shallow','mid','deep','unit','config','query','covered'):ledger[domain+'_'+k]=d[k]
    for ri,rule in enumerate(RULES):
        fused=[G.fuse(d['sensor'],d['coverage'],rule) for d in ds]
        maxima=np.concatenate([v.max(-1)[d['clear']] for v,d in zip(fused,ds)])
        thr=cutoff(maxima,budget,np.concatenate(fused));cost=int((maxima>=thr).sum());assert cost<=budget
        metrics=[simple_metrics(events(v,thr,d['ranges']),d) for v,d in zip(fused,ds)]
        contacts=sum(m['all_contact']['stops'] for m in metrics);shallow=sum(m['shallow']['stops'] for m in metrics);deep=sum(m['deep']['stops'] for m in metrics)
        candidates.append(dict(rule=rule,threshold=thr,clear_stops=cost,clear_n=total_n,allowed_stops=budget,
            pooled_timely=contacts,pooled_shallow_timely=shallow,pooled_deep_timely=deep,
            domains={d['domain']:dict(metrics=m,single=s,clear_within_single_x1p1=m['clear']['stops']<=s['clear']['stops']*1.1+1e-12) for d,m,s in zip(ds,metrics,single_metrics)},
            rank_key=[contacts,shallow,deep,-ri]))
        for d,v in zip(ds,fused):ledger[d['domain']+'_'+rule]=v
    selected=max(candidates,key=lambda c:c['rank_key'])
    save_npz(out/'fusion/calibration_ledger.npz',**ledger)
    result=dict(status='COMPLETE',selected=selected,candidates=candidates,pooled_single_clear_stops=single_stops,
        pooled_allowed_clear_stops=budget,pooled_clear_n=total_n,calibration_domains=[d['domain'] for d in ds],
        plan_sha256=sha(out/'fusion/PLAN.json'),input_sha256=provenance,ledger_sha256=sha(out/'fusion/calibration_ledger.npz'),
        seconds=time.monotonic()-tick)
    write(target,result);print('SELECTED',selected['rule'],selected['threshold'],selected['pooled_timely'],flush=True);return result

def natural_metrics(d,arms,thresholds):
    units=np.unique(d['unit']);ui=np.searchsorted(units,d['unit']);rng=np.random.default_rng(2026100543)
    boot=np.array([np.bincount(rng.integers(len(units),size=len(units)),minlength=len(units)) for _ in range(1000)])
    evs={a:events(v,thresholds[a],d['ranges']) for a,v in arms.items()};cells={}
    for group in ('all','mode0','mode1','mode2'):
        keep=np.ones(len(ui),bool) if group=='all' else d['mode']==int(group[-1]);cells[group]={}
        for arm,ev in evs.items():
            cell={}
            for category in ('clear','shallow','contact0_2','contact2_5','deep'):
                den=keep&d[category];flag=ev['stopped'] if category=='clear' else ev['timely'];num=int((den&flag).sum());n=int(den.sum())
                count=np.bincount(ui,weights=den&flag,minlength=len(units));total=np.bincount(ui,weights=den,minlength=len(units))
                with np.errstate(invalid='ignore',divide='ignore'):draws=(boot@count)/(boot@total)
                finite=draws[np.isfinite(draws)]
                metric=dict(stops=num,n=n,rate=num/n if n else None,ci95=np.quantile(finite,[.025,.975]).tolist() if len(finite) else [None,None])
                baseline=evs['single']['stopped'] if category=='clear' else evs['single']['timely']
                metric.update(rescues=int((den&flag&~baseline).sum()),losses=int((den&~flag&baseline).sum()))
                base_count=np.bincount(ui,weights=den&baseline,minlength=len(units))
                with np.errstate(invalid='ignore',divide='ignore'):delta=(boot@(count-base_count))/(boot@total)
                df=delta[np.isfinite(delta)]
                metric['paired_delta_ci95']=np.quantile(df,[.025,.975]).tolist() if len(df) else [None,None]
                if category=='clear':metric.update(proxy_minutes=n*2.6/60,stops_per_proxy_minute=num/(n*2.6/60) if n else None,
                    rate_ci95_per_proxy_minute=(np.array(metric['ci95'])*60/2.6).tolist() if len(finite) else [None,None])
                cell[category]=metric
            # Physical episode union, only if both queries fully clear.
            assert np.array_equal(d['query'].reshape(-1,2),np.broadcast_to([0,1],(len(ui)//2,2)))
            clear=d['clear'].reshape(-1,2).all(1)&keep.reshape(-1,2).all(1)
            joint=ev['stopped'].reshape(-1,2).any(1);n=int(clear.sum());num=int((joint&clear).sum())
            cell['joint_clear']=dict(stops=num,n=n,proxy_minutes=n*2.6/60,stops_per_proxy_minute=num/(n*2.6/60) if n else None)
            cell['coverage_missing_frames']=int(((d['coverage'].sum(1)==0)&keep[:,None]).sum())
            cell['abstention_frames']=int((~np.isfinite(arms[arm])&keep[:,None]).sum())
            cell['episodes_with_abstention']=int((~np.isfinite(arms[arm]).all(1)&keep).sum())
            cell['episodes_all_frames_abstention']=int((~np.isfinite(arms[arm]).any(1)&keep).sum())
            cell['covered']=int((keep&d['covered']).sum());cell['censored']=int((keep&~d['covered']).sum())
            cells[group][arm]=cell
    return cells,evs

def angular(out,selected):
    out=Path(out);existing=out/'fusion/angular_result.json'
    if existing.exists():
        result=read(existing)
        assert result['selected_rule']==selected['rule'] and result['selected_threshold']==selected['threshold']
        assert result['selection_sha256']==sha(out/'fusion/selection.json')
        assert result['ledger_sha256']==sha(out/'fusion/angular_ledger.npz')
        return result
    import cnh_dual_sensor_evaluate as D
    baseline=D.load_baseline();parent=read(PILOT/'PLAN.json');plan=read(ENVELOPE/'PLAN.json')
    units=sorted(set(parent['evaluation_units'])&set(plan['A']['units']))
    indices=[baseline['units'].tolist().index(u) for u in units];scenes=[baseline['scenes'][i] for i in indices]
    raw=[];noisy=[];hashes={};angles=None
    for u in units:
        p=ENVELOPE/'angular/scores'/f'unit{u}.npz'
        with np.load(p,allow_pickle=False) as z:
            if angles is None:angles=z['angles_deg'].tolist()
            assert angles==z['angles_deg'].tolist();raw.append(z['raw'].copy())
        op=PILOT/'observations'/f'unit{u}.npz'
        with np.load(op,allow_pickle=False) as z:noisy.append(z['noisy'][0]@np.linalg.inv(G.extrinsic(-15)))
        hashes[str(p)]=sha(p);hashes[str(op)]=sha(op)
    scores=D.smooth_full(np.stack(raw));lookup={float(a):i for i,a in enumerate(angles)}
    covered=baseline['covered'][indices];categories=baseline['all_categories'][indices]
    clear=np.broadcast_to((categories=='clear').all(2)[:,:,None,:],(len(units),7,4,2)).reshape(-1)
    ranges=np.broadcast_to(np.array([s['front_range_m'] for s in scenes])[:,None,None,None,:],(len(units),7,4,2,13)).reshape(-1,13)
    evaluation={};ledger={};support=[]
    for psi in plan['A']['psi_deg']:
        b=scores[:,[lookup[float(psi-15)],lookup[float(psi+15)]]].transpose(0,2,3,5,1,4).reshape(-1,2,13)
        cov=np.stack([np.stack([G.query_coverage(G.estimated_query_poses(n,angles=(psi-30,psi))) for n in nn]) for nn in noisy])
        cov=np.broadcast_to(cov.transpose(0,1,4,2,3)[:,None],(len(units),7,4,2,2,13)).reshape(-1,2,13)
        single=scores[:,lookup[float(psi)]].transpose(0,1,2,4,3).reshape(-1,13)
        arms=dict(single=single,OR=b.max(1),gated=G.fuse(b,cov,selected['rule']))
        thresholds=dict(single=SINGLE,OR=SINGLE,gated=selected['threshold'])
        masks={}
        for group in ('all','opposite','same'):
            shallow=np.zeros((len(units),7,4,2),bool);deep=shallow.copy()
            for i,s in enumerate(scenes):
                if group=='all' or (group=='same')==bool(s['fov_in']):
                    shallow[i,:2,:,s['group']]=covered[i,:2,None];deep[i,2,:,s['group']]=covered[i,2]
            masks[group]=(shallow.reshape(-1),deep.reshape(-1))
        cells={}
        for arm,v in arms.items():
            ev=events(v,thresholds[arm],ranges);cells[arm]={}
            for group,(shallow,deep) in masks.items():
                cell={}
                for name,den in [('shallow',shallow),('deep',deep)]:
                    n=int(den.sum());num=int((den&ev['timely']).sum());cell[name]=dict(stops=num,n=n,rate=num/n if n else None)
                cells[arm][group]=cell
            num=int((clear&ev['stopped']).sum());n=int(clear.sum());cells[arm]['clear_query']=dict(stops=num,n=n,proxy_minutes=n*2.6/60,stops_per_proxy_minute=num/(n*2.6/60) if n else None)
            ledger[f'psi{psi}_{arm}']=v
            for key,val in ev.items():ledger[f'psi{psi}_{arm}_{key}']=val
        if psi<=25:support.append(cells['gated']['opposite']['shallow']['rate'] is not None and cells['gated']['opposite']['shallow']['rate']>=.9-1e-12)
        evaluation[str(psi)]=cells;ledger[f'psi{psi}_coverage']=cov;ledger[f'psi{psi}_sensor']=b
    ledger.update(units=np.array(units),covered=covered,clear_query=clear,ranges=ranges,opposite=np.array([not s['fov_in'] for s in scenes]))
    save_npz(Path(out)/'fusion/angular_ledger.npz',**ledger)
    result=dict(status='COMPLETE',units=units,scenes=len(units),opposite_scenes=sum(not s['fov_in'] for s in scenes),psi_deg=plan['A']['psi_deg'],sequence=evaluation,
        selected_rule=selected['rule'],selected_threshold=selected['threshold'],selection_sha256=sha(out/'fusion/selection.json'),
        tolerance_psi_le25_pass=all(support),input_sha256=hashes,ledger_sha256=sha(Path(out)/'fusion/angular_ledger.npz'))
    write(Path(out)/'fusion/angular_result.json',result);return result

def evaluate(out=OUT):
    out=Path(out);tick=time.monotonic();selection=read(out/'fusion/selection.json');selected=selection['selected']
    if (out/'fusion/result.json').exists():
        result=read(out/'fusion/result.json')
        assert result['selection_sha256']==sha(out/'fusion/selection.json') and result['ledger_sha256']==sha(out/'fusion/natural97000_ledger.npz')
        return result
    assert selection['plan_sha256']==sha(out/'fusion/PLAN.json')
    d=load_natural(out,97000);arms=dict(single=d['single'],OR=d['sensor'].max(1),gated=G.fuse(d['sensor'],d['coverage'],selected['rule']))
    cells,evs=natural_metrics(d,arms,dict(single=SINGLE,OR=SINGLE,gated=selected['threshold']))
    ar=angular(out,selected);a=cells['all'];single=a['single'];gated=a['gated']
    ratio=gated['clear']['stops']/single['clear']['stops'] if single['clear']['stops'] else (1. if gated['clear']['stops']==0 else None)
    shallow_delta=gated['shallow']['rate']-single['shallow']['rate'] if gated['shallow']['rate'] is not None and single['shallow']['rate'] is not None else None
    deep_delta=gated['deep']['rate']-single['deep']['rate'] if gated['deep']['rate'] is not None and single['deep']['rate'] is not None else None
    clearpass=ratio is not None and ratio<=1.15+1e-12;tolpass=ar['tolerance_psi_le25_pass']
    if ratio is None or shallow_delta is None or deep_delta is None:branch='NOT_EVALUABLE'
    elif clearpass and shallow_delta>=-.03-1e-12 and deep_delta>=-.03-1e-12 and tolpass:branch='GATED_DUAL_SUPPORTED_SIM'
    elif (clearpass and not tolpass) or (tolpass and ratio>1.3):branch='GATING_TRADEOFF'
    else:branch='GATING_NOT_SUPPORTED'
    ledger={k:d[k] for k in ('single','sensor','coverage','ranges','clear','shallow','mid','deep','unit','config','query','covered','contact0_2','contact2_5')}
    ledger['gated']=arms['gated']
    for arm,ev in evs.items():
        for key,val in ev.items():ledger[arm+'_'+key]=val
    save_npz(out/'fusion/natural97000_ledger.npz',**ledger)
    result=dict(status='COMPLETE',selected_rule=selected['rule'],selected_threshold=selected['threshold'],natural97000=cells,
        units=d['units'],configs=d['configs'],uniform_config_count=len(d['configs']),query_episodes=len(d['unit']),
        angular_evaluation=ar,decision=dict(branch=branch,clear_ratio=ratio,clear_pass=clearpass,shallow_delta_rate=shallow_delta,deep_delta_rate=deep_delta,tolerance_pass=tolpass),
        selection_sha256=sha(out/'fusion/selection.json'),input_sha256=d['hashes'],ledger_sha256=sha(out/'fusion/natural97000_ledger.npz'),seconds=time.monotonic()-tick,
        limits=['Synthetic Development, not hardware/real-world/safety proof','Integrated noisy pose model assumes public gravity and exact initial anchor; estimator not validated on hardware','Query exposure and joint physical episode proxies are not real walking incidence','Coverage no eligible branch/unknown pose is absent alarm, never clear evidence','96000 used only for diagnostic, not calibration or evaluation'])
    write(out/'fusion/result.json',result);print(json.dumps(result['decision']),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','calibrate','evaluate','angular','coverage95000','coverage97000']);p.add_argument('--out',type=Path,default=OUT);args=p.parse_args()
    if args.stage.startswith('coverage'):precompute_coverage(args.out,int(args.stage[-5:]))
    elif args.stage=='angular':angular(args.out,read(args.out/'fusion/selection.json')['selected'])
    else:{'freeze':freeze,'calibrate':calibrate,'evaluate':evaluate}[args.stage](args.out)
