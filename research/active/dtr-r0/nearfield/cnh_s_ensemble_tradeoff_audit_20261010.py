"""Independent pooled-negative / fresh-hold tradeoff audit, saved outputs only.

No producer evaluator imports, rendering, fitting or protected-data access.
Every canonical threshold is checked using independent clip maxima: gap1 must
emit the first nonzero grade, so clip-any and timely counts equal grade support.
The scalar state machine independently checks all selected grades and ledgers.
"""
import argparse
import ast
import csv
import json
import pickle
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
import audit_cnh_cost_v2_holdout_dev_20261010 as A
import cnh_task_cost_audit_20261010 as I
import cnh_s_ensemble_confirm_audit_20261010 as P

ROOT=A.ROOT
ARMS=('E','S955','S956','S957','Sensemble')
PRIOR=ROOT/'artifacts.local/work/cnh-task-cost-retrain-dev-20261010'
read,sha=A.read,A.sha


def save_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def threshold(record):
    tau=I.curve_tau(record)
    return np.inf if tau is None else tau


def thresholds(score,fixed):
    eligible=(fixed==0)&np.isfinite(score)
    ties=np.unique(score[eligible])
    return np.r_[-np.inf,np.nextafter(ties.astype(float),np.inf),np.inf]


def count_at(maxima,taus):
    vals=np.sort(np.asarray(maxima)[np.isfinite(maxima)])
    return len(vals)-np.searchsorted(vals,taus,side='left')


def clip_curve(score,fixed,rows,taus):
    selected=I.masks(rows)['far_pass_plus_clear']
    strong=(fixed[selected]>0).any((-2,-1))
    eligible=(fixed[selected]==0)&np.isfinite(score[selected])
    maximum=np.where(eligible,score[selected],-np.inf).max((-2,-1))
    return int(strong.sum())+count_at(maximum[~strong],taus),int(selected.sum()*fixed.shape[1])


def contact_curve(score,fixed,category,rows,taus,group='all'):
    before=(fixed[:,:,:11]>0).any(-2)
    maximum=np.where((fixed[:,:,:11]==0)&np.isfinite(score[:,:,:11]),
                     score[:,:,:11],-np.inf).max(-2)
    selected=np.ones(len(rows),dtype=bool)
    if group!='all':selected=np.array([r[group] if group=='dark_thin' else r['shape_family']=='sign_edge' for r in rows])
    result={}
    for q,h in enumerate(A.HEIGHTS):
        physical=selected&(category[:,q]=='contact')
        base=before[physical,:,q]
        ma=maximum[physical,:,q]
        net=count_at(ma[~base],taus)
        result[h]=dict(net=net,timely=int(base.sum())+net,rescue=net,
                       loss=np.zeros(len(taus),dtype=int),denominator=int(physical.sum()*fixed.shape[1]))
    return result


def family_rates(emission,baseline,rows):
    fc=I.masks(rows)['far_pass_plus_clear']
    a=(emission>0).any((-2,-1));b=(baseline>0).any((-2,-1))
    families=np.array([r['background_family'] for r in rows])
    result={}
    for family in sorted(set(families)):
        mask=fc&(families==family);den=int(mask.sum()*a.shape[1])
        result[family]=dict(clip_denominator=den,clips_with_any_notification=int(a[mask].sum()),
                           baseline_clips_with_any_notification=int(b[mask].sum()),
                           increment_pp=float(100*(a[mask].sum()-b[mask].sum())/den))
    return result


def frozen_features(raw,current,current_valid):
    smooth=A.smooth(raw)
    slope=np.zeros_like(raw,dtype=float);residual=np.zeros_like(raw,dtype=float)
    for frame in range(13):
        y=raw[...,max(0,frame-4):frame+1,:]
        x=np.arange(y.shape[-2],dtype=float);x-=x.mean()
        if len(x)>1:
            b=(y*x[:,None]).sum(-2)/(x*x).sum()
            slope[...,frame,:]=b
            residual[...,frame,:]=np.sqrt(((y-y.mean(-2)[...,None,:]-b[...,None,:]*x[:,None])**2).mean(-2))
    cuts=read(ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
    valid=current_valid&np.isfinite(current)
    spatial=np.concatenate((np.where(valid,current,np.nan),(~valid).astype(float)),-1)
    result=[]
    for i,seed in enumerate((2026100955,2026100956,2026100957)):
        first3=np.stack((smooth[i]-cuts[str(seed)]['single'],slope[i],residual[i]),-1)
        result.append(np.concatenate((first3,spatial),-1).astype(np.float32))
    return np.asarray(result)


def predict_sample(features):
    result=np.empty(features.shape[:-1],dtype=np.float32)
    with threadpool_limits(limits=2):
        for ai,seed in enumerate((2026100955,2026100956,2026100957)):
            for q,h in enumerate(A.HEIGHTS):
                with (PRIOR/'models'/f'S_{seed}_{h}.pickle').open('rb') as f:model=pickle.load(f)
                x=features[ai,...,q,:]
                result[ai,...,q]=model.decision_function(x.reshape(-1,47)).reshape(x.shape[:-1])
    return result


def identities(out):
    manifest=read(out/'execution_manifest.json');frozen=manifest['frozen_spec']
    repaired=[]
    for p,h in {**frozen['frozen_dependencies'],**manifest['new_sources']}.items():
        if sha(p)!=h and Path(p).name=='cnh_s_ensemble_tradeoff_20261010.py':
            receipt=read(out/'source_serialization_repair_receipt.json')
            snapshot=out/receipt['original_controller_snapshot']
            assert receipt['original_execution_manifest_sha256']==sha(out/'execution_manifest.json')
            assert sha(snapshot)==h==receipt['original_controller_sha256']
            assert sha(p)==receipt['delivery_controller_sha256'] and receipt['scientific_recipe_changed'] is False
            before={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(snapshot.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
            after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
            for function in ('spec','pool','seal','data','evaluate','identity'):
                assert before[function]==after[function],function
            repaired.append(dict(path=p,science_functions_AST_equal=True,original_sha256=h,delivery_sha256=sha(p)))
        else:assert sha(p)==h,p
    expected={p:h for p,h in read(PRIOR/'trained_models_manifest.json')['models'].items() if Path(p).name.startswith('S_')}
    assert frozen['models']==expected and len(expected)==6
    previous=read(PRIOR/'execution_manifest.json')
    for p,h in frozen['feature_sources'].items():assert previous['new_sources'][p]==h
    assert manifest['training']==0 and manifest['protected_access']==0
    seal=read(out/'sealed_calibration.json')
    for p,h in seal['bindings'].items():assert sha(out/p)==h,p
    assert (out/'PLAN.json').stat().st_mtime_ns<(out/'data/hold/hist.npy').stat().st_mtime_ns
    assert (out/'sealed_calibration.json').stat().st_mtime_ns<(out/'PLAN.json').stat().st_mtime_ns
    assert (out/'sealed_calibration.json').stat().st_mtime_ns<(out/'data/hold/hist.npy').stat().st_mtime_ns
    receipt=read(out/'data_hold_receipt.json')
    assert receipt['cal_seal_sha256']==sha(out/'sealed_calibration.json') and receipt['training']==0
    candidate=read(out/'candidate_manifest.json')
    assert candidate['features']['dimensions']==47 and candidate['threshold']==seal['calibrations']['Sensemble']
    for entry in candidate['models']+candidate['features']['implementation']+candidate['strong_source']['frozen_dependencies']:
        assert sha(ROOT/entry['path'])==entry['sha256']
    assert {str((ROOT/e['path']).resolve()):e['sha256'] for e in candidate['models']}=={str(Path(p).resolve()):h for p,h in expected.items()}
    assert candidate['calibration']['sha256']==sha(out/'sealed_calibration.json')
    assert candidate['notifier']['gap']==1 and candidate['notifier']['nominal_frames']==list(range(3,16))
    assert sha(ROOT/candidate['notifier']['implementation'])==candidate['notifier']['sha256']
    assert candidate['weights']==dict(near_pass_max_gap_m=.10,near_light=.25,near_strong=1,far_clear_any=1)
    assert candidate['hold_selection'] is False and candidate['training']==0 and candidate['protected_access']==0
    return dict(frozen_models=6,frozen_feature_sources=len(frozen['feature_sources']),candidate_manifest_sha256=sha(out/'candidate_manifest.json'),
                metadata_repairs=repaired,training=0,protected_access=0)


def source(out,rows):
    with np.load(out/'data/hold/geometry.npz') as a:category,sensor=a['category'],a['sensor']
    keys,counts=A.geometry(rows,sensor,category)
    assert len(rows)==1536 and counts=={'contact':512,'pass':512,'clear':512}
    assert (category=='contact').sum(0).tolist()==[256,256]
    assert len({r['background_family'] for r in rows})==4
    for layer in ('0-5cm','5-10cm','10-20cm','20-35cm'):
        sub=[r for r in rows if r['pass_layer']==layer]
        assert len(sub)==128 and {r['group'] for r in sub}=={0,1}
    assert all(r['lateral_gap_m']>=.35 for r in rows if r['placement']=='clear')
    inventory=A.inventory_check(out,{'hold':keys},{'hold':rows})
    assert any('cnh-s-ensemble-confirm-dev-20261010/scene_rows.json' in e['path'].replace('\\','/') for e in read(out/'inventory.json')['files'])
    oldd,oldrho=set(),set()
    def collect(v):
        if isinstance(v,dict):
            b=v.get('target_box')
            if isinstance(b,dict) and set(('lo','hi','rho'))<=set(b):
                oldd.update(round(float(hi)-float(lo),10) for lo,hi in zip(b['lo'],b['hi']))
                oldrho.add(round(float(b['rho']),10))
            if isinstance(v.get('target_thickness_m'),(int,float)):oldd.add(round(float(v['target_thickness_m']),10))
            for item in v.values():collect(item)
        elif isinstance(v,list):
            for item in v:collect(item)
    for entry in read(out/'inventory.json')['files']:collect(read(ROOT/entry['path']))
    d={round(float(hi)-float(lo),10) for r in rows for lo,hi in zip(r['target_box']['lo'],r['target_box']['hi'])}
    d|={round(float(r['target_thickness_m']),10) for r in rows}
    rho={round(float(r['target_box']['rho']),10) for r in rows}
    assert not d&oldd and not rho&oldrho
    return dict(physical_scenes=1536,contact_events=1024,families=4,inventory=inventory,target_parameter_overlaps=0)


def pooled(out,deadline):
    rows=read(out/'pool_rows.json');receipt=read(out/'pool_input_receipt.json')
    assert len(rows)==3584 and all(r['placement'] in ('pass','clear') for r in rows)
    assert len({r['physical_key'] for r in rows})==len(rows)
    assert len({r['background_family'] for r in rows})==14
    assert receipt['contact_utility_access'] is False and receipt['negative_scenes']==len(rows)
    assert receipt['archive_sha256']==sha(out/'pool_scores.npz') and receipt['rows_sha256']==sha(out/'pool_rows.json')
    for p,h in receipt['bindings'].items():assert sha(p)==h,p
    with np.load(out/'pool_scores.npz') as a:archive={k:a[k].copy() for k in a.files}
    assert tuple(archive['arm_keys'])==ARMS and not (archive['category']=='contact').any()
    np.testing.assert_array_equal(archive['light_scores'][4],archive['light_scores'][1:4].astype(np.float32).mean(0))
    counts=[]
    cuts=read(ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
    jointcuts=read(ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010/calibrations.json')
    sample_predictions=0
    cohorts={(r['source_dataset'],r['source_split']) for r in rows}
    expected={(d,s) for d in ('cnh-cost-v2-holdout-dev-20261010','cnh-task-cost-retrain-dev-20261010','cnh-s-ensemble-confirm-dev-20261010') for s in ('cal','hold')}
    assert cohorts==expected
    for dataset,split in sorted(cohorts):
        assert time.monotonic()<deadline
        ids=np.array([i for i,r in enumerate(rows) if (r['source_dataset'],r['source_split'])==(dataset,split)])
        folder=out.parent/dataset/'data'/split
        source_rows=read(out.parent/dataset/'scene_rows.json')[split]
        wanted=[i for i,r in enumerate(source_rows) if r['placement'] in ('pass','clear')]
        original=[rows[i]['source_scene_index'] for i in ids]
        assert original==wanted
        for n,i in zip(ids,original):
            value={k:v for k,v in rows[n].items() if k not in ('source_dataset','source_split','source_scene_index')}
            assert value==source_rows[i]
        selected=original[::max(1,len(original)//16)]
        saved_indices=ids[::max(1,len(original)//16)]
        with np.load(folder/'scores.npz') as a:
            np.testing.assert_array_equal(archive['old5'][ids],2*((A.smooth(a['m3_raw'][original])>=.9404184587540165)|(A.smooth(a['local_raw'][original])>=4.625390338985158)))
            raw=a['ordinary_raw'][:,selected]
            current=a['current'][selected].astype(float);valid=a['current_valid'][selected]
        features=frozen_features(raw,current,valid)
        predictions=predict_sample(features)
        np.testing.assert_array_equal(predictions,archive['light_scores'][1:4,saved_indices].astype(np.float32))
        sample_predictions+=predictions.size
        readout=folder/'frozen_readout.npz'
        if not readout.exists():readout=out.parent/dataset/('cal_scores_readout.npz' if split=='cal' else 'hold_grades_notifications.npz')
        with np.load(readout) as a:joint=a['joint_scores'][:,selected]
        om=A.smooth(raw)-np.array([cuts[str(s)]['single'] for s in (2026100955,2026100956,2026100957)])[:,None,None,None,None]
        jm=joint-np.array([jointcuts[f'{s}/score_current/c15_p64']['theta'] for s in (2026100955,2026100956,2026100957)])[:,None,None,None,None]
        margin=np.maximum(om.mean(0),jm.mean(0))
        np.testing.assert_allclose(archive['light_scores'][0,saved_indices],np.where(margin>=0,margin,-np.inf),rtol=0,atol=1e-12)
        if (folder/'task_scores.npz').exists():
            with np.load(folder/'task_scores.npz') as a:
                for ai,arm in enumerate(ARMS[1:],1):
                    np.testing.assert_array_equal(archive['light_scores'][ai,ids],a['light_scores'][list(a['arm_keys']).index(arm),original])
        counts.append(dict(dataset=dataset,split=split,negative_scenes=len(ids)))
    seal=read(out/'sealed_calibration.json')
    curves=list(csv.DictReader((out/'pool_calibration_curve.csv').open(encoding='utf-8-sig',newline='')))
    with np.load(out/'pool_grades_notifications.npz') as a:
        grade,notice=a['grades'],a['notifications'];np.testing.assert_array_equal(a['baseline_notifications'],A.scalar_gap1(archive['old5']))
        assert tuple(a['keys'])==ARMS
    baseline=A.scalar_gap1(archive['old5']);results={}
    for ai,arm in enumerate(ARMS):
        assert time.monotonic()<deadline
        score=archive['light_scores'][ai];fixed=archive['old5']
        taus=thresholds(score,fixed);counts,den=clip_curve(score,fixed,rows,taus)
        theirs=[r for r in curves if r['arm']==arm]
        assert len(theirs)==len(taus)
        basecount=int((fixed[I.masks(rows)['far_pass_plus_clear']]>0).any((-2,-1)).sum())
        cap=basecount+.03*den;chosen=int(np.flatnonzero(counts<=cap)[0])
        for index,(tau,count,r) in enumerate(zip(taus,counts,theirs)):
            assert threshold(r)==tau
            assert int(r['far_pass_plus_clear_notified_clips'])==count
            assert int(r['far_clear_clip_denominator'])==den
            assert (r['feasible_clip_rate_cap'].lower()=='true')==bool(count<=cap)
            assert all(int(r[h+'_'+f])==0 for h in A.HEIGHTS for f in ('timely','rescue','loss','net'))
        rec=seal['calibrations'][arm]
        assert threshold(rec)==taus[chosen] and rec['candidate_index']==chosen
        assert rec['contact_utility_access'] is False and rec['candidate_thresholds']==len(taus)
        assert rec['notified_clips']==counts[chosen] and rec['maximum_notified_clips']==cap
        np.testing.assert_array_equal(grade[ai],I.grade(score,fixed,taus[chosen]))
        emission=A.scalar_gap1(grade[ai]);np.testing.assert_array_equal(emission,notice[ai])
        rates=family_rates(emission,baseline,rows)
        for family in rec['families']:
            ours=rates[family['family']]
            assert family['notified_clips']==ours['clips_with_any_notification']
            assert family['old5_notified_clips']==ours['baseline_clips_with_any_notification']
            assert family['increment_pp']==ours['increment_pp']
        x=np.array([v['increment_pp'] for v in rates.values()])
        summary=rec['family_increment_pp']
        assert summary==dict(median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(x.max()))
        results[arm]=dict(ties=len(taus)-2,tau=float(taus[chosen]),notified_clips=int(counts[chosen]),clip_denominator=den)
    return dict(cohorts=len(cohorts),negative_scenes=len(rows),negative_clips=2*len(rows),families=14,
                frozen_HGB_sample_predictions=sample_predictions,calibrations=results)


def hold_scores(out):
    folder=out/'data/hold'
    with np.load(folder/'task_scores.npz') as a:values={k:a[k].copy() for k in a.files}
    assert tuple(values['arm_keys'])==ARMS
    with np.load(folder/'scores.npz') as a:
        raw=a['ordinary_raw'];current=a['current'].astype(float);valid=a['current_valid']
        np.testing.assert_array_equal(values['old5'],2*((A.smooth(a['m3_raw'])>=.9404184587540165)|(A.smooth(a['local_raw'])>=4.625390338985158)))
        np.testing.assert_array_equal(values['m3'],2*(A.smooth(a['m3_raw'])>=.8557642486787612))
    expected=frozen_features(raw,current,valid)
    np.testing.assert_array_equal(np.load(folder/'S_features.npy',mmap_mode='r'),expected)
    ids=np.arange(0,len(values['old5']),max(1,len(values['old5'])//32))
    np.testing.assert_array_equal(predict_sample(expected[:,ids]),values['light_scores'][1:4,ids].astype(np.float32))
    np.testing.assert_array_equal(values['light_scores'][4],values['light_scores'][1:4].astype(np.float32).mean(0))
    with np.load(folder/'frozen_readout.npz') as a:joint=a['joint_scores']
    cuts=read(ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json')
    jointcuts=read(ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010/calibrations.json')
    ordinary=A.smooth(raw);om,jm=[],[]
    for i,seed in enumerate((2026100955,2026100956,2026100957)):
        cut=jointcuts[f'{seed}/score_current/c15_p64']['theta']
        om.append(ordinary[i]-cuts[str(seed)]['single']);jm.append(joint[i]-cut)
        both=np.where((values['old5']>0)|(ordinary[i]>=cuts[str(seed)]['addition']),2,
                      np.where(ordinary[i]>=cuts[str(seed)]['single'],1,0)).astype(np.int8)
        both[(both==0)&np.isfinite(joint[i])&(joint[i]>=cut)]=1
        np.testing.assert_array_equal(values['both'][i],both)
    margin=np.maximum(np.array(om).mean(0),np.array(jm).mean(0))
    np.testing.assert_allclose(values['light_scores'][0],np.where(margin>=0,margin,-np.inf),rtol=0,atol=1e-12)
    for p,h in read(out/'data_hold_receipt.json')['outputs_sha256'].items():assert sha(folder/p)==h
    return values


def bootstrap_vectors(emission,baseline,category,rows):
    first,before=I.outcomes(emission),I.outcomes(baseline)
    timely=(first>=0)&(first<11);btime=(before>=0)&(before<11)
    joint,bjoint=emission.max(-1),baseline.max(-1)
    masks=I.masks(rows);near=masks['near_pass']
    vectors=[]
    for q in range(2):
        contact=category[:,q]=='contact'
        vectors.extend([((timely[:,:,q].astype(int)-btime[:,:,q])*contact[:,None]).sum(1),contact.astype(int)*emission.shape[1]])
    full=masks['far_pass_plus_clear']
    vectors.extend([(((joint>0).any(-1).astype(int)-(bjoint>0).any(-1))*full[:,None]).sum(1),full.astype(int)*emission.shape[1]])
    vectors.extend([(I.clip_quarters(emission,near[:,None])/4*masks['all_negative'][:,None]).sum(1),
                    (I.clip_quarters(baseline,near[:,None])/4*masks['all_negative'][:,None]).sum(1)])
    return np.column_stack(vectors).astype(float)


def bootstrap_recompute(values,rows,replicates=2000):
    families=np.array([r['background_family'] for r in rows])
    indices=[np.flatnonzero(families==f) for f in sorted(set(families))]
    rng=np.random.default_rng(20261010)
    output={k:np.empty((replicates,8)) for k in ('physical_scene','family','hierarchical_family_scene')}
    for b in range(replicates):
        output['physical_scene'][b]=values[rng.integers(len(rows),size=len(rows))].sum(0)
        chosen=rng.integers(len(indices),size=len(indices));clusters=[indices[j] for j in chosen]
        output['family'][b]=sum((values[idx].sum(0) for idx in clusters),start=np.zeros(8))
        output['hierarchical_family_scene'][b]=sum((values[idx[rng.integers(len(idx),size=len(idx))]].sum(0) for idx in clusters),start=np.zeros(8))
    result={}
    for kind,t in output.items():
        metric=dict(HEAD_net=t[:,0],BODY_net=t[:,2],HEAD_BODY_net=t[:,0]+t[:,2],
            HEAD_net_pp=100*t[:,0]/t[:,1],BODY_net_pp=100*t[:,2]/t[:,3],
            HEAD_BODY_net_pp=100*(t[:,0]+t[:,2])/(t[:,1]+t[:,3]),
            far_clear_increment_pp=100*t[:,4]/t[:,5],weighted_cost=t[:,6],weighted_cost_increment=t[:,6]-t[:,7])
        result[kind]={k:dict(lower=float(np.quantile(v,.025)),upper=float(np.quantile(v,.975))) for k,v in metric.items()}
    return result


def hold(out,rows,deadline):
    values=hold_scores(out);category=values['category'];fixed=values['old5']
    seal=read(out/'sealed_calibration.json');metrics=read(out/'metrics.json')
    with np.load(out/'hold_grades_notifications.npz') as a:
        grades=dict(zip(a['keys'].tolist(),a['grades']));saved=dict(zip(a['keys'].tolist(),a['notifications']))
        np.testing.assert_array_equal(a['category'],category)
    np.testing.assert_array_equal(grades['fixed/old5'],fixed)
    np.testing.assert_array_equal(grades['fixed/m3'],values['m3'])
    np.testing.assert_array_equal(grades['both955'],values['both'][0])
    emission={key:A.scalar_gap1(g) for key,g in grades.items()}
    bootstrap=read(out/'bootstrap_intervals.json');family_result={}
    for key,notice in emission.items():
        assert time.monotonic()<deadline
        np.testing.assert_array_equal(notice,saved[key])
        if key.endswith('/selected'):
            arm=key.split('/')[0];ai=ARMS.index(arm)
            np.testing.assert_array_equal(grades[key],I.grade(values['light_scores'][ai],fixed,threshold(seal['calibrations'][arm])))
            np.testing.assert_array_equal(grades[key]==2,fixed==2)
        independent=dict(costs=I.costs(notice,rows),contacts={
            ref:{group:A.paired(notice,emission[base],category,rows,group) for group in ('all','dark_thin','sign_edge')}
            for ref,base in (('old5','fixed/old5'),('both','both955'))})
        report=metrics['reports']['hold/'+key];I.compare_metric(independent,report)
        rates=family_rates(notice,emission['fixed/old5'],rows)
        for r in report['families']:
            ours=rates[r['family']]
            assert r['notified_clips']==ours['clips_with_any_notification'] and r['old5_notified_clips']==ours['baseline_clips_with_any_notification']
            assert r['increment_pp']==ours['increment_pp']
        fc=independent['costs']['far_pass_plus_clear'];bfc=I.costs(emission['fixed/old5'],rows)['far_pass_plus_clear']
        assert report['far_clear_increment_pp']==100*(fc['clips_with_any_notification']-bfc['clips_with_any_notification'])/fc['clip_denominator']
        family_result[key]=rates
        if key in bootstrap:
            record=bootstrap[key];vectors=bootstrap_vectors(notice,emission['fixed/old5'],category,rows)
            np.testing.assert_array_equal(vectors,record['scene_vectors'])
            assert record['replicates']==2000 and record['seed']==20261010 and record['family_count']==4
            if key=='Sensemble/selected':assert record['intervals']==bootstrap_recompute(vectors,rows)
    ledger=I.verify_ledgers(out,{'hold':dict(emissions=emission,category=category)},{'hold':rows})
    assert metrics['contract']['hold_selection'] is False and metrics['contract']['binary_pass_fail_policy'] is False
    curve_rows=list(csv.DictReader((out/'hold_tradeoff_curve.csv').open(encoding='utf-8-sig',newline='')))
    curve_result={}
    for arm in ('Sensemble','E','both955'):
        assert time.monotonic()<deadline
        if arm=='both955':
            score=np.where((values['both'][0]>0)&(fixed==0),values['margin'][0],-np.inf)
            taus=thresholds(score,fixed);taus[0]=0.
        else:score=values['light_scores'][ARMS.index(arm)];taus=thresholds(score,fixed)
        theirs=[r for r in curve_rows if r['arm']==arm];assert len(theirs)==len(taus)
        full,den=clip_curve(score,fixed,rows,taus)
        groups={g:contact_curve(score,fixed,category,rows,taus,g) for g in ('all','dark_thin','sign_edge')}
        for index,(r,tau) in enumerate(zip(theirs,taus)):
            assert threshold(r)==tau and int(r['far_pass_plus_clear_notified_clips'])==full[index]
            for group,heights in groups.items():
                for h,counts in heights.items():
                    name=h+'_net' if group=='all' else h+'_'+group+'_net'
                    assert int(r[name])==counts['net'][index]
            if index in np.unique(np.linspace(0,len(taus)-1,9,dtype=int)):
                g=I.grade(score,fixed,tau)
                if arm=='both955':g=np.where(fixed==2,2,np.where(np.isfinite(score)&(score>=tau),values['both'][0],0)).astype(np.int8)
                e=A.scalar_gap1(g);cost=I.costs(e,rows)
                assert float(r['weighted_cost'])==cost['all_negative']['weighted_cost']
                assert int(r['far_pass_plus_clear_notifications'])==cost['far_pass_plus_clear']['notifications']
        curve_result[arm]=dict(all_ties=len(taus)-2,representative_scalar_costs=9)
    return dict(ledger_counts=ledger,curve_verification=curve_result,bootstrap='All arms independent physical-scene vectors; S ensemble all 2000 scene/family/hierarchical interval replicates exact',families=family_result)


def run(out):
    began=time.monotonic();prior=sum(read(p)['seconds'] for p in out.glob('independent_audit_failure_*.json'))
    assert prior<150 and not (out/'independent_audit.json').exists()
    deadline=began+150-prior
    identity=identities(out);rows=read(out/'scene_rows.json')['hold']
    source_result=source(out,rows)
    print('AUDIT_SOURCE',round(time.monotonic()-began,3),flush=True)
    pool_result=pooled(out,deadline)
    print('AUDIT_POOL',round(time.monotonic()-began,3),flush=True)
    hold_result=hold(out,rows,deadline)
    for p,h in read(out/'evaluation_receipt.json')['outputs_sha256'].items():assert sha(out/p)==h
    elapsed=time.monotonic()-began;assert elapsed+prior<150
    result=dict(status='PASS',seconds=elapsed,audit_command_wall_seconds=elapsed+prior,source_sha256=sha(__file__),
        identities=identity,source=source_result,pool=pool_result,hold=hold_result,
        checks=['All frozen model/feature source byte hashes; freshhold seal chronology',
                'All 6 consumed negative cohorts only, 3584 unique physical scenes and 14 families',
                'All pooled score ties clip-any counts, exact lowest feasible threshold, family distribution',
                'Independent frozen 47 features and deterministic six-HGB sample predictions',
                'Independent selected-arm scalar gap1, contact/cost/family metrics, complete ledgers',
                'Every hold curve clip/timely/subgroup tie, representative scalar costs',
                'Physical scene vectors for all arms; exact S ensemble 2000 paired bootstrap replicates'],
        rendering=0,ToF_inference=0,CPU_frozen_HGB_sample_inference=True,training=0,protected_access=0)
    save_new(out/'independent_audit.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010')
    args=p.parse_args();started=time.monotonic()
    try:run(args.out)
    except BaseException as error:
        args.out.mkdir(parents=True,exist_ok=True)
        save_new(args.out/f'independent_audit_failure_{time.time_ns()}.json',dict(status='FAILED',seconds=time.monotonic()-started,error=repr(error),source_sha256=sha(__file__)))
        raise
