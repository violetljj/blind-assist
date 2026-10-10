"""Independent weak-mass score/cost/timing audit; never imports its producer."""
import csv
import json
from pathlib import Path
import time

import numpy as np
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E
import audit_cnh_graded_peak_body_only_dev as A

OUT = C.ROOT/'artifacts.local/work/cnh-graded-weak-mass-dev-20261010'
PARENT = C.ROOT/'artifacts.local/work/cnh-graded-peak-head-cal-dev-20261010'


def clock(grade, end=13):
    # Min reduction is independent of producer first()/argmax implementation.
    frames = np.arange(3, 3+end)[None,None,:,None]
    first = np.where(grade[...,:end,:] > 0, frames, 99).min(-2)
    return np.where(first == 99, -1, first)


def costs(grade, category):
    joint = (grade > 0).any(-1)
    masks = dict(clear=(category == 'clear').all(-1),
        pass_=(category == 'pass').any(-1)&~(category == 'contact').any(-1))
    return {name:dict(slots=int(joint[mask].sum()),
        clips=int(joint[mask].any(-1).sum()),
        slot_denominator=int(joint[mask].size),
        clip_denominator=int(mask.sum())*grade.shape[1]) for name,mask in masks.items()}


def paired(old, new, end=11):
    a,b = clock(old,end),clock(new,end)
    both=(a>=0)&(b>=0)
    return dict(denominator=int(a.size),before=int((a>=0).sum()),after=int((b>=0).sum()),
        rescue=int(((a<0)&(b>=0)).sum()),loss=int(((a>=0)&(b<0)).sum()),
        earlier=int((both&(b<a)).sum()),later=int((both&(b>a)).sum()),same=int((both&(b==a)).sum()))


def contact_report(old, new, category, q, mask=None):
    take = category[:,q]=='contact'
    if mask is not None: take &= mask
    a,b = old[take,...,q:q+1],new[take,...,q:q+1]
    report = dict(timely=paired(a,b,11),full=paired(a,b,13))
    for name,grade in (('before',a),('after',b)):
        first=clock(grade)
        report[name+'_outcomes'] = dict(timely=int(((first>=3)&(first<=13)).sum()),
            late=int((first>=14).sum()),silent=int((first<0).sum()))
    return report


def cutoff_for_union(score, baseline, category, caps):
    """Price only new union slots and wholly new pure-pass clips.

    Order statistics retain complete score ties. Baseline-cost portions are
    constants, so maxima over new opportunities determine the minimum cutoff.
    """
    flags = baseline > 0
    eligible = (baseline == 0)&np.isfinite(score)
    joint = flags.any(-1)
    clear=(category=='clear').all(-1)
    passed=(category=='pass').any(-1)&~(category=='contact').any(-1)
    old = costs(baseline,category)
    remaining_clear = caps['clear']-old['clear']['slots']
    remaining_pass = caps['pass']-old['pass_']['clips']
    assert remaining_clear >= 0 and remaining_pass >= 0
    values = np.where(eligible,score,-np.inf).max(-1)
    cv = values[clear][~joint[clear]]
    pv = values[passed].max(-1)[~joint[passed].any(-1)]
    def bound(x,n):
        finite=np.sort(x[np.isfinite(x)])
        if len(finite)<=n: return -np.inf
        return float(np.nextafter(finite[-n-1],np.inf))
    return max(bound(cv,remaining_clear),bound(pv,remaining_pass))


def load_npz(path):
    with np.load(path,allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def read_csv(path):
    with path.open(encoding='utf8',newline='') as stream:
        return list(csv.DictReader(stream))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def flat_cost(grade,category):
    a=costs(grade,category)
    return dict(clear_slots=a['clear']['slots'],clear_clips=a['clear']['clips'],
        pass_slots=a['pass_']['slots'],pass_clips=a['pass_']['clips'],
        clear_clip_denominator=a['clear']['clip_denominator'],pass_clip_denominator=a['pass_']['clip_denominator'],
        clear_slot_denominator=a['clear']['slot_denominator'],pass_slot_denominator=a['pass_']['slot_denominator'])


def close(a,b,label):
    if isinstance(b,dict):
        assert set(a)==set(b),(label,set(a)^set(b))
        for key,value in b.items(): close(a[key],value,label+'/'+str(key))
    elif isinstance(b,(list,tuple)):
        assert len(a)==len(b),label
        for i,value in enumerate(b): close(a[i],value,label+'/'+str(i))
    elif isinstance(b,np.ndarray): np.testing.assert_array_equal(a,b,err_msg=label)
    elif isinstance(b,(float,np.floating)): assert np.isclose(a,b,rtol=1e-11,atol=1e-11),(label,a,b)
    else: assert a==b,(label,a,b)


def feature_spots(split):
    """FP64 scalar definitions on 24 vectors, independent of GPU producer."""
    schema=C.read(OUT/'features/schema.json')
    cache=load_npz(OUT/'features'/f'{split}_features.npz')
    source=C.SOURCE/'data'/split
    ambient=load_npz(source/'physics.npz')['ambient']
    bias=np.load(C.ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy').astype(np.float32)
    geom=load_npz(C.OUT/'features'/f'{split}_public_geometry.npz')
    hist=np.load(source/'hist.npy',mmap_mode='r')
    count=0; worst=0.
    for n in (0,383):
        den=np.sqrt(np.maximum(16*ambient.astype(np.float32)[...,None]+np.maximum(bias,0),1e-9))
        z=((np.asarray(hist[n,0],np.float32)-bias)/den).astype(np.float16).astype(np.float64).reshape(16,1024)
        log=np.sign(z)*np.log1p(np.abs(z))
        for j in (0,4,12):
            f=j+3; length=int(geom['length'][j]); begin=f-length+1
            for q in range(2):
                answers=[]; validity=[]
                for r in (0,2):
                    w=geom['membership'][j,q,r,8-length:].astype(np.float64)
                    mass=w.sum(-1); supported=mass>0; number=int(supported.sum())
                    safe=np.maximum(mass,1e-20)
                    signed=log[begin:f+1]*w
                    pos=np.maximum(signed,0); neg=np.maximum(-signed,0)
                    pd=pos.sum(-1)/safe; nd=neg.sum(-1)/safe
                    mean=pd[supported].mean() if number else 0.
                    peak=float(pd.max()); current=pos[-1]; total=current.sum(); squares=(current*current).sum(); bins=int((w[-1]>0).sum())
                    signed_mean=(signed/safe[:,None]).sum(0)/max(number,1)
                    bounded=(np.maximum(log[begin:f+1],0)/(1+np.maximum(log[begin:f+1],0))*w).sum(-1)/safe
                    values=[pd[-1],nd[-1],total**2/max(squares,1e-20)/max(bins,1),
                        np.sort(current)[-8:].sum()/max(total,1e-20),mean,peak,
                        np.sqrt(max((pd[supported]**2).mean()-mean**2,0)) if number else 0,
                        nd[supported].mean() if number else 0.,pd.sum()**2/max((pd*pd).sum(),1e-20),
                        pd[-1]/max(peak,1e-20),pd[-1]/max(mean,1e-20),
                        bounded[supported].mean() if number else 0.,mean-np.maximum(signed_mean,0).sum(),float(length)]
                    v=np.isfinite(values)&(number>0)
                    v[:4]&=supported[-1];v[2]&=(squares>0)&(bins>0);v[3]&=total>0
                    v[8]&=(pd*pd).sum()>0;v[9]&=(peak>0)&supported[-1];v[10]&=(mean>0)&supported[-1]
                    answers.extend(values);validity.extend(v.tolist())
                expected=np.array(answers);valid=np.array(validity)
                close(cache['valid'][n,0,j,q],valid,f'{split}/{n}/{j}/{q}/feature validity')
                actual=cache['features'][n,0,j,q]
                np.testing.assert_allclose(actual[valid],expected[valid],atol=3e-4,rtol=3e-5)
                assert np.isnan(actual[~valid]).all()
                worst=max(worst,float(np.max(np.abs(actual[valid]-expected[valid]))));count+=1
    close(cache['names'].tolist(),schema['names'],split+'/feature names')
    close(cache['frames'],np.arange(3,16),split+'/causal frames')
    assert cache['features'].dtype==np.float32 and cache['valid'].dtype==bool
    assert cache['features'].shape==(384,4,13,2,28)
    return dict(vectors=count,max_absolute_error=worst,definition_backend='independent FP64 scalar',producer_imported=False)


def run():
    began=time.monotonic(); audit=OUT/'verification'
    if (audit/'PLAN.json').exists(): raise FileExistsError('Preserve independent audit attempt')
    save(audit/'PLAN.json',dict(task='WEAK_MASS_INDEPENDENT_AUDIT',CPU_command_wall_seconds_cap=180,
        producer_imported=False,new_fit=0,new_prediction=0,
        scope='All12 score-grade cells, six independent lowest whole-tie cuts and dual union cost caps, baseline/strong exact, 360 shape/rho cohorts and full timing; FP64 public-feature spots',
        source_sha256=C.sha(Path(__file__))))
    try:
        plan=C.read(OUT/'PLAN.json')
        for path,digest in plan['inputs_sha256'].items(): close(C.sha(C.ROOT/path),digest,'input/'+path)
        close(C.sha(Path(__file__).with_name('cnh_graded_weak_mass_dev.py')),plan['source_sha256'],'producer source')
        data=C.load();thresholds=C.read(C.PARENT/'thresholds.json')
        models=C.read(OUT/'models.json');cuts=C.read(OUT/'calibrations.json')
        metrics=C.read(OUT/'metrics.json');cohorts=C.read(OUT/'cohorts.json')
        fit=np.array([r['background_id'] in (8,10) for r in data['cal']['rows']])
        close(C.read(OUT/'cal_partition.json'),dict(fit_background_ids=[8,10],calibrate_background_ids=[9,11]),'partition')
        close(plan['params'],C.read(C.ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010/PLAN.json')['params'],'fixed HGB recipe')
        full_reports={};calcheck={};featurecheck={};cells=0;cohort_count=0
        archives={};parents={}
        for split,d in data.items():
            archives[split]=dict(scores=load_npz(OUT/f'{split}_scores.npz'),grades=load_npz(OUT/f'{split}_grades.npz'))
            parent=load_npz(PARENT/f'{split}_grades.npz');parents[split]=dict(zip(parent['keys'].tolist(),parent['grades']))
            expected_keys=[f'{seed}/{arm}' for seed in G.SEEDS for arm in ('current_control','weak_mass')]
            for kind in ('scores','grades'):
                close(archives[split][kind]['keys'].tolist(),expected_keys,split+'/'+kind+'/keys')
                close(archives[split][kind]['scene_ids'],d['scene_ids'],split+'/'+kind+'/identity')
            featurecheck[split]=feature_spots(split)
        for si,seed in enumerate(G.SEEDS):
            all_grades={};all_base={}
            for split,d in data.items():
                ordinary=d['candidates'][0,si];th=thresholds[str(seed)]
                strong=(d['m3']>=E.OLD_RAISED)|(d['local']>=E.OLD_LOCAL)|(ordinary>=th['addition'])
                base=np.where(strong,2,np.where(ordinary>=th['single'],1,0)).astype(np.int8)
                close(base,parents[split][f'{seed}/baseline'],split+'/independent baseline')
                all_base[split]=base
                scores=dict(zip(archives[split]['scores']['keys'].tolist(),archives[split]['scores']['scores']))
                saved=dict(zip(archives[split]['grades']['keys'].tolist(),archives[split]['grades']['grades']))
                old_scores=load_npz(C.ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'/f'{split}_scores.npz')
                old=dict(zip(old_scores['keys'].tolist(),old_scores['scores']))
                np.testing.assert_allclose(scores[f'{seed}/current_control'],old[f'{seed}/score_current'],rtol=1e-10,atol=1e-10)
                for arm in ('current_control','weak_mass'):
                    if time.monotonic()-began>=180: raise TimeoutError('Independent audit180 CPU-command-wall seconds cap')
                    key=f'{seed}/{arm}';score=scores[key];cut=cuts[key]
                    theta=-np.inf if cut['nonbinding'] else cut['theta']
                    grade=base.copy();grade[(base==0)&np.isfinite(score)&(score>=theta)]=1
                    close(saved[key],grade,split+'/'+key+'/score to grade')
                    close(grade==2,strong,split+'/'+key+'/strong exact')
                    close(grade[base>0],base[base>0],split+'/'+key+'/base retained')
                    all_grades[(split,arm)]=grade
                    head=parents[split][f'{seed}/head50'];pubkey=f'{split}/{key}'
                    expected=A.describe(grade,d,dict(prior_light=head>0,ordinary_OR=strong,M3=d['m3']>=E.M3_THETA,old_fusion=(d['m3']>=E.OLD_RAISED)|(d['local']>=E.OLD_LOCAL)))
                    for field,value in expected.items(): close(metrics[pubkey][field],value,pubkey+'/'+field)
                    close(metrics[pubkey]['cost'],flat_cost(grade,d['category']),pubkey+'/cost')
                    close(metrics[pubkey]['head50_cost'],flat_cost(head,d['category']),pubkey+'/head50 cost')
                    close(metrics[pubkey]['paired_vs_head50'],A.paired(head>0,grade>0,d['category']),pubkey+'/paired timely')
                    close(metrics[pubkey]['physical_vs_head50'],A.physical_pair(head>0,grade>0,d['category']),pubkey+'/physical timely')
                    full_reports[pubkey]={}
                    for q,height in enumerate(E.HEIGHTS):
                        for family in ('all',*sorted({r['shape_family'] for r in d['rows']})):
                            for rho in ('all',.25,.65):
                                mask=np.array([(family=='all' or r['shape_family']==family) and (rho=='all' or r['rho']==rho) for r in d['rows']])
                                report=contact_report(head,grade,d['category'],q,mask)
                                group=f'{height}/{family}/{rho}'
                                outcome=dict(events=report['timely']['denominator'],**report['after_outcomes'])
                                close(cohorts[pubkey]['outcomes'][group],outcome,pubkey+'/'+group+'/outcomes')
                                pair=report['timely'];paired_values=dict(events=pair['denominator'],**{n:pair[n] for n in ('rescue','loss','earlier','later')})
                                close(cohorts[pubkey]['paired_vs_head50'][group],paired_values,pubkey+'/'+group+'/paired')
                                full_reports[pubkey][group]=report;cohort_count+=1
                    cells+=1
            for arm in ('current_control','weak_mass'):
                key=f'{seed}/{arm}';cut=cuts[key]
                score=dict(zip(archives['cal']['scores']['keys'].tolist(),archives['cal']['scores']['scores']))[key]
                cat=data['cal']['category'][~fit];base=all_base['cal'][~fit]
                head=parents['cal'][f'{seed}/head50'][~fit]
                refcost=flat_cost(head,cat);basecost=flat_cost(base,cat)
                theta=cutoff_for_union(score[~fit],base,cat,dict(clear=refcost['clear_slots'],**{'pass':refcost['pass_clips']}))
                actual_theta=-np.inf if cut['nonbinding'] else cut['theta']
                assert theta==actual_theta,('independent minimal cut',key,theta,actual_theta)
                actualcost=flat_cost(all_grades[('cal',arm)][~fit],cat)
                assert actualcost['clear_slots']<=refcost['clear_slots'] and actualcost['pass_clips']<=refcost['pass_clips']
                close(cut['reference_cost'],refcost,key+'/reference cost')
                close(cut['fixed_base_cost'],basecost,key+'/baseline cost')
                close(cut['actual_cost'],actualcost,key+'/actual cost')
                close(cut['additional_caps'],dict(clear_slots=refcost['clear_slots']-basecost['clear_slots'],pass_clips=refcost['pass_clips']-basecost['pass_clips']),key+'/caps')
                for q,height in enumerate(E.HEIGHTS):
                    eligible=all_base['cal'][fit,...,q]==0;count=eligible.sum(-1)
                    yy=np.broadcast_to((data['cal']['category'][fit,q]=='contact')[:,None,None],eligible.shape)[eligible]
                    model=models[f'{key}/{height}']
                    close(model['rows'],int(eligible.sum()),key+'/'+height+'/fit rows')
                    close(model['contact_rows'],int(yy.sum()),key+'/'+height+'/fit labels')
                    close(model['eligible_scene_replica_groups'],int((count>0).sum()),key+'/'+height+'/group weights')
                    close(model['dimensions'],47 if arm=='current_control' else 103,key+'/'+height+'/dimensions')
                    assert model['iterations']<=100
                    close(C.sha(OUT/model['path']),model['sha256'],key+'/'+height+'/model hash')
                calcheck[key]=dict(reference_cost=refcost,actual_cost=actualcost,fixed_base_cost=basecost,minimum_cutoff_exact=True)
            for split,d in data.items():
                old,new=all_grades[(split,'current_control')],all_grades[(split,'weak_mass')]
                pubkey=f'{split}/{seed}/weak_mass'
                close(metrics[pubkey]['paired_vs_current_control'],A.paired(old>0,new>0,d['category']),pubkey+'/vs control')
                for q,height in enumerate(E.HEIGHTS):
                    for family in ('all',*sorted({r['shape_family'] for r in d['rows']})):
                        for rho in ('all',.25,.65):
                            mask=np.array([(family=='all' or r['shape_family']==family) and (rho=='all' or r['rho']==rho) for r in d['rows']])
                            report=contact_report(old,new,d['category'],q,mask)
                            pair=report['timely'];value=dict(events=pair['denominator'],**{n:pair[n] for n in ('rescue','loss','earlier','later')})
                            group=f'{height}/{family}/{rho}'
                            close(cohorts[pubkey]['paired_vs_current_control'][group],value,pubkey+'/'+group+'/vs control')
                            full_reports[pubkey][group]['versus_current_control']=report
        close(len(models),12,'fixed fit count');close(len(metrics),12,'complete metrics');close(len(cohorts),12,'complete cohorts')
        save(audit/'comparison.json',full_reports);save(audit/'calibration.json',calcheck)
        save(audit/'receipt.json',dict(status='PASS',seconds=time.monotonic()-began,cells=cells,
            models=12,cuts=6,cohorts=cohort_count,feature_checks=featurecheck,
            full_window_timing_recomputed=True,producer_imported=False,new_fit=0,new_prediction=0,
            source_sha256=C.sha(Path(__file__)),GPU_seconds=0))
        print(f'PASS cells{cells} cohorts{cohort_count} cuts6 models12 seconds{time.monotonic()-began:.3f}')
    except BaseException as error:
        save(audit/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began));raise


if __name__=='__main__': run()
