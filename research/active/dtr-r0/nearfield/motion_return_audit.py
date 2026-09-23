"""Independent saved-array motion audit; no production AUC/aggregation imports."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[4]
ROOT=REPO/'artifacts.local/evidence/ba-motion-return-audit-20260923'
SOURCE=ROOT.parent/'ba-motion-return-20260923-run'
COHORTS={'stability':'ba-local-stability-20260923','rescue':'ba-local-rescue-fresh-20260923'}
SCORES=('residual','static_residual','wrong_residual')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)

def pair_auc(labels, values):
    """Probability positive residual beats negative residual; ties count half."""
    y=np.asarray(labels,bool);v=np.asarray(values,float)
    positive=v[y];negative=v[~y]
    if not len(positive) or not len(negative):return None
    comparisons=positive[:,None]-negative[None,:]
    return float((np.sum(comparisons<0)+.5*np.sum(comparisons==0))/comparisons.size)

def summarize(rows):
    result={}
    for label in ('contributor','corridor_target'):
        usable=[r for r in rows if r[label]['residual'] is not None]
        result[label]=dict(evaluable_zones=len(usable),total_zones=len(rows),
            geometry_groups=len(set(r['group'] for r in usable)),
            auc={k:sum(r[label][k] for r in usable)/len(usable) if usable else None for k in SCORES},
            common_points=sum(r['common_points'] for r in usable))
    return result

def prepare():
    write(ROOT/'plan/seal-v3.json',dict(script_sha256=sha(Path(__file__))))
    inputs=[dict(alias='plan',path=str(ROOT/'plan'),role='configuration',purpose='independent-motion-audit'),
            dict(alias='source',path=str(SOURCE),role='evaluator',purpose='sealed-arrays-labels-rows')]
    for c,stem in COHORTS.items():
        inputs.append(dict(alias=c+'_spec',path=str(ROOT.parent/stem/'plan/spec.json'),role='configuration',purpose='recount-strata-reset'))
        inputs.append(dict(alias=c+'_obs',path=str(ROOT.parent/(stem+'-prepared')/'observations'),role='observation',purpose='saved-input-hash-and-range-check'))
    write(ROOT/'run-spec-v3.json',dict(schema='blindassist-asset-run-v1',id='motion-return-audit-20260923-v3',
        route='ue-motion-return',question='Do independent arrays ranks and missing denominators reproduce motion attribution?',
        evaluator='research/active/dtr-r0/nearfield/motion_return_audit.py',
        evidence_boundary='Consumed saved-array audit; no independent LK or native lineage generation',
        reuse=dict(mode='diagnostic',query='motion return attribution audit'),inputs=inputs,
        outputs=[dict(alias='result',path=str(ROOT.with_name(ROOT.name+'-run-v3')/'result.json'),role='result',required=True)],
        result_output='result',command=[sys.executable,str(Path(__file__)),'execute','--result','{{output:result}}']))

def execute(result):
    assert read(Path(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL']))['state']=='running'
    assert sha(Path(__file__))==read(ROOT/'plan/seal-v3.json')['script_sha256']
    assertions=0
    def check(ok):
        nonlocal assertions
        assert ok
        assertions+=1
    def equal(a,b):
        if isinstance(a,dict):
            check(set(a)==set(b))
            for k in a:equal(a[k],b[k])
        elif isinstance(a,float):check(abs(a-b)<1e-12)
        else:check(a==b)
    check(pair_auc([True,False],[0.,1.])==1.)
    check(pair_auc([True,False],[1.,0.])==0.)
    check(pair_auc([True,False],[1.,1.])==.5)
    check(pair_auc([True,True],[0.,1.]) is None)
    for f,h in read(SOURCE/'output-seal.json')['files'].items():check(sha(SOURCE/f)==h)
    source_seal=read(ROOT.parent/'ba-motion-return-20260923/plan/seal.json')
    for f,h in source_seal['files'].items():check(sha(REPO/f)==h)
    runner=(REPO/'research/active/dtr-r0/nearfield/motion_return_diagnostic.py').read_text()
    check('extract(rgb[i],rgb[i-1],tof[i],tof[i-1])' in runner)
    check('extract(rgb[i+1]' not in runner)
    exported=read(SOURCE/'zone-rows.json');global_result=read(SOURCE/'result.json');outcomes={};same_pose={}
    for cohort,stem in COHORTS.items():
        spec=read(ROOT.parent/stem/'plan/spec.json')['cases'];obs=ROOT.parent/(stem+'-prepared')/'observations'
        seal=read(SOURCE/(cohort+'-public-seal.json'))
        check(seal['evaluator_opened'] is False)
        check(sha(SOURCE/(cohort+'-public.npz'))==seal['sha256'])
        for f in ('rgb','tof'):check(sha(obs/(f+'.npy'))==seal[f+'_sha256'])
        with np.load(SOURCE/(cohort+'-public.npz')) as archive:
            a={k:archive[k] for k in archive.files}
        with np.load(SOURCE/(cohort+'-labels.npz')) as archive:
            lab={k:archive[k] for k in archive.files}
        tof=np.load(obs/'tof.npy');report=read(SOURCE/(cohort+'-report.json'))
        rows=[r for r in exported if r['cohort']==cohort]
        check(len(rows)==576*64);check(a['residual'].shape==(576,3072))
        rebuilt=[];counts=Counter();strata=defaultdict(Counter)
        for i,case in enumerate(spec):
            candidate=a['candidate'][i];matched=a['matched'][i];wrong=a['wrong_matched'][i]
            common=matched & wrong
            for s in SCORES:common &= np.isfinite(a[s][i])
            y=lab['contributor'][i];relevant=lab['corridor_target'][i]
            check(not np.any(relevant & ~y));check(not np.any(y & ~a['current_return_valid'][i]))
            check(np.array_equal(a['current_zone'][i],a['current_zone'][0]))
            check(np.array_equal(a['low_flat_index'][i],a['low_flat_index'][0]))
            check(np.array_equal(a['current_xy'][i],a['current_xy'][0]))
            check(not np.any(a['tracked'][i] & ~candidate))
            check(np.array_equal(matched,a['tracked'][i]&a['current_return_valid'][i]&a['previous_return_valid'][i]))
            if case['frame_in_clip']==0:
                for k in ('tracking_raw','tracked','matched','wrong_matched','previous_return_valid'):check(not a[k][i].any())
                for k in (*SCORES,'fb_error','previous_xy'):check(np.isnan(a[k][i]).all())
                check((a['previous_zone'][i]==-1).all());check((a['missing_reason'][i]==8).all())
            else:
                check(spec[i-1]['clip_id']==case['clip_id'])
                q=a['current_xy'][i].astype(float);p=a['previous_xy'][i].astype(float)
                cz=a['current_zone'][i];pz=a['previous_zone'][i];wz=a['wrong_previous_zone'][i]
                check(np.array_equal(wz,np.where(pz>=0,pz//8*8+(pz%8+4)%8,-1)))
                current=tof[i,np.maximum(cz,0),0].astype(float)*8
                for score,prior,points,available in (
                    ('residual',pz,p,matched),('static_residual',pz,q,matched),('wrong_residual',wz,p,wrong)):
                    check(np.isnan(a[score][i][~available]).all())
                    old=tof[i-1,np.maximum(prior,0),0].astype(float)*8
                    take=np.flatnonzero(available)
                    prediction=np.array([159.5,89.5])+(points[take]-[159.5,89.5])*(old[take]/current[take])[:,None]
                    check(np.allclose(a[score][i][take],np.linalg.norm(q[take]-prediction,axis=1),atol=1e-12))
            c=Counter(points=int(candidate.sum()),current_valid=int(a['current_return_valid'][i].sum()),
                tracked=int((a['tracked'][i]&candidate).sum()),matched=int(matched.sum()),common=int(common.sum()),
                contributors=int(y.sum()),matched_contributors=int((y&matched).sum()),common_contributors=int((y&common).sum()),
                relevant_contributors=int(relevant.sum()),matched_relevant=int((relevant&matched).sum()),
                frames=1,first_frames=int(case['frame_in_clip']==0))
            c.update({'reason_'+str(int(k)):int(n) for k,n in zip(*np.unique(a['missing_reason'][i][candidate],return_counts=True))})
            counts.update(c)
            for axis in ('shape','layout_relation','phase','base_group_id'):strata[axis+'/'+str(case[axis])].update(c)
            for zone in range(64):
                mask=common&(a['current_zone'][i]==zone);row=rows[i*64+zone]
                check((row['frame'],row['zone'])==(i,zone));check(row['common_points']==int(mask.sum()))
                new=dict(row)
                for name,truth in (('contributor',y),('corridor_target',relevant)):
                    new[name]={s:pair_auc(truth[mask],a[s][i][mask]) for s in SCORES}
                    equal(new[name],row[name])
                rebuilt.append(new)
        equal(dict(counts),report['coverage']);equal({k:dict(v) for k,v in strata.items()},report['strata_coverage'])
        total=summarize(rebuilt);equal(total,report['summary'])
        stationary=[i for i,c in enumerate(spec) if c['frame_in_clip'] and
                    c['camera']==spec[i-1]['camera'] and c['objects']==spec[i-1]['objects']]
        check(len(stationary)==96)
        same_pose[cohort]=dict(frames=len(stationary),summary=summarize([r for r in rebuilt if r['frame'] in stationary]),
            note='Exact declared camera and objects equal; source dwell phase also includes first-arrival frame4')
        for axis in ('phase','shape','relation','group'):
            for value in report['strata_auc'][axis]:equal(summarize([r for r in rebuilt if r[axis]==value]),report['strata_auc'][axis][value])
        coverage=counts['matched_contributors']/counts['contributors'];equal(coverage,report['matched_contributor_fraction'])
        primary=total['contributor'];scores=primary['auc']
        passed=scores['residual'] is not None and scores['residual']>=.7 and scores['residual']-scores['static_residual']>=.05 and scores['residual']-scores['wrong_residual']>=.05 and coverage>=.5 and primary['geometry_groups']>=4
        equal(passed,report['pass_gate']);equal(global_result['cohorts'][cohort],{k:report[k] for k in ('summary','coverage','matched_contributor_fraction','pass_gate')})
        outcomes[cohort]=passed
    equal(global_result['decision'],'MOTION_RETURN_ATTRIBUTION_COMPONENT' if all(outcomes.values()) else 'MOTION_RETURN_ATTRIBUTION_NOT_ESTABLISHED')
    write(result.parent/'same-pose.json',same_pose)
    write(result,dict(status='PASS',assertions=assertions,frames=1152,zone_rows=len(exported),pass_gate=outcomes,
        limitation='No independent LK tracking or native contributor lineage regeneration; inspected current/previous-only call and replayed saved-array math'))
    print(read(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute']);p.add_argument('--result',type=Path)
    a=p.parse_args();prepare() if a.command=='prepare' else execute(a.result)
