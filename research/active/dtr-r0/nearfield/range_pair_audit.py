"""Independent exported-row and saved-public-input audit; no production summaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[4]
ROOT=REPO/'artifacts.local/evidence/ba-range-pair-audit-20260923'
SOURCE=ROOT.parent/'ba-range-pair-20260923-run'
COHORTS={'stability':'ba-local-stability-20260923','rescue':'ba-local-rescue-fresh-20260923'}
RULES=('global_min','corridor_min','corridor_median3')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)

def prepare():
    write(ROOT/'plan-v2/seal.json',{'script_sha256':sha(Path(__file__))})
    inputs=[dict(alias='plan',path=str(ROOT/'plan-v2'),role='configuration',purpose='independent-audit'),
            dict(alias='source',path=str(SOURCE),role='evaluator',purpose='exported-pair-results')]
    for c,stem in COHORTS.items():
        for suffix,path,role in [('spec',ROOT.parent/stem/'plan/spec.json','configuration'),
            ('obs',ROOT.parent/(stem+'-prepared')/'observations','observation')]:
            inputs.append(dict(alias=c+'_'+suffix,path=str(path),role=role,purpose='independent-saved-input-recount'))
    write(ROOT/'run-spec-v2.json',dict(schema='blindassist-asset-run-v1',id='range-pair-audit-20260923-v2',
        route='ue-range-pair',question='Do independent exported counts and public saved readouts reproduce range-pair results?',
        evaluator='research/active/dtr-r0/nearfield/range_pair_audit.py',
        evidence_boundary='Consumed controlled Development audit; no new scientific result',
        reuse=dict(mode='diagnostic',query='matched range pair independent audit'),inputs=inputs,
        outputs=[dict(alias='result',path=str(ROOT.with_name(ROOT.name+'-run')/'result.json'),role='result',required=True)],
        result_output='result',command=[sys.executable,str(Path(__file__)),'execute','--result','{{output:result}}']))

def recount(rows,rule):
    pos=[r['readouts'][rule] for r in rows if r['relation']!='OUTSIDE']
    neg=[r['readouts'][rule][k] for r in rows if r['relation']=='OUTSIDE' for k in ('near_m','far_m')]
    available=lambda r:r['near_m'] is not None and r['far_m'] is not None
    crosses=lambda r:available(r) and .3<=r['near_m']<=3 and r['far_m']>3
    witness=lambda r:r['near_target'] and r['far_target']
    n=len(pos);a=sum(available(r) for r in pos);correct=sum(crosses(r) for r in pos)
    ranked=sum(available(r) and r['near_m']<r['far_m'] for r in pos)
    w=sum(witness(r) for r in pos);alerts=sum(x is not None and .3<=x<=3 for x in neg)
    return dict(pairs=n,available_pairs=a,missing_pairs=n-a,correct_crossings=correct,joint_rate=correct/n,
        near_ranked_closer=ranked,ranking_all_pairs=ranked/n,target_witness_both=w,
        target_witness_joint=sum(witness(r) and crosses(r) for r in pos),missing_target_witness=n-w,
        nonwitness_joint=sum(not witness(r) and crosses(r) for r in pos),outside_endpoints=len(neg),
        outside_missing=sum(x is None for x in neg),outside_alerts=alerts,outside_FPR=alerts/len(neg))

def execute(result):
    assert read(Path(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL']))['state']=='running'
    assert sha(Path(__file__))==read(ROOT/'plan-v2/seal.json')['script_sha256']
    checks=0
    def check(value):
        nonlocal checks
        assert value
        checks+=1
    for f,h in read(SOURCE/'output-seal.json')['files'].items():check(sha(SOURCE/f)==h)
    rows=read(SOURCE/'pairs.json');saved=[r for r in rows if r['draw']==-1]
    check(len(rows)==192*17);check(len(saved)==192);check(sum(r['unique'] for r in saved)==144)
    check(len({r['group'] for r in saved})==16)
    descriptive={}
    for cohort,stem in COHORTS.items():
        spec=read(ROOT.parent/stem/'plan/spec.json')['cases']
        tof=np.load(ROOT.parent/(stem+'-prepared')/'observations/tof.npy')
        public=read(SOURCE/(cohort+'-public.json'));report=read(SOURCE/(cohort+'-report.json'))
        selected=[r for r in rows if r['cohort']==cohort]
        check(len(selected)==96*17)
        seen=set()
        def sig(c):
            return tuple((tuple(o['size_m']),tuple(round(o['center_m'][j]-c['camera'][k],6)
                for j,k in enumerate(('x','y','z')))) for o in c['objects'])
        for r in [r for r in selected if r['draw']==-1]:
            ni,fi=r['near'],r['far'];n,f=spec[ni],spec[fi]
            check(n['clip_id']==f['clip_id']==r['clip_id'])
            check((n['frame_in_clip'],f['frame_in_clip'])==((3,2) if r['phase']=='entry' else (9,10)))
            key=tuple(sig(spec[i]) for i in (ni,fi,ni-1,fi-1))
            check(r['unique']==(key not in seen));seen.add(key)
            check(n['objects']==f['objects'])
        check(len(seen)==72)
        for i_text,draws in public.items():
            i=int(i_text);t=tof[i].astype(float);z=t[:,0]*8;b=t[:,2:]
            valid=(t[:,1]==1)&np.isfinite(z)&(z>=.1)&(z<8)
            focal=160/np.tan(np.deg2rad(50))
            compatible=valid&((b[:,3]*320-160)*z/focal>=-.3)&((b[:,1]*320-160)*z/focal<=.3)&((b[:,2]*180-90)*z/focal>=-.2)&((b[:,0]*180-90)*z/focal<=.9)
            for rule in RULES:
                ids=sorted(np.flatnonzero(valid if rule=='global_min' else compatible),key=lambda j:(z[j],j))
                count=3 if rule=='corridor_median3' else 1
                ids=ids[:count] if len(ids)>=count else []
                expected=float(np.median(z[ids])) if ids else None
                p=draws['-1'][rule]
                check(p['selected_zone_ids']==ids);check(p['estimate_m']==expected)
                common=[j for j in ids if tof[i-1,j,1]==1]
                delta=float(np.median((tof[i,common,0]-tof[i-1,common,0])*8)) if common else None
                check(p['causal_delta_m']==delta);check(p['history_common_zones']==len(common))
        for r in selected:
            for rule in RULES:
                v=r['readouts'][rule]
                for prefix,idx in (('near',r['near']),('far',r['far'])):
                    p=public[str(idx)][str(r['draw'])][rule];support=v[prefix+'_support']
                    check(v[prefix+'_m']==p['estimate_m']);check(v[prefix+'_delta']==p['causal_delta_m'])
                    check(0<=support['target_pixels']<=support['total_pixels'])
                    check(v[prefix+'_target']==support['target_witnessed']==(support['target_pixels']>0))
                    check((support['total_pixels']>0)==bool(p['selected_zone_ids']))
        descriptive[cohort]={}
        for part in ('saved','noise'):
            use=[r for r in selected if r['unique'] and (r['draw']==-1 if part=='saved' else r['draw']>=0)]
            for rule in RULES:check(recount(use,rule)==report['summary'][part][rule])
            for axis in ('phase','shape'):
                for value in report['strata'][axis]:
                    for rule in RULES:check(recount([r for r in use if r[axis]==value],rule)==report['strata'][axis][value][part][rule])
            history={}
            for rule in RULES:
                history[rule]={}
                for phase in ('entry','exit'):
                    values=[r['readouts'][rule][k] for r in use if r['relation']!='OUTSIDE' and r['phase']==phase
                            for k in ('near_delta','far_delta')]
                    observed=[x for x in values if x is not None]
                    correct=sum(x<0 if phase=='entry' else x>0 for x in observed)
                    history[rule][phase]=dict(endpoints=len(values),available=len(observed),missing=len(values)-len(observed),
                        zero=sum(x==0 for x in observed),direction_correct=correct,
                        all_rate=correct/len(values),available_rate=correct/len(observed) if observed else None)
            descriptive[cohort][part]=dict(causal_history=history)
        def stats(xs):return dict(count=len(xs),median=float(np.median(xs)),minimum=min(xs),maximum=max(xs))
        descriptive[cohort]['rgb']=dict(pairs=stats(report['saved_rgb_pair_L1']),dwell=stats(report['dwell_rgb_L1']))
    write(result.parent/'descriptive.json',descriptive)
    write(result,dict(status='PASS',assertions=checks,raw_pairs=192,unique_pairs=144,groups=16,
        saved_public_frames=sum(len(read(SOURCE/(c+'-public.json'))) for c in COHORTS),
        scope='Independent exported counts and full saved public readout replay; native lineage not independently regenerated'))
    print(read(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute']);p.add_argument('--result',type=Path)
    a=p.parse_args();prepare() if a.command=='prepare' else execute(a.result)
