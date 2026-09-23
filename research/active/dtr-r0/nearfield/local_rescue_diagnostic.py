"""Fixed causal public rescue controls on consumed sources; no fitting."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from query_occupancy_data import read, write, sha
from inherit_spatial_model import canonical_tof, QUERIES, FOCAL
from ba_camera_corridor_metrics import evaluate_rows
from local_transfer_metrics import comparison

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = REPO/'artifacts.local/evidence/ba-local-rescue-20260923'
COHORTS = {'transfer':'ba-local-transfer-20260922', 'stability':'ba-local-stability-20260923'}
GATES = ('two_frames','same_query','nominal_range','upper_range','range_same_query')
ARMS = ('A_current','local',*GATES)


def range_evidence(tof, query):
    z, valid, boxes, low, high = canonical_tof(tof)
    yl = (boxes[:,0]*180-90)/FOCAL*z
    yh = (boxes[:,2]*180-90)/FOCAL*z
    xl = (boxes[:,1]*320-160)/FOCAL*z
    xh = (boxes[:,3]*320-160)/FOCAL*z
    overlap = valid & (xh >= query[0]) & (xl <= query[1]) & (yh >= query[2]) & (yl <= query[3])
    nominal = overlap & (z >= .3) & (z <= 3.)
    upper = nominal & (high <= 3.)
    return dict(nominal_count=int(nominal.sum()), upper_count=int(upper.sum()),
        overlapping_count=int(overlap.sum()),
        minimum_range=float(z[overlap].min()) if overlap.any() else 8.,
        minimum_high=float(high[overlap].min()) if overlap.any() else 8.)


def causal_controls(prob, baseline, tof, features, sequence, cutoff):
    """sequence contains only clip reset and order, never evaluator metadata."""
    flags, records = [], []
    previous = None
    for i, (clip, frame) in enumerate(sequence):
        if frame == 0:
            previous = None
        elif previous is None or previous['clip'] != clip or previous['frame']+1 != frame:
            raise ValueError('Noncontiguous causal stream')
        q = (1,4)[int(np.argmax(prob[i,[1,4]]))]
        p = float(prob[i,q]); eligible = p >= cutoff
        geom = range_evidence(tof[i], QUERIES[q])
        prior_any = previous is not None and previous['eligible']
        prior_same = previous is not None and previous['prob'][q] >= cutoff
        admits = dict(two_frames=prior_any, same_query=prior_same,
            nominal_range=geom['nominal_count']>0,upper_range=geom['upper_count']>0,
            range_same_query=geom['nominal_count']>0 and prior_same)
        a = bool(baseline[i]['alert'])
        flags.append(dict(A_current=a,local=a or eligible,
            **{k:bool(a or (eligible and v)) for k,v in admits.items()}))
        f = features[i,q]
        records.append(dict(winning_query=q,local_score=p,A_score=float(baseline[i]['score']),
            definite_fraction=float(f[924]),possible_fraction=float(f[939]),
            possible_range_mean=float(f[937]*8),
            rgb_definite_contrast=float(np.linalg.norm(f[955:958])),
            rgb_possible_contrast=float(np.linalg.norm(f[958:961])),
            prior_local=float(prior_any),prior_same=float(prior_same),
            score_delta=p-float(previous['prob'][q]) if previous is not None else 0.,
            **geom))
        previous = dict(clip=clip,frame=frame,prob=prob[i].copy(),eligible=eligible)
    return flags,records


def prepare():
    plan = ROOT/'plan'
    files = [Path(__file__),HERE/'LOCAL_RESCUE_PROTOCOL_20260923.md',
        HERE/'inherit_spatial_model.py',HERE/'ba_camera_corridor_metrics.py',
        HERE/'local_transfer_metrics.py',HERE/'query_occupancy_data.py']
    write(plan/'seal.json',dict(files={p.relative_to(REPO).as_posix():sha(p) for p in files}))
    inputs = [dict(alias='plan',path=str(plan),role='configuration',purpose='fixed-diagnostic-definition')]
    for name,stem in COHORTS.items():
        root=REPO/'artifacts.local/evidence'/stem
        for alias,path,role in [('pred',root.with_name(stem+'-predictions'),'evaluator'),
            ('tof',root.with_name(stem+'-prepared')/'observations/tof.npy','observation'),
            ('labels',root.with_name(stem+'-prepared')/'labels/evaluation.npz','evaluator'),
            ('manifest',root.with_name(stem+'-prepared')/'materialization.json','configuration')]:
            inputs.append(dict(alias=name+'_'+alias,path=str(path),role=role,purpose='consumed-causal-rescue-diagnostic'))
    write(ROOT/'run-spec.json',dict(schema='blindassist-asset-run-v1',id='local-rescue-20260923-v1',
        route='ue-local-rescue',question='Can public causal gates distinguish LOCAL-only rescues and false alerts?',
        evaluator=str(Path(__file__).relative_to(REPO)).replace('\\','/'),
        evidence_boundary='Consumed same-generator Development; no fit or fresh claim',
        reuse=dict(mode='diagnostic',query='LOCAL transfer stability causal rescue false alert public evidence'),
        inputs=inputs,outputs=[dict(alias='result',path=str(ROOT/'run/result.json'),role='result',required=True)],
        result_output='result',command=[sys.executable,str(Path(__file__)),'execute','--result','{{output:result}}']))
    print(ROOT/'run-spec.json')


def execute(result):
    start=time.perf_counter()
    assert read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])['state']=='running'
    seal=read(ROOT/'plan/seal.json')
    for name,h in seal['files'].items():assert sha(REPO/name)==h,name
    out=result.parent
    sys.path.insert(0,str(REPO))
    from tools.research_backend import BackendCandidate,DeviceObservation,select_backend
    select_backend('scalar-scoring',cpu=BackendCandidate('numpy-statistics','cpu',lambda:np.sum([1,2]),
        lambda _:DeviceObservation('cpu',platform.processor(),'numpy '+np.__version__)),
        cpu_reason='TASK_NOT_GPU_SUITABLE',record_path=out/'backend.json')
    data={}; hashes={}
    # All candidate decisions across both cohorts are saved before opening labels.
    for name,stem in COHORTS.items():
        base=REPO/'artifacts.local/evidence'/stem
        pred=base.with_name(stem+'-predictions'); prep=base.with_name(stem+'-prepared')
        ps=read(pred/'prediction-seal.json')
        assert ps['status']=='PASS' and ps['evaluation_labels_opened'] is False
        for f,h in ps['hashes'].items():assert sha(pred/f)==h,f
        man=read(prep/'materialization.json')
        assert sha(prep/'observations/tof.npy')==man['hashes']['observations/tof.npy']
        ids=read(pred/'identities.json'); baseline=read(pred/'baseline.json')
        prob=np.load(pred/'probabilities.npz')['local']; feat=np.load(pred/'features.npz')['local']
        tof=np.load(prep/'observations/tof.npy')
        flags,features=causal_controls(prob,baseline,tof,feat,
            [(r['clip_id'],r['frame_in_clip']) for r in ids],ps['thresholds']['local'])
        write(out/(name+'-public-decisions.json'),dict(flags=flags,features=features))
        hashes[name]=dict(prediction_seal=sha(pred/'prediction-seal.json'),
            tof=sha(prep/'observations/tof.npy'),labels=man['hashes']['labels/evaluation.npz'])
        data[name]=(prep,ids,baseline,flags,features)
    write(out/'candidate-seal.json',dict(inputs=hashes,
        files={name+'-public-decisions.json':sha(out/(name+'-public-decisions.json')) for name in COHORTS},
        labels_opened=False,fits=0,cutoff_selections=0))
    reports={}
    for name,(prep,ids,baseline,flags,features) in data.items():
        assert sha(prep/'labels/evaluation.npz')==hashes[name]['labels']
        lab=np.load(prep/'labels/evaluation.npz'); assert np.array_equal(lab['indices'],np.arange(len(ids)))
        rows=[]
        for i,m in enumerate(ids):
            truth=bool((lab['classes'][i,[1,4]]<6).any()) if lab['valid'][i,[1,4]].all() else None
            rows.append(dict(id=m['id'],clip_id=m['clip_id'],frame_in_clip=m['frame_in_clip'],time_s=m['time_s'],
                base_group_id=m['base_group_id'],truth=truth,boundary=m['layout_relation']=='BOUNDARY',
                predictions={arm:dict(alert=flags[i][arm],unknown=bool(baseline[i]['unknown']),
                    ambiguous=bool(flags[i][arm] and baseline[i]['unknown'])) for arm in ARMS}))
        metrics=evaluate_rows(rows,arms=ARMS)
        comparisons={arm:comparison(rows,metrics,arm,'local') for arm in GATES}
        inc=[i for i,f in enumerate(flags) if f['local'] and not f['A_current'] and rows[i]['truth'] is not None]
        positive=[i for i in inc if rows[i]['truth']];negative=[i for i in inc if not rows[i]['truth']]
        distributions={}
        for key in features[0]:
            if key=='winning_query':continue
            y=[rows[i]['truth'] for i in inc];x=[features[i][key] for i in inc]
            distributions[key]=dict(auc=float(roc_auc_score(y,x)) if positive and negative else None,
                true_quantiles=np.quantile([features[i][key] for i in positive],[0,.5,1]).tolist() if positive else [],
                false_quantiles=np.quantile([features[i][key] for i in negative],[0,.5,1]).tolist() if negative else [])
        feasibility={}
        for arm in GATES:
            kept=sum(flags[i][arm] for i in positive); rejected=sum(not flags[i][arm] for i in negative)
            c=comparisons[arm]; delays=[e for e in c['event_differences'] if e['delay_s'] is not None and e['delay_s']>1e-9]
            lost=[e for e in c['event_differences'] if e['lost']]
            feasibility[arm]=dict(kept_incremental_TP=kept,total_incremental_TP=len(positive),
                removed_incremental_FP=rejected,total_incremental_FP=len(negative),
                lost_events=len(lost),delayed_events=len(delays),
                pass_gate=bool(kept>=.8*len(positive) and rejected>=.5*len(negative) and not lost and not delays))
        # All-frame and per-event coverage, including entirely missed events.
        coverage={arm:[dict(clip_id=e['clip_id'],start_frame=e['start_frame'],
            fraction=sum(r['predictions'][arm]['alert'] for r in rows if r['clip_id']==e['clip_id'] and
                e['start_frame']<=r['frame_in_clip']<=e['end_frame'])/(e['end_frame']-e['start_frame']+1))
            for e in metrics['arms'][arm]['events']] for arm in ARMS}
        write(out/(name+'-rows.json'),rows)
        write(out/(name+'-incremental.json'),[dict(**rows[i],features=features[i]) for i in inc])
        reports[name]=dict(metrics=metrics,comparisons=comparisons,feasibility=feasibility,
            distributions=distributions,coverage=coverage)
        write(out/(name+'-report.json'),reports[name])
    passed=[arm for arm in GATES if all(reports[c]['feasibility'][arm]['pass_gate'] for c in COHORTS)]
    write(result,dict(status='PASS',decision='FIXED_GATE_FEASIBLE' if passed else 'NO_FIXED_GATE_JOINT_GAIN',
        passed=passed,feasibility={c:reports[c]['feasibility'] for c in COHORTS},
        elapsed_s=time.perf_counter()-start,fits=0,cutoff_selections=0,frames=1152,
        scope='CONSUMED_DEVELOPMENT',source_seal_sha256=sha(ROOT/'plan/seal.json')))
    for name,h in seal['files'].items():assert sha(REPO/name)==h,name
    write(out/'output-seal.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(read(result)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute']);p.add_argument('--result',type=Path)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    else:execute(a.result)
