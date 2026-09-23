"""Single transfer-selected scalar gate; no stability cutoff tuning."""
import argparse
import os
from pathlib import Path
import sys
import time
from query_occupancy_data import read,write,sha
from ba_camera_corridor_metrics import evaluate_rows
from local_transfer_metrics import comparison

REPO=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
ROOT=REPO/'artifacts.local/evidence/ba-local-rescue-gate-20260923'
INPUT=REPO/'artifacts.local/evidence/ba-local-rescue-20260923-run'


def select_cutoff(positive,negative):
    if not positive or not negative or max(positive)>=min(negative):
        raise ValueError('No separated transfer interval; no cutoff search authorized')
    return (max(positive)+min(negative))/2


def apply_gate(rows,features,cutoff):
    import copy
    rows=copy.deepcopy(rows)
    for row,f in zip(rows,features,strict=True):
        a=row['predictions']['A_current'];local=row['predictions']['local']
        alert=bool(a['alert'] or (local['alert'] and f['possible_fraction']<=cutoff))
        row['predictions']['ambiguity_gate']=dict(alert=alert,unknown=a['unknown'],ambiguous=bool(alert and a['unknown']))
    return rows


def evaluate_gate(rows):
    arms=('A_current','local','two_frames','ambiguity_gate')
    metrics=evaluate_rows(rows,arms=arms)
    comp=comparison(rows,metrics,'ambiguity_gate','local')
    inc=[r for r in rows if r['predictions']['local']['alert'] and not r['predictions']['A_current']['alert']]
    pos=[r for r in inc if r['truth'] is True];neg=[r for r in inc if r['truth'] is False]
    kept=sum(r['predictions']['ambiguity_gate']['alert'] for r in pos)
    removed=sum(not r['predictions']['ambiguity_gate']['alert'] for r in neg)
    lost=sum(e['lost'] for e in comp['event_differences'])
    delays=[e['delay_s'] for e in comp['event_differences'] if e['delay_s'] is not None and e['delay_s']>1e-9]
    passed=len(pos)>=8 and len(neg)>=2 and kept>=.8*len(pos) and removed>=.5*len(neg) and lost==0 and not delays
    return dict(metrics=metrics,comparison=comp,summary=dict(incremental_TP=len(pos),retained_TP=kept,
        incremental_FP=len(neg),removed_FP=removed,lost_events=lost,delayed_events=len(delays),
        maximum_delay_s=max(delays,default=0),pass_gate=passed,
        opportunity_evaluable=len(pos)>=8 and len(neg)>=2))


def prepare():
    write(ROOT/'plan/seal.json',dict(files={p.relative_to(REPO).as_posix():sha(p) for p in
        [Path(__file__),HERE/'LOCAL_RESCUE_GATE_PROTOCOL_20260923.md',HERE/'ba_camera_corridor_metrics.py',HERE/'local_transfer_metrics.py']}))
    write(ROOT/'run-spec.json',dict(schema='blindassist-asset-run-v1',id='local-rescue-gate-20260923-v1',
        route='ue-local-rescue',question='Does one transfer-selected ambiguity fraction gate retain causal LOCAL rescues?',
        evaluator='research/active/dtr-r0/nearfield/local_rescue_gate.py',evidence_boundary='Consumed Development feature selection; no independent claim',
        reuse=dict(mode='development',query='LOCAL rescue public possible-only fraction gate'),
        inputs=[dict(alias='plan',path=str(ROOT/'plan'),role='configuration',purpose='fixed-single-rule'),
            dict(alias='diagnostic',path=str(INPUT),role='evaluator',purpose='consumed-selection-and-replay')],
        outputs=[dict(alias='result',path=str(ROOT.with_name(ROOT.name+'-run')/'result.json'),role='result',required=True)],
        result_output='result',command=[sys.executable,str(Path(__file__)),'execute','--result','{{output:result}}']))


def execute(result):
    assert read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])['state']=='running'
    seal=read(ROOT/'plan/seal.json')
    for f,h in seal['files'].items():assert sha(REPO/f)==h,f
    source_seal=read(INPUT/'output-seal.json')
    for f,h in source_seal['files'].items():assert sha(INPUT/f)==h,f
    inc=read(INPUT/'transfer-incremental.json')
    positive=[r['features']['possible_fraction'] for r in inc if r['truth'] is True]
    negative=[r['features']['possible_fraction'] for r in inc if r['truth'] is False]
    cutoff=select_cutoff(positive,negative)
    out=result.parent
    write(out/'gate.json',dict(feature='winning_query_possible_only_pixel_fraction',slot=939,operator='<=',cutoff=cutoff,
        selector='midpoint(max transfer incremental TP, min transfer incremental FP)',positive_frames=len(positive),
        negative_frames=len(negative),feature_selection_scope='Both consumed cohorts inspected',
        cutoff_selection_scope='Transfer only',source_sha256=sha(INPUT/'transfer-incremental.json')))
    reports={}
    for cohort in ('transfer','stability'):
        rows=read(INPUT/(cohort+'-rows.json'));features=read(INPUT/(cohort+'-public-decisions.json'))['features']
        rows=apply_gate(rows,features,cutoff)
        write(out/(cohort+'-rows.json'),rows)
        reports[cohort]=evaluate_gate(rows)
        write(out/(cohort+'-report.json'),reports[cohort])
    write(result,dict(status='PASS',decision='ADVANCE_ONE_FRESH_SOURCE' if all(r['summary']['pass_gate'] for r in reports.values()) else 'STOP_SCALAR_GATE_TRANSFER_FAILED',
        cutoff=cutoff,summaries={k:v['summary'] for k,v in reports.items()},fits=0,scalar_cutoff_selections=1,
        backend='CPU TASK_NOT_GPU_SUITABLE scalar selection',scope='CONSUMED_DEVELOPMENT'))
    write(out/'output-seal.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(read(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute']);p.add_argument('--result',type=Path)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    else:execute(a.result)
