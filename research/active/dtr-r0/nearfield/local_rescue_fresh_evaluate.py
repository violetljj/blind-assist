"""Seal fixed scalar rescue predictions on new instances, then evaluate."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
from query_occupancy_data import read,write,sha
from local_rescue_diagnostic import causal_controls
from local_rescue_gate import evaluate_gate
from ba_camera_corridor_metrics import evaluate_rows

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
ROOT=REPO/'artifacts.local/evidence/ba-local-rescue-fresh-20260923'
GATE=REPO/'artifacts.local/evidence/ba-local-rescue-gate-20260923-run/gate.json'


def sibling(suffix):return ROOT.with_name(ROOT.name+'-'+suffix)


def runspec(stage):
    files=[Path(__file__),HERE/'local_rescue_diagnostic.py',HERE/'local_rescue_gate.py',
        HERE/'inherit_spatial_model.py',HERE/'ba_camera_corridor_metrics.py',HERE/'local_transfer_metrics.py']
    seal=ROOT/'gate-stage-seals'/(stage+'.json')
    write(seal,dict(files={p.relative_to(REPO).as_posix():sha(p) for p in files},gate_sha256=sha(GATE)))
    inputs=[dict(alias='stage_seal',path=str(ROOT/'gate-stage-seals'),role='configuration',purpose='fixed-gate-stage-code'),
        dict(alias='gate',path=str(GATE),role='configuration',purpose='unchanged-transfer-selected-rule'),
        dict(alias='predictions',path=str(sibling('predictions')),role='evaluator',purpose='sealed-frozen-public-model-outputs')]
    if stage=='gate':
        inputs.append(dict(alias='tof',path=str(sibling('prepared')/'observations/tof.npy'),role='observation',purpose='fixed-causal-controls'))
        inputs.append(dict(alias='manifest',path=str(sibling('prepared')/'materialization.json'),role='configuration',purpose='observation-identity'))
    else:
        inputs.extend([dict(alias='candidates',path=str(sibling('gated')),role='evaluator',purpose='sealed-candidate-decisions'),
            dict(alias='labels',path=str(sibling('prepared')/'labels'),role='evaluator',purpose='post-seal-union-truth')])
    result=sibling('gated' if stage=='gate' else 'evaluated')/'result.json'
    path=ROOT/(stage+'-rescue-run-spec.json')
    write(path,dict(schema='blindassist-asset-run-v1',id='local-rescue-fresh-'+stage+'-20260923-v1',route='ue-local-rescue-fresh',
        question='Does unchanged ambiguity admission transfer to new controlled geometries and trajectories?',
        evaluator='research/active/dtr-r0/nearfield/local_rescue_fresh_evaluate.py',
        evidence_boundary='New same-generator Development instances; feature and cutoff fixed before capture',
        reuse=dict(mode='development',query='LOCAL fresh rescue scalar ambiguity gate sampled events'),
        inputs=inputs,outputs=[dict(alias='result',path=str(result),role='result',required=True)],result_output='result',
        command=[sys.executable,str(Path(__file__)),'execute','--stage',stage,'--result','{{output:result}}']))
    print(path)


def execute(stage,result):
    assert read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])['state']=='running'
    seal=read(ROOT/'gate-stage-seals'/(stage+'.json'))
    for f,h in seal['files'].items():assert sha(REPO/f)==h,f
    assert sha(GATE)==seal['gate_sha256']
    pred=sibling('predictions');out=result.parent
    ps=read(pred/'prediction-seal.json')
    assert ps['status']=='PASS' and not ps['evaluation_labels_opened']
    for f,h in ps['hashes'].items():assert sha(pred/f)==h,f
    ids=read(pred/'identities.json');baseline=read(pred/'baseline.json')
    if stage=='gate':
        manifest=read(sibling('prepared')/'materialization.json')
        tofpath=sibling('prepared')/'observations/tof.npy'
        assert sha(tofpath)==manifest['hashes']['observations/tof.npy']
        prob=np.load(pred/'probabilities.npz')['local'];feat=np.load(pred/'features.npz')['local']
        flags,features=causal_controls(prob,baseline,np.load(tofpath),feat,
            [(r['clip_id'],r['frame_in_clip']) for r in ids],ps['thresholds']['local'])
        cutoff=read(GATE)['cutoff']
        for f,x in zip(flags,features,strict=True):
            f['ambiguity_gate']=bool(f['A_current'] or (f['local'] and x['possible_fraction']<=cutoff))
        write(out/'decisions.json',dict(flags=flags,features=features))
        write(out/'prediction-seal.json',dict(status='PASS',labels_opened=False,gate_sha256=sha(GATE),
            decisions_sha256=sha(out/'decisions.json'),expected_labels_sha256=manifest['hashes']['labels/evaluation.npz'],
            input_prediction_seal_sha256=sha(pred/'prediction-seal.json')))
        write(result,dict(status='PASS',frames=len(ids),labels_opened=False,fits=0,cutoff_selections=0))
    else:
        gated=sibling('gated');gs=read(gated/'prediction-seal.json')
        assert sha(GATE)==gs['gate_sha256'] and not gs['labels_opened']
        assert sha(gated/'decisions.json')==gs['decisions_sha256']
        assert sha(pred/'prediction-seal.json')==gs['input_prediction_seal_sha256']
        labpath=sibling('prepared')/'labels/evaluation.npz'
        assert sha(labpath)==gs['expected_labels_sha256']
        lab=np.load(labpath);assert np.array_equal(lab['indices'],np.arange(len(ids)))
        decisions=read(gated/'decisions.json');rows=[]
        for i,m in enumerate(ids):
            truth=bool((lab['classes'][i,[1,4]]<6).any()) if lab['valid'][i,[1,4]].all() else None
            rows.append(dict(**{k:m[k] for k in ('id','clip_id','frame_in_clip','time_s','base_group_id','trajectory','shape','size_level','layer','layout_relation')},
                truth=truth,boundary=m['layout_relation']=='BOUNDARY',
                features=decisions['features'][i],predictions={a:dict(alert=f,unknown=bool(baseline[i]['unknown']),
                    ambiguous=bool(f and baseline[i]['unknown'])) for a,f in decisions['flags'][i].items()}))
        report=evaluate_gate(rows);arms=('A_current','local','two_frames','ambiguity_gate')
        report['strata']={axis:{v:evaluate_rows([r for r in rows if r[axis]==v],arms=arms)
            for v in sorted({r[axis] for r in rows})} for axis in ('trajectory','shape','size_level','layer','layout_relation')}
        report['coverage']={a:[dict(clip_id=e['clip_id'],fraction=sum(r['predictions'][a]['alert'] for r in rows
            if r['clip_id']==e['clip_id'] and e['start_frame']<=r['frame_in_clip']<=e['end_frame'])/(e['end_frame']-e['start_frame']+1))
            for e in report['metrics']['arms'][a]['events']] for a in arms}
        write(out/'rows.json',rows);write(out/'report.json',report)
        admission=read(sibling('prepared')/'labels/source-admission.json')
        summary=report['summary']
        decision=('RESCUE_GATE_NEW_INSTANCE_COMPONENT' if summary['pass_gate'] else 'RESCUE_GATE_NEW_INSTANCE_NEGATIVE')
        if not admission['admissible'] or not summary['opportunity_evaluable']:decision='RESCUE_GATE_NEW_INSTANCE_NOT_EVALUABLE'
        write(result,dict(status='PASS',decision=decision,summary=summary,source_admission=admission,
            metrics={a:dict(**report['metrics']['arms'][a]['frames']['all_known'],
                events=report['metrics']['arms'][a]['detected_events'],false_segments=report['metrics']['arms'][a]['false_alert_segment_count']) for a in arms},
            fits=0,cutoff_selections=0,gate_sha256=sha(GATE),scope='NEW_SAME_GENERATOR_CONTROLLED_DEVELOPMENT'))
        print(read(result))
    write(out/'output-seal.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['runspec','execute']);p.add_argument('--stage',choices=['gate','evaluate']);p.add_argument('--result',type=Path)
    a=p.parse_args()
    if a.command=='runspec':runspec(a.stage)
    else:execute(a.stage,a.result)
