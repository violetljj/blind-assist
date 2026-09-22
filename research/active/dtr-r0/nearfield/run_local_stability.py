"""Governed one-pass frozen LOCAL composite/trajectory transfer stages."""
import argparse
from pathlib import Path
import os
import shutil
import sys

from query_occupancy_data import read,write,sha,stage_path

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ROOT = REPO/'artifacts.local/evidence/ba-local-stability-20260923'
FROZEN = REPO/'artifacts.local/evidence/ba-inherit-spatial-20260922-run'


def prepare():
    from local_stability_spec import freeze
    print(freeze(ROOT/'plan'))
    shutil.copyfile(HERE/'LOCAL_STABILITY_PROTOCOL_20260923.md',ROOT/'plan/stability-protocol.md')


def runspec(stage,attempt):
    inputs = [dict(alias='plan',path=str(ROOT/'plan'),role='configuration',purpose='fixed-composite-transfer-plan')]
    result = stage_path(ROOT,dict(materialize='prepared',predict='predictions',evaluate='evaluated').get(stage,stage))/'result.json'
    if stage == 'capture':
        inputs.append(dict(alias='scene',asset='unreal/BlindAssistStreetLab',relative_path='Content/StreetLab',
                           role='configuration',purpose='unchanged-map-and-opaque-materials'))
        result = stage_path(ROOT,'capture')/'launcher-terminal.json'
        command = [sys.executable,str(HERE/'launch_local_stability.py'),'--spec',str(ROOT/'plan/spec.json'),
                   '--protocol',str(ROOT/'plan/protocol.json'),'--output',str(result.parent)]
    elif stage == 'materialize':
        for alias,part,role in [('rgb','observations','observation'),('native','evaluator','evaluator'),
                ('receipt','receipt.json','configuration'),('launch','launch-receipt.json','configuration'),
                ('release','process-release.json','configuration')]:
            inputs.append(dict(alias=alias,path=str(stage_path(ROOT,'capture')/part),role=role,purpose='scoped-materialization'))
    elif stage == 'predict':
        for alias,part,role in [('observations','observations','observation'),('materialization','materialization.json','configuration')]:
            inputs.append(dict(alias=alias,path=str(stage_path(ROOT,'prepared')/part),role=role,purpose='public-frozen-inference'))
        for i,name in enumerate(('raw.pkl','local.pkl','selection.json','model-seal.json','freeze.json','feature-seal.json','source-snapshot')):
            inputs.append(dict(alias=f'frozen_{i}',path=str(FROZEN/name),role='configuration',purpose='unchanged-model-state'))
    else:
        inputs.append(dict(alias='materialization',path=str(stage_path(ROOT,'prepared')/'materialization.json'),
                           role='configuration',purpose='sealed-materialization-identity'))
        for alias,path,role in [('predictions',stage_path(ROOT,'predictions'),'evaluator'),
                               ('labels',stage_path(ROOT,'prepared')/'labels/evaluation.npz','evaluator')]:
            inputs.append(dict(alias=alias,path=str(path),role=role,purpose='post-prediction-seal-evaluation'))
        if stage == 'audit':
            for alias,path in [('evaluated',stage_path(ROOT,'evaluated')),('native',stage_path(ROOT,'capture')/'evaluator'),
                              ('materialized_labels',stage_path(ROOT,'prepared')/'labels')]:
                inputs.append(dict(alias=alias,path=str(path),role='evaluator',purpose='independent-geometry-and-metric-audit'))
    if stage != 'capture':
        code = [HERE/'run_local_stability.py',HERE/('local_stability_'+dict(materialize='data',predict='inference',evaluate='metrics',audit='audit')[stage]+'.py'),
                HERE/'LOCAL_STABILITY_PROTOCOL_20260923.md']
        extra = ('local_stability_spec.py','query_occupancy_data.py','inherit_spatial_model.py','ba_camera_corridor_metrics.py','local_transfer_metrics.py',
                 'local_transfer_inference.py','ba_camera_corridor.py','tof_fov45_core.py','tof_corridor_calibration.py')
        code += [HERE/n for n in extra]
        if stage == 'audit':code.append(HERE/'local_support_audit.py')
        seal = ROOT/'stage-seals'/f'{stage}-{attempt}.json'
        write(seal,dict(files={p.relative_to(REPO).as_posix():sha(p) for p in code}))
        for p in code:
            target = ROOT/'stage-seals'/f'{stage}-{attempt}-source'/p.relative_to(REPO)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        inputs.append(dict(alias='stage_seal',path=str(seal),role='configuration',purpose='immutable-stage-code'))
        command = [sys.executable,str(Path(__file__).resolve()),'execute','--stage',stage,'--attempt',attempt,
                   '--result','{{output:result}}']
    path = ROOT/(stage+'-run-spec-'+attempt+'.json')
    write(path,dict(schema='blindassist-asset-run-v1',id='local-stability-20260923-'+stage+'-'+attempt,
        route='ue-local-stability',question='Does frozen LOCAL retain earlier detections on new composite shapes sizes and posed approaches?',
        evaluator='research/active/dtr-r0/nearfield/run_local_stability.py',
        evidence_boundary='Same-generator controlled Development; no fit or continuous physical motion claim',
        reuse=dict(mode='development',query='LOCAL frozen HGB composite shape size approach dwell event timing transfer'),
        inputs=inputs,outputs=[dict(alias='result',path=str(result),role='result',required=True)],
        result_output='result',command=command,parameters=dict(frames=576,fit=False,threshold_selection=False,stage=stage)))
    print(path)


def execute(stage,attempt,result):
    journal = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    assert journal and read(journal)['state'] == 'running', 'Use governed research-ue entry'
    seal = read(ROOT/'stage-seals'/f'{stage}-{attempt}.json')
    for name,digest in seal['files'].items():
        assert sha(REPO/name) == digest, name
    if stage == 'materialize':
        from local_stability_data import materialize
        write(result,materialize(ROOT))
    elif stage == 'predict':
        from local_stability_inference import predict
        predict(argparse.Namespace(observations=stage_path(ROOT,'prepared')/'observations',
            materialization=stage_path(ROOT,'prepared')/'materialization.json',frozen_run=FROZEN,
            protocol=ROOT/'plan/stability-protocol.md',result=result))
    elif stage == 'evaluate':
        from local_stability_metrics import evaluate
        evaluate(ROOT,result)
    else:
        from local_stability_audit import audit
        audit(ROOT,result)
    for name,digest in seal['files'].items():
        assert sha(REPO/name) == digest, name


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__);p.add_argument('command',choices=('prepare','runspec','execute'))
    p.add_argument('--stage',choices=('capture','materialize','predict','evaluate','audit'))
    p.add_argument('--attempt',default='v1');p.add_argument('--result',type=Path);a=p.parse_args()
    if a.command == 'prepare':prepare()
    elif a.command == 'runspec':runspec(a.stage,a.attempt)
    else:execute(a.stage,a.attempt,a.result)
