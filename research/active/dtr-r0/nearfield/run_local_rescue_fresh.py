"""Governed fresh rescue source capture, materialization and frozen inference."""
import argparse
from pathlib import Path
import os
import shutil
import sys

from query_occupancy_data import read,write,sha,stage_path

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ROOT = REPO/'artifacts.local/evidence/ba-local-rescue-fresh-20260923'
FROZEN = REPO/'artifacts.local/evidence/ba-inherit-spatial-20260922-run'


def prepare():
    from local_rescue_source import freeze
    print(freeze(ROOT/'plan'))
    shutil.copyfile(HERE/'LOCAL_RESCUE_FRESH_PROTOCOL_20260923.md',ROOT/'plan/rescue-fresh-protocol.md')
    shutil.copyfile(HERE/'LOCAL_RESCUE_GATE_PROTOCOL_20260923.md',ROOT/'plan/gate-protocol.md')
    shutil.copyfile(REPO/'artifacts.local/evidence/ba-local-rescue-gate-20260923-run/gate.json',ROOT/'plan/gate.json')


def runspec(stage,attempt):
    if stage not in ('capture','materialize','predict'):
        raise ValueError('Unsupported source stage')
    inputs = [dict(alias='plan',path=str(ROOT/'plan'),role='configuration',purpose='fixed-composite-transfer-plan')]
    inputs.append(dict(alias='gate',path=str(REPO/'artifacts.local/evidence/ba-local-rescue-gate-20260923-run/gate.json'),
                       role='configuration',purpose='frozen-gate-before-source-capture'))
    result = stage_path(ROOT,dict(materialize='prepared',predict='predictions').get(stage,stage))/'result.json'
    if stage == 'capture':
        inputs.append(dict(alias='scene',asset='unreal/BlindAssistStreetLab',relative_path='Content/StreetLab',
                           role='configuration',purpose='unchanged-map-and-opaque-materials'))
        result = stage_path(ROOT,'capture')/'launcher-terminal.json'
        command = [sys.executable,str(HERE/'launch_local_rescue_fresh.py'),'--spec',str(ROOT/'plan/spec.json'),
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
        raise ValueError('Only capture/materialize/predict belong to source runner')
    if stage != 'capture':
        code = [HERE/'run_local_rescue_fresh.py',HERE/('local_rescue_'+dict(materialize='data',predict='inference')[stage]+'.py'),
                HERE/'LOCAL_RESCUE_FRESH_PROTOCOL_20260923.md']
        extra = ('local_rescue_source.py','query_occupancy_data.py','inherit_spatial_model.py','ba_camera_corridor_metrics.py','local_transfer_metrics.py',
                 'local_transfer_inference.py','ba_camera_corridor.py','tof_fov45_core.py','tof_corridor_calibration.py')
        code += [HERE/n for n in extra]
        seal = ROOT/'stage-seals'/f'{stage}-{attempt}.json'
        write(seal,dict(files={p.relative_to(REPO).as_posix():sha(p) for p in code}))
        for p in code:
            target = ROOT/'stage-seals'/f'{stage}-{attempt}-source'/p.relative_to(REPO)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        inputs.append(dict(alias='stage_seal',path=str(seal),role='configuration',purpose='immutable-stage-code'))
        command = [sys.executable,str(Path(__file__).resolve()),'execute','--stage',stage,'--attempt',attempt,
                   '--result','{{output:result}}']
    path = ROOT/(stage+'-run-spec-'+attempt+'.json')
    write(path,dict(schema='blindassist-asset-run-v1',id='local-rescue-fresh-20260923-'+stage+'-'+attempt,
        route='ue-local-rescue-fresh',question='Does a preselected causal LOCAL rescue admission retain benefits on new dimensions and schedules?',
        evaluator='research/active/dtr-r0/nearfield/run_local_rescue_fresh.py',
        evidence_boundary='Same-generator controlled Development; no fit or continuous physical motion claim',
        reuse=dict(mode='development',query='LOCAL frozen HGB composite shape size approach dwell event timing transfer'),
        inputs=inputs,outputs=[dict(alias='result',path=str(result),role='result',required=True)],
        result_output='result',command=command,parameters=dict(frames=576,fit=False,threshold_selection=False,stage=stage)))
    print(path)


def execute(stage,attempt,result):
    journal = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    assert journal and read(journal)['state'] == 'running', 'Use governed research-ue entry'
    from launch_local_rescue_fresh import verify_protocol
    verify_protocol(ROOT/'plan/protocol.json',ROOT/'plan/spec.json',REPO)
    seal = read(ROOT/'stage-seals'/f'{stage}-{attempt}.json')
    for name,digest in seal['files'].items():
        assert sha(REPO/name) == digest, name
    if stage == 'materialize':
        from local_rescue_data import materialize
        write(result,materialize(ROOT))
    elif stage == 'predict':
        from local_rescue_inference import predict
        predict(argparse.Namespace(observations=stage_path(ROOT,'prepared')/'observations',
            materialization=stage_path(ROOT,'prepared')/'materialization.json',frozen_run=FROZEN,
            protocol=ROOT/'plan/rescue-fresh-protocol.md',result=result))
    else:
        raise ValueError('Unsupported source stage')
    for name,digest in seal['files'].items():
        assert sha(REPO/name) == digest, name


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__);p.add_argument('command',choices=('prepare','runspec','execute'))
    p.add_argument('--stage',choices=('capture','materialize','predict'))
    p.add_argument('--attempt',default='v1');p.add_argument('--result',type=Path);a=p.parse_args()
    if a.command == 'prepare':prepare()
    elif a.command == 'runspec':runspec(a.stage,a.attempt)
    else:execute(a.stage,a.attempt,a.result)
