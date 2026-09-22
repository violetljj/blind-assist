"""Prepare and execute the fixed, evaluator-only LOCAL support diagnostic."""
import argparse
from pathlib import Path
import shutil
import sys

from query_occupancy_data import read, write, sha

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ROOT = REPO/'artifacts.local/evidence/ba-local-support-20260922'
SOURCE = REPO/'artifacts.local/evidence/ba-local-transfer-20260922'
FILES = [HERE/name for name in (
    'LOCAL_SUPPORT_PROTOCOL_20260922.md', 'run_local_support.py',
    'local_support_diagnostic.py', 'local_support_audit.py',
    'test_local_support_diagnostic.py',
    'inherit_spatial_model.py', 'tof_fov45_core.py', 'ba_camera_corridor.py',
    'query_occupancy_data.py', 'local_transfer_metrics.py',
    'ba_camera_corridor_metrics.py')]+[REPO/'tools/research_backend.py']


def sibling(path, suffix):
    return path.with_name(path.name+'-'+suffix)


def prepare():
    plan = ROOT/'plan'
    if plan.exists():
        raise FileExistsError('Frozen plan already exists')
    for path in FILES:
        assert path.is_file(), path
    plan.mkdir(parents=True)
    frozen = {path.relative_to(REPO).as_posix():sha(path) for path in FILES}
    write(plan/'freeze.json', dict(schema='local-support-freeze-v1', sources=frozen,
        source_spec_sha256=sha(SOURCE/'plan/spec.json'),
        prediction_seal_sha256=sha(sibling(SOURCE,'predictions')/'prediction-seal.json'),
        saved_rows_sha256=sha(sibling(SOURCE,'evaluated')/'frame-results.json')))
    for path in FILES:
        target=plan/'source-snapshot'/path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path,target)
    inputs=[]
    for alias,path,role in (
        ('freeze',plan,'configuration'),
        ('source_plan',SOURCE/'plan','configuration'),
        ('native',sibling(SOURCE,'capture')/'evaluator','evaluator'),
        ('observations',sibling(SOURCE,'prepared')/'observations','observation'),
        ('materialization',sibling(SOURCE,'prepared')/'materialization.json','configuration'),
        ('labels',sibling(SOURCE,'prepared')/'labels/evaluation.npz','evaluator'),
        ('predictions',sibling(SOURCE,'predictions'),'evaluator'),
        ('evaluated',sibling(SOURCE,'evaluated'),'evaluator')):
        inputs.append(dict(alias=alias,path=str(path),role=role,purpose='frozen-native-support-diagnostic'))
    for stage in ('diagnostic','audit'):
        stage_inputs=list(inputs)
        if stage=='audit':
            stage_inputs.append(dict(alias='diagnostic',path=str(sibling(ROOT,'run')),
                role='evaluator',purpose='independent-saved-output-native-recount'))
        result=sibling(ROOT,'run' if stage=='diagnostic' else 'audit')/'result.json'
        spec=dict(schema='blindassist-asset-run-v1',id='local-support-20260922-'+stage+'-v1',
            route='ue-local-support',question='Which frozen LOCAL gains have observed native support in the score-winning query?',
            evaluator='research/active/dtr-r0/nearfield/run_local_support.py',
            evidence_boundary='Consumed evaluator-only support opportunity; no public prediction or ownership learning claim',
            reuse=dict(mode='diagnostic',query='LOCAL query support native strongest return pixel bands correspondence'),
            inputs=stage_inputs,outputs=[dict(alias='result',path=str(result),role='result',required=True)],
            result_output='result',command=[sys.executable,str(Path(__file__).resolve()),'execute',
                '--stage',stage,'--result','{{output:result}}'],
            parameters=dict(frames=576,model_inference=False,fit=False,threshold_search=False,
                privileged=True,primary_retained_rescues=32,source_freeze_sha256=sha(plan/'freeze.json')))
        write(ROOT/(stage+'-run-spec-v1.json'),spec)
    print('Frozen protocol, source snapshots and two governed RunSpecs prepared.')


def check_freeze():
    frozen=read(ROOT/'plan/freeze.json')
    for relative,digest in frozen['sources'].items():
        assert sha(REPO/relative)==digest, relative
        assert sha(ROOT/'plan/source-snapshot'/relative)==digest, relative
    assert sha(SOURCE/'plan/spec.json')==frozen['source_spec_sha256']
    assert sha(sibling(SOURCE,'predictions')/'prediction-seal.json')==frozen['prediction_seal_sha256']
    assert sha(sibling(SOURCE,'evaluated')/'frame-results.json')==frozen['saved_rows_sha256']


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=('prepare','execute'))
    parser.add_argument('--stage',choices=('diagnostic','audit'))
    parser.add_argument('--result',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare()
    else:
        check_freeze()
        if args.stage=='diagnostic':
            from local_support_diagnostic import run
            run(REPO,args.result)
        elif args.stage=='audit':
            from local_support_audit import audit
            audit(REPO,sibling(ROOT,'run'),args.result)
        else:
            raise ValueError('Stage required')
        check_freeze()
