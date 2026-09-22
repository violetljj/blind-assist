"""Freeze and govern the fixed public-interval correspondence feasibility check."""
import argparse
from pathlib import Path
import shutil
import sys

from query_occupancy_data import read, write, sha

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ROOT = REPO/'artifacts.local/evidence/ba-return-correspondence-20260923'
FILES = [HERE/name for name in (
    'RETURN_CORRESPONDENCE_PROTOCOL_20260923.md','run_return_correspondence.py',
    'return_correspondence_check.py','return_correspondence_data.py',
    'return_correspondence_audit.py','test_return_correspondence_data.py',
    'inherit_spatial_model.py','local_support_diagnostic.py','query_occupancy_data.py',
    'local_transfer_metrics.py','ba_camera_corridor_metrics.py','ba_camera_corridor.py',
    'tof_fov45_core.py')]+[REPO/'tools/research_backend.py']


def prepare():
    plan = ROOT/'plan'
    if plan.exists():
        raise FileExistsError('Preserve frozen plan')
    for p in FILES:
        assert p.is_file(), p
    write(plan/'freeze.json',dict(files={p.relative_to(REPO).as_posix():sha(p) for p in FILES}))
    for p in FILES:
        dst = plan/'source-snapshot'/p.relative_to(REPO)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dst)
    base = REPO/'artifacts.local/evidence'
    inputs = [dict(alias=a,path=str(p),role=r,purpose='fixed-interval-association-feasibility') for a,p,r in (
        ('plan',plan,'configuration'),
        ('lineage',base/'ba-local-support-20260922-run','evaluator'),
        ('observations',base/'ba-local-transfer-20260922-prepared/observations','observation'),
        ('materialization',base/'ba-local-transfer-20260922-prepared/materialization.json','configuration'),
        ('saved_rows',base/'ba-local-transfer-20260922-evaluated/frame-results.json','evaluator'))]
    for stage in ('diagnostic','audit'):
        stage_inputs = list(inputs)
        if stage == 'audit':
            stage_inputs.append(dict(alias='diagnostic',path=str(ROOT.with_name(ROOT.name+'-run')),
                                     role='evaluator',purpose='independent-analytical-recount'))
        result = ROOT.with_name(ROOT.name+('-run' if stage=='diagnostic' else '-audit'))/'result.json'
        write(ROOT/(stage+'-run-spec-v1.json'),dict(schema='blindassist-asset-run-v1',
            id='return-correspondence-20260923-'+stage+'-v1',route='ue-return-correspondence',
            question='Can perfect contributor association remove LOCAL errors using unchanged public range intervals?',
            evaluator='research/active/dtr-r0/nearfield/run_return_correspondence.py',
            evidence_boundary='Consumed privileged feasibility only; no public learner or runtime claim',
            reuse=dict(mode='diagnostic',query='LOCAL contributor angular correspondence public interval support'),
            inputs=stage_inputs,outputs=[dict(alias='result',path=str(result),role='result',required=True)],
            result_output='result',command=[sys.executable,str(Path(__file__).resolve()),'execute',
                '--stage',stage,'--result','{{output:result}}'],
            parameters=dict(fit=False,capture=False,frames=576,interval_change=False,
                            source_freeze_sha256=sha(plan/'freeze.json'))))
    print('Prepared frozen source and two governed specifications')


def check():
    for name,digest in read(ROOT/'plan/freeze.json')['files'].items():
        assert sha(REPO/name) == digest, name
        assert sha(ROOT/'plan/source-snapshot'/name) == digest, name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=('prepare','execute'))
    parser.add_argument('--stage',choices=('diagnostic','audit'))
    parser.add_argument('--result',type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        check()
        if args.stage == 'diagnostic':
            from return_correspondence_check import run
            run(REPO,args.result)
        else:
            from return_correspondence_audit import audit
            audit(REPO,args.result)
        check()
