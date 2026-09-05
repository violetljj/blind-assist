"""Seal the single corrected source design before its first durable raster.

This seals source capture only. It does not seal an eleven-arm inference runner
or authorize opening fitting/final algorithm outcomes.
"""
import argparse
import json
import shutil
from pathlib import Path
import materialize_dtr_final_roster_execution as execution


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def seal(root):
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    execution.shell.roster_validator.validate(execution.shell.base.read_json(execution.shell.base.ROSTER_PROTOCOL), repo_root=repo)
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root/'source_snapshot'
    files = sorted(list(here.glob('*.py')) + list(here.glob('*.json')) + [
        repo/'tools/run_dtr_carla_c2_rich_scene.ps1', repo/'tools/assert_carla_storage_capacity.ps1'])
    identities = {}
    for path in files:
        relative = path.relative_to(repo)
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        identities[relative.as_posix()] = execution.shell.base.c2.sha256_file(target)
    groups = {}
    for group in ['FIT_ONLY', 'FINAL_A', 'FINAL_B']:
        protocol, annex = execution.materialize(group)
        protocol['source_disjoint_contract']['capture_authorized'] = True
        protocol['source_disjoint_contract']['reason'] = 'SINGLE_PREPIXEL_SOURCE_DESIGN_SEALED_NO_METHOD_AUTHORITY'
        protocol['objective'] = 'Frozen R1 source capture; full source gates required before any model execution'
        directory = root/group
        directory.mkdir()
        write(directory/'protocol.json', protocol)
        annex.update(status='PREPIXEL_SOURCE_CAPTURE_SEALED', capture_authorized=True,
                     protocol_sha256=execution.shell.base.c2.sha256_file(directory/'protocol.json'),
                     implementation_snapshot_sha256=identities,
                     detector_ledger_and_inference_authorized=False,
                     fit_or_final_algorithm_truth_open_authorized=False)
        write(directory/'annex.json', annex)
        groups[group] = {name: execution.shell.base.c2.sha256_file(directory/name) for name in ['protocol.json','annex.json']}
    receipt = {'status':'PREPIXEL_SOURCE_CAPTURE_SEALED', 'groups':groups,
               'source_snapshot':identities, 'all_method_implementations_preserved':True,
               'source_gate_failure':'NOT_EVALUABLE_NO_SCENE_REPAIR_OR_RETRY_AFTER_DURABLE_FRAMES',
               'algorithm_outcome_access_authorized':False}
    write(root/'source-freeze.json', receipt)
    return {'status':receipt['status'], 'root':str(root.resolve()), 'snapshot_files':len(files)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(seal(args.output_root)))
