"""R1 join bridge: preserve C2 integrity, explicitly decline its occlusion test.

R1 non-collision visual shells have no collision polygon by design. C2's old
occlusion routine therefore raises before producing any result. This bridge
records that routine as NOT_APPLICABLE and false, never as passed. The frozen
R1 finalizer still requires every other C2 check and all ten sealed source gates.
No capture, prediction, fitting, or evaluator scoring is performed here.
"""
import argparse
import json
from pathlib import Path
import sys

import join_dtr_carla_c2_rich_scene as legacy
import finalize_dtr_final_roster_join as finalizer


def inapplicable_occlusion(protocol, contract, rows_by_episode):
    ids=[str(ep) for ep in contract['episodes']]
    if not all(ep in rows_by_episode for ep in ids):
        raise ValueError('missing_occlusion_episode')
    return {'contract_id':str(contract['contract_id']),
            'episodes':{ep:{'runs':[], 'selected':None, 'passed':False} for ep in ids},
            'pair_occlusion_indices_identical':False,
            'selected_indices':{ep:[] for ep in ids}, 'passed':False,
            'status':'NOT_APPLICABLE_NONCOLLISION_R1_VISUAL_SHELL',
            'authority':'roster-source-gate.json; no legacy occlusion pass claimed'}


def join(root, protocol, annex):
    root=root.resolve(strict=True);protocol=protocol.resolve(strict=True)
    scope=json.loads(annex.read_text(encoding='utf-8'))
    for ref in scope['code_files']:
        if legacy.sha256_file(Path(ref['path']))!=ref['sha256']:
            raise ValueError('R1_join_bridge_code_drift')
    source=root/'roster-source-gate.json'
    gate=json.loads(source.read_text())
    if scope['source_gate_sha256'].get(root.name)!=legacy.sha256_file(source):
        raise ValueError('R1_join_bridge_source_gate_binding')
    if gate['status']!='SOURCE_GATE_MET' or gate['provenance']['capture_protocol_sha256']!=legacy.sha256_file(protocol):
        raise ValueError('R1_join_bridge_source_not_admitted')
    if (root/'result.json').exists():raise FileExistsError('joined_result_already_exists')
    # The failed legacy invocation created these directories before its KeyError.
    # rmdir accepts empty directories only; all partial payloads are preserved.
    for name in ('model','evaluator'):
        path=root/name
        if path.exists():path.rmdir()
    original=legacy.evaluate_occlusion_contract
    arguments=sys.argv
    try:
        legacy.evaluate_occlusion_contract=inapplicable_occlusion
        sys.argv=[str(Path(legacy.__file__)), '--root',str(root),'--protocol',str(protocol)]
        code=legacy.main()
        if code not in (0,2):raise RuntimeError('legacy_join_incomplete')
    finally:
        legacy.evaluate_occlusion_contract=original
        sys.argv=arguments
    return finalizer.finalize(root,protocol)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--annex',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(join(args.root,args.protocol,args.annex)))
