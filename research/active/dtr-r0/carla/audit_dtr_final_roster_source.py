"""Verify native raw shards then evaluate R1 source semantics, without inference."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

import dtr_carla_c2_rich_scene as c2
import evaluate_dtr_final_visual_shell_probe as probe
import evaluate_dtr_final_roster_source as source


def audit(protocol_path, root, annex_path, *, historical_diagnostic=False):
    protocol=json.loads(protocol_path.read_text(encoding='utf-8-sig'))
    annex_document=json.loads(annex_path.read_text(encoding='utf-8-sig'))
    roster_path=Path(__file__).with_name('dtr_final_reckoning_roster_protocol.json')
    roster=json.loads(roster_path.read_text(encoding='utf-8-sig'))
    provenance={'capture_protocol_sha256':c2.sha256_file(protocol_path),
        'roster_sha256':c2.sha256_file(roster_path),'annex_sha256':c2.sha256_file(annex_path),
        'source_root':str(root),'historical_diagnostic':historical_diagnostic,
        'fit_final_scores_or_predictions_opened':False,'verified_sensors':[]}
    try:
        instance=probe.load_shard(protocol,protocol_path,root,'instance')
        provenance['verified_sensors'].append('instance')
        witness=probe.load_shard(protocol,protocol_path,root,'witness')
        provenance['verified_sensors'].append('witness')
    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as error:
        return {'status':'MECHANICAL_SOURCE_VERIFICATION_FAILURE','reason':str(error),
                'provenance':provenance,'source_semantics':None}
    annexes=copy.deepcopy(annex_document['strata'])
    scenarios={s['episode_id']:s for s in protocol['scenarios']}
    pairs={p['episode_id']:p['reference_episode_id'] for p in annex_document.get('reference_pairs',[])}
    bundles={};plan_mismatches=[]
    for key,annex in annexes.items():
        ep=annex['episode_id']
        if ep not in instance or ep not in witness:
            continue
        native=c2.build_plan_receipt(scenarios[ep].get('issued_plan'))
        if annex.get('issued_plan_receipt') != native:
            plan_mismatches.append(ep)
            if not historical_diagnostic:
                return {'status':'MECHANICAL_ANNEX_CAPTURE_BINDING_FAILURE','reason':'issued_plan_receipt_mismatch',
                        'episodes':plan_mismatches,'provenance':provenance,'source_semantics':None}
        annex['issued_plan_receipt']=native
        annex['source_authority']='VERIFIED_NATIVE_INSTANCE_AND_INDEPENDENT_WITNESS_NO_WEARABLE_REQUIRED_FOR_SOURCE_SEMANTICS'
        if key.startswith('S09'):
            try:
                instance[ep]=source.add_mask_components(instance[ep],root/'shards/instance',annex['target_asset'])
            except (ImportError,OSError,ValueError,KeyError) as error:
                return {'status':'MECHANICAL_INSTANCE_COMPONENT_DECODE_FAILURE','reason':str(error),
                        'provenance':provenance,'source_semantics':None}
        bundles[key]={'source_rows':instance[ep],'instance_rows':instance[ep],'witness_rows':witness[ep],
                      'reference_rows':instance.get(pairs.get(ep))}
    provenance['annex_plan_mismatches_replaced_with_native_for_historical_diagnostic']=plan_mismatches
    semantics=source.evaluate_roster(roster,bundles,annexes)
    return {'status':semantics['status'],'provenance':provenance,'source_semantics':semantics,
            'scope':'SOURCE_ONLY; historical removals are prospective source eligibility, not old frozen interventions' if historical_diagnostic else 'SOURCE_ONLY_NO_DETECTOR_CREDENTIAL_OR_METHOD_ADMISSION'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True,help='Native raw capture protocol')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--annex',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--historical-diagnostic',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=audit(args.protocol.resolve(),args.root.resolve(),args.annex.resolve(),historical_diagnostic=args.historical_diagnostic)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False))
    return 0 if result['status']==source.PASS else 2


if __name__=='__main__':raise SystemExit(main())
