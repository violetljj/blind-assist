"""Join native calibration, complete live geometry and retained-engine parity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'research/active/dtr-r0/unreal'))
from evaluate_street_closed_loop import evaluate_episode
from street_bank_loader import load_scenarios

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)

def verify(alignment, comparison, parity):
    evidence={}
    def record(path):
        evidence[str(path.resolve())]=sha(path)
        return read(path)
    calibration=record(alignment/'alignment-validation.json')
    native=record(alignment/'result.json')
    require(native['script_sha256']==sha(REPO/'research/active/dtr-r0/unreal/ue_depth_calibration.py'),
            'Calibration implementation changed after native capture')
    require('StreetLabV4' in native['inherited_map'] and native['project_config_sha256']==
            sha(REPO/'artifacts.local/unreal/BlindAssistStreetLab/Config/DefaultEngine.ini'),
            'Calibration must use the actual V4 render settings')
    require(record(alignment/'process-release.json')['released'],'Calibration resources not released')
    require(calibration['status']=='PASS' and calibration['metric_cases_verified']==6 and
            calibration['alignment_cases_verified']==10 and calibration['edge_scanlines_verified']==120 and
            calibration['moving_alignment_status']=='PASS' and calibration['moving_cases_verified']==4,
            'Native metric depth and RGB-D alignment must both pass')
    differential=record(parity/'receipt.json')
    require(differential['status']=='PASS' and differential['mismatches']==0 and
            differential['source_dt_s']==[.1] and differential['compared_prefixes']>=4,
            'Retained engine must agree at actual 10 Hz')
    result=record(comparison/'comparison.json')
    require(result['status']=='COMPLETE','Full paired controller comparison required')
    plan=record(comparison/'plan.json')
    parity_identity=record(parity/'input-identity.json')
    matching_sources=[]
    for mode in result['modes']:
        root=comparison/mode.lower()
        if (root/'sensor-worker/backend.json').is_file() and sha(root/'sensor-worker/backend.json')==parity_identity['worker_backend_sha256']:
            if all((root/name).is_file() and sha(root/name)==digest for name,digest in parity_identity['candidate_cache_sha256'].items()):
                matching_sources.append(mode)
    require(matching_sources,'Parity candidates do not belong to this comparison')
    for name,digest in plan['sources'].items():
        require(sha(REPO/name)==digest,'Runtime source changed: '+name)
    for name,digest in plan['inputs'].items():
        require(sha(Path(name))==digest,'Comparison input changed: '+name)
    frames=0;actor_checks=0;episodes=0;outcomes={}
    for mode in result['modes']:
        root=comparison/mode.lower()
        require(not (root/'owner.lock').exists(),'Controller run is still owned')
        evaluation=record(root/'evaluation.json')
        require(evaluation['status']=='COMPLETE' and evaluation['expected_pairs']==8 and
                evaluation['all_open_loop_contrasts_pass'],'Incomplete or invalid control denominator')
        identity=record(root/'identity.json')
        specs=load_scenarios(Path(identity['scenario_selection']['manifest_path']),identity['scenario_selection']['split'])
        catalog={spec['id']:spec for spec in specs}
        seen=set()
        for path in sorted((root/'evaluator/episodes').glob('*.json')):
            episode=record(path)
            key=(episode['scenario_id'],episode['arm'])
            require(key not in seen,'Duplicate episode');seen.add(key)
            spec=catalog[key[0]]
            checked=evaluate_episode(spec,episode)
            require(checked['status']=='COMPLETE' and checked['visual_geometry']['status']=='PASS',
                    'Contact, action or visual geometry validation failed: '+str(key))
            dt=[b['time_s']-a['time_s'] for a,b in zip(episode['frames'],episode['frames'][1:])]
            require(dt and all(abs(value-.1)<1e-7 for value in dt),'Actual timestamps violate 10 Hz')
            frames+=len(episode['frames']);actor_checks+=checked['visual_geometry']['actor_checks'];episodes+=1
        require(seen=={(s,arm) for s in catalog for arm in ('OPEN_LOOP','ASSISTED')},'Missing actual branch')
        outcomes[mode]={'successes':evaluation['assisted_successes'],'denominator':evaluation['assisted_complete_denominator']}
        release=record(root/'process-release.json')
        require(release.get('released') is True,'Owned live resources were not released')
    require(episodes==32,'Expected 32 newly executed branches')
    return {'status':'PASS','scope':'UE_AVOIDANCE_DEVELOPMENT_LAB_ACCEPTANCE',
            'static_alignment_views':6,'moving_alignment_views':4,'edge_scanlines':120,'live_branches':episodes,
            'live_frames':frames,'native_actor_checks':actor_checks,'sensor_hz':10,
            'parity_frames':differential['frames'],'controller_outcomes':outcomes,
            'parity_source_modes':matching_sources,
            'algorithm_promotion':False,'precise_mesh_collision':False,'carla_feature_parity':False,
            'evidence_sha256':evidence}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('alignment','comparison','parity','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    require(output.is_relative_to((REPO/'artifacts.local').resolve()),'Output must stay in artifacts.local')
    require(not output.exists(),'Use a new acceptance receipt')
    output.parent.mkdir(parents=True,exist_ok=True)
    try:result=verify(args.alignment,args.comparison,args.parity)
    except Exception as exc:
        result={'status':'FAIL','reason':str(exc),'exception_type':type(exc).__name__}
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
    return 0 if result['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
