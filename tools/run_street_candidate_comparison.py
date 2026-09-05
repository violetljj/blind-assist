"""Paired action-conditioner check on the eight consumed UE regression scenes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MODES=('CANDIDATE_DEPTH','CANDIDATE_DTR')


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def summarize(root):
    evaluations={mode:read(root/mode.lower()/'evaluation.json') for mode in MODES}
    identities=[{k:v for k,v in read(root/mode.lower()/'identity.json').items() if k!='controller_mode'} for mode in MODES]
    if identities[0]!=identities[1]: raise RuntimeError('Comparison inputs or runtime sources changed')
    rows=[]
    for first in evaluations[MODES[0]]['pairs']:
        row={'scenario_id':first['scenario_id'],'modes':{}}
        for mode,evaluation in evaluations.items():
            pair=next(pair for pair in evaluation['pairs'] if pair['scenario_id']==first['scenario_id'])
            arm=pair['ASSISTED']
            row['modes'][mode]={key:arm.get(key) for key in ('status','success','contact','goal_reached','duration_s','action_dwell_s')}
            row['modes'][mode].update(open_loop_contrast_pass=pair['open_loop_contrast_pass'],delay_s=pair['assisted_delay_s'])
        rows.append(row)
    complete=all(e['status']=='COMPLETE' and e['all_open_loop_contrasts_pass'] for e in evaluations.values())
    baseline=evaluations[MODES[0]]['assisted_successes']
    challenger=evaluations[MODES[1]]['assisted_successes']
    regressions=[r['scenario_id'] for r in rows if r['modes'][MODES[0]]['success'] and not r['modes'][MODES[1]]['success']]
    interventions={}
    for mode in MODES:
        episodes=[read(path) for path in sorted((root/mode.lower()/'evaluator/episodes').glob('*.json'))]
        commands=[frame.get('response',{}).get('command',{}) for ep in episodes if ep['arm']=='ASSISTED' for frame in ep['frames']]
        interventions[mode]={'frames':len(commands),'changed_motion_frames':sum(bool(c.get('candidate_intervention')) for c in commands),
            'supported_footprint_frames':sum(c.get('candidate_evaluation',{}).get('admitted_tracks',0)>0 for c in commands),
            'nominal_predicted_intersection_frames':sum(bool(c.get('action_conditioned_risk')) for c in commands)}
    result={'status':'COMPLETE' if complete else 'INCOMPLETE','modes':list(MODES),'rows':rows,
        'evidence':'Actual rerendered UE synthetic Development on the manifest-selected conditions; no exact-pixel pairing',
        'mechanism':'Both arms enumerate and score the same candidate motions; only CANDIDATE_DTR selects using footprint intersections',
        'claim_boundary':'Geometry proxy outcomes; no fresh confirmation, exact-pixel pairing, deployment or safety claim',
        'compute_boundary':'Both arms compute the same detector, X73 state and candidate scores',
        'successes':{m:e['assisted_successes'] for m,e in evaluations.items()},
        'denominators':{m:e['assisted_complete_denominator'] for m,e in evaluations.items()},
        'interventions':interventions,'success_regressions':regressions,
        'decision':'INCOMPLETE' if not complete else 'NOT_EVALUABLE_NO_ACTION_FOOTPRINT_SUPPORT' if not interventions[MODES[1]]['supported_footprint_frames'] else
                   'REJECT_DEFAULT_PROMOTION_REGRESSION' if regressions else
                   'DEVELOPMENT_SUCCESS_GAIN' if challenger>baseline else 'NO_INCREMENTAL_SUCCESS_GAIN',
        'default_promotion':False}
    write(root/'comparison.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenario-manifest',type=Path,required=True)
    parser.add_argument('--scenario-split',choices=('regression','development'),default='regression')
    parser.add_argument('--action-footprint-state',choices=('cadence','frozen'),default='cadence')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    scripts=repo/'research/active/dtr-r0/unreal'
    sys.path.insert(0,str(scripts))
    from ue_incremental import source_paths
    from street_bank_loader import load_scenarios
    from scenario_bank import validate_specs
    selected=load_scenarios(args.scenario_manifest,args.scenario_split)
    if not validate_specs(selected)['passed'] or len(selected)!=8:
        parser.error('This bounded comparison requires eight valid scenes')
    output=args.output.resolve()
    if not output.is_relative_to((repo/'artifacts.local').resolve()): parser.error('Output must remain under artifacts.local')
    output.mkdir(parents=True,exist_ok=args.resume)
    names=('capture_street_closed_loop.py','street_live_server.py','street_live_policy.py','street_action_risk.py',
           'street_scenarios.py','scenario_bank.py','evaluate_street_closed_loop.py','reuse_street_open_loop.py',
           'street_process_lifecycle.py','ue_action_footprints.py','visual_geometry.py','street_bank_loader.py','discriminating_bank.py')
    sources=sorted(set(source_paths()+[scripts/name for name in names]+[Path(__file__),repo/'tools/run_street_closed_loop.py']))
    project=repo/'artifacts.local/unreal/BlindAssistStreetLab'
    inputs=[args.scenario_manifest.resolve(),project/'Content/StreetLab/StreetLabV4.umap',
            project/'Config/DefaultEngine.ini',repo/'artifacts.local/models/yolo11n-seg.pt',
            args.engine/'Engine/Build/Build.version']
    plan={'schema':'street-action-conditioner-comparison-v2','modes':list(MODES),'map':'StreetLabV4',
        'action_footprint_state':args.action_footprint_state,
        'scenario_split':args.scenario_split,
        'source_contract_correction':'Action-only fit window spans four nominal source observations while retained X73 and its batch oracle stay unchanged',
        'sources':{str(path.relative_to(repo)):sha(path) for path in sources},
        'inputs':{str(path.resolve()):sha(path) for path in inputs},
        'acceptance':'Exact engine parity is checked separately. Complete both arms, retain failures, report per-scene success and time without changing any parameters.',
        'scope':'Eight disclosed Development cases; two controllers with new open-loop and assisted branches (32 branches total)',
        'stop':'One complete comparison; no tuning, retry of algorithm failures, or time-budget extension'}
    if args.resume:
        if read(output/'plan.json')!=plan: raise RuntimeError('Frozen comparison changed')
    else:
        write(output/'plan.json',plan)
        for path in sources:
            snapshot=output/'source_snapshot'/path.relative_to(repo)
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,snapshot)
    for mode in MODES:
        for name,digest in plan['sources'].items():
            if sha(repo/name)!=digest or sha(output/'source_snapshot'/name)!=digest:
                raise RuntimeError('Runtime source or frozen snapshot changed: '+name)
        for name,digest in plan['inputs'].items():
            if sha(Path(name))!=digest: raise RuntimeError('Frozen input changed: '+name)
        run=output/mode.lower()
        if (run/'evaluation.json').exists() and read(run/'evaluation.json')['status']=='COMPLETE': continue
        command=[sys.executable,'-B',str(repo/'tools/run_street_closed_loop.py'),'--engine',str(args.engine),
            '--output',str(run),'--map','StreetLabV4','--controller-mode',mode,'--prediction-engine','incremental',
            '--action-footprint-state',args.action_footprint_state,
            '--scenario-manifest',str(args.scenario_manifest),'--scenario-split',args.scenario_split]
        if run.exists(): command.append('--resume')
        write(output/'progress.json',{'status':'RUNNING','mode':mode,'owner_pid':os.getpid(),'started_utc':time.time()})
        try: subprocess.run(command,check=True)
        except BaseException as exc:
            write(output/'progress.json',{'status':'INTERRUPTED','mode':mode,'error':str(exc),'checkpoints_retained':True})
            raise
    result=summarize(output)
    write(output/'progress.json',{'status':result['status'],'completed_utc':time.time()})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
