"""UE-first pedestrian avoidance research: replay, closed loop and comparisons."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

REPO=Path(__file__).resolve().parents[1]
UE=REPO/'research/active/dtr-r0/unreal'
DATA=REPO/'artifacts.local/unreal'
DATASET=DATA/'fixed-sensory-v4-20260905-a'
BANK=DATA/'scenario-bank-v2-20260905.json'
MODES=('DEPTH_ONLY','DTR_ONLY','JOINT','CANDIDATE_DEPTH','CANDIDATE_DTR')


def engine_root(explicit=None):
    if explicit:return Path(explicit).resolve()
    configured=os.environ.get('UE_ENGINE_ROOT')
    if configured:return Path(configured).resolve()
    project=json.loads((DATA/'BlindAssistStreetLab/BlindAssistStreetLab.uproject').read_text())
    installation=Path(os.environ.get('PROGRAMDATA','C:/ProgramData'))/'Epic/UnrealEngineLauncher/LauncherInstalled.dat'
    if installation.is_file():
        for item in json.loads(installation.read_text())['InstallationList']:
            if item.get('AppName')=='UE_'+project['EngineAssociation']:
                return Path(item['InstallLocation']).resolve()
    raise ValueError('Specify --engine or UE_ENGINE_ROOT for the installed project engine')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('status',help='Read local entrypoint availability; does not launch UE')
    replay=sub.add_parser('replay',help='Recorded RGB-D perception only; no new motion outcome')
    replay.add_argument('--dataset',type=Path,default=DATASET)
    replay.add_argument('--output',type=Path,required=True)
    replay.add_argument('--episode',action='append',default=[])
    calibration=sub.add_parser('calibrate',help='Known-distance depth check in an unsaved temporary world')
    calibration.add_argument('--engine',type=Path)
    calibration.add_argument('--output',type=Path,required=True)
    for name in ('closed-loop','compare'):
        child=sub.add_parser(name)
        child.add_argument('--engine',type=Path)
        child.add_argument('--scenario-manifest',type=Path,default=BANK)
        child.add_argument('--output',type=Path,required=True)
        child.add_argument('--resume',action='store_true')
        if name=='closed-loop':
            child.add_argument('--controller-mode',choices=MODES,default='DEPTH_ONLY')
            child.add_argument('--split',choices=('regression','development'),default='regression')
            child.add_argument('--case',action='append',default=[])
    return p


def command(args):
    if not args.output.resolve().is_relative_to((REPO/'artifacts.local').resolve()):
        raise ValueError('Output must remain under artifacts.local')
    if args.command=='replay':
        target=UE/'ue_fixed_replay.py'
        argv=['replay','--dataset',str(args.dataset),'--output',str(args.output),'--engine','incremental']
        for episode in args.episode:argv+=['--episode',episode]
    else:
        engine=engine_root(args.engine)
        if not (engine/'Engine/Binaries/Win64/UnrealEditor-Cmd.exe').is_file():
            raise ValueError('Unreal Editor executable missing: '+str(engine))
        if args.command=='calibrate':
            return REPO/'tools/run_ue_depth_calibration.py',['--engine',str(engine),'--output',str(args.output)]
        target=REPO/'tools'/('run_street_closed_loop.py' if args.command=='closed-loop' else 'run_street_candidate_comparison.py')
        argv=['--engine',str(engine),'--output',str(args.output),'--scenario-manifest',str(args.scenario_manifest)]
        if args.resume:argv+=['--resume']
        if args.command=='closed-loop':
            argv+=['--map','StreetLabV4','--controller-mode',args.controller_mode,
                   '--prediction-engine','incremental','--scenario-split',args.split]
            for case in args.case:argv+=['--case',case]
    return target,argv


def main():
    p=parser();args=p.parse_args()
    if args.command=='status':
        paths={'project':DATA/'BlindAssistStreetLab/BlindAssistStreetLab.uproject',
               'map':DATA/'BlindAssistStreetLab/Content/StreetLab/StreetLabV4.umap',
               'fixed_input_manifest':DATASET/'integrity.json','scenario_bank':BANK}
        print(json.dumps({'primary_backend':'UE','map':'StreetLabV4','motion_reference':'DEPTH_ONLY',
                          'prediction_engine':'incremental','check_scope':'EXISTENCE_ONLY_NOT_SOURCE_VALIDATION',
                          'inputs':{k:{'path':str(v),'exists':v.is_file()} for k,v in paths.items()}},indent=2))
        return
    try:target,argv=command(args)
    except ValueError as error:p.error(str(error))
    print(json.dumps({'backend':'UE','entrypoint':str(target),'arguments':argv}),flush=True)
    sys.argv=[str(target),*argv]
    sys.path.insert(0,str(target.parent))
    runpy.run_path(str(target),run_name='__main__')


if __name__=='__main__':main()
