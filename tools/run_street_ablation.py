"""Frozen three-controller Development comparison with fresh UE controls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def summarize(root):
    modes=('JOINT','DTR_ONLY','DEPTH_ONLY')
    evaluations={m:read(root/m.lower()/'evaluation.json') for m in modes}
    identities={m:read(root/m.lower()/'identity.json') for m in modes}
    comparable=[]
    for identity in identities.values():
        comparable.append({k:v for k,v in identity.items() if k!='controller_mode'})
    if any(v!=comparable[0] for v in comparable): raise RuntimeError('Ablation inputs changed')
    rows=[]
    for pair in evaluations['JOINT']['pairs']:
        row={'scenario_id':pair['scenario_id'],'modes':{}}
        for mode, result in evaluations.items():
            p=next(p for p in result['pairs'] if p['scenario_id']==pair['scenario_id'])
            a=p['ASSISTED']
            row['modes'][mode]={'success':a['success'],'contact':a['contact'],
                'goal_reached':a['goal_reached'],'duration_s':a['duration_s'],
                'delay_s':p['assisted_delay_s'],'action_dwell_s':a.get('action_dwell_s',{}),
                'trigger_sources':p['causal_trajectory']['trigger_sources']}
        rows.append(row)
    result={'status':'COMPLETE' if all(e['status']=='COMPLETE' and e['all_open_loop_contrasts_pass'] for e in evaluations.values()) else 'INCOMPLETE',
        'evidence':'Fresh UE closed-loop synthetic Development, identical frozen scripts and scene',
        'controller_removal':'DTR_ONLY disables all depth control including validity and steering; DEPTH_ONLY disables DTR control. Both observations remain logged in all modes.',
        'compute_boundary':'All modes compute both raw perception branches; worker timing is not branch compute ablation.',
        'claim_boundary':'Curated geometry-proxy outcomes; not safety, generalization, or a new DTR planner.',
        'successes':{m:e['assisted_successes'] for m,e in evaluations.items()},
        'denominators':{m:e['assisted_complete_denominator'] for m,e in evaluations.items()},'rows':rows}
    write(root/'comparison.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenario-manifest',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    scripts=repo/'research/active/dtr-r0/unreal'
    root=args.output.resolve()
    if not root.is_relative_to((repo/'artifacts.local').resolve()): p.error('Output must remain under artifacts.local')
    root.mkdir(parents=True,exist_ok=args.resume)
    sources=[*sorted(scripts.glob('*.py')),repo/'tools/run_street_closed_loop.py',Path(__file__)]
    project=repo/'artifacts.local/unreal/BlindAssistStreetLab'
    inputs=[project/'Content/StreetLab/StreetLabV4.umap',project/'Config/DefaultEngine.ini',
            repo/'artifacts.local/models/yolo11n-seg.pt',args.scenario_manifest.resolve()]
    plan={'modes':['JOINT','DTR_ONLY','DEPTH_ONLY'],'map':'StreetLabV4',
          'scenario_manifest_sha256':sha(args.scenario_manifest),
          'inputs':{str(f):sha(f) for f in inputs},
          'sources':{str(f.relative_to(repo)):sha(f) for f in sources},
          'claim':'One fixed Development comparison; no tuning or retry after observed algorithm failure'}
    if args.resume:
        if read(root/'plan.json')!=plan: raise RuntimeError('Frozen plan changed')
    else: write(root/'plan.json',plan)
    for mode in plan['modes']:
        for name,digest in plan['inputs'].items():
            if sha(Path(name))!=digest: raise RuntimeError('Input changed during ablation: '+name)
        for name,digest in plan['sources'].items():
            if sha(repo/name)!=digest: raise RuntimeError('Source changed during ablation: '+name)
        out=root/mode.lower()
        if (out/'evaluation.json').exists() and read(out/'evaluation.json')['status']=='COMPLETE': continue
        command=[sys.executable,str(repo/'tools/run_street_closed_loop.py'),'--engine',str(args.engine),
            '--output',str(out),'--map','StreetLabV4','--controller-mode',mode,
            '--scenario-manifest',str(args.scenario_manifest),'--scenario-split','regression']
        if out.exists(): command.append('--resume')
        write(root/'progress.json',{'status':'RUNNING','mode':mode,'owner_pid':__import__('os').getpid(),'started_utc':time.time()})
        try:
            subprocess.run(command,check=True)
        except BaseException as exc:
            write(root/'progress.json',{'status':'INTERRUPTED','mode':mode,'error':str(exc),'checkpoints_retained':True})
            raise
    result=summarize(root)
    write(root/'progress.json',{'status':result['status'],'completed_utc':time.time()})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
