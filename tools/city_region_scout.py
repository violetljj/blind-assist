"""Prepare source-only City region scouting specs and summarize completed captures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def prepare(candidates,template,project,output):
    project=Path(project).resolve();output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest=[]
    for region in candidates['regions']:
        x,y=region['candidate_center_xy_m'];name=region['region_id']
        spec=dict(template);asset=region['map_asset']
        if not asset.startswith('/Game/'):raise ValueError('Project map required')
        path=project.parent/'Content'/Path(asset[len('/Game/'):]+'.umap')
        spec.update(map_asset=asset,map_file=str(path),map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            export_appearance=False,export_native_inventory=True,export_hlod_membership=True,inventory_indices=[1,2,3,4],
            world_partition_region_m=dict(min=[x-150,y-150,-5],max=[x+150,y+150,400]),
            source_floor_grid=dict(bounds_xy_m=[x-45,y-45,x+45,y+45],step_m=3,top_z_m=3,bottom_z_m=-3),
            scope='SOURCE_ONLY_RECONNAISSANCE_NOT_RESEARCH_COHORT',cases=[])
        poses=[(0,0,125,0,-90),(-6,0,2.45,0,-5),(6,0,2.45,180,-5),(0,-6,2.45,90,-5),(0,6,2.45,-90,-5)]
        for i,(dx,dy,z,yaw,pitch) in enumerate(poses):
            spec['cases'].append(dict(name=f'{name}_scout_{i}',camera=dict(x=x+dx,y=y+dy,z=z,yaw=yaw,pitch=pitch,roll=0),
                objects=[],probe_native_floor=i>0,region_id=name,split=region['split']))
        dest=output/(name+'.json');dest.write_text(json.dumps(spec,indent=2),encoding='utf-8')
        manifest.append(dict(region_id=name,split=region['split'],spec_file=dest.name,
            spec_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),status='NOT_ADMITTED'))
    (output/'manifest.json').write_text(json.dumps(dict(regions=manifest,scope='No route admission or model execution'),indent=2),encoding='utf-8')


def summarize(capture,output):
    from PIL import Image,ImageDraw
    capture=Path(capture);output=Path(output)
    if output.exists():raise FileExistsError(output)
    receipt=json.loads((capture/'receipt.json').read_bytes());spec=json.loads((capture/'source/spec.json').read_bytes())
    if receipt['status']!='PASS':raise ValueError('Completed successful capture required')
    output.mkdir(parents=True)
    sheet=Image.new('RGB',(1280,((len(spec['cases'])+1)//2)*384),'#181818');draw=ImageDraw.Draw(sheet)
    for i,c in enumerate(spec['cases']):
        x,y=(i%2)*640,(i//2)*384
        with Image.open(capture/f'model/sample/{i:04d}.png') as im:sheet.paste(im.convert('RGB'),(x,y+24))
        draw.text((x+5,y+5),c['name'],fill='white')
    sheet.save(output/'overview.png')
    result=dict(status='REQUIRES_VISUAL_ROUTE_REVIEW',frames=receipt['frame_count'],
        native_floor_probes=receipt.get('native_floor_probes',[]),source_unchanged=receipt['source_unchanged'],
        scope='Source scout; successful transport does not establish routes or isolated backgrounds',
        receipt_sha256=hashlib.sha256((capture/'receipt.json').read_bytes()).hexdigest())
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def run_specs(args):
    """Run the same verified source specs on a host-local project; retain failures."""
    root=Path(__file__).resolve().parents[1];out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    manifest=json.loads(args.manifest.read_bytes());rows=manifest['regions']
    wanted=set(args.region or [r['region_id'] for r in rows])
    if wanted-{r['region_id'] for r in rows}:raise ValueError('Unknown region')
    live=subprocess.check_output(['pwsh','-NoProfile','-Command',
        "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'UnrealEditor*' }).Count"],text=True).strip()
    if live!='0':raise RuntimeError('Another Unreal process is active on this host')
    out.mkdir(parents=True);results=[]
    try:
        for row in rows:
            if row['region_id'] not in wanted:continue
            path=args.manifest.parent/row['spec_file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['spec_sha256']:raise ValueError('Scout spec identity changed')
            spec=json.loads(path.read_bytes())
            if spec.get('scope')!='SOURCE_ONLY_RECONNAISSANCE_NOT_RESEARCH_COHORT':raise ValueError('Source-only spec required')
            map_file=args.project.resolve().parent/'Content'/Path(spec['map_asset'][len('/Game/'):]+'.umap')
            if hashlib.sha256(map_file.read_bytes()).hexdigest()!=spec['map_sha256']:raise ValueError('Worker map identity mismatch')
            spec['map_file']=str(map_file)
            spec_path=out/(row['region_id']+'-spec.json');spec_path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
            capture=out/row['region_id']
            subprocess.run([sys.executable,str(root/'tools/run_city_pcg_capture.py'),'--project',str(args.project),
                '--plugin',str(args.plugin),'--engine',str(args.engine),'--spec',str(spec_path),'--output',str(capture),
                '--timeout',str(args.timeout)],check=True,cwd=root)
            summarize(capture,out/(row['region_id']+'-review'))
            results.append(dict(region_id=row['region_id'],status='PASS',capture=str(capture)))
            (out/'progress.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    except BaseException as exc:
        (out/'completion.json').write_text(json.dumps(dict(status='FAIL',completed=results,error=str(exc)),indent=2),encoding='utf-8')
        raise
    (out/'completion.json').write_text(json.dumps(dict(status='PASS',completed=results,
        scope='Source reconnaissance; route and background admission remains separate'),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--candidates',type=Path,required=True);a.add_argument('--project',type=Path,required=True)
    a.add_argument('--template',type=Path,default=Path('experiments/city-field/capture-template-v1.json'));a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('summarize');a.add_argument('--capture',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--manifest',type=Path,required=True);a.add_argument('--region',action='append')
    for key in ('project','plugin','engine','output'):a.add_argument('--'+key,type=Path,required=True)
    a.add_argument('--timeout',type=float,default=1800)
    a=p.parse_args()
    canonical=(Path(__file__).resolve().parents[1]/'artifacts.local').resolve()
    if not a.output.resolve().is_relative_to(canonical):raise ValueError('Canonical artifact output required')
    if a.command=='prepare':prepare(json.loads(a.candidates.read_bytes()),json.loads(a.template.read_bytes()),a.project,a.output)
    elif a.command=='summarize':summarize(a.capture,a.output)
    else:run_specs(a)
