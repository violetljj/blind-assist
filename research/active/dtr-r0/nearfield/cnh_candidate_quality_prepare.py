"""Task-owned repaired-v2 Development preparation; no outcome metrics."""
import concurrent.futures as cf
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import cnh_track_a_v13_sensor as sensor
BASE=Path(__file__).resolve().parents[4]
ROOT=BASE/'artifacts.local/work/cnh-track-a-scale-v2-20260926-dev-repaired'
SOURCE=BASE/'artifacts.local/evidence/cnh-track-a-scale-v2-20260926-v1/geometry'
FAMILY='cnh-track-a-scale-v2-20260926'
START=time.monotonic()
def write(name,obj):
    (ROOT/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('unit*/unit*.json'))}
def budget():
    if time.monotonic()-START>2700: raise RuntimeError('45 minute preparation budget')
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if size>8*1024**3: raise RuntimeError('8 GiB storage budget')
    return size
def command(script,args,log):
    with (ROOT/log).open('w',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,str(Path(__file__).parent/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f'{script} failed {r.returncode}')
def render(u):
    command('cnh_track_a_v13_sensor.py',['--geometry',ROOT/'geometry','--output',ROOT/'sensor','--unit',u,'--mount',-10,'--family',FAMILY,'--rate',5,'--oracle-calib'],f'logs/unit{u:02d}.log')
    return u
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'logs').mkdir(exist_ok=True)
    if (ROOT/'geometry').exists(): raise RuntimeError('Refuse overwrite existing geometry')
    before=hashes(SOURCE)
    write('source_hashes_before.json',before)
    write('provenance.json',dict(source=str(SOURCE),destination=str(ROOT/'geometry'),
          source_geometry_bytes=sum(p.stat().st_size for p in SOURCE.rglob('*') if p.is_file()),
          original_readout_files=len(list((SOURCE.parent/'readouts').rglob('*'))),
          authority='v2 repaired Development; historical failed source, not fresh formal data',
          family=FAMILY,split_counts=[96,32,64],excluded_original_incomplete=[143]))
    write('progress.json',dict(stage='copy',completed=0,total=95,pid=os.getpid(),backend='CPU TASK_NOT_GPU_SUITABLE',workers=4))
    shutil.copytree(SOURCE,ROOT/'geometry')
    command('cnh_track_a_v13_repair.py',['--root',ROOT/'geometry','--n-units',192,'--family',FAMILY,'--split-counts',96,32,64,'--fast-margin'],'logs/repair.log')
    command('cnh_track_a_v13_merge.py',['--root',ROOT/'geometry','--n-units',192,'--v2-gates','--split-counts',96,32,64],'logs/merge.log')
    gate=json.loads((ROOT/'geometry/result.json').read_text())
    if gate['status']!='GEOMETRY_READY': raise RuntimeError('Scientific geometry gate failed')
    units=[u for u in range(96,192) if (ROOT/f'geometry/unit{u:02d}/unit{u:02d}.json').exists()]
    if len(units)!=95 or 143 in units: raise RuntimeError('Unexpected cohort')
    sensor.FAMILY=FAMILY; sensor.FORCE_RATE=5
    params,g3=sensor.reference_parameters()
    if not g3['pass_gate']: raise RuntimeError('G3 failure')
    c=json.loads((ROOT/'geometry/unit96/unit96.json').read_text())['configs'][0]
    a,no=sensor.render_config(c,-10,params,5,False)
    b,yes=sensor.render_config(c,-10,params,5,True)
    equality={k:bool(np.array_equal(a[k],b[k])) for k in a}
    if not all(equality.values()) or no is not None or yes is None: raise RuntimeError('Oracle toggle changes observations')
    # Exercise run default and explicit option on a one-config calib geometry.
    cg=ROOT/'canary/geometry/unit96'; cg.mkdir(parents=True)
    d=json.loads((ROOT/'geometry/unit96/unit96.json').read_text());d['configs']=d['configs'][:1]
    (cg/'unit96.json').write_text(json.dumps(d))
    sensor.run(cg.parent,ROOT/'canary/default',96,-10)
    sensor.run(cg.parent,ROOT/'canary/optin',96,-10,True)
    if (ROOT/'canary/default/unit96-mount-10-oracle.npz').exists(): raise RuntimeError('Default calib oracle exists')
    with np.load(ROOT/'canary/default/unit96-mount-10-observations.npz') as x, np.load(ROOT/'canary/optin/unit96-mount-10-observations.npz') as y:
        if not all(np.array_equal(x[k],y[k]) for k in x.files): raise RuntimeError('run compatibility failed')
    write('canary.json',dict(observation_arrays_equal=equality,default_calib_oracle=False,optin_calib_oracle=True,G3=g3,elapsed_s=time.monotonic()-START))
    done=[]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(render,u):u for u in units}
        for f in cf.as_completed(pending):
            done.append(f.result()); size=budget()
            write('progress.json',dict(stage='sensor',completed=len(done),total=len(units),units=sorted(done),pid=os.getpid(),bytes=size,elapsed_s=time.monotonic()-START))
    frames=0;problems=[]
    for u in units:
        g=json.loads((ROOT/f'geometry/unit{u:02d}/unit{u:02d}.json').read_text())
        meta=json.loads((ROOT/f'sensor/unit{u:02d}-mount-10.json').read_text())
        if meta['G3']!=g3 or not meta['oracle']: problems.append([u,'meta'])
        with np.load(ROOT/f'sensor/unit{u:02d}-mount-10-observations.npz') as a,np.load(ROOT/f'sensor/unit{u:02d}-mount-10-oracle.npz') as o:
            if int(a['rate'])!=5 or a['hist'].shape[0]!=3 or not np.isfinite(a['hist']).all():problems.append([u,'rate/snr/finite'])
            for c in g['configs']:
                sel=a['config']==c['config']
                if sel.sum()!=len(c['labels']) or not np.allclose(a['world_from_Q'][sel],c['world_from_Q']):problems.append([u,c['config'],'pairing'])
            if len(o['object_id'])!=len(a['config']):problems.append([u,'oracle frames'])
            frames+=a['hist'].shape[1]
    after=hashes(SOURCE);write('source_hashes_after.json',after)
    if before!=after: raise RuntimeError('Original source changed')
    write('sensor_gates.json',dict(G3=g3['pass_gate'],G4=not problems,problems=problems,units=units,frames=frames,mounts=[-10],scope='calib/audit only; train intentionally not rendered'))
    if problems:raise RuntimeError('G4 failed')
    write('terminal.json',dict(status='PREPARED',source_unchanged=True,source_geometry_json_files=len(before),calib=32,audit=63,excluded=[143],elapsed_s=time.monotonic()-START,bytes=budget(),backend='CPU TASK_NOT_GPU_SUITABLE',workers=4))
if __name__=='__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    try:main()
    except BaseException as e:
        ROOT.mkdir(parents=True,exist_ok=True);write('terminal.json',dict(status='FAILED',error=repr(e),elapsed_s=time.monotonic()-START));raise
