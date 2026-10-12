"""Frozen Development replay of two bin1 residual hypotheses, no default changes."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import real_noise_ablation_diagnosis as D
ROOT=D.P.ROOT
OUT=ROOT/'artifacts.local/work/cnh-near-residual-dev-20261012/compensation'
QUIET=('phone_100702','phone_100754','phone_103924','dorm_4','dorm_4b','dorm_9')
MANIFEST=ROOT/'artifacts.local/work/cnh-pose-floor-noise-dr-dev-20261012/real_manifest.json'
def save(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def transform(z,method,template=None):
    """Any (...,8,8,16) input. Current-frame only; no output clipping.
    fixed_template requires separately frozen training-recording 8x8 template.
    """
    out=np.array(z,copy=True); b=np.asarray(z[...,1],np.float32)
    if method=='spatial_median': estimate=np.maximum(np.median(b,axis=(-2,-1),keepdims=True),0)
    elif method=='fixed_template':
        if template is None or np.shape(template)!=(8,8):raise ValueError('8x8 training template required')
        estimate=np.asarray(template,np.float32)
    else:raise ValueError(method)
    out[...,1]=b-estimate
    return out

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(lane='EXPLORE consumed Development',gpu_command_wall_budget_s=900,cpu_command_wall_budget_s=1800,
      methods=['spatial_median','fixed_template'],models_cuts='frozen joint manifest',
      spatial='Subtract max(0,current-frame 64zone bin1 median), no rectification of residual.',
      template='Per quiet recording time median bin1; equal-record median templates; leave current quiet recording out. Nonquiet uses all six. Clip template >=0 only; residual signed.',
      quiet_recordings=QUIET,scope='36 existing real pose cells, no training/new capture. No target labels select magnitude.',
      limits='Spatial median can subtract broad real objects; fixed template can oversubtract lower-offset runs. Real adapter residual asymmetric, do not interpret centred input as recovered physical noise.')
    save(OUT/'PLAN.json',plan)
    templates={n:np.median(np.load(D.DEFAULT/'inputs'/f'{n}.npz')['z'][...,1],axis=0) for n in QUIET}
    arrays={'all':np.maximum(np.median(np.stack(list(templates.values())),axis=0),0)}
    arrays.update({n:np.maximum(np.median(np.stack([v for k,v in templates.items() if k!=n]),axis=0),0) for n in QUIET})
    np.savez_compressed(OUT/'templates.npz',**arrays)
    stats={}
    for path in sorted((D.DEFAULT/'inputs').glob('*.npz')):
        z=np.load(path)['z'];name=path.stem;row={}
        for method in ('original','spatial_median','fixed_template'):
            a=z if method=='original' else transform(z,method,arrays.get(name,arrays['all']))
            b=a[...,1];diff=np.diff(b,axis=0)
            row[method]=dict(mean=float(b.mean()),negative_fraction=float((b<0).mean()),zero_fraction=float((b==0).mean()),
             quantiles=np.quantile(b,[0,.01,.1,.5,.9,.99,1]).tolist(),median_zone_time_mean=float(np.median(b.mean(0))),
             median_difference_sd=float(np.median(np.sqrt(np.mean(diff**2,axis=0)/2))),
             positive_mass=float(np.maximum(b,0).sum()),unchanged_other_bins=bool(np.array_equal(a[...,[0,*range(2,16)]],z[...,[0,*range(2,16)]])))
        stats[name]=row
    save(OUT/'input_statistics.json',stats)
    print('PREPARED',len(stats),flush=True)

def run(device):
    import torch
    from local_signed_bias_adapter import ContractReplay
    torch.set_num_threads(4);started=time.monotonic()
    templates=dict(np.load(OUT/'templates.npz'))
    cells=[c for c in json.loads((D.DEFAULT/'cells.json').read_text()) if not c.get('drop_rows') and not c.get('diagnostic')]
    receipt=dict(device=device,manifest=str(MANIFEST),manifest_sha256=D.P.sha(MANIFEST),cells=len(cells),status='running',gpu_name=torch.cuda.get_device_name() if device=='cuda' else None)
    save(OUT/'receipt.json',receipt)
    engine=ContractReplay(str(MANIFEST),device=device)
    try:
      for method in ('spatial_median','fixed_template'):
        folder=OUT/method;folder.mkdir(exist_ok=True);records=[]
        for c in cells:
          dest=folder/(c['id']+'.npz');records.append(dict(c,npz=str(dest)))
          if dest.exists():continue
          if time.monotonic()-started>850:raise RuntimeError('GPU budget margin stop')
          d=np.load(D.DEFAULT/'inputs'/(c['input']+'.npz'))
          z=transform(d['z'],method,templates.get(c['input'],templates['all']))
          engine.engine.configure_geometry(*c['pose'])
          result=engine.infer(z,d['seq'],d['ms'])
          np.savez_compressed(dest,**result,seq=d['seq'],ms=d['ms'])
          print(method,c['id'],round(time.monotonic()-started,2),flush=True)
        save(folder/'index.json',dict(records=records,status='complete'))
    finally:
      engine.close();torch.cuda.empty_cache()
      receipt.update(seconds=time.monotonic()-started,status='released');save(OUT/'receipt.json',receipt)
    summarize()

def summarize():
    report={}
    for method in ('spatial_median','fixed_template'):
      rows=[]
      for c in json.loads((OUT/method/'index.json').read_text())['records']:
        a=np.load(D.DEFAULT/'joint'/(c['id']+'.npz'));b=np.load(c['npz']);ready=a['ready']
        assert np.array_equal(ready,b['ready']) and np.array_equal(a['seq'],b['seq'])
        x,y=a['S_grade'][ready],b['S_grade'][ready]
        row=dict(id=c['id'],input=c['input'],pose=c['pose'],expect=c['expect'],ready=int(ready.sum()))
        for mode in ('strong','light','any'):
          u=x==2 if mode=='strong' else x==1 if mode=='light' else x>0
          v=y==2 if mode=='strong' else y==1 if mode=='light' else y>0
          row[mode]=dict(original=u.sum(0).tolist(),candidate=v.sum(0).tolist(),rescue=(~u&v).sum(0).tolist(),loss=(u&~v).sum(0).tolist())
          assert np.array_equal(u.sum(0)+(~u&v).sum(0)-(u&~v).sum(0),v.sum(0))
        rows.append(row)
      report[method]=rows
    save(OUT/'comparison.json',report)
    print('COMPARISON_READY',flush=True)
def validate_backend():
    import torch
    from local_signed_bias_adapter import ContractReplay
    torch.set_num_threads(4);start=time.monotonic();rows=[]
    engine=ContractReplay(str(MANIFEST),device='cuda')
    try:
      for c in json.loads((OUT/'spatial_median/index.json').read_text())['records']:
        d=np.load(D.DEFAULT/'inputs'/(c['input']+'.npz'))
        engine.engine.configure_geometry(*c['pose']);b=engine.infer(d['z'],d['seq'],d['ms'])
        a=np.load(D.DEFAULT/'joint'/(c['id']+'.npz'))
        row=dict(id=c['id'],ready_equal=bool(np.array_equal(a['ready'],b['ready'])),grade_equal=bool(np.array_equal(a['S_grade'],b['S_grade'])),max_m3_difference=float(np.max(np.abs(a['m3_score'][a['ready']]-b['m3_score'][a['ready']]))))
        rows.append(row)
        assert row['ready_equal'] and row['grade_equal'],row
    finally:
      engine.close();torch.cuda.empty_cache()
      save(OUT/'backend_validation.json',dict(seconds=time.monotonic()-start,rows=rows))
    print('BACKEND_VALIDATED',len(rows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary','validate']);ap.add_argument('--device',default='cuda');a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run(a.device)
    elif a.stage=='validate':validate_backend()
    else:summarize()
