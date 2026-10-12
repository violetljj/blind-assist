"""Additional real scenes for frozen DEPTHOR calibration, not new confirmation."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import h5py
import numpy as np
import cnh_zju_budget_readout as base
import cnh_zju_rank_localization as grid

ROOT=grid.ROOT
DATA=ROOT/'artifacts.local/datasets/zjul5/ZJUL5'
OLD=ROOT/'artifacts.local/work/mz140-depthor-20260915'
OUT=ROOT/'artifacts.local/work/cnh-zju-scene-calibration-20261002'
RUN_ID='CNH_ZJU_SCENE_CALIBRATION_20261002'
ARMS=('raw','tof_idw4','depthor')
KEYS=('sensor','known','near','main','far','mixed','thin',*ARMS)
read=grid.read
save=grid.save
sha=grid.sha


def prepare():
    assert not (OUT/'PLAN.json').exists()
    meta=read(DATA/'data.json');groups=defaultdict(list)
    for row in meta['train']:groups[row['filename'].split('/')[0]].append(row['filename'])
    assert len(groups)==7 and sum(map(len,groups.values()))==483
    cal=[]
    for scene,names in sorted(groups.items()):
        names=sorted(names)
        for k in np.unique(np.linspace(0,len(names)-1,20).astype(int)):
            name=names[k];path=DATA/name
            cal.append(dict(id=name,scene=scene,role='cal',input=str(path),input_sha256=sha(path),
                prediction=str(OUT/'predictions'/f'{len(cal):03d}-depthor.npz')))
    old=read(base.OUT/'PLAN.json');evaluation=[]
    for row in old['inputs']:
        evaluation.append(dict(id=row['id'],scene=row['scene'],role='eval',input=row['input'],input_sha256=row['input_sha256'],
            prediction=row['depthor'],prediction_sha256=row['depthor_sha256']))
    assert len(cal)==140 and len(evaluation)==160 and not ({r['scene'] for r in cal}&{r['scene'] for r in evaluation})
    assert sha(OLD/'depthor-zju-small.pt')==read(grid.OLD/'protocol.json')['checkpoint_sha256']
    dependencies={str(x):sha(x) for x in (Path(__file__),Path(base.__file__),Path(grid.__file__),Path(base.anchor.__file__),base.OUT/'PLAN.json',
        Path(__file__).with_name('mz140_depthor.py'),DATA/'data.json',OLD/'depthor-zju-small.pt')}
    dependencies.update({str(x):sha(x) for x in sorted((OLD/'upstream').rglob('*.py'))})
    OUT.mkdir(parents=True,exist_ok=True)
    line=f'| 2026-10-02 | {RUN_ID} | PRE_RUN; official train7scenes x20 lexical evenly spaced=140 additional RGB calibration, existing test160/8scenes eval; all consumed Development incl old full SNR truth access; original DEPTHOR strict frozen new140 predictions only, raw/IDW4; no train/download | NOT_RUN; cal10/20%far: pooled control vs primary worst-scene minimum threshold; eval main1.2-2.1/closer<1.2/FAR>=2.1 all same sensor domain, scene costs; no eval tuning | Primary both budgets: main>=calbestToF+3pp, closer>=ToF, pooledfar<=ToF and every8eval scene<=budget, >=6/8main wins. Pass SCENE_CALIBRATED_COMPONENT_DEV else SCENE_CALIBRATION_NOT_ESTABLISHED; no body/safety/freshconfirmation claim. Preserve all failures and no same-cohort scalar retuning | `artifacts.local/work/cnh-zju-scene-calibration-20261002/REPORT.md` |\n'
    body=grid.RUNS.read_text(encoding='utf-8');assert RUN_ID not in body
    grid.RUNS.write_text(body.rstrip()+'\n'+line,encoding='utf-8');(OUT/'prerun-row.txt').write_text(line,encoding='utf-8')
    save(OUT/'PLAN.json',dict(run_id=RUN_ID,cal=cal,eval=evaluation,dependencies=dependencies,
        rules=dict(arms=ARMS,budgets=[.1,.2],policies=['pooled','worst_scene'],primary='worst_scene',
            prediction='Original mz140_depthor.load_model and official dtof_to_sparse_depth; RGB480x640/255, output[1], clip.001..10 exactly old ba_depth_probe; prediction stage never opens depth',
            sampling='Each official train scene 20 evenly spaced lexical names; all seven, no outcome selection; oldtest160 unchanged',
            calibration='strict score<threshold, scene far budgets; worst_scene min finite allowed threshold, all_finite if all scenes allow; undefined scene far => NOT_EVALUABLE; ToF chosen cal maxmain/lowestfar/raw first',
            gate='Both budgets primary: main>=ToF+.03, closer>=ToF, pooledfar<=ToF, all8eval scenes far<=budget, >=6/8scene main wins; undefined => NOT_EVALUABLE; otherwise fail SCENE_CALIBRATION_NOT_ESTABLISHED',
            boundaries='All project consumed Development. Official DEPTHOR config trains Hypersim, validates ZJU; no proof base models never saw scenes. Different scene names not independent physical-layout proof. No body pose/ToF extrinsics/return source; far pixels not body clear costs. No model or old-source-result promotion'),preregistration=line))
    print('PREPARED140cal160eval',flush=True)


def verify(plan):
    for path,digest in plan['dependencies'].items():assert sha(path)==digest,path


def predict():
    p=read(OUT/'PLAN.json');verify(p);assert not (OUT/'prediction-seal.json').exists()
    import torch
    from mz140_depthor import load_model
    torch.set_num_threads(4);assert torch.cuda.is_available()
    model,info=load_model(OLD/'upstream',OLD/'depthor-zju-small.pt','cuda')
    from src.utils.dataloader import dtof_to_sparse_depth
    (OUT/'predictions').mkdir(exist_ok=True);timings=[];start=time.perf_counter()
    try:
        with torch.inference_mode():
            for i,row in enumerate(p['cal']):
                target=Path(row['prediction']);assert not target.exists()
                assert sha(row['input'])==row['input_sha256']
                with h5py.File(row['input'],'r') as f:rgb=f['rgb'][:];hist=f['hist_data'][:];fr=f['fr'][:];mask=f['mask'][:]
                sparse=dtof_to_sparse_depth(torch.tensor(hist).float(),torch.tensor(fr),torch.tensor(mask))
                image=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].cuda()/255.
                torch.cuda.synchronize();t=time.perf_counter()
                prediction=model(dict(image=image,sparse_depth=sparse[None].cuda()))[1]
                torch.cuda.synchronize();milliseconds=1000*(time.perf_counter()-t)
                value=prediction[0,0].float().cpu().numpy()
                assert value.shape==(480,640) and np.isfinite(value).all()
                np.savez_compressed(target,depth=np.clip(value,.001,10))
                timings.append(dict(id=row['id'],ms=milliseconds,sparse_points=int((sparse>0).sum())))
                if (i+1)%20==0:print(f'PREDICTED {i+1}/140',flush=True)
        save(OUT/'prediction-seal.json',dict(plan_sha256=sha(OUT/'PLAN.json'),files={str(Path(r['prediction'])):sha(r['prediction']) for r in p['cal']}))
        save(OUT/'runtime.json',dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,model=info,
            backend='CUDA original cached-model backend; equivalent previous workload GPU evidence reused; CPU I/O/evaluation TASK_NOT_GPU_SUITABLE',
            seconds=time.perf_counter()-start,frames=timings))
    finally:
        del model
        torch.cuda.empty_cache()
    print('SEALED140',flush=True)


def frame(row):
    assert sha(row['input'])==row['input_sha256'] and sha(row['prediction'])==row['prediction_sha256']
    with h5py.File(row['input'],'r') as f:boxes=f['fr'][:];mask=f['mask'][:];returns=f['hist_data'][:,0]
    with np.load(row['prediction']) as f:prediction=f['depth']
    nodes=[grid.public_nodes(box,prediction.shape)[0] for box in boxes]
    valid=mask&np.isfinite(returns)&(returns>.001)&(returns<10)
    centres=[];ranges=[]
    for k,box in enumerate(boxes):
        y0,x0,y1,x1=map(int,box);y0,y1=np.clip([y0,y1],0,480);x0,x1=np.clip([x0,x1],0,640)
        if valid[k] and y1>y0 and x1>x0:centres.append([(y0+y1-1)/2,(x0+x1-1)/2]);ranges.append(returns[k])
    centres=np.asarray(centres).reshape(-1,2);ranges=np.asarray(ranges)
    observations=[dict(raw=np.full(len(n),returns[k] if valid[k] else np.nan),tof_idw4=base.idw4(n,centres,ranges),
        depthor=prediction[n[:,0],n[:,1]].astype(float)) for k,n in enumerate(nodes)]
    with h5py.File(row['input'],'r') as f:gt=f['depth'][:]
    chunks={k:[] for k in KEYS}
    for k,n in enumerate(nodes):
        d=gt[n[:,0],n[:,1]];known=np.isfinite(d)&(d>.001)&(d<10);near=known&(d<2.1);far=known&(d>=2.1)
        for key,value in dict(sensor=np.full(len(n),valid[k]),known=known,near=near,main=near&(d>=1.2),far=far,
            mixed=np.full(len(n),near.any() and far.any()),thin=np.full(len(n),1<=near.sum()<=8 and far.any()),**observations[k]).items():chunks[key].append(value)
    return dict(id=row['id'],scene=row['scene'],role=row['role'],zones=64,empty=sum(len(n)==0 for n in nodes),
        invalid_sensor=int((~valid).sum()),**{k:np.concatenate(v) for k,v in chunks.items()})


def merge(frames):return {k:np.concatenate([f[k] for f in frames]) for k in KEYS}


def worst(rules):
    if not rules or any(not r['defined'] for r in rules):return dict(defined=False,all_finite=False,threshold=None)
    finite=[r['threshold'] for r in rules if not r['all_finite']]
    return dict(defined=True,all_finite=not finite,threshold=min(finite) if finite else None)


def paired(data,other,rules):
    a=base.calls(data['depthor'],rules['depthor']);b=base.calls(data[other],rules[other]);use=data['sensor']
    out={}
    for name,m in (('main',data['main']),('closer',data['near']&~data['main']),('far',data['far'])):
        out[name]=dict(added=int((use&m&a&~b).sum()),removed=int((use&m&~a&b).sum()))
    return out


def evaluate():
    p=read(OUT/'PLAN.json');verify(p);assert not (OUT/'result.json').exists()
    seal=read(OUT/'prediction-seal.json');assert seal['plan_sha256']==sha(OUT/'PLAN.json')
    rows=[dict(r,prediction_sha256=seal['files'][r['prediction']]) for r in p['cal']]+p['eval'];start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:frames=list(pool.map(frame,rows))
    cal=merge([f for f in frames if f['role']=='cal']);evaluation=merge([f for f in frames if f['role']=='eval'])
    cal_scenes=sorted({r['scene'] for r in p['cal']});eval_scenes=sorted({r['scene'] for r in p['eval']})
    scene={s:merge([f for f in frames if f['scene']==s]) for s in cal_scenes+eval_scenes}
    output={}
    for budget in (.1,.2):
        scene_rules={s:{a:base.select_threshold(scene[s][a],scene[s]['sensor']&scene[s]['far'],budget) for a in ARMS} for s in cal_scenes}
        policies=dict(pooled={a:base.select_threshold(cal[a],cal['sensor']&cal['far'],budget) for a in ARMS},
            worst_scene={a:worst([scene_rules[s][a] for s in cal_scenes]) for a in ARMS})
        answers={}
        for policy,rules in policies.items():
            c={a:base.counts(cal,a,rules[a]) for a in ARMS};e={a:base.counts(evaluation,a,rules[a]) for a in ARMS}
            sc={s:{a:base.counts(v,a,rules[a]) for a in ARMS} for s,v in scene.items()}
            eligible=[a for a in ARMS[:2] if c[a]['main_recall'] is not None and c[a]['far_fpr'] is not None and rules[a]['defined']]
            selected=max(eligible,key=lambda a:(c[a]['main_recall'],-c[a]['far_fpr'],-ARMS.index(a))) if eligible else None
            estimable=selected is not None and all(r['defined'] for r in rules.values()) and all(e[a][k] is not None for a in ('depthor',selected) for k in ('main_recall','closer_recall','far_fpr')) and all(sc[s]['depthor']['far_fpr'] is not None for s in eval_scenes)
            wins=[s for s in eval_scenes if selected is not None and sc[s]['depthor']['main_recall'] is not None and sc[s][selected]['main_recall'] is not None and sc[s]['depthor']['main_recall']>sc[s][selected]['main_recall']+1e-12]
            compliant=[s for s in eval_scenes if sc[s]['depthor']['far_fpr'] is not None and sc[s]['depthor']['far_fpr']<=budget+1e-12]
            gates=dict(main_gain=estimable and e['depthor']['main_recall']>=e[selected]['main_recall']+.03-1e-12,
                closer=estimable and e['depthor']['closer_recall']>=e[selected]['closer_recall']-1e-12,
                matched_cost=estimable and e['depthor']['far_fpr']<=e[selected]['far_fpr']+1e-12,
                all_scene_cost=len(compliant)==8,scene_wins=len(wins)>=6)
            answers[policy]=dict(rules=rules,cal=c,eval=e,scenes=sc,chosen_tof=selected,gates=gates,estimable=estimable,scene_wins=wins,cost_compliant=compliant,
                scene_paired={s:{a:paired(scene[s],a,rules) for a in ARMS[:2]} for s in eval_scenes},
                paired={a:paired(evaluation,a,rules) for a in ARMS[:2]},subgroups={g:{a:base.counts(evaluation,a,rules[a],g) for a in ARMS} for g in ('mixed','thin')})
        output[str(budget)]=dict(cal_scene_rules=scene_rules,policies=answers)
    principal=[o['policies']['worst_scene'] for o in output.values()]
    estimable=all(o['estimable'] for o in principal);passed=estimable and all(all(o['gates'].values()) for o in principal)
    verdict=('SCENE_CALIBRATED_COMPONENT_DEV' if passed else 'SCENE_CALIBRATION_NOT_ESTABLISHED') if estimable else 'NOT_EVALUABLE'
    save(OUT/'frame-ledger.json',[dict(id=f['id'],scene=f['scene'],role=f['role'],zones=f['zones'],empty=f['empty'],invalid_sensor=f['invalid_sensor'],
        counts={b:{policy:{a:base.counts(f,a,v['rules'][a]) for a in ARMS} for policy,v in out['policies'].items()} for b,out in output.items()}) for f in frames])
    save(OUT/'result.json',dict(verdict=verdict,budgets=output,cal_scenes=cal_scenes,eval_scenes=eval_scenes,seconds=time.perf_counter()-start))
    print(json.dumps(dict(verdict=verdict,gates={b:o['policies']['worst_scene']['gates'] for b,o in output.items()})),flush=True)


def selftest():
    rules=[dict(defined=True,all_finite=False,threshold=2.),dict(defined=True,all_finite=False,threshold=1.)]
    assert worst(rules)['threshold']==1
    assert worst([dict(defined=True,all_finite=True,threshold=None)])['all_finite']
    assert not worst([dict(defined=False)])['defined']
    print('PASS_WORST_SCENE_THRESHOLD')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','predict','evaluate','selftest'))
    globals()[parser.parse_args().action]()
