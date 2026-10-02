"""RGB-only frozen relative depth, and quantity-preserving zone readout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import h5py
import numpy as np
import cnh_zju_depthor_stages as stages

ROOT=stages.ROOT
OUT=ROOT/'artifacts.local/work/cnh-zju-relative-reorder-20261003'
RUN_ID='CNH_ZJU_RELATIVE_REORDER_20261003'
ARMS=('final','relative_reorder','shuffle')
read,save,sha=stages.read,stages.save,stages.sha


def prepare():
    assert not (OUT/'PLAN.json').exists()
    old=read(stages.OUT/'PLAN.json');seal=read(stages.OUT/'prediction-seal.json')
    assert seal['plan_sha256']==sha(stages.OUT/'PLAN.json')
    rows=[dict(r,stages_sha256=seal['files'][r['stages']],relative=str(OUT/'predictions'/f'{i:03d}-relative.npz')) for i,r in enumerate(old['rows'])]
    deps={str(p):sha(p) for p in (Path(__file__),Path(stages.__file__),Path(stages.prior.grid.__file__),
        stages.OUT/'PLAN.json',stages.OUT/'prediction-seal.json',Path(__file__).with_name('mz140_depthor.py'),stages.prior.OLD/'depthor-zju-small.pt')}
    deps.update({str(p):sha(p) for p in sorted((stages.prior.OLD/'upstream').rglob('*.py'))})
    OUT.mkdir(parents=True,exist_ok=True)
    row=f'| 2026-10-03 | {RUN_ID} | PRE_RUN; same300 consumed ZJU140cal/160eval, original embedded RGB relative module normalize/resize no ToF/GT; final distances conserved per public grid, assign sorted distances by relative rank vs fixedshuffle/final | NOT_RUN; all5old thresholds main/closer/far paired, quantity-only lowerbounds, near-far ordering .1/.5m; primary worst10/20 both main/closer nondecrease far nonincrease +one strict improvement, >=6/8scene three-way nonworse each | Pass REORDERING_COMPONENT_DEV else STOP_FIXED_REORDERING; relative scale not metres, reference never allocates candidates, no scene deletion/threshold tuning, zone-dependent scores not coherent global depth/bodyalert; no confirmation | `artifacts.local/work/cnh-zju-relative-reorder-20261003/REPORT.md` |\n'
    runs=stages.prior.grid.RUNS;body=runs.read_text(encoding='utf-8');assert RUN_ID not in body
    runs.write_text(body.rstrip()+'\n'+row,encoding='utf-8');(OUT/'prerun-row.txt').write_text(row,encoding='utf-8')
    save(OUT/'PLAN.json',dict(run_id=RUN_ID,rows=rows,thresholds=old['thresholds'],dependencies=deps,case=old['case'],preregistration=row,
        rules=dict(arms=ARMS,relative='Original compute_rel_depth with normalized RGB and zju_s resize, embedded checkpoint DA; no ToF input or GT; relative low=near',
            reorder='Within every clipped unique public8x8 zone nodes sort final ascending and assign ascending by relative; stable native lexicographic coordinate ties. Shuffle SHA256(run|id|zone) PCG64 permutation.',
            unknown='Every node retained, no GT filtering in allocation. Unknown separate; any nonfinite relative/stage causes NOT_EVALUABLE execution stop with evidence retained, no filtering. Repeated overlapping zones retained.',
            decision='Both oldworst10/20: eval mainTP>=final,closerTP>=final,farFP<=final,one strict; at least6/8scene same3 nonworse per budget. No numerical posthoc tolerance/retune. Not-evaluable if any main/closer/far denominator absent.',
            bounds='For each valid zone Kpredictednear, Nknownnear,Uunknown: unavoidablefar=max(0,K-N-U), minimumnearFN=max(0,N-K). GT only evaluator.',
            pairs='all within-zone known near<2.1 vsfar>=2.1 pairs with separation>=.1/.5; sign correct/tie/wrong of rgb/pre/final and joint27cells; correlated, descriptive',
            limits='Consumed Development; same scenes/model selection histories; zone readout not full-depth image; no ToF-independent causal attribution, metric scale, physical return origin/bodyclear/safety claim')))
    print('PREPARED300',flush=True)


def verify(p):
    for path,digest in p['dependencies'].items():assert sha(path)==digest,path


def predict():
    p=read(OUT/'PLAN.json');verify(p);assert not (OUT/'prediction-seal.json').exists()
    import torch
    from mz140_depthor import load_model
    torch.set_num_threads(4);assert torch.cuda.is_available()
    model,info=load_model(stages.prior.OLD/'upstream',stages.prior.OLD/'depthor-zju-small.pt','cuda')
    from src.utils.set_mde import compute_rel_depth,transforms_cfg
    (OUT/'predictions').mkdir(exist_ok=True);start=time.perf_counter();times=[]
    try:
        with torch.inference_mode():
            for i,r in enumerate(p['rows']):
                target=Path(r['relative']);assert not target.exists() and sha(r['input'])==r['input_sha256']
                with h5py.File(r['input'],'r') as f:rgb=f['rgb'][:]
                image=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].cuda()/255
                torch.cuda.synchronize();t=time.perf_counter()
                feature,rel,inv=compute_rel_depth(model.depth_anything,(image-model._mean)/model._std,**transforms_cfg['zju_s'])
                torch.cuda.synchronize()
                value=rel[0,0].cpu().numpy();inverse=inv[0,0].cpu().numpy()
                assert value.shape==(480,640) and np.isfinite(value).all() and value.min()>=0 and value.max()<=1
                assert np.isfinite(inverse).all()
                np.savez_compressed(target,relative=value,inverse=inverse)
                times.append(dict(id=r['id'],ms=1000*(time.perf_counter()-t)))
                if (i+1)%50==0:print(f'RGB_RELATIVE {i+1}/300',flush=True)
        save(OUT/'prediction-seal.json',dict(plan_sha256=sha(OUT/'PLAN.json'),files={r['relative']:sha(r['relative']) for r in p['rows']}))
        save(OUT/'runtime.json',dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,model=info,seconds=time.perf_counter()-start,frames=times,
            backend='Prior CUDA backend reused; scoring on CPU TASK_NOT_GPU_SUITABLE; no full fusion forward or ToF input'))
    finally:
        del model
        torch.cuda.empty_cache()
    print('SEALED_RGB300',flush=True)


def assign(values,relative,key):
    # Stable coordinate order resolves relative ties; no reference is accepted.
    out=np.empty_like(values);out[np.argsort(relative,kind='stable')]=np.sort(values,kind='stable')
    generator=np.random.default_rng(int.from_bytes(__import__('hashlib').sha256((RUN_ID+'|'+key).encode()).digest()[:8],'little'))
    shuffled=values[generator.permutation(len(values))]
    assert np.array_equal(np.sort(out),np.sort(values)) and np.array_equal(np.sort(shuffled),np.sort(values))
    return dict(final=values,relative_reorder=out,shuffle=shuffled)


def frame(r,p,seal):
    for key in ('input','stages'):assert sha(r[key])==r[key+'_sha256']
    assert sha(r['relative'])==seal['files'][r['relative']]
    with np.load(r['relative']) as f:relative=f['relative']
    with np.load(r['stages']) as f:pre,final=f['before'],f['final']
    assert all(v.shape==(480,640) and np.isfinite(v).all() for v in (relative,pre,final)), 'Nonfinite stage: stop without filtering nodes'
    with h5py.File(r['input'],'r') as f:boxes,mask,returns=f['fr'][:],f['mask'][:],f['hist_data'][:,0]
    observations=[]
    for k,box in enumerate(boxes):
        n=stages.prior.grid.public_nodes(box)[0];a,b=n[:,0],n[:,1]
        observations.append(dict(nodes=n,relative=relative[a,b],pre=pre[a,b],
            arms=assign(final[a,b],relative[a,b],r['id']+'|'+str(k)),valid=bool(mask[k] and np.isfinite(returns[k]) and .001<returns[k]<10)))
    with h5py.File(r['input'],'r') as f:gt=f['depth'][:]
    zones=[]
    for k,o in enumerate(observations):
        n=o['nodes'];d=gt[n[:,0],n[:,1]];known=np.isfinite(d)&(d>.001)&(d<10)
        near=known&(d<2.1);far=known&(d>=2.1);main=near&(d>=1.2);closer=near&~main
        table={}
        for name,threshold in p['thresholds'].items():
            calls={arm:v<threshold for arm,v in o['arms'].items()};base=calls['final'];K=int(base.sum());N=int(near.sum());U=int((~known).sum())
            assert all(int(c.sum())==K for c in calls.values()), 'Threshold call quantity changed'
            table[name]=dict(bounds=dict(unavoidable_far=max(0,K-N-U),minimum_near_fn=max(0,N-K),K=K),
                arms={arm:{label:dict(n=int(m.sum()),tp=int((c&m).sum()),added=int((c&~base&m).sum()),removed=int((~c&base&m).sum()))
                    for label,m in (('main',main),('closer',closer),('far',far),('unknown',~known))} for arm,c in calls.items()})
        ni,fi=np.meshgrid(np.flatnonzero(near),np.flatnonzero(far),indexing='ij');ni=ni.ravel();fi=fi.ravel()
        pair={}
        for margin in (.1,.5):
            take=d[fi]-d[ni]>=margin
            states=[np.sign(v[fi]-v[ni]).astype(int)+1 for v in (o['relative'],o['pre'],o['arms']['final'])]
            joint=states[0]*9+states[1]*3+states[2]
            pair[str(margin)]=np.bincount(joint[take],minlength=27).tolist()
        zones.append(dict(zone=k,valid=o['valid'],nodes=len(n),near=int(near.sum()),far=int(far.sum()),unknown=int((~known).sum()),
            mixed=bool(near.any() and far.any()),thin=bool(1<=near.sum()<=8 and far.any()),table=table,pairs=pair))
    return dict(id=r['id'],scene=r['scene'],role=r['role'],zones=zones)


def summarize(frames,p):
    allzones=[z for f in frames for z in f['zones']];zs=[z for z in allzones if z['valid']]
    output=dict(frames=len(frames),zones=len(allzones),valid_zones=len(zs),all_nodes=sum(z['nodes'] for z in allzones),groups={})
    for group,items in (('all',zs),('mixed',[z for z in zs if z['mixed']]),('thin',[z for z in zs if z['thin']])):
        output['groups'][group]=dict(zones=len(items),nodes=sum(z['nodes'] for z in items),near=sum(z['near'] for z in items),far=sum(z['far'] for z in items),unknown=sum(z['unknown'] for z in items),
            table={name:dict(bounds={k:sum(z['table'][name]['bounds'][k] for z in items) for k in ('unavoidable_far','minimum_near_fn','K')},
                arms={arm:{label:{k:sum(z['table'][name]['arms'][arm][label][k] for z in items) for k in ('n','tp','added','removed')}
                    for label in ('main','closer','far','unknown')} for arm in ARMS}) for name in p['thresholds']},
            pairs={str(m):np.sum([z['pairs'][str(m)] for z in items],axis=0,dtype=np.int64).tolist() if items else [0]*27 for m in (.1,.5)})
    return output


def dominates(table):
    a,b=table['arms']['relative_reorder'],table['arms']['final']
    return a['main']['tp']>=b['main']['tp'] and a['closer']['tp']>=b['closer']['tp'] and a['far']['tp']<=b['far']['tp']


def evaluate():
    p=read(OUT/'PLAN.json');verify(p);assert not (OUT/'result.json').exists()
    seal=read(OUT/'prediction-seal.json');assert seal['plan_sha256']==sha(OUT/'PLAN.json')
    with ThreadPoolExecutor(max_workers=4) as pool:frames=list(pool.map(lambda r:frame(r,p,seal),p['rows']))
    roles={s:summarize([f for f in frames if f['role']==s],p) for s in ('cal','eval')}
    scenes={s:summarize([f for f in frames if f['scene']==s],p) for s in sorted({f['scene'] for f in frames})}
    gates={};supported=True
    for name in ('worst_scene_0.1','worst_scene_0.2'):
        table=roles['eval']['groups']['all']['table'][name];a,b=table['arms']['relative_reorder'],table['arms']['final']
        supported &= all(b[k]['n']>0 for k in ('main','closer','far'))
        wins=[s for s in scenes if any(f['scene']==s and f['role']=='eval' for f in frames)
            and all(scenes[s]['groups']['all']['table'][name]['arms']['final'][k]['n']>0 for k in ('main','closer','far'))
            and dominates(scenes[s]['groups']['all']['table'][name])]
        gates[name]=dict(nonworse=dominates(table),strict=any(a[k]['tp']!=b[k]['tp'] for k in ('main','closer','far')),scene_nonworse=wins,scene_gate=len(wins)>=6)
    passed=all(g['nonworse'] and g['strict'] and g['scene_gate'] for g in gates.values())
    verdict=('REORDERING_COMPONENT_DEV' if passed else 'STOP_FIXED_REORDERING') if supported else 'NOT_EVALUABLE'
    save(OUT/'frame-ledger.json',frames)
    save(OUT/'result.json',dict(verdict=verdict,gates=gates,roles=roles,scenes=scenes,
        case=summarize([f for f in frames if f['id']==p['case']],p)))
    print(verdict,gates,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','predict','evaluate'))
    globals()[parser.parse_args().action]()
