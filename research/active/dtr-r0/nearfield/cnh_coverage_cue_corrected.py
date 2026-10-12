"""Frozen corrected EXPLORE replay; legacy streams remain immutable."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import copy
import os
import time
import numpy as np
import cnh_coverage_cue_corrected_core as S

OUT = S.OUT
OLD = S.ROOT/'artifacts.local/work/cnh-coverage-cue-geometry-20261005'
MODELS = ('E_orig', 'E_oracle', 'E_grav')


def freeze():
    assert not (OUT/'PLAN.json').exists(), 'Never overwrite frozen plan'
    plan = copy.deepcopy(S.read(OLD/'PLAN.json'))
    plan.update(run='CNH_COVERAGE_CUE_CORRECTED_20261005',
                created_utc=datetime.now(timezone.utc).isoformat(),
                started_unix=time.time(), budget_seconds=5400,
                deadline_unix=time.time()+5400)
    plan['cells']['height_centers_m']['BODY'] = [.54]
    plan['cells']['per_side_height_arrivals_per_min'] = {'HEAD':1200, 'BODY':600}
    plan['cells']['excluded_vertical_boundary_y_m'] = .78
    plan['head_motion']['scans']['trajectory'] = '0 -> -20 -> +20 -> 0; inherited executed 6s trajectory'
    plan['models'] = {
        'E_orig':'Old stored closed loops, metric-only reaggregation; legacy v1 still used 16 cells. Historical confounded comparator, excluded from corrected software choice.',
        'E_oracle':'Current true rotation and position for controllers; estimated-angle formula unchanged. Software logic reference, not deployable; future evaluator-only.',
        'E_grav':'Inherited scale, translation and local N(0,0.1deg) step innovations. Keep inherited bias Y sign, remove X/Z constant biases; constant 1deg/s rotation LEFT-multiplied about world gravity Y. No new correction/reset of accumulated step noise. Gravity-referenced synthetic AHRS approximation, not calibrated device model.'}
    plan['interpretation'] = {
        'main':'12 yaw-restorable cells only; assert zero k3=0. Elimination=(none-policy)/(none-zero), per sigma/tau K4 mean. Actual auditory events (both sides=one) and unnecessary side tags/min.',
        'frontier':'For each model and sigma/tau cell, minimum residual among non-diagnostic software configurations with mean auditory events<=5/10/20 per min; no interpolation. All 80 configurations retained. Zero-latency diagnostics excluded.',
        'decision':'At <=10/min: >=8/12 cells software>=dual-15pp -> SOFTWARE_COMPETITIVE_SYNTH; else >=8/12 cells software<=dual-30pp -> GEOMETRY_DOMINATED_SYNTH; else MIXED. No feasible configuration = NOT_COMPARABLE, not a zero-elimination observation.',
        'estimator':'At same budget, |E_grav best elimination - E_oracle best elimination|<=5pp in >=8/12 cells -> ESTIMATION_NOT_BOTTLENECK_SYNTH. Compare selected configurations and matched configuration differences separately.',
        'orig':'E_orig branch computed descriptively only; old v1 defect preserved per metric-only instruction.',
        'scope':'Synthetic descriptive geometry/control, not hardware, safety, real burden, product thresholds or inference/training.'}
    plan['budget_policy'] = '90min from corrected freeze; reserve300s for delivery. If projected runtime too long, drop T=.4 & p=1/3, then L=1.0 globally in order, retain NOT_RUN configurations. No result-driven tuning.'
    plan['inputs_sha256'] = {str(p.relative_to(S.ROOT)):S.sha(p) for p in [OLD/'PLAN.json',OLD/'independent_yaw0_geometry.json',OLD/'step1_gate.json',OLD/'geometry_engineering_check.json']}
    plan['legacy_ledgers_sha256'] = {p.name:S.sha(p) for p in sorted((OLD/'streams').glob('*.npz'))}
    assert len(plan['legacy_ledgers_sha256'])==48
    plan['source_sha256'] = {Path(p).name:S.sha(p) for p in [__file__,S.__file__,S.G.__file__]}
    S.save(OUT/'PLAN.json',plan)
    (OUT/'PLAN.sha256').write_text(S.sha(OUT/'PLAN.json')+'\n',encoding='utf8')
    print('FROZEN',S.sha(OUT/'PLAN.json'),flush=True)


def metrics(model, source, jp):
    sigma,tau,k=source['sigma'],source['tau'],source['replica']
    lp=jp.with_suffix('.npz')
    if model=='E_orig':
        assert S.sha(lp)==S.read(OUT/'PLAN.json')['legacy_ledgers_sha256'][lp.name]
    else:
        assert S.sha(lp)==source['ledger_sha256']
    with np.load(lp) as z:
        counts=z['coverage_counts']
        points,side,band=S.cell_centers(S.read(OLD/'PLAN.json') if model=='E_orig' else S.read(OUT/'PLAN.json'))
        retained=points[:,1]!=.78
        counts=counts[:,:,retained];points=points[retained];side=side[retained]
        assert counts.shape==(80,600,12)
        evalmask=(np.arange(600)>=50)&(np.arange(600)<590)
        assert np.all(counts[2,evalmask]==3), 'zero must have no k3 deficits'
        cfg=z['cue_config'];frames=z['cue_frame'];tags=z['cue_sides']
        excess=z['restorable_unnecessary']
        rows=[]
        for j,c in enumerate(S.configurations()):
            domains={}
            for s in (0,1):
                for y in (-.045,.265,.54):
                    a=counts[j,evalmask][:,(side==s)&(points[:,1]==y)]
                    domains[f'{("LEFT","RIGHT")[s]}/y{y}']=dict(arrivals=int(a.size),insufficient_k3=int((a<3).sum()),k3_per_min=float((a<3).sum()/1.8))
            selected=(cfg==j)&evalmask[frames]
            event_frames=frames[selected];intervals=np.diff(event_frames)
            rows.append(dict(**c,domains=domains,total_k3_per_min=sum(d['k3_per_min'] for d in domains.values()),
                             auditory_cues_per_min=float(selected.sum()/1.8),
                             unnecessary_side_cues_per_min=((tags[selected]&excess[selected]).sum(0)/1.8).tolist(),
                             cue_intervals_frames=intervals.tolist(),dual_tag_events=int(tags[selected].all(1).sum()),
                             auditory_events=int(selected.sum())))
    return dict(model=model,sigma=sigma,tau=tau,replica=k,rows=rows,ledger_sha256=S.sha(lp))


def summarize():
    streams=[]
    for model in MODELS:
        folder=(OLD if model=='E_orig' else OUT/model)/'streams'
        for jp in sorted(folder.glob('*.json')):
            source=S.read(jp)
            if source['status']=='COMPLETE':streams.append(metrics(model,source,jp))
    cells=[];frontier=[];branches={};gap=[]
    for model in MODELS:
        for sigma in (5,10,15):
            for tau in (.5,1.,2.,4.):
                ss=[s for s in streams if s['model']==model and s['sigma']==sigma and s['tau']==tau]
                if len(ss)!=4:continue
                aggregate=[]
                for j,c in enumerate(S.configurations()):
                    rr=[s['rows'][j] for s in ss]
                    row=dict(model=model,sigma=sigma,tau=tau,K=4,**c)
                    for key in ('total_k3_per_min','auditory_cues_per_min','unnecessary_side_cues_per_min'):
                        row[key]=np.mean([r[key] for r in rr],axis=0).tolist()
                    row['domains']={d:{key:float(np.mean([r['domains'][d][key] for r in rr])) for key in ('arrivals','insufficient_k3','k3_per_min')} for d in rr[0]['domains']}
                    aggregate.append(row)
                none,dual,zero=aggregate[:3]
                assert zero['total_k3_per_min']==0
                den=none['total_k3_per_min']-zero['total_k3_per_min'];assert den>0
                for row in aggregate:
                    row['elimination']=(none['total_k3_per_min']-row['total_k3_per_min'])/den
                cells.extend(aggregate)
                for budget in (5,10,20):
                    feasible=[r for r in aggregate if r['method'] in ('A','B','v1') and not r['diagnostic'] and r['auditory_cues_per_min']<=budget]
                    best=min(feasible,key=lambda r:(r['total_k3_per_min'],r['auditory_cues_per_min'],r['name'])) if feasible else None
                    frontier.append(dict(model=model,sigma=sigma,tau=tau,budget=budget,feasible_configs=len(feasible),
                                         best=best,dual_elimination=dual['elimination'],status='COMPARABLE' if best else 'NOT_COMPARABLE'))
        ff=[f for f in frontier if f['model']==model and f['budget']==10]
        competitive=sum(f['best'] is not None and f['best']['elimination']>=f['dual_elimination']-.15 for f in ff)
        dominated=sum(f['best'] is not None and f['best']['elimination']<=f['dual_elimination']-.30 for f in ff)
        verdict='SOFTWARE_COMPETITIVE_SYNTH' if competitive>=8 else 'GEOMETRY_DOMINATED_SYNTH' if dominated>=8 else 'MIXED'
        branches[model]=dict(verdict=verdict,competitive_cells=competitive,dominated_cells=dominated,comparable_cells=sum(f['best'] is not None for f in ff),expected_cells=12,historical_only=model=='E_orig')
    for budget in (5,10,20):
        for sigma in (5,10,15):
            for tau in (.5,1.,2.,4.):
                fs=[next((f for f in frontier if f['model']==m and f['sigma']==sigma and f['tau']==tau and f['budget']==budget),None) for m in ('E_oracle','E_grav')]
                valid=all(f and f['best'] is not None for f in fs)
                gap.append(dict(sigma=sigma,tau=tau,budget=budget,status='COMPARABLE' if valid else 'NOT_COMPARABLE',
                                delta_pp=100*(fs[1]['best']['elimination']-fs[0]['best']['elimination']) if valid else None,
                                oracle_config=fs[0]['best']['name'] if valid else None,grav_config=fs[1]['best']['name'] if valid else None))
    close=sum(g['budget']==10 and g['delta_pp'] is not None and abs(g['delta_pp'])<=5 for g in gap)
    S.save(OUT/'result.json',dict(status='COMPLETE' if len(streams)==144 else 'PARTIAL',streams=len(streams),expected_streams=144,
                               branches=branches,cells=cells,frontier=frontier,estimator_gap=gap,
                               estimator_branch_at10='ESTIMATION_NOT_BOTTLENECK_SYNTH' if close>=8 else 'ESTIMATION_GAP_REPORTED',estimator_close_cells_at10=close,
                               per_stream=streams,plan_sha256=S.sha(OUT/'PLAN.json'),elapsed_s=time.time()-S.read(OUT/'PLAN.json')['started_unix']))
    print('SUMMARY',len(streams),branches,'estimator close',close,flush=True)


def run(workers):
    plan=S.read(OUT/'PLAN.json')
    assert S.sha(OUT/'PLAN.json')==(OUT/'PLAN.sha256').read_text().strip()
    for name,expected in plan['source_sha256'].items():
        assert S.sha(Path(__file__).parent/name)==expected, name
    assert S.read(OLD/'step1_gate.json')['status']=='PASS_PROXY_LIMITED'
    assert S.read(OLD/'geometry_engineering_check.json')['status']=='PASS'
    S.save(OUT/'request.json',dict(pid=os.getpid(),started_unix=time.time(),deadline_unix=plan['deadline_unix'],workers=workers,backend='FROZEN_PROTOCOL_CPU_ONLY; numpy'))
    jobs=[(m,s,t,k) for m in ('E_oracle','E_grav') for s in (5,10,15) for t in (.5,1.,2.,4.) for k in range(4)]
    complete=0
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(S.simulate,j) for j in jobs]
            for f in as_completed(futures):
                r=f.result();complete+=r['status']=='COMPLETE'
                S.save(OUT/'progress.json',dict(completed=complete,total=96,last_model=r.get('model'),last_activity_utc=datetime.now(timezone.utc).isoformat()))
        summarize()
        S.save(OUT/'terminal.json',dict(status='COMPLETE' if complete==96 else 'PARTIAL',completed=complete,total=96,elapsed_s=time.time()-plan['started_unix']))
    except BaseException as e:
        S.save(OUT/'terminal.json',dict(status='FAILED',error=repr(e),completed=complete,elapsed_s=time.time()-plan['started_unix']))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','run','summarize'));p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    {'freeze':freeze,'run':lambda:run(a.workers),'summarize':summarize}[a.stage]()
