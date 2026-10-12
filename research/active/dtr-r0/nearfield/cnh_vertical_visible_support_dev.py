"""Read frozen per-frame features/scores; no projection, inference or new gate."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'artifacts.local/work'
OUT=WORK/'cnh-vertical-visible-support-dev-20261009'
CACHE=WORK/'cnh-bar-vertical-evidence-dev-20261009'
BOUNDARY=WORK/'cnh-bar-boundary-contrast-dev-20261009'
JOIN=WORK/'cnh-vertical-geometry-join-dev-20261009'
SHAPE=WORK/'cnh-aligned-shapes-dev-20261008'
T0,TM,TL=.8557642486787612,.9404184587540165,4.625390338985158
FEATURES=('history_abs_share','current_inner_positive_share','total_density','current_scaled_mass')

def readcsv(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def save(p,d):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(d,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
def csvsave(name,rr):
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def finite(x):return float(x) if np.isfinite(x) else None
def stat(xx):
    values=np.array([x for x in xx if x is not None],float)
    return dict(n=len(xx),finite=len(values),mean=finite(values.mean()) if len(values) else None,
                median=finite(np.median(values)) if len(values) else None)

def prepare():
    t=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    inputs=[SHAPE/'PLAN.json',SHAPE/'geometry.npz',JOIN/'dwin_loss_events.csv',JOIN/'clear_geometry_join.csv',
            CACHE/'masks.npz',*[CACHE/f'angle_{a:+d}.npz' for a in (-3,3)],
            *[BOUNDARY/f'scores_{a:+d}.npz' for a in (-3,3)]]
    plan=dict(task='CNH_VERTICAL_VISIBLE_SUPPORT_DEV_20261009',base_commit='2537c921',
        authorization='User 推进 approving preceding proposed60s cumulativeCPU cache diagnosis',budget_seconds=60,
        budget_unit='Cumulative cache read/diagnosis/repair/focused check wall seconds; no carry-over from closed geometry run',
        independent_preliminary_command_seconds=3.322,
        scope='Fixed±3 cache; farf3..9 all24clearK4/bothsides; nearf10..13 fusionDwin-loss-condition clear+15cm vsin1/in4, same/mirror separately',
        observables='Saved tot,current,cnt under identical public query inner/outer masks; raw/smooth M3/local. No evaluator shape/side enters feature computation.',
        H1='Far clear alarm cells may carry more historical absolute support than nonalarmK4 at same scene/frame; raw-versus-last5 response may locate smoothing contribution, not causal history proof',
        H2='Near shallow contacts may retain higher current positive mass fraction inside public query than condition-matched clear; read allK4xK4 continuous pairs, not just alarms',
        feature_contract='History residual=FP16tot-FP16current. history_abs_share=L1(history inner)/(L1(history inner)+L1(current inner)); current_inner_positive_share=positivecurrent inner/(inner+outer). Density tot/cumulativecnt; current_scaled_mass current/samecnt is NOT single-frame density. Background included, no photon attribution.',
        visible_support='Far public HEAD+BODYunion, near respectiveheight. Outer|x|.3..6 samepublicy,z; equal total lateral width. cnt cumulative coverage, not noisevariance.',
        decision='A signal tied to namedvisiblequantity and an actionable falsifiable prediction may justify minimumcontrolled validation; mirror consistency strengthens, not necessary/sufficient. No numeric success/stop threshold, no N64/dprime/info-bound. Noise/seed confounding retained.',
        stop='Finish fixed extraction/comparisons and focusedcheck or60s including failures. No newprojection/inference/training/sampling, no Dwin/Dmaxrefinement, no policy promotion.',
        input_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    plan['prepare_seconds']=time.monotonic()-t;save(OUT/'PLAN.json',plan)
    print('PREPARED',round(plan['prepare_seconds'],3),flush=True)

def run():
    t=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf-8'))
    previous=plan['prepare_seconds']+plan['independent_preliminary_command_seconds']
    previous+=sum(json.loads(p.read_text(encoding='utf-8'))['seconds'] for p in OUT.glob('failure_*.json'))
    def check():
        if previous+time.monotonic()-t>52:raise TimeoutError('Reserve final8s of60s for focusedcheck; no expansion')
    try:
        rows=json.loads((SHAPE/'PLAN.json').read_text(encoding='utf-8'))['scene_rows']
        loss=[r for r in readcsv(JOIN/'dwin_loss_events.csv') if r['base']=='fusion']
        truth=np.load(SHAPE/'geometry.npz');cat=truth['category'];sensor=truth['sensor'];truth.close()
        with np.load(CACHE/'masks.npz') as z:qmask=z['query_masks'].astype(float)
        # Public support only: preserve y/z fractions, extend into outer x halves.
        outer=np.zeros_like(qmask);yz=qmask.sum(1)/12
        outer[:,:6]=yz[:,None];outer[:,18:]=yz[:,None]
        masks=np.stack([qmask[0],outer[0],qmask[1],outer[1],qmask.sum(0),outer.sum(0)]).reshape(6,-1)
        assert np.allclose(masks.sum(1)[::2],masks.sum(1)[1::2])
        features={};far=[];pairrows=[];nearcounts=[];condition_stats=[]
        for a in (-3,3):
            check()
            with np.load(CACHE/f'angle_{a:+d}.npz') as z:
                ids=z['scene_ids'];echo=z['echo'];cnt=z['count'];raw=z['logits'][:,:,:,0];localraw=z['local_raw']
            with np.load(BOUNDARY/f'scores_{a:+d}.npz') as z:sm=z['m3'];sl=z['local']
            assert echo.shape==(528,13,2,24,17,33)
            index={int(s):i for i,s in enumerate(ids)}
            flags=(sm>=TM)|(sl>=TL);rawflags=(raw>=TM)|(localraw>=TL)
            values=np.empty((132,4,13,3,4),float)
            for j in range(13):
                check();total=echo[:,j,0].astype(float).reshape(528,-1);current=echo[:,j,1].astype(float).reshape(528,-1)
                cov=masks@cnt[j].astype(float).ravel()
                ts=total@masks.T;cs=current@masks.T
                ca=np.abs(current)@masks.T;ha=np.abs(total-current)@masks.T;cp=np.maximum(current,0)@masks.T
                for qi in range(3):
                    inside,outside=2*qi,2*qi+1
                    with np.errstate(divide='ignore',invalid='ignore'):
                        vv=np.stack([ha[:,inside]/(ha[:,inside]+ca[:,inside]),
                                     cp[:,inside]/(cp[:,inside]+cp[:,outside]),ts[:,inside]/cov[inside],cs[:,inside]/cov[inside]],1)
                    values[:,:,j,qi]=vv.reshape(132,4,4)
                del total,current
            features[a]=(values,ids,raw,sm,flags)
            clearids=[int(s) for s in ids if (cat[int(s)]=='clear').all()]
            for s in clearids:
                i=index[s]
                for j in range(7):
                    for k in range(4):
                        r=rows[s]
                        far.append(dict(angle=a,scene=s,replica=k,frame=j+3,side=r['side'],width_cm=round((r['hi'][0]-r['lo'][0])*100),
                            rho=r['rho'],span=r['group'],distance_m=float(r['lo'][2]-sensor[j+3,2,3]),
                            fusion=int(flags[i,k,j].any()),raw_fusion=int(rawflags[i,k,j].any()),
                            last5_only_alarm=int(flags[i,k,j].any() and not rawflags[i,k,j].any()),
                            raw_M3_max=float(raw[i,k,j].max()),smooth_M3_max=float(sm[i,k,j].max()),
                            **{name:finite(values[i,k,j,2,n]) for n,name in enumerate(FEATURES)}))
            def key(r):return (round((r['hi'][0]-r['lo'][0])*100),r['rho'],r['group'],r['variant'],r['side'])
            selected={(int(float(r['width_cm'])),float(r['rho']),r['span'],r['variant'],int(r['side']),r['height']) for r in loss if int(r['angle'])==a}
            for relation in ('same_side','mirrored_side'):
                selected_clear={(w,rho,span,var,side if relation=='same_side' else -side) for w,rho,span,var,side,h in selected}
                cc=[s for s in clearids if key(rows[s]) in selected_clear]
                nearcounts.append(dict(angle=a,relation=relation,condition_cells=len(selected_clear),clear_joint_denominator=len(cc)*4*4,
                                       clear_joint_alarms=int(flags[[index[s] for s in cc],:,7:11].any(-1).sum())))
                for w,rho,span,var,side,h in sorted(selected):
                    cs=next(s for s in clearids if key(rows[s])==(w,rho,span,var,side if relation=='same_side' else -side))
                    qi=0 if h=='HEAD' else 1
                    for depth in ('in_1cm','in_4cm'):
                        ps=next(int(s) for s in ids if rows[int(s)]['placement']==depth and key(rows[int(s)])==(w,rho,span,var,side))
                        for j in range(7,11):
                            for ck in range(4):
                                for pk in range(4):
                                    cv,pv=values[index[cs],ck,j,qi],values[index[ps],pk,j,qi]
                                    rec=dict(angle=a,relation=relation,width_cm=w,rho=rho,span=span,variant=var,height=h,depth=depth,frame=j+3,
                                        clear_scene=cs,contact_scene=ps,clear_replica=ck,contact_replica=pk,
                                        clear_alarm=int(flags[index[cs],ck,j,qi]),contact_alarm=int(flags[index[ps],pk,j,qi]))
                                    for n,name in enumerate(FEATURES):
                                        rec['clear_'+name]=finite(cv[n]);rec['contact_'+name]=finite(pv[n]);rec['delta_'+name]=finite(pv[n]-cv[n])
                                    rec['delta_raw_M3']=float(raw[index[ps],pk,j,qi]-raw[index[cs],ck,j,qi])
                                    rec['delta_smooth_M3']=float(sm[index[ps],pk,j,qi]-sm[index[cs],ck,j,qi])
                                    pairrows.append(rec)
            del echo,cnt;print('EXTRACTED',a,round(time.monotonic()-t,3),'seconds',flush=True)
        farstats=[];mixed=[]
        for a in (-3,3):
            for side in (-1,1):
                ss=[r for r in far if r['angle']==a and r['side']==side]
                for alarm in (0,1):
                    ff=[r for r in ss if r['fusion']==alarm]
                    farstats.append(dict(angle=a,side=side,alarm=alarm,n=len(ff),last5_only=sum(r['last5_only_alarm'] for r in ff),
                                         **{name:stat([r[name] for r in ff]) for name in FEATURES}))
            cells=defaultdict(list)
            for r in far:
                if r['angle']==a:cells[r['scene'],r['frame']].append(r)
            for (s,f),rr in cells.items():
                yes=[r for r in rr if r['fusion']];no=[r for r in rr if not r['fusion']]
                if not yes or not no:continue
                mixed.append(dict(angle=a,scene=s,frame=f,side=rows[s]['side'],alarm_K=len(yes),nonalarm_K=len(no),
                    **{'delta_'+name:finite(np.mean([r[name] for r in yes])-np.mean([r[name] for r in no]))
                       if all(r[name] is not None for r in rr) else None for name in FEATURES}))
        mixedsummary=[]
        for a in (-3,3):
            for side in (-1,1):
                mm=[r for r in mixed if r['angle']==a and r['side']==side]
                mixedsummary.append(dict(angle=a,side=side,mixed_cells=len(mm),
                    **{name:stat([r['delta_'+name] for r in mm]) for name in FEATURES}))
        nearstats=[]
        for a in (-3,3):
            for relation in ('same_side','mirrored_side'):
                for depth in ('in_1cm','in_4cm'):
                    rr=[r for r in pairrows if r['angle']==a and r['relation']==relation and r['depth']==depth]
                    nearstats.append(dict(angle=a,relation=relation,depth=depth,pair_occurrences=len(rr),
                        **{name:stat([r['delta_'+name] for r in rr]) for name in FEATURES},
                        delta_raw_M3=stat([r['delta_raw_M3'] for r in rr]),delta_smooth_M3=stat([r['delta_smooth_M3'] for r in rr])))
        check();csvsave('far_all_windows.csv',far);csvsave('far_mixed_K4_cells.csv',mixed);csvsave('near_all_pairs.csv',pairrows)
        save(OUT/'result.json',dict(status='COMPLETE',seconds=time.monotonic()-t,cumulative_before_final_audit=previous+time.monotonic()-t,
            far_rows=len(far),far_by_alarm=farstats,far_mixed_summary=mixedsummary,near_counts=nearcounts,near_summary=nearstats,
            near_pair_rows=len(pairrows),visible_metrics=list(FEATURES),new_inference=0,new_projection=0,training=0,new_samples=0,
            limits='Far alarm-conditioned noise selection; near differentseed/background/noise, Kcrosspairs correlated; no causal test or matchedcost detection trial'))
        print('COMPLETE',len(far),len(pairrows),'cumulative',round(previous+time.monotonic()-t,3),flush=True)
    except BaseException as e:
        save(OUT/f'failure_{time.time_ns()}.json',dict(seconds=time.monotonic()-t,error=repr(e)));raise

def counts():
    """Deduplicate height windows; joint clear windows remain in result.json."""
    t=time.monotonic();rr=readcsv(OUT/'near_all_pairs.csv');out=[]
    for a in (-3,3):
        for rel in ('same_side','mirrored_side'):
            for depth in ('in_1cm','in_4cm'):
                rows=[r for r in rr if int(r['angle'])==a and r['relation']==rel and r['depth']==depth]
                contacts={tuple(r[k] for k in ('height','contact_scene','contact_replica','frame')):int(r['contact_alarm']) for r in rows}
                clear={tuple(r[k] for k in ('height','clear_scene','clear_replica','frame')):int(r['clear_alarm']) for r in rows}
                out.append(dict(angle=a,relation=rel,depth=depth,height_clear_windows=len(clear),height_clear_alarms=sum(clear.values()),
                    contact_windows=len(contacts),contact_alarms=sum(contacts.values()),
                    positive_mass_pairs=sum(float(r['delta_current_scaled_mass'])>0 for r in rows),
                    positive_share_pairs=sum(float(r['delta_current_inner_positive_share'])>0 for r in rows),pairs=len(rows)))
    save(OUT/'near_counts_supplement.json',dict(rows=out,seconds=time.monotonic()-t))

if __name__=='__main__':globals()[sys.argv[1]]()
