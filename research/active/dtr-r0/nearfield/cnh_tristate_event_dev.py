"""Frozen-output, evaluator-only event-level CNH Development analysis."""
from __future__ import annotations
import argparse,csv,json,os,time
from pathlib import Path
import numpy as np
import cnh_tristate_dev as R

OUT=R.WORK/'cnh-tristate-event-dev-20261006'
R3=R.WORK/'cnh-tristate-dev-r3-20261006'
GEOMETRY=[R.WORK/p for p in ('cnh-observed-sequence-20261002/geometry.npz','cnh-dual-gated-fusion-20261005/natural97000/geometry.npz','cnh-extrinsic-aug-20261006/geometry/calibration.npz','cnh-extrinsic-aug-20261006/continuation-r1/geometry/evaluation.npz')]
CATS=['contact0-2cm','contact2-5cm','contact>5cm']
EPS=1e-10

def causal_index(fraction,frames=R.FRAMES):
    return np.searchsorted(frames,np.asarray(fraction)+EPS,side='right')-1

def burden(unknown):
    u=np.asarray(unknown,bool)[:,2:12] # states5..14; intervals[t5,t15)
    return dict(seconds=u.sum(1)*.2,total_seconds=np.full(len(u),2.),
                starts=(u[:,1:]&~u[:,:-1]).sum(1),initial_unknown=u[:,0].astype(int),
                obstacle_free_intervals=10)

def partition(score,gate,tau,fraction,contact):
    alarm=score>=R.THRESHOLD;clear=~alarm&gate&(score<=tau)
    ix=causal_index(fraction);rr=np.flatnonzero(contact)
    if not ((ix[rr]>=0)&(ix[rr]<13)).all():raise ValueError('Contact deadline outside saved outputs')
    timely=np.maximum.accumulate(alarm,axis=1)[rr,ix[rr]]
    state=np.full(len(score),-1,np.int8)
    state[rr]=np.where(timely,0,np.where(clear[rr,ix[rr]],2,1))
    return state,clear,~alarm&~clear,alarm

def freeze(start):
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[R3/'online.npz',R3/'rows.json',R3/'truth.npz',R3/'calibration.json',*GEOMETRY]
    R.save(OUT/'PLAN.json',dict(phase='EXPLORE; consumed synthetic Development; descriptive',started_unix=start,deadline_unix=start+2400,budget_wall_seconds=2400,
        goal='Event silent-miss/unknown burden tradeoff, exact existing .9m deadline; output and model unchanged',
        event='Existing designated boxes[0] first true front crossing .9m; all-box query surface labels at that reference; mergeHEAD/BODY by(unit,config), deepest contact category. Not per-object events.',
        censor='Covered contacts only in main denominator; all other scene categories and right/left censored separately, no censored misses',
        state='Last already-produced frame <= reference_fraction with numerical tolerance1e-10 frame; never interpolate score or take right frame. Any original alarm before deadline timely; else unknown or clear.',
        output='R3 original noise m3, marginindex1; exact20 common tau; A/B both retained; no new rendering/inference/training or gate/threshold change',
        control='Both original query clear_all at all13 observed poses; original physical surfaces0.3..3m, width+/-.4m, includes graze exclusion. Sampled clear, not continuous or hardware walking.',
        burden='ZOH statesframes5..14 cover[t5,t15), 10*.2=2.0s/sequence; new unknown entries only6..14; initialframe5 unknown separately left-censored, no invented startup. Warmup[frame3,frame5) separate.',
        secondary='R3 strict-clear frame unknown ratio retained separately, different labels/denominators; saved-frame entry counts are not human cue frequency',
        intervals='1000 whole-unit bootstrap, strata batch/mode/turn, paired arms; percentile95 descriptive CI, zero denominators undefined',
        matching='Common event silent-rate budgets from20 point curves; perreplicate feasible grid minimization, undefined when no point feasible; no superiority gate',
        sources='R3 same-frame proxies only, overlapping; fov witness only if r3 contact label present, core gap only if evaluated; not target-specific or causal proof',
        fact_checks='Natural target U(.6,2.6);1.77m controlledpsi20/25 opposite shallow only; proxy978 estimated-near not exact event',
        prior_observed='Review found960 physicalcontacts/963queries and crossing tradeoffs; not a fresh blinded experiment, do not precommit dominance',
        restrictions='CPU only; certificates and fresh paper confirmation deferred; no automatic model optimization; finish docs/scopedmaster push',
        hashes={str(p.relative_to(R.ROOT)):R.sha(p) for p in inputs},source_sha256=R.sha(__file__)))
    (OUT/'PLAN.sha256').write_text(R.sha(OUT/'PLAN.json')+'\n',encoding='utf8')
    (OUT/'source').mkdir();(OUT/'source/analysis.py').write_bytes(Path(__file__).read_bytes())

def check_time():
    if time.time()>R.read(OUT/'PLAN.json')['deadline_unix']:raise TimeoutError('40min wall budget')

def query_scores(rows,d):
    """Retain individual HEAD/BODY to crosscheck original query timely metric."""
    scores=np.empty((len(rows),13,2,2));lookup={(r['unit'],r['config']):i for i,r in enumerate(rows)}
    early=np.load(R.MARGIN/'frame_scores_M3_early.npz');late=np.load(R.MARGIN/'frame_scores_M3.npz')
    for batch,folder in R.score_sources().items():
        for p in sorted(folder.glob('unit*.npz')):
            check_time()
            with np.load(p) as z:
                unit=int(z['unit']);configs=z['configs'];sm=R.smooth(z['reference'] if batch>=98000 else z['raw'])
                single=R.smooth(np.concatenate((early[str(unit)],late[str(unit)]),axis=1))[configs] if batch<97000 else sm[0]
                dual=sm.max(0) if batch<97000 else sm[1:].max(0)
                for j,c in enumerate(configs):scores[lookup[unit,int(c)]]=np.stack((single[j],dual[j]),axis=1)
    early.close();late.close()
    np.testing.assert_array_equal(scores.max(-1),d['score'])
    return scores

def load_geometry(rows):
    n=len(rows);lookup={(r['unit'],r['config']):i for i,r in enumerate(rows)}
    cats=np.full((n,2),'MISSING',dtype='<U20');controls=np.zeros((n,2),bool);covered=np.zeros((n,2),bool);frac=np.full((n,2),np.nan);censor=np.full((n,2),'MISSING',dtype='<U20')
    for path in GEOMETRY:
        with np.load(path) as z:
            for j in range(len(z['unit'])):
                key=(int(z['unit'][j]),int(z['config'][j]));i=lookup[key];q=int(z['query'][j]);assert cats[i,q]=='MISSING'
                cats[i,q]=z['ref_category'][j];controls[i,q]=z['clear_all'][j];covered[i,q]=z['covered'][j];frac[i,q]=z['reference_fraction'][j];censor[i,q]=z['censor_reason'][j]
    assert not (cats=='MISSING').any();np.testing.assert_array_equal(covered[:,0],covered[:,1]);np.testing.assert_allclose(frac[:,0],frac[:,1],equal_nan=True)
    contact_query=covered&np.isin(cats,CATS);contact=contact_query.any(1);depth=np.full(n,-1,np.int8)
    for k,cat in enumerate(CATS):depth[(contact_query&(cats==cat)).any(1)]=k
    return dict(category=cats,control=controls.all(1),covered=covered[:,0],fraction=frac[:,0],censor=censor[:,0],contact_query=contact_query,contact=contact,depth=depth)

def analyze():
    tick=time.monotonic();check_time();plan=R.read(OUT/'PLAN.json')
    for p,h in plan['hashes'].items():assert R.sha(R.ROOT/p)==h
    rows=R.read(R3/'rows.json')
    with np.load(R3/'online.npz') as z:d={k:z[k] for k in ('score','gate','thresholds')}
    with np.load(R3/'truth.npz') as z:t={k:z[k] for k in z.files}
    g=load_geometry(rows);qscore=query_scores(rows,d);n=len(rows)
    units=np.array([r['unit'] for r in rows]);batches=np.array([r['batch'] for r in rows]);modes=np.array([r['mode'] for r in rows]);turns=np.array([r['turn'] for r in rows])
    uu,inv=np.unique(units,return_inverse=True);rng=np.random.default_rng(2026100617);boot=np.zeros((1000,len(uu)),np.int16)
    strat=np.array([f"{rows[np.flatnonzero(units==u)[0]]['batch']}/{rows[np.flatnonzero(units==u)[0]]['mode']}/{rows[np.flatnonzero(units==u)[0]]['turn']}" for u in uu])
    for s in np.unique(strat):
        ids=np.flatnonzero(strat==s);draw=rng.integers(0,len(ids),size=(1000,len(ids)))
        for k in range(1000):boot[k,ids]=np.bincount(draw[k],minlength=len(ids))
    def ratio(num,den,mask):
        nn=np.bincount(inv,weights=np.where(mask,num,0),minlength=len(uu));dd=np.bincount(inv,weights=np.where(mask,den,0),minlength=len(uu));bn=boot@nn;bd=boot@dd;ok=bd>0
        nv,dv=float(nn.sum()),float(dd.sum());return dict(numerator=nv,denominator=dv,value=nv/dv if dv else None,ci95=np.quantile(bn[ok]/bd[ok],[.025,.975]).tolist() if ok.any() else [None,None],bootstrap_defined=int(ok.sum()))
    groups={'all':np.ones(n,bool)}|{f'mode{k}':modes==k for k in range(3)}|{f'turn_{k}':turns==k for k in ('left','right','none')}|{f'batch{k}':batches==k for k in np.unique(batches)}
    rr=np.flatnonzero(g['contact']);ix=causal_index(g['fraction']);safe=~t['contact']&~t['graze'];results={};states={};burdens={};bootstrap_curves={}
    for group,mask in groups.items():
        results[group]={'sequences':int(mask.sum()),'contact_events':int((mask&g['contact']).sum()),'control_sequences':int((mask&g['control']).sum()),'curves':{}}
        for a,arm in enumerate(('single','dual')):
            for variant in ('A','B'):
                key=arm+'/'+variant;curve=[];brisk=[];bcost=[];stateall=[]
                for j,tau in enumerate(d['thresholds']):
                    state,clear,unknown,alarm=partition(d['score'][...,a],d['gate'][:,:,1,a],tau,g['fraction'],g['contact'])
                    if variant=='B':
                        clear&=~np.maximum.accumulate(alarm,axis=1);unknown=~alarm&~clear
                    # B changes costs; any prior alarm already makes an event timely.
                    b=burden(unknown);control=mask&g['control'];ones=np.ones(n);evt=mask&g['contact'];silent=state==2
                    point={'index':j,'tau':float(tau),'timely':ratio(state==0,g['contact'],mask),'unknown_miss':ratio(state==1,g['contact'],mask),'silent':ratio(silent,g['contact'],mask),
                        'unknown_given_not_timely':ratio(state==1,(state==1)|silent,mask),'unknown_time':ratio(b['seconds'],b['total_seconds'],control),
                        'unknown_entries_per_min':ratio(b['starts'],b['total_seconds']/60,control),'initial_unknown':ratio(b['initial_unknown'],ones,control),
                        'control_obstacle_time':ratio(alarm[:,2:12].sum(1)*.2,b['total_seconds'],control),
                        'warmup_unknown':ratio(unknown[:,:2].sum(1)*.2,np.full(n,.4),control),
                        'r3_frame_unknown_secondary':ratio((unknown&safe).sum(1),safe.sum(1),mask),
                        'depth':{CATS[k]:ratio(silent,g['contact'],mask&(g['depth']==k)) for k in range(3)}}
                    sources={}
                    for field in ('core_truth_gap','fov_in','unchecked_contact','near_contact','edge_contact','side_contact','boundary'):
                        arr=t[field][...,a] if t[field].ndim==3 else t[field];val=np.zeros(n,bool);val[rr]=arr[rr,ix[rr]];sources[field]=int((evt&silent&val).sum())
                    sources['r3_contact_evaluable']=int((evt[rr]&silent[rr]&t['contact'][rr,ix[rr]]).sum());point['sources']=sources
                    curve.append(point);stateall.append(state)
                    nn=np.bincount(inv,weights=evt&silent,minlength=len(uu));dd=np.bincount(inv,weights=evt,minlength=len(uu));den=boot@dd;num=boot@nn;brisk.append(np.divide(num,den,out=np.full(1000,np.nan),where=den>0))
                    nn=np.bincount(inv,weights=np.where(control,b['seconds'],0),minlength=len(uu));dd=np.bincount(inv,weights=np.where(control,b['total_seconds'],0),minlength=len(uu));den=boot@dd;num=boot@nn;bcost.append(np.divide(num,den,out=np.full(1000,np.nan),where=den>0))
                results[group]['curves'][key]=curve
                if group=='all':states[key]=np.asarray(stateall);bootstrap_curves[key]=(np.asarray(brisk),np.asarray(bcost))
    # Paired budget comparison, including feasible-point selection per replicate.
    matches={}
    for variant in ('A','B'):
        matches[variant]=[]
        for cap in (0,2,5,10,15,30,50):
            cells={};samples={}
            for arm in ('single','dual'):
                key=arm+'/'+variant;curve=results['all']['curves'][key];eligible=[p for p in curve if p['silent']['numerator']<=cap and p['unknown_time']['value'] is not None]
                chosen=min(eligible,key=lambda p:p['unknown_time']['value']) if eligible else None
                cells[arm]=None if chosen is None else {'index':chosen['index'],'tau':chosen['tau'],'silent':chosen['silent'],'unknown_time':chosen['unknown_time']}
                risk,cost=bootstrap_curves[key];feasible=(risk<=cap/g['contact'].sum())&np.isfinite(cost);values=np.where(feasible,cost,np.inf).min(0);values[~np.isfinite(values)]=np.nan;samples[arm]=values
            delta=samples['dual']-samples['single'];ok=np.isfinite(delta)
            matches[variant].append(dict(silent_budget_count=cap,silent_budget_rate=cap/int(g['contact'].sum()),cells=cells,
                dual_minus_single_burden=None if any(v is None for v in cells.values()) else cells['dual']['unknown_time']['value']-cells['single']['unknown_time']['value'],
                paired_ci95=np.quantile(delta[ok],[.025,.975]).tolist() if ok.any() else [None,None],bootstrap_joint_defined=int(ok.sum())))
    # Query-level original timely denominator kept distinct from physical episodes.
    qtim=np.maximum.accumulate(qscore>=R.THRESHOLD,axis=1);query_metrics={}
    for a,arm in enumerate(('single','dual')):
        q=np.zeros((n,2),bool);cov=np.flatnonzero(g['covered']);q[cov]=qtim[cov,ix[cov],a,:]
        query_metrics[arm]={cat:ratio((q&g['contact_query']&(g['category']==cat)).sum(1),(g['contact_query']&(g['category']==cat)).sum(1),np.ones(n,bool)) for cat in CATS}
        query_metrics[arm]['all']=ratio((q&g['contact_query']).sum(1),g['contact_query'].sum(1),np.ones(n,bool))
        rescue=np.zeros(n,bool);rescue[rr]=np.maximum.accumulate(d['score'][...,a]>=R.THRESHOLD,axis=1)[rr,ix[rr]]&~(q[rr]&g['contact_query'][rr]).any(1)
        query_metrics[arm]['physical_alarm_without_contact_query_alarm']=int(rescue.sum())
    counts=dict(sequences=n,units=len(uu),physical_contact=int(g['contact'].sum()),contact_queries=int(g['contact_query'].sum()),physical_controls=int(g['control'].sum()),
                covered_sequences=int(g['covered'].sum()),censor={k:int((g['censor']==k).sum()) for k in np.unique(g['censor'])},depth={CATS[k]:int((g['depth']==k).sum()) for k in range(3)},
                last_deadline_state_frames={str(int(k+3)):int(v) for k,v in zip(*np.unique(ix[rr],return_counts=True))})
    np.savez_compressed(OUT/'ledger.npz',**g,query_score=qscore,states=np.stack([states[k] for k in ('single/A','dual/A','single/B','dual/B')]),unit=units,batch=batches,mode=modes,turn=turns)
    R.save(OUT/'result.json',dict(status='DESCRIPTIVE_COMPLETE',counts=counts,groups=results,matched=matches,query_timely=query_metrics,thresholds=d['thresholds'].tolist(),seconds=time.monotonic()-tick,
        limits=['Old consumed synthetic Development, no risk guarantee or human cue effectiveness','Existing scene episode identity and sampled-clear controls only','Warmup excluded from maincost; initial unknown not counted as new entry','Source proxies are same-frame overlap, not object-specific causes','Paired matching CI reselects grid per replicate; missing feasibility reported, no dominance gate'],
        provenance={'PLAN.json':R.sha(OUT/'PLAN.json'),'ledger.npz':R.sha(OUT/'ledger.npz')}))
    with (OUT/'curves.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.writer(f);w.writerow(['group','arm_variant','tau','events','timely','unknown_miss','silent','silent_rate','unknown_time','unknown_entries_per_min'])
        for group,v in results.items():
            for key,curve in v['curves'].items():
                for p in curve:w.writerow([group,key,p['tau'],p['silent']['denominator'],p['timely']['numerator'],p['unknown_miss']['numerator'],p['silent']['numerator'],p['silent']['value'],p['unknown_time']['value'],p['unknown_entries_per_min']['value']])
    print(json.dumps({'counts':counts,'seconds':time.monotonic()-tick},ensure_ascii=False,indent=2),flush=True)

def audit_receipt():
    """Add proxy evaluability/source identity without changing saved analysis."""
    r=R.read(OUT/'result.json')
    with np.load(OUT/'ledger.npz') as z:g={k:z[k] for k in z.files}
    with np.load(R3/'truth.npz') as z:t={k:z[k] for k in z.files}
    rr=np.flatnonzero(g['contact']);ix=causal_index(g['fraction']);groups={'all':np.ones(len(g['unit']),bool)}
    groups.update({f'mode{k}':g['mode']==k for k in range(3)})
    groups.update({f'turn_{k}':g['turn']==k for k in ('left','right','none')})
    availability={}
    for group,mask in groups.items():
        availability[group]={}
        for v,key in enumerate(('single/A','dual/A','single/B','dual/B')):
            a=v%2;availability[group][key]=[]
            for state in g['states'][v]:
                silent=mask[rr]&(state[rr]==2);ce=t['core_gap_evaluated'][rr,ix[rr],a]
                availability[group][key].append({'silent':int(silent.sum()),'core_gap_evaluated':int((silent&ce).sum()),
                    'r3_contact_evaluable':int((silent&t['contact'][rr,ix[rr]]).sum())})
    paths=[R.MARGIN/'frame_scores_M3_early.npz',R.MARGIN/'frame_scores_M3.npz']
    for folder in R.score_sources().values():paths.extend(sorted(folder.glob('unit*.npz')))
    R.save(OUT/'audit_receipt.json',dict(source_hashes={str(p.relative_to(R.ROOT)):R.sha(p) for p in paths},
        proxy_availability=availability,analysis_sha256=R.sha(OUT/'result.json'),
        final_source_sha256=R.sha(__file__),note='Adds evaluability denominators and original query input identity; saved curves unchanged. Frozen source retained; post-analysis changes only receipt/plot.'))
    (OUT/'source/final_analysis.py').write_bytes(Path(__file__).read_bytes())

def plot(variant='A'):
    os.environ['MPLCONFIGDIR']=str(OUT/'mpl-cache');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    r=R.read(OUT/'result.json');fig,axes=plt.subplots(3,2,figsize=(11,12),layout='constrained')
    for ax,group in zip(axes.flat,('all','mode0','mode1','mode2','turn_left','turn_right')):
        for arm,color in (('single','#2864b5'),('dual','#c76b28')):
            p=r['groups'][group]['curves'][arm+'/'+variant];x=np.array([100*v['unknown_time']['value'] for v in p]);y=np.array([np.nan if v['silent']['value'] is None else 100*v['silent']['value'] for v in p]);ax.plot(x,y,'o-',color=color,label=arm)
            lo=[np.nan if v['silent']['ci95'][0] is None else 100*v['silent']['ci95'][0] for v in p];hi=[np.nan if v['silent']['ci95'][1] is None else 100*v['silent']['ci95'][1] for v in p];ax.fill_between(x,lo,hi,color=color,alpha=.13)
        h=r['groups'][group]
        ax.set(title=f"{group}: {h['contact_events']} events / {h['control_sequences']} controls",xlabel='Unknown time on sampled-clear controls (%)',ylabel='Silent misses / contact episodes (%)');ax.grid(alpha=.2);ax.legend()
    fig.suptitle(f'Exact 0.9m deadline / {variant} — consumed synthetic Development, descriptive')
    fig.savefig(OUT/f'event_curves_{variant}.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','analyze','plot','audit']);p.add_argument('--started-unix',type=float,default=time.time());a=p.parse_args()
    if a.stage=='freeze':freeze(a.started_unix)
    elif a.stage=='analyze':analyze()
    elif a.stage=='audit':audit_receipt()
    else:
        plot('A');plot('B')
