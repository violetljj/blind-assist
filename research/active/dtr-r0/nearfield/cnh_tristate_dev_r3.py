"""Consumed CNH Development R3. Calibration is geometry-only and sealed first."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import cnh_tristate_dev as R1
import cnh_tristate_dev_r2 as R2

ROOT=R1.ROOT
OUT=R1.WORK/'cnh-tristate-dev-r3-20261006'


def deadline():
    if time.time()>=R1.read(OUT/'PLAN.json')['deadline_unix']:
        raise TimeoutError('R3 60minute wall limit; preserve completed stages')


def freeze(started):
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=R1.WORK/'cnh-track-a-v5-20260928/data/source/cnh_track_a_readout.py'
    R1.save(OUT/'PLAN.json',dict(task='CNH_TRISTATE_DEV_R3_20261006',phase='EXPLORE consumed synthetic Development, descriptive; no certificate',
        started_unix=started,deadline_unix=started+3600,budget_wall_seconds=3600,r2_revision='5ed0c72ac0be4a8cb3a52b09e368d3e736128c35',
        calibration=dict(no_scene=True,no_scores=True,no_labels=True,replicas=512,seed_prefix=2026100631,
            trajectory='16frames,dt.2s,speed.8m/s,aligned head,nominal pitch-10deg',
            noise='Frozen noisy_poses, same draws: scale uniform+/-.2,bias+/−1deg/s peraxis,stepN(0,.1deg)',
            gravity='Earlier E_grav convention: remove constantX/Z biases; keep constantY around worldgravity,left-multiply; retain original local3axis stepnoise and scale/translation draws; gate-only synthetic assumption',
            margins_degrees=np.arange(0,20.0001,.5).tolist(),rule='Minimum .5deg margin with >=99% passed nonwarm frames pooled512x11; empty mask not a pass',
            volume='Weighted5cm voxel midpoint quadrature across512x11 sampled estimated poses, byHEAD/BODY and distance; not sentinel counts, not exact continuous boundary',
            primary_selection='Use original m/model unless its mean core/check volume fraction <.50, then use gravity m/model; do not inspect downstream risk',
            significance='Geometry selfcheck conditional on frozen simulator; correlatedframes, not confidence bound or formal coverage guarantee'),
        gate=dict(query_forward_m=[.9,2.1],width_m=.29,heights=R1.HEIGHTS,range_m=R2.RANGE,age_seconds=.6,
            definition='R2 estimated current worldgrid+boundary sentinels; nominalhistory union with halfFOV22.5deg-m both axes; actual sensor union unshrunk45deg FOV',
            direction='past1s displacement>=.05m,frames<5 unknown; currentheightorigin',validity='Simulation assumption finiteSE3 and availabledirection; hardware tracking quality still absent'),
        outputs=dict(single_dual='Frozen originalM3score/query/threshold',threshold=R1.THRESHOLD,
            tau='R1exact20common labelblind scorequantiles retained, no re-fit',A='no hold',B='prior obstacle output within2.6s forbids clear, noalarm current becomes unknown; no invented earlier scores'),
        label=dict(forward_m=[0,2.1],width_m=.30,heights=R1.HEIGHTS,contact='Closed boxes,currenttrue travel,anyheight; outsidecore remains mainerror',
            graze='samewindow,width.40 shell minus corecontact',person_front='savedtravelorigin proxy, anatomical offset NOT_AVAILABLE'),
        metrics='all20points risk(error/clear),unknown/frames,stricttrueclear availability,alarm; wholeunit1000bootstrap stratifiedbatch/mode/turn, pairedarms',
        source_diagnostics='Trueposes geometric corecoverage gaps, contact angular/rangeFOV witnesses, uncheckedcontact witnesses/near/edge, labelboundary. Overlap allowed. No occlusion/cane efficacy proof',
        sensitivity='m-1 and m+1 clippedat0, only corevolume and single/dual A/B availability; no risk-based selection',
        stage2='Recommendation only;alpha5%first,zeroerrorCP perarmdelta.025,one preselectedconfig/frame perindependentunit; no generation',
        expected_only='singlefewclear acrossnatural modes;dualmore inmode0/1;errorsmostlyreadout/unchecked; not criteria',
        restrictions='CPUonly,existingdata only; no photonrender/train/modelinference/GPU; retainr1/r2; no scientific pass/fail gate in development',
        source_sha256={str(p.relative_to(ROOT)):R1.sha(p) for p in [Path(__file__),Path(R1.__file__),Path(R2.__file__),frozen]}))
    (OUT/'PLAN.sha256').write_text(R1.sha(OUT/'PLAN.json')+'\n')
    (OUT/'source').mkdir();(OUT/'source/main_at_freeze.py').write_bytes(Path(__file__).read_bytes())


def unit_worker(job):
    """Label-blind public geometry; one unit's configs share metadata load."""
    import cnh_tristate_dev_r3_geometry as G
    batch,unit,configs,model,margins=job
    sensor=travel=None;cached_noisy=None
    if batch>=98000:
        op=(R1.AUG if batch==98000 else R1.CONT)/f'observations/{"calibration" if batch==98000 else "evaluation"}/unit{unit}.npz'
        with np.load(op) as z:
            sensor=z['sensor_center'];cached_noisy=z['noisy_center'];oc=z['configs'].tolist()
    poses=[];gates=[];counts=[];missing=[];volumes=[]
    for c in configs:
        if model=='original':
            est=cached_noisy[oc.index(c)] if batch>=98000 else R1.original_motion(unit,c)[2]
        else:
            if batch<98000:
                true_sensor=R1.original_motion(unit,c)[0];seed=2026092900+unit*1000+c*10+7
            else:
                # Original frozen aug data applies a reflection after noisy generation.
                mirrored=bool((unit//3)%2);reflect=np.diag([-1.,1.,1.,1.]);true_sensor=reflect@sensor@reflect if mirrored else sensor
                meta_plan=R1.read((R1.AUG if batch==98000 else R1.CONT)/'PLAN.json')
                seed=int(np.random.SeedSequence([meta_plan.get('noise_seed_prefix',2026100607),unit,c,0]).generate_state(1)[0])
            est=G.noise_pose(true_sensor,seed,'gravity')
            if batch>=98000 and mirrored:est=reflect@est@reflect
        seqgate=[];seqcount=[];seqmissing=[]
        for f in R1.FRAMES:
            data=G.prepared(est,int(f));ev=G.gates(data,margins)
            seqgate.append(ev['passed']);seqcount.append(ev['mask_count']);seqmissing.append(ev['missing_count'])
        poses.append(est);gates.append(seqgate);counts.append(seqcount);missing.append(seqmissing)
    return dict(unit=unit,poses=np.asarray(poses),gate=np.asarray(gates),mask_count=np.asarray(counts),missing_count=np.asarray(missing))


def online():
    tick=time.monotonic();deadline();selection=R1.read(OUT/'calibration.json')
    model=selection['selected_model'];m=selection['selected_m'];margins=np.array([max(0,m-1),m,m+1.])
    # Scientific scores are opened only after geometric margin selection is sealed.
    rows=R1.read(R1.OUT/'rows.json')
    with np.load(R1.OUT/'online.npz') as z:base={k:z[k] for k in ('score','aug_score','thresholds')}
    grouped={}
    for r in rows:grouped.setdefault((r['batch'],r['unit']),[]).append(r['config'])
    jobs=[(b,u,cc,model,margins) for (b,u),cc in grouped.items()];result=[]
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,unitdata in enumerate(pool.map(unit_worker,jobs,chunksize=1)):
            deadline();result.append(unitdata)
            if (i+1)%24==0:print('onlineunits',i+1,'/',len(jobs),round(time.monotonic()-tick,1),'s',flush=True)
    arrays={key:np.concatenate([r[key] for r in result]) for key in ('poses','gate','mask_count','missing_count')}
    arrays.update(base);arrays['margins']=margins
    np.savez_compressed(OUT/'online.npz',**arrays);R1.save(OUT/'rows.json',rows)
    R1.save(OUT/'online_receipt.json',dict(seconds=time.monotonic()-tick,units=len(jobs),rows=len(rows),frames=len(rows)*13,
        selected_model=model,selected_m=m,online_sha256=R1.sha(OUT/'online.npz'),rows_sha256=R1.sha(OUT/'rows.json'),
        calibration_sha256=R1.sha(OUT/'calibration.json'),r1_online_sha256=R1.sha(R1.OUT/'online.npz'),
        r1_input_receipt_sha256=R1.sha(R1.OUT/'online_receipt.json'),source_sha256=R1.sha(__file__),
        limits='Geometrygate-only correction; originalM3inputs/scores unchanged ifgravitymodel selected'))


def summarize():
    tick=time.monotonic();deadline();rows=R1.read(OUT/'rows.json')
    with np.load(OUT/'online.npz') as z:d={k:z[k] for k in z.files}
    with np.load(OUT/'truth.npz') as z:t={k:z[k] for k in z.files}
    contact,graze=t['contact'],t['graze'];safe=~contact&~graze;whole=np.ones_like(contact,bool)
    units=np.asarray([r['unit'] for r in rows]);batches=np.asarray([r['batch'] for r in rows]);modes=np.asarray([r['mode'] for r in rows]);turns=np.asarray([r['turn'] for r in rows])
    groups={'all':np.ones(len(rows),bool)}|{f'mode{m}':modes==m for m in range(3)}|{f'turn_{x}':turns==x for x in ('left','right','none')}|{f'batch{x}':batches==x for x in R1.score_sources()}
    for b in R1.score_sources():
        for m in range(3):groups[f'batch{b}/mode{m}']=(batches==b)&(modes==m)
    alarm=d['score']>=R1.THRESHOLD
    hold=np.maximum.accumulate(alarm,axis=1) # All13retained frames span2.4s<2.6s.
    results={};rng=np.random.default_rng(2026100632)
    for name,mask in groups.items():
        if not mask.any():continue
        uu,inv=np.unique(units[mask],return_inverse=True);ncluster=len(uu)
        stratum=np.array([f'{u//1000}/{u%3}/{"left" if u>=98000 and (u//3)%2 else "right"}' for u in uu]);boot=np.zeros((1000,ncluster),np.int16)
        for s in np.unique(stratum):
            ids=np.flatnonzero(stratum==s);draw=rng.integers(0,len(ids),size=(1000,len(ids)))
            for j in range(1000):boot[j,ids]=np.bincount(draw[j],minlength=len(ids))
        def ratio(num,den):
            nn=np.bincount(inv,weights=num[mask].sum(1),minlength=ncluster);dd=np.bincount(inv,weights=den[mask].sum(1),minlength=ncluster)
            bn=boot@nn;bd=boot@dd;ok=bd>0;ci=np.quantile(bn[ok]/bd[ok],[.025,.975]).tolist() if ok.any() else [None,None]
            n,total=int(nn.sum()),int(dd.sum());return dict(numerator=n,denominator=total,value=n/total if total else None,ci95=ci,bootstrap_defined=int(ok.sum()))
        curves={}
        for a,arm in enumerate(('single','dual')):
            for variant in ('A','B'):
                curve=[]
                for index,tau in enumerate(d['thresholds']):
                    clear=~alarm[...,a]&d['gate'][:,:,1,a]&(d['score'][...,a]<=tau)
                    if variant=='B':clear &=~hold[...,a]
                    error=clear&contact;unknown=~alarm[...,a]&~clear
                    sources={}
                    for key in ('core_truth_gap','fov_in','fov_angular_in','fov_unresolved','unchecked_contact','near_contact','edge_contact','side_contact','boundary'):
                        value=t[key][...,a] if t[key].ndim==3 else t[key]
                        sources[key]=int((error&value)[mask].sum())
                    curve.append(dict(index=index,tau=float(tau),risk=ratio(error,clear),unknown=ratio(unknown,whole),clear=ratio(clear,whole),
                        obstacle=ratio(alarm[...,a],whole),availability=ratio(clear&safe,safe),no_contact_availability=ratio(clear&~contact,~contact),
                        true_clear_obstacle=ratio(alarm[...,a]&safe,safe),true_clear_unknown=ratio(unknown&safe,safe),
                        graze_clear=ratio(clear&graze,clear),sources=sources))
                curves[arm+'/'+variant]=curve
        results[name]=dict(units=ncluster,sequences=int(mask.sum()),frames=int(mask.sum()*13),curves=curves)
    sensitivity={};calibration=R1.read(OUT/'calibration.json')
    volume_curve=calibration['models'][calibration['selected_model']]['curve']
    for mi,m in enumerate(d['margins']):
        volume_entry=next(p for p in volume_curve if p['margin_deg']==float(m))
        sensitivity[str(float(m))]={'core_fraction':volume_entry['core_fraction']}
        for a,arm in enumerate(('single','dual')):
            for variant in ('A','B'):
                # Availability across all20thresholds, no risk in sensitivity.
                entries=[]
                for tau in d['thresholds']:
                    c=~alarm[...,a]&d['gate'][:,:,mi,a]&(d['score'][...,a]<=tau)
                    if variant=='B':c&=~hold[...,a]
                    entries.append(dict(numerator=int((c&safe).sum()),denominator=int(safe.sum()),value=float((c&safe).sum()/safe.sum())))
                sensitivity[str(float(m))][arm+'/'+variant]=entries
    # Planning uses descriptive largest grid point <=5% and nonempty output.
    planning={}
    from scipy.stats import binom
    for key,curve in results['all']['curves'].items():
        candidates=[p for p in curve if p['risk']['value'] is not None and p['risk']['value']<=.05]
        if not candidates:planning[key]=dict(status='NO_DEVELOPMENT_POINT_AT5PERCENT');continue
        p=candidates[-1];yield_rate=p['clear']['value'];needed=math.ceil(72/yield_rate)
        lower,upper=needed,max(needed+1,needed*2)
        while binom.sf(71,upper,yield_rate)<.95:upper*=2
        while lower<upper:
            mid=(lower+upper)//2
            if binom.sf(71,mid,yield_rate)>=.95:upper=mid
            else:lower=mid+1
        n95=lower
        planning[key]=dict(tau=p['tau'],index=p['index'],empirical_risk=p['risk'],clear_yield=yield_rate,expected_zero_error_units=needed,
            units95pct_atleast72clear=n95,scope='Yieldplanning only; observederrors may prevent CPcertificate even with enough outputs; oldframes not independent')
    augmented={};ath=R1.read(R1.CONT/'evaluator/final_calibration.json')['threshold']
    for b in (98000,99000):
        mm=batches==b;augmented[str(b)]={}
        for a,arm in enumerate(('single','dual')):
            aa=d['aug_score'][...,a]>=ath;hh=np.maximum.accumulate(aa,axis=1)
            for v in ('A','B'):
                c=~aa&d['gate'][:,:,1,a]&(d['aug_score'][...,a]<=d['thresholds'][-1])
                if v=='B':c&=~hh
                den=int(c[mm].sum());err=int((c&contact)[mm].sum());safe_n=int(safe[mm].sum())
                augmented[str(b)][arm+'/'+v]=dict(clear=den,error=err,risk=err/den if den else None,strictclear_n=safe_n,
                    available=int((c&safe)[mm].sum()),obstacle=int(aa[mm].sum()),unknown=int((~aa&~c)[mm].sum()),
                    alarm_threshold=ath,tau_lo=float(d['thresholds'][-1]),scope='Appendixonly: unchangedensemble3alarm threshold, use maximumsharedM3tau; not calibrated enhancedclear threshold')
    result=dict(status='DESCRIPTIVE_COMPLETE',guarantee=False,scientific_binary_verdict='NOT_APPLICABLE',groups=results,sensitivity=sensitivity,planning=planning,augmented=augmented,
        labels=dict(contact=int(contact.sum()),graze=int(graze.sum()),strict_clear=int(safe.sum()),no_contact=int((~contact).sum())),
        calibration=calibration,summary_seconds=time.monotonic()-tick,
        provenance={x:R1.sha(OUT/x) for x in ('PLAN.json','online.npz','truth.npz','calibration.json')},
        limits=['Consumed syntheticDevelopment only; clustereddescriptive intervals, no formalrisk certificate',
            'Outsidecore contact remains mainerror; labelcurrenttravelorigin proxy',
            'FOV witnesses and geometriccore gaps are proxies, no unoccludedreturn or causalreadout proof',
            'Source categories overlap; no witness does not prove completeobjectinside/outsidecore',
            'B hold history startsframe3; no earlier modelalarm was available',
            'Sensitivity reports availabilityonly; do not select m using empiricalrisk',
            'Syntheticgravity uses new gateestimate but unchanged originalM3scoreinputs; hardwareestimator not measured',
            'Single/dual optical/noise distributions and batchdirectionrecipes remain consumedDevelopment'])
    R1.save(OUT/'result.json',result)
    with (OUT/'curves.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['group','arm_variant','tau','errors','clear','risk','risk_lo','risk_hi','unknown','availability'])
        for name,group in results.items():
            for key,curve in group['curves'].items():
                for p in curve:w.writerow([name,key,p['tau'],p['risk']['numerator'],p['risk']['denominator'],p['risk']['value'],*p['risk']['ci95'],p['unknown']['value'],p['availability']['value']])
    print(json.dumps({'labels':result['labels'],'planning':planning,'summary_seconds':result['summary_seconds']},ensure_ascii=False,indent=2),flush=True)


def plot():
    os.environ['MPLCONFIGDIR']=str(OUT/'mpl-cache')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=R1.read(OUT/'result.json');fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,v in zip(axes,('A','B')):
        for arm,col in (('single','#2864b5'),('dual','#c76b28')):
            p=r['groups']['all']['curves'][arm+'/'+v];x=np.array([100*a['unknown']['value'] for a in p]);y=np.array([np.nan if a['risk']['value'] is None else 100*a['risk']['value'] for a in p]);
            ax.plot(x,y,'o-',color=col,label=arm+' M3')
            lower=[np.nan if a['risk']['ci95'][0] is None else 100*a['risk']['ci95'][0] for a in p];upper=[np.nan if a['risk']['ci95'][1] is None else 100*a['risk']['ci95'][1] for a in p]
            ax.fill_between(x,lower,upper,color=col,alpha=.15)
        ax.axhline(5,color='#777',ls=':',lw=1);ax.set(title='A: no hold' if v=='A' else 'B: obstacle hold2.6s',xlabel='Unable-to-judge outputs (%)',ylabel='Empirical false-clear risk (%)');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('R3 consumed synthetic Development — descriptive, not a guarantee')
    fig.savefig(OUT/'tristate_r3_curves.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','online','summarize','plot']);p.add_argument('--started-unix',type=float,default=time.time());a=p.parse_args()
    if a.stage=='freeze':freeze(a.started_unix)
    elif a.stage=='online':online()
    elif a.stage=='summarize':summarize()
    else:plot()
