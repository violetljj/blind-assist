"""Record-constrained dorm hypotheses, mosaic bias, frozen diagnostic replay.

No training/threshold selection/protected split. Geometry precedes any scores;
all statistics are descriptive and invalid firmware range is not empty truth.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
NEAR=ROOT/'research/active/dtr-r0/nearfield'
sys.path.insert(0,str(NEAR))
import cnh_proposal_attribution_scenes as S
from cnh_route_sensor import synthesize_response
from cnh_track_a_fov import rx
import real_pilot_calibration as P

OUT=ROOT/'artifacts.local/work/dorm-twin-discrimination-dev-20261011'
PLAN=Path(__file__).resolve().parents[1]/'DORM_TWIN_DISCRIMINATION_PLAN_20261011.md'
FROZEN_BIAS=ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy'
PITCHES=tuple(range(-25,6,5))
WIDTH=.3002784
SEEDS=(20261011031,20261011032,20261011033)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def read(name):return json.loads((OUT/name).read_text(encoding='utf8'))
def arr_stats(x):
    x=np.asarray(x,float).ravel();x=x[np.isfinite(x)]
    return dict(n=len(x),median=float(np.median(x)) if len(x) else None,p10=float(np.percentile(x,10)) if len(x) else None,p90=float(np.percentile(x,90)) if len(x) else None)

def box(name,lo,hi,rho):
    if np.any(np.array(hi)<=np.array(lo)):raise ValueError(name+' invalid positive-volume box')
    return dict(name=name,lo=list(lo),hi=list(hi),rho=float(rho))

def scene(condition,gap=.25,profile='nominal'):
    height={'near':1.4,'nominal':1.55,'far':1.7}.get(profile,1.55)
    longitudinal={'near':-.2,'nominal':0.,'far':.2}.get(profile,0.)
    dh={'near':.1,'nominal':0.,'far':-.1}.get(profile,0.)
    end={'near':2.8,'nominal':3.2,'far':4.5}.get(profile,5. if profile=='open' else 3.2)
    b=[box('floor',[-4,height,-2],[4,height+.15,6],.3),box('ceiling',[-4,height-2.95,-2],[4,height-2.8,6],.35)]
    if condition in ('2','3'):
        z={'2':.623,'3':1.452}[condition]
        b.append(box('wall_target',[-1.25,height-2.8,z],[1.25,height,z+.1],.5))
    else:
        b.append(box('end_curtain',[-1,height-2.8,end],[1,height,end+.1],.35))
        if profile!='open':
            b.append(box('left_wall',[-1.25,height-2.8,-.5],[-1.15,height,4.],.5))
            inner=.3+gap
            b.append(box('suitcase',[-inner-.40,height-(.65+dh),.60+longitudinal],[-inner,height,1.10+longitudinal],.35))
            b.append(box('desk_top',[inner,height-(.75+dh),1.4+longitudinal],[inner+.7,height-(.75+dh)+.08,2.8+longitudinal],.35))
            for z in (1.45,2.7):
                b.append(box('desk_leg',[inner+.04,height-(.75+dh),z+longitudinal],[inner+.09,height,z+.06+longitudinal],.35))
            for z in (.85,1.15):
                b.append(box('ladder_upright',[inner,height-2.,z+longitudinal],[inner+.04,height,z+.04+longitudinal],.35))
            for h in np.arange(.3,2.,.3):
                b.append(box('ladder_rung',[inner,height-h,.85+longitudinal],[inner+.04,height-h+.04,1.19+longitudinal],.35))
        if condition in ('5','6','7'):
            z=2. if condition=='6' else 1.;x=.5 if condition=='7' else 0.
            b.append(box('rod',[x-.015,height-1.9,z-.015],[x+.015,height,z+.015],.45))
            b.append(box('cylinder_proxy',[x+.04,height-.65,z-.09],[x+.22,height,z+.09],.35))
        if condition=='8':b.append(box('head_bar',[-.45,-.0125,1.2375],[.45,.0125,1.2625],.45))
    return dict(condition=condition,gap_m=gap,profile=profile,sensor_height_m=height,boxes=b)

def intersects_query(b,lo,hi):
    # AABB surface intersection, excluding mere bounding-volume containment.
    low=np.array(b['lo']);high=np.array(b['hi']);lo=np.array(lo);hi=np.array(hi)
    return bool(np.all(high>lo)&np.all(low<hi)&any(low[a]>=lo[a] or high[a]<=hi[a] for a in range(3)))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'scenes.json').exists():raise FileExistsError('Preserve existing scenes')
    scenes=[]
    for c in ('2','3','4','5','6','7','8'):
        for pitch in PITCHES:scenes.append(dict(key=f'c{c}_p{pitch}_g025_nominal',pitch_deg=pitch,kind='condition',**scene(c)))
    for gap in (.2,.3,.4,.55):
        for pitch in PITCHES:scenes.append(dict(key=f'c4_p{pitch}_g{int(gap*100):02d}_scan',pitch_deg=pitch,kind='gap',**scene('4',gap)))
    for profile,gap in [('near',.2),('far',.3),('open',.55)]:
        for pitch in PITCHES:scenes.append(dict(key=f'c4_p{pitch}_{profile}',pitch_deg=pitch,kind='sensitivity' if profile!='open' else 'open_control',**scene('4',gap,profile)))
    scenes.append(dict(key='noise_only',pitch_deg=-10,kind='noise_control',condition='noise',gap_m=0.,profile='none',sensor_height_m=1.55,boxes=[]))
    for s in scenes:
        s['ideal_query_surface_occupied']=[any(intersects_query(b,lo,hi) for b in s['boxes']) for lo,hi in zip(S.QUERY_LOW,S.QUERY_HIGH)]
    params,receipt=S.nominal_parameters()
    save('scenes.json',scenes)
    save('prepare_receipt.json',dict(plan_sha256=sha(PLAN),code_sha256=sha(__file__),prior_inventory_sha256=sha(P.OUT/'inventory.json'),
         frozen_bias_sha256=sha(FROZEN_BIAS),sensor_parameters=asdict(params),reference_receipt=receipt,
         source_sha256={str(Path(sys.modules[m].__file__).relative_to(ROOT)):sha(sys.modules[m].__file__) for m in ('cnh_route_sensor','cnh_track_a_fov','cnh_track_a_v13_sensor')},
         scenes_sha256=sha(OUT/'scenes.json'),cells=len(scenes),seeds=list(SEEDS),frames_per_seed=16,
         photo_limit='No complete corridor photo; parameterized authoring hypotheses with metadata constraints, not measured replica'))
    print('PREPARED',len(scenes),'cells',flush=True)

def mosaic():
    ds=P.data();p=P.read('instrument_parameters.json')
    with np.load(P.OUT/'instrument_arrays.npz') as f:b2=f['bias']
    coeff=np.array(p['noise']['adapter_coefficients_a_b_c']);floor=p['noise']['numerical_variance_floor']
    bias=np.zeros((64,16));beta=np.zeros_like(bias);aref=np.zeros_like(bias);coverage=np.zeros_like(bias,int);available=np.zeros_like(bias,bool)
    details=[]
    for z,b in np.ndindex(64,4):
        values=[];amb=[];sources=[];raw_sources=[];per=[];stable_qualified=0;all_qualified=0
        for sid,d in ds.items():
            h=d['H'][:,z];a=d['A'][:,z];var=coeff[0]*max(b2[z,b],0)+coeff[1]*np.maximum(a,0)+coeff[2]
            sigma=np.sqrt(np.maximum(var,floor))
            no_peak=np.maximum(h[:,1:4],0).max(-1)<=np.maximum(5*sigma,.25*np.maximum(h[:,4:8],0).max(-1))
            eligible=P.good(d)[:,z]&(d['D'][:,z]/1000>=max(1.2,(b+1.5)*WIDTH))&no_peak
            stable_qualified+=int((eligible&d['stable']).sum());all_qualified+=int(eligible.sum())
            raw_sources.append((sid,d,eligible))
        prefer=stable_qualified>=16
        for sid,d,mask in raw_sources:
            if prefer:mask=mask&d['stable']
            v=d['H'][mask,z,b];a=d['A'][mask,z]
            if len(v):values.extend(v);amb.extend(a);sources.extend([sid]*len(v))
            if len(v)>=8:per.append(dict(segment=sid,n=len(v),median_h=float(np.median(v)),median_a=float(np.median(a))))
        v=np.array(values);a=np.array(amb);n=len(v);coverage[z,b]=n
        record=dict(zone=z,bin=b,n=n,stable_candidates=stable_qualified,all_candidates=all_qualified,stable_only=prefer,
                    source_counts={sid:sources.count(sid) for sid in sorted(set(sources))},segment_medians=per,
                    status='NOT_EVALUABLE',median=arr_stats(v),ambient=arr_stats(a))
        if n>=16:
            available[z,b]=True;ar=float(np.median(a));slopes=[]
            if len(per)>=3:
                for j,q in enumerate(per):
                    for r in per[:j]:
                        da=q['median_a']-r['median_a']
                        if abs(da)>=1:slopes.append((q['median_h']-r['median_h'])/da)
            be=float(np.median(slopes)) if slopes else 0.
            intercept=float(np.median(v-be*(a-ar)));res=v-(intercept+be*(a-ar))
            loo=[]
            for sid in set(sources):
                keep=np.array(sources)!=sid
                if keep.sum()>=16:loo.append(float(np.median(v[keep]-be*(a[keep]-ar))))
            bias[z,b]=intercept;beta[z,b]=be;aref[z,b]=ar
            record.update(status='AMBIENT_REGRESSION_WEAK' if slopes else 'POOLED_MEDIAN_WEAK',intercept=intercept,beta=be,a_ref=ar,
                          residual_MAD=float(1.4826*np.median(abs(res-np.median(res)))),residual=arr_stats(res),loo_intercept=arr_stats(loo))
        details.append(record)
    np.savez_compressed(OUT/'mosaic_bias_arrays.npz',bias=bias,beta=beta,a_ref=aref,coverage=coverage,available=available)
    comparisons={}
    for label,path in [('v1',P.V1/'instrument_arrays.npz'),('v2',P.OUT/'instrument_arrays.npz')]:
        with np.load(path) as f:old=f['bias'];oldmask=f['bias_available']
        mask=available&oldmask
        comparisons[label]=dict(prior_covered=int(oldmask.sum()),common_cells=int(mask.sum()),difference_at_v3_reference=arr_stats((bias-old)[mask]),
                                pearson=float(np.corrcoef(bias[mask],old[mask])[0,1]) if mask.sum()>1 and np.std(bias[mask])*np.std(old[mask])>0 else None)
    report=dict(plan_sha256=sha(PLAN),scope='All ten consumed recordings reused; firmware far/no-peak cannot exclude weak near surfaces',
                covered_cells=int(available.sum()),total_cells=1024,near_cell_denominator=256,coverage_by_bin=available.sum(0).tolist(),
                unavailable_policy='Subtract0, NOT_EVALUABLE, no spatial interpolation',cells=details,comparison=comparisons,
                v2_noise_parameters_fixed=p['noise'],v2_bias_sha256=sha(P.OUT/'instrument_arrays.npz'),
                normalization='Only numerator instrument template replaced; v2 denominator still uses v2 bias, coefficients/floor unchanged')
    save('mosaic_bias_report.json',report)
    print('MOSAIC',report['covered_cells'],'/1024',report['coverage_by_bin'],flush=True)

def bias3(a):
    with np.load(OUT/'mosaic_bias_arrays.npz') as p:
        return np.where(p['available'],p['bias']+p['beta']*(a[...,None]-p['a_ref']),0.)

def real_input(d,path):
    if path!='v3':return P.inputs(d)[path]
    params=P.read('instrument_parameters.json')['noise'];c=np.array(params['adapter_coefficients_a_b_c'])
    with np.load(P.OUT/'instrument_arrays.npz') as f:b2=f['bias']
    variance=c[0]*np.maximum(b2,0)+c[1]*np.maximum(d['A'],0)[...,None]+c[2]
    return (d['H']-bias3(d['A']))/np.sqrt(np.maximum(variance,params['numerical_variance_floor']))

def generate():
    start=time.monotonic();params,_=S.nominal_parameters();rays,weights=S.ray_grid();native_bias=np.load(FROZEN_BIAS)
    receipts=[]
    for s in read('scenes.json'):
        directions=rays@rx(s['pitch_deg']).T
        hit=S.raycast_boxes(np.zeros(3),directions,s['boxes'])
        distance=hit['distance'];rho=hit['rho'];cos=hit['cos']
        hs=[];ambient=[];sequences=[];ms=[];expected=None
        for k,seed in enumerate(SEEDS):
            # Static repeated physical observation, independent electronic noise.
            response=synthesize_response(np.broadcast_to(distance,(16,*distance.shape)),rho,cos,weights,params=params,seed=seed)
            h=response['histogram'].reshape(16,8,8,16,8).sum(-1)
            hs.append(h);ambient.append(response['ambient']);sequences.extend(range(16));ms.extend(range(0,3200,200))
            if expected is None:expected=response['diagnostics']['expected_signal'][0].reshape(8,8,16,8).sum(-1)
        h=np.concatenate(hs);a=np.concatenate(ambient)
        z=(h-native_bias)/np.sqrt(np.maximum(16*a[...,None]+np.maximum(native_bias,0),1e-9))
        geo_range=np.full((8,8),np.nan)
        for rr,cc in np.ndindex(8,8):
            vals=distance[rr,cc];vals=vals[np.isfinite(vals)]
            if len(vals):geo_range[rr,cc]=np.median(vals)
        np.savez_compressed(OUT/(s['key']+'_input.npz'),H=h,A=a,Z=z,seq=np.array(sequences),ms=np.array(ms),
                            expected_signal=expected,geometric_range=geo_range,object_id=hit['object_id'])
        receipts.append(dict(key=s['key'],sha256=sha(OUT/(s['key']+'_input.npz')),frames=len(h),range=arr_stats(geo_range),expected_signal=arr_stats(expected)))
        if time.monotonic()-start>400:raise TimeoutError('CPU generation400s cap')
    save('generation_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-start,cells=receipts,
         frozen_bias_sha256=sha(FROZEN_BIAS),sensor_parameters=asdict(params),code_sha256=sha(__file__)))
    print('GENERATED',len(receipts),'cells',round(time.monotonic()-start,3),flush=True)

def counts(r):
    ready=r['ready'];n=int(ready.sum());arms={}
    for arm in ('m3','old5','Sensemble'):
        grade=r['S_grade'] if arm=='Sensemble' else np.where(r[arm],2,0)
        arms[arm]=[dict(query=('HEAD','BODY')[q],strong=int((grade[ready,q]==2).sum()),light=int((grade[ready,q]==1).sum()),
                        strong_rate=float((grade[ready,q]==2).mean()),light_rate=float((grade[ready,q]==1).mean())) for q in range(2)]
    return dict(ready=n,warmup=int((~ready).sum()),resets=int(r['reset'].sum()),arms=arms)

def curtain_probe():
    # Post-primary engineering component diagnostic, declared before these scores.
    from cnh_sim2real_frozen import FrozenReplay
    start=time.monotonic();params,_=S.nominal_parameters();rays,weights=S.ray_grid();native_bias=np.load(FROZEN_BIAS);rows=[];engine=None
    try:
        engine=FrozenReplay()
        for profile,gap,end in [('near',.2,3.2),('nominal',.25,2.8)]:
            for pitch in (-15,-10,-5,0):
                s=scene('4',gap,profile);s.update(key=f'probe_{profile}_curtain{end}_p{pitch}',pitch_deg=pitch)
                for b in s['boxes']:
                    if b['name']=='end_curtain':b['lo'][2]=end;b['hi'][2]=end+.1
                hit=S.raycast_boxes(np.zeros(3),rays@rx(pitch).T,s['boxes']);hs=[];aa=[]
                for seed in SEEDS:
                    response=synthesize_response(np.broadcast_to(hit['distance'],(16,*hit['distance'].shape)),hit['rho'],hit['cos'],weights,params=params,seed=seed)
                    hs.append(response['histogram'].reshape(16,8,8,16,8).sum(-1));aa.append(response['ambient'])
                h=np.concatenate(hs);a=np.concatenate(aa);z=(h-native_bias)/np.sqrt(np.maximum(16*a[...,None]+np.maximum(native_bias,0),1e-9))
                seq=np.tile(np.arange(16),3);ms=seq*200;engine.configure_geometry(pitch);r=engine.infer(z,seq,ms)
                np.savez_compressed(OUT/(s['key']+'_input.npz'),H=h,A=a,Z=z,seq=seq,ms=ms)
                np.savez_compressed(OUT/(s['key']+'_scores.npz'),**r)
                rows.append(dict(scene=s,counts=counts(r)))
        save('curtain_probe_report.json',dict(status='COMPLETE',seconds=time.monotonic()-start,rows=rows,diagnostic_only=True,plan_sha256=sha(PLAN),resources_released=True))
    finally:
        if engine:engine.close()
    print('CURTAIN_PROBE',round(time.monotonic()-start,3),flush=True)

def infer(cap):
    from cnh_sim2real_frozen import FrozenReplay
    start=time.monotonic();engine=None;report=dict(status='RUNNING',twin={},real={},code_sha256=sha(__file__),plan_sha256=sha(PLAN))
    def check():
        if time.monotonic()-start>=cap:raise TimeoutError('Frozen diagnostic GPU cap')
    try:
        engine=FrozenReplay(batch_size=32);report.update(models=engine.used_hashes,thresholds=dict(m3=.8557642486787612,raised=.9404184587540165,local=4.625390338985158,S=engine.s_tau),
                    backend=dict(device=str(engine.device),gpu=engine.torch.cuda.get_device_name(),precision=engine.precision))
        scenes=read('scenes.json')
        for pitch in PITCHES:
            check();engine.configure_geometry(pitch)
            for s in [s for s in scenes if s['pitch_deg']==pitch]:
                with np.load(OUT/(s['key']+'_input.npz')) as d:r=engine.infer(d['Z'],d['seq'],d['ms'],check)
                np.savez_compressed(OUT/(s['key']+'_matched_scores.npz'),**r)
                report['twin'].setdefault(s['key'],{})['matched']=counts(r)
            save('inference_partial.json',report);print('MATCHED',pitch,'elapsed',round(time.monotonic()-start,2),flush=True)
        engine.configure_geometry(-10.)
        for s in scenes:
            with np.load(OUT/(s['key']+'_input.npz')) as d:r=engine.infer(d['Z'],d['seq'],d['ms'],check)
            np.savez_compressed(OUT/(s['key']+'_legacy_scores.npz'),**r)
            report['twin'][s['key']]['legacy']=counts(r)
        for sid,d in P.data().items():
            z=real_input(d,'v3');r=engine.infer(z.reshape(-1,8,8,16),d['seq'],d['ms'],check)
            np.savez_compressed(OUT/(sid+'_v3_scores.npz'),**r)
            report['real'][sid]={'v3':counts(r)}
            for path in ('v1','v2'):
                source=P.OUT/(sid+'_'+path+'_scores.npz')
                with np.load(source) as prior:report['real'][sid][path]=counts(dict(prior))
            print('REAL',sid,'v3',[(a,[q['strong'] for q in report['real'][sid]['v3']['arms'][a]]) for a in ('m3','old5','Sensemble')],flush=True)
        report['status']='COMPLETE'
    except BaseException as e:report['status']='PARTIAL';report['error']=repr(e);raise
    finally:
        if engine is not None:engine.close();report['resources_released']=True
        report['seconds']=time.monotonic()-start;save('inference_report.json',report)
    print('INFER_SECONDS',round(report['seconds'],3),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','mosaic','generate','infer','curtain_probe']);parser.add_argument('--cap',type=float,default=390)
    a=parser.parse_args();started=time.monotonic()
    globals()[a.stage](a.cap) if a.stage=='infer' else globals()[a.stage]()
    print('COMMAND_SCRIPT_SECONDS',round(time.monotonic()-started,3),flush=True)
