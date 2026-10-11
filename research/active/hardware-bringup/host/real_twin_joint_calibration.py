"""Histogram-only noise/geometry fit and separated frozen replay closure."""
from __future__ import annotations
import argparse,json,time,copy
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, minimize
import real_twin_joint_instrument as I
import real_pilot_calibration as P
import dorm_twin_discrimination as D
OUT=I.OUT
PLAN=Path(__file__).resolve().parents[1]/'REAL_TWIN_JOINT_CALIBRATION_PLAN_20261011.md'
GNAMES=('pitch_deg','curtain_distance_m','gap_left_m','gap_right_m','curtain_rho_effective')
GBOUNDS=[(-25,5),(2.8,4.5),(.2,.30),(.2,.30),(.02,.7)]
SEEDS=(20261011111,20261011112,20261011113)
def read(name):return json.loads((OUT/name).read_text(encoding='utf8'))
def save(name,obj):I.save(name,obj)
def recording(sid):return dict(np.load(P.OUT/(sid+'.npz')))
@lru_cache(maxsize=1)
def instrument():return read('instrument_fit.json')['instrument_parameters'],np.load(OUT/'instrument_fit_arrays.npz')['bias'].reshape(8,8,16)
@lru_cache(maxsize=1)
def noise_model():return read('noise_fit.json')

def noise():
    start=time.monotonic();params,bias=instrument();bias=bias.reshape(64,16);xs=[];ys=[];labels=[];ids=[]
    for sid in ('4','4b'):
        d=recording(sid)
        for j,w in enumerate(P.read('motion_windows.json')[sid]):
            if not w['stable']:continue
            h=d['H'][w['start']:w['end']+1];a=d['A'][w['start']:w['end']+1]
            mean=np.maximum(h.mean(0)-bias,0);var=h.var(0,ddof=1)
            x=np.stack([mean,np.broadcast_to(a.mean(0)[:,None],mean.shape),np.ones_like(mean)],-1)
            xs.extend(x.reshape(-1,3));ys.extend(var.ravel());labels.extend([sid]*1024);ids.extend([j]*1024)
    x=np.asarray(xs);y=np.asarray(ys);c=P.nnls_small(x,y);pred=x@c
    def metric(y,p):return dict(r_squared=float(1-np.sum((y-p)**2)/max(np.sum((y-y.mean())**2),1e-12)),relative_rmse=float(np.linalg.norm(y-p)/max(np.linalg.norm(y),1e-12)))
    cross=[]
    for parity in (0,1):
        train=np.array(ids)%2==parity;cc=P.nnls_small(x[train],y[train]);cross.append(dict(train_parity=parity,coefficients=cc.tolist(),test=metric(y[~train],x[~train]@cc)))
    positive=y[y>0];floor=float(max(np.percentile(positive,1)*.01,1e-8))
    result=dict(status='EFFECTIVE_DECODED_NOISE',coefficients_a_b_c=c.tolist(),fit=metric(y,pred),observations=len(y),numerical_floor=floor,
                segment={sid:metric(y[np.array(labels)==sid],pred[np.array(labels)==sid]) for sid in ('4','4b')},odd_even_window_check=cross,
                limit='All stable window cells retained; motion/weak returns enter effective variance, no photon unit identification',
                instrument_sha256=I.sha(OUT/'instrument_fit.json'),plan_sha256=I.sha(PLAN),seconds=time.monotonic()-start)
    np.savez_compressed(OUT/'noise_fit_arrays.npz',x=x,y=y,prediction=pred,window=np.array(ids),segment=np.array(labels))
    save('noise_fit.json',result);print('NOISE',result,flush=True)

def variance(signal,ambient):
    n=noise_model();a,b,c=n['coefficients_a_b_c']
    return np.maximum(a*np.maximum(signal,0)+b*np.maximum(ambient,0)[...,None]+c,n['numerical_floor'])

def boxes(g,condition='4',position=None):
    pitch,end,gl,gr,rho=g;s=D.scene(condition,.25)
    for box in s['boxes']:
        name=box['name']
        if name=='end_curtain':box['lo'][2]=end;box['hi'][2]=end+.1;box['rho']=rho
        shift=-gl+.25 if name=='suitcase' else gr-.25 if name in ('desk_top','desk_leg','ladder_upright','ladder_rung') else 0
        box['lo'][0]+=shift;box['hi'][0]+=shift
        if position is not None and name in ('rod','cylinder_proxy','head_bar'):
            for axis,dx in enumerate(position):box['lo'][axis]+=dx;box['hi'][axis]+=dx
    return s['boxes']

def expected(g,condition='4',position=None,samples=4):
    params,bias=instrument();rays,weights=I.plane_rays(samples)
    hit=D.S.raycast_boxes(np.zeros(3),rays@D.rx(g[0]).T,boxes(g,condition,position))
    return I.predict(hit['distance'],hit['rho'],hit['cos'],weights,params,bias=bias).reshape(64,16)

def geometry(cap):
    start=time.monotonic();observations=[];ambients=[];scales=[];ds={}
    for sid in ('4','4b','9'):
        d=recording(sid);ds[sid]=d;mean=d['H'][d['stable']].mean(0);a=d['A'][d['stable']].mean(0)
        _,bg=instrument();var=variance(mean-bg.reshape(64,16),a)
        observations.append(mean);ambients.append(a);scales.append(np.sqrt(var))
    obs=np.array(observations);scale=np.array(scales);calls=0
    def objective(g):
        nonlocal calls
        calls+=1
        if time.monotonic()-start>cap:raise TimeoutError('Geometry CPU cap')
        pred=expected(g);return float(np.mean(((pred-obs)/scale)**2))
    # No score imports/calls. Deterministic objective-only multistart.
    coarse=[]
    for pitch in (-25,-20,-15,-10,-5,0,5):
        for end in (2.8,3.,3.2,3.6,4.,4.5):
            g=np.array([pitch,end,.25,.25,.35]);coarse.append((objective(g),g))
    starts=sorted(coarse,key=lambda q:q[0])[:3];attempts=[]
    de=differential_evolution(objective,GBOUNDS,popsize=7,maxiter=35,seed=202610111,polish=False,tol=.002,workers=1)
    starts.append((float(de.fun),de.x))
    for _,g in starts:
        opt=minimize(objective,g,method='Powell',bounds=GBOUNDS,options={'maxiter':30,'ftol':1e-5,'xtol':1e-3})
        attempts.append(dict(objective=float(opt.fun),parameters=opt.x.tolist(),converged=bool(opt.success)))
    best=min(attempts,key=lambda q:q['objective']);g=np.array(best['parameters']);minimum=objective(g);profiles={}
    for j,name in enumerate(GNAMES):
        rows=[];free=np.arange(5)!=j
        for value in np.unique(np.r_[np.linspace(*GBOUNDS[j],7),g[j]]):
            trial=g.copy();trial[j]=value
            def fn(v):
                x=trial.copy();x[free]=v;return objective(x)
            opt=minimize(fn,trial[free],method='Powell',bounds=[b for k,b in enumerate(GBOUNDS) if k!=j],options={'maxiter':12,'ftol':5e-4,'xtol':.005})
            full=trial.copy();full[free]=opt.x
            rows.append(dict(value=float(value),objective=float(opt.fun),free_parameters=opt.x.tolist(),parameters=full.tolist()))
        admitted=[q['value'] for q in rows if q['objective']<=1.05*minimum]
        profiles[name]=dict(grid=rows,descriptive_set=[min(admitted),max(admitted)] if admitted else None,bound_included=[GBOUNDS[j][0] in admitted,GBOUNDS[j][1] in admitted])
    old_centre=dict(parameters=g.tolist(),objective=minimum)
    candidates=[q for p in profiles.values() for q in p['grid']]
    improved=min(candidates,key=lambda q:q['objective'])
    if improved['objective']<minimum:
        g=np.array(improved['parameters']);minimum=objective(g)
    for j,name in enumerate(GNAMES):
        profiles[name]['grid'].append(dict(value=float(g[j]),objective=minimum,parameters=g.tolist(),centre=True))
        admitted=[q['value'] for q in profiles[name]['grid'] if q['objective']<=1.05*minimum]
        profiles[name]['descriptive_set']=[min(admitted),max(admitted)]
        profiles[name]['bound_included']=[GBOUNDS[j][0] in admitted,GBOUNDS[j][1] in admitted]
    pred4=expected(g);pred=expected(g,samples=16);_,bg=instrument();bg=bg.reshape(64,16)
    metrics={}
    for k,sid in enumerate(('4','4b','9')):
        metrics[sid]=dict(signal_relative_l2=float(np.linalg.norm(pred-obs[k])/max(np.linalg.norm(obs[k]-bg),1e-12)),weighted_rmse=float(np.sqrt(np.mean(((pred-obs[k])/scale[k])**2))),raw_relative_l2=float(np.linalg.norm(pred-obs[k])/max(np.linalg.norm(obs[k]),1e-12)))
    result=dict(status='HISTOGRAM_ONLY_FIXED_INSTRUMENT_GEOMETRY',parameters=dict(zip(GNAMES,map(float,g))),bounds=dict(zip(GNAMES,GBOUNDS)),objective=minimum,
                profiles=profiles,attempts=attempts,pre_profile_centre=old_centre,de_objective=float(de.fun),roll_deg=0,metrics=metrics,
                quadrature_4_to_16_relative_l2=float(np.linalg.norm(pred-pred4)/max(np.linalg.norm(pred),1e-12)),seconds=time.monotonic()-start,objective_calls=calls,
                instrument_sha256=I.sha(OUT/'instrument_fit.json'),noise_sha256=I.sha(OUT/'noise_fit.json'),plan_sha256=I.sha(PLAN),
                fit_conditions=['4','4b','9'],evaluation_only=['5','6','7','8'],profile_kind='Objective <=1.05 x minimum, conditional approximate optimization, not CI',
                limits=['Geometry authoring hypotheses and unsigned NIR reflectance nuisance; no complete corridor photo or metric extrinsics.', 'Instrument offset/distance prior confounding is carried into geometry, not propagated as a full joint posterior.'])
    np.savez_compressed(OUT/'geometry_fit_arrays.npz',observations=obs,ambient=np.array(ambients),scale=scale,prediction=pred,prediction4=pred4,parameters=g)
    save('geometry_fit.json',result);print('GEOMETRY',result['parameters'],result['metrics'],'seconds',result['seconds'],flush=True)

def normalized(d,path):
    if path=='v2':return P.inputs(d)['v2']
    _,bg=instrument();bg=bg.reshape(64,16);return (d['H']-bg)/np.sqrt(variance(bg,d['A']))

def generate():
    start=time.monotonic();fit=read('geometry_fit.json');g=np.array([fit['parameters'][n] for n in GNAMES]);params,bg=instrument();bg=bg.reshape(64,16);cells=[]
    for sid in ('4','4b','9','5','6','7','8'):
        d=recording(sid);a=d['A'][d['stable']].mean(0);obs=d['H'][d['stable']].mean(0)
        positions={'centre':np.zeros(3)}
        if sid in ('5','6','7','8'):
            dz={'5':.3,'6':.4,'7':.4,'8':.4}[sid]
            positions.update(near=np.array([0,0,-dz]),far=np.array([0,0,dz]))
            if sid=='8':positions.update(low=np.array([0,.1,0]),high=np.array([0,-.1,0]))
            else:
                dx=.15 if sid=='7' else .1;positions.update(left=np.array([-dx,0,0]),right=np.array([dx,0,0]))
        for label,pos in positions.items():
            mean=expected(g,'4' if sid in ('4b','9') else sid,pos,samples=16);var=variance(mean-bg,a)
            h=np.concatenate([mean+np.random.default_rng(seed).normal(size=(16,64,16))*np.sqrt(var) for seed in SEEDS]);aa=np.broadcast_to(a,(48,64));seq=np.tile(np.arange(16),3);ms=seq*200
            key=f'c{sid}_{label}';np.savez_compressed(OUT/(key+'_input.npz'),H=h,A=aa,seq=seq,ms=ms,expected=mean,variance=var,real_mean=obs)
            residual=mean-obs;scale=np.sqrt(variance(obs-bg,a))
            metrics=dict(signal_relative_l2=float(np.linalg.norm(residual)/max(np.linalg.norm(obs-bg),1e-12)),weighted_rmse=float(np.sqrt(np.mean((residual/scale)**2))),raw_relative_l2=float(np.linalg.norm(residual)/max(np.linalg.norm(obs),1e-12)),
                         by_bin_l1=np.abs(residual).sum(0).tolist(),top_cells=[dict(zone=int(i//16),bin=int(i%16),difference=float(residual.ravel()[i])) for i in np.argsort(abs(residual).ravel())[::-1][:8]])
            cells.append(dict(key=key,condition=sid,position=label,delta_xyz_m=pos.tolist(),metrics=metrics,input_sha256=I.sha(OUT/(key+'_input.npz'))))
    save('closure_inputs.json',dict(status='COMPLETE',cells=cells,geometry_sha256=I.sha(OUT/'geometry_fit.json'),instrument_sha256=I.sha(OUT/'instrument_fit.json'),noise_sha256=I.sha(OUT/'noise_fit.json'),plan_sha256=I.sha(PLAN),seconds=time.monotonic()-start))
    print('GENERATED',len(cells),'seconds',time.monotonic()-start,flush=True)

def replay(cap):
    from cnh_sim2real_frozen import FrozenReplay
    start=time.monotonic();e=None;report=dict(status='RUNNING',twin={},real={})
    def check():
        if time.monotonic()-start>cap:raise TimeoutError('GPU replay cap')
    try:
        e=FrozenReplay();report.update(models=e.used_hashes,thresholds=dict(m3=.8557642486787612,raised=.9404184587540165,local=4.625390338985158,S=e.s_tau),backend=str(e.device),precision=e.precision)
        for cell in read('closure_inputs.json')['cells']:
            d=dict(np.load(OUT/(cell['key']+'_input.npz')));report['twin'][cell['key']]={}
            for path in ('v2','joint'):
                z=normalized(d,path).reshape(-1,8,8,16);r=e.infer(z,d['seq'],d['ms'],check);np.savez_compressed(OUT/(cell['key']+'_'+path+'_scores.npz'),**r);report['twin'][cell['key']][path]=D.counts(r)
        for sid in ('4','4b','9','5','6','7','8'):
            d=recording(sid);report['real'][sid]={}
            for path in ('v2','joint'):
                r=e.infer(normalized(d,path).reshape(-1,8,8,16),d['seq'],d['ms'],check);np.savez_compressed(OUT/(sid+'_'+path+'_scores.npz'),**r);report['real'][sid][path]=D.counts(r)
        report['status']='COMPLETE'
    finally:
        if e:e.close()
        report.update(seconds=time.monotonic()-start,resources_released=True,fit_parameters_sha256=I.sha(OUT/'geometry_fit.json'));save('closure_replay.json',report)
    print('REPLAY',report['status'],report['seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['noise','geometry','generate','replay']);p.add_argument('--cap',type=float,default=500);a=p.parse_args()
    globals()[a.stage](a.cap) if a.stage in ('geometry','replay') else globals()[a.stage]()
