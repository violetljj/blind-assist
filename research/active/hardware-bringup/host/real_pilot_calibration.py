"""Ten-condition EXPLORE instrument calibration and frozen replay.

Only selected raw pilot bytes and frozen dependencies are opened. No hardware,
training, threshold search, whole-scene subtraction or protected split access.
See the companion prewritten PLAN for all selection/interpretation rules.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'research/active/dtr-r0/nearfield'))
from cnh_h3_recording_diagnostics import load_recording
from cnh_sim2real_instrument import nnls_small, _sensor, _nominal, _expected_plane

OUT = ROOT/'artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot'
CAP = ROOT/'artifacts.local/work/real-labeled-capture-20261011'
V1 = ROOT/'artifacts.local/work/sim2real-input-contract-dev-20261011'
PLAN = Path(__file__).resolve().parents[1]/'REAL_PILOT_CALIBRATION_REPLAY_PLAN_20261011.md'
FIT = ('1', '2', '3', '4', '4b')
WIDTH = .3002784


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
def read(name): return json.loads((OUT/name).read_text(encoding='utf8'))
def data(): return {r['id']:dict(np.load(OUT/(r['id']+'.npz'))) for r in read('inventory.json')}
def good(d): return (d['S']==5)&(d['N']>0)&(d['D']>0)
def stats(x):
    x=np.asarray(x,float).ravel();x=x[np.isfinite(x)]
    return dict(n=len(x), median=float(np.median(x)) if len(x) else None,
                p10=float(np.percentile(x,10)) if len(x) else None,
                p90=float(np.percentile(x,90)) if len(x) else None)


def windows(d):
    """User-authorized relaxed pilot proxy; range-unavailable stays explicit."""
    n=len(d['H']);valid=good(d);echo=np.maximum(d['H'],0).sum(-1)
    dt=np.diff(d['ms']);cut=np.flatnonzero((np.diff(d['seq'])!=1)|(dt<=0)|(dt>500))+1
    records=[];stable=np.zeros(n,bool);blocks=[]
    for run in np.split(np.arange(n),cut):
        for start in range(0,len(run)-7,8):
            ix=run[start:start+8];pairs=[]
            for i,j in zip(ix[:-1],ix[1:]):
                common=valid[i]&valid[j]&(d['D'][i]>150)&(d['D'][j]>150);change=np.abs(d['D'][j,common]-d['D'][i,common])
                rel=2*np.abs(echo[j]-echo[i])/np.maximum(echo[j]+echo[i],1)
                p=dict(common=int(common.sum()),distance_median_mm=float(np.median(change)) if len(change) else None,
                       distance_p90_mm=float(np.percentile(change,90)) if len(change) else None,
                       echo_median=float(np.median(rel)),echo_p90=float(np.percentile(rel,90)),
                       validity_change=float(np.mean(valid[i]!=valid[j])))
                p['total_echo_relative_change']=float(2*abs(echo[j].sum()-echo[i].sum())/max(echo[j].sum()+echo[i].sum(),1))
                pairs.append(p)
            has_range=[p for p in pairs if p['common']>=1]
            echo_ok=np.median([p['total_echo_relative_change'] for p in pairs])<=.20 and np.percentile([p['total_echo_relative_change'] for p in pairs],90)<=.40
            distance_ok=len(has_range)<4 or (np.median([p['distance_median_mm'] for p in has_range])<=80 and np.median([p['distance_p90_mm'] for p in has_range])<=250)
            ok=bool(echo_ok and distance_ok)
            records.append(dict(start=int(ix[0]),end=int(ix[-1]),stable=ok,range_mode='DISTANCE_AND_ECHO' if len(has_range)>=4 else 'ECHO_ONLY',pairs=pairs))
            if ok:stable[ix]=True;blocks.append(ix)
    return stable, blocks, records


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'inventory.json').exists():raise FileExistsError('Preserve existing pilot outputs')
    manifest=json.loads((CAP/'capture.json').read_text(encoding='utf8'))
    selected={str(r['condition_index']):r for r in manifest['attempts'] if r['status'] in ('complete','needs_review') and r.get('frames',0)>0}
    inventory=[];motion={}
    for sid in ('1','2','3','4','4b','5','6','7','8','9'):
        r=selected[sid];folder=CAP/r['path'];raw=folder/'tof/raw.bin'
        if not raw.exists():raw=folder/'raw.bin'
        d=load_recording(raw);meta=json.loads((folder/'metadata.json').read_text(encoding='utf8'))
        if sha(raw)!=meta['raw_sha256'] or len(d['H'])!=r['frames']:raise ValueError('Capture identity mismatch '+sid)
        stable,blocks,records=windows(d)
        np.savez_compressed(OUT/(sid+'.npz'),**{k:d[k] for k in ('H','A','D','S','N','seq','ms')},stable=stable)
        inventory.append(dict(id=sid,condition=r['condition'],frames=len(d['H']),role='fit/diagnostic' if sid in FIT else 'evaluation_only',
                              raw_path=raw.relative_to(ROOT).as_posix(),raw_sha256=sha(raw),metadata_sha256=sha(folder/'metadata.json'),
                              quality=d['quality'],stable_frames=int(stable.sum()),stable_fraction=float(stable.mean()),stable_windows=len(blocks),
                              echo_only_windows=sum(r['stable'] and r['range_mode']=='ECHO_ONLY' for r in records)))
        motion[sid]=records
    if sum(r['frames'] for r in inventory)!=1507:raise ValueError('Unexpected pilot denominator')
    save('inventory.json',inventory);save('motion_windows.json',motion)
    save('prepare_receipt.json',dict(plan_sha256=sha(PLAN),code_sha256=sha(__file__),capture_manifest_sha256=sha(CAP/'capture.json')))
    print('STABLE',[(r['id'],r['stable_frames'],r['frames']) for r in inventory],flush=True)


def regression(x,y):
    if len(y)<12 or np.linalg.matrix_rank(x)<3:
        return dict(status='NOT_EVALUABLE',observations=len(y)),None
    c=nnls_small(x,y);pred=x@c
    return dict(status='DESCRIPTIVE_NNLS',observations=len(y),a=float(c[0]),b=float(c[1]),c=float(c[2]),
                r_squared=float(1-np.sum((pred-y)**2)/max(np.sum((y-y.mean())**2),1e-12)),
                relative_rmse=float(np.linalg.norm(pred-y)/max(np.linalg.norm(y),1e-12)),
                rank=int(np.linalg.matrix_rank(x)),scaled_condition_number=float(np.linalg.cond(x/np.maximum(np.linalg.norm(x,axis=0),1e-12)))),c


def estimate():
    all_d=data(); fit_d={s:all_d[s] for s in FIT};arrays={};motion=read('motion_windows.json')
    blocks=lambda s:[np.arange(r['start'],r['end']+1) for r in motion[s] if r['stable']]
    # Instrument subtraction: balcony only, possible railing half excluded.
    d=fit_d['1'];valid=good(d);pool=[]
    for ix in blocks('1'):
        near=valid[ix,:,None] & (d['D'][ix,:,None]/1000<=np.maximum(1.5,(np.arange(16)+2.5)*WIDTH))
        eligible=~near.any(0)
        eligible &= (np.arange(64)[:,None]//8>=4)&(np.arange(16)[None,:]<4)
        pool.append(np.where(eligible[None],d['H'][ix],np.nan))
    coverage=np.zeros((64,16),int);bias=np.zeros((64,16))
    if pool:
        samples=np.concatenate(pool);coverage=np.isfinite(samples).sum(0)
        for z,b in np.ndindex(64,16):
            if coverage[z,b]>=8:bias[z,b]=np.median(samples[:,z,b][np.isfinite(samples[:,z,b])])
    available=coverage>=8
    arrays.update(bias=bias,bias_coverage=coverage,bias_available=available)
    # Noise fit: only paired lighting calibration scenes, stable windows.
    xs=[];ys=[];yd=[];labels=[];block_ids=[];cells=[]
    for sid in ('4','4b'):
        d=fit_d[sid]
        for bi,ix in enumerate(blocks(sid)):
            h=d['H'][ix];a=d['A'][ix];ranges=d['D'][ix]
            zone=np.ptp(a,axis=0)/np.maximum(a.mean(0),1e-12)<=.5
            centers=np.floor(ranges/WIDTH/1000).astype(int)
            not_surface=(~good(d)[ix])[:,:,None] | (abs(np.arange(16)[None,None,:]-centers[:,:,None])>2)
            mask=zone[:,None]&np.all(not_surface,axis=0)&np.any(h!=0,axis=0)
            mean=np.maximum(h.mean(0),0);var=h.var(0,ddof=1);diff=(np.diff(h,axis=0)**2).mean(0)/2
            for z,b in zip(*np.where(mask)):
                xs.append([mean[z,b],a.mean(0)[z],1.]);ys.append(var[z,b]);yd.append(diff[z,b]);labels.append(sid);block_ids.append(bi);cells.append([z,b])
    x=np.array(xs).reshape(-1,3);y=np.array(ys);diff=np.array(yd);labels=np.array(labels);bis=np.array(block_ids)
    noise,c=regression(x,y);noise['difference_variance_fit'],_=regression(x,diff)
    noise['by_lighting']={s:regression(x[labels==s],y[labels==s])[0] for s in ('4','4b')}
    noise['odd_even_validation']={}
    for parity in (0,1):
        train=(bis%2)==parity;test=~train;res,cc=regression(x[train],y[train])
        if cc is not None and test.any():res.update(test_n=int(test.sum()),test_r_squared=float(1-np.sum((x[test]@cc-y[test])**2)/max(np.sum((y[test]-y[test].mean())**2),1e-12)))
        noise['odd_even_validation'][str(parity)]=res
    if c is None or not np.any(c>0):
        c=np.array([1.,16.,0.]);noise['fallback']='Frozen synthetic 16*A+positive_bias; no physical unit calibration'
    noise['adapter_coefficients_a_b_c']=c.tolist()
    positive=x@c if len(x) else np.array([]);positive=positive[positive>0]
    floor=max(float(np.median(positive))*1e-3 if len(positive) else 0.,1e-12)
    noise['numerical_variance_floor']=floor
    noise['variance_over_difference_variance']=stats(y/np.maximum(diff,1e-12));noise['ambient_span']=stats(x[:,1])
    noise['units']='Decoded CNH units squared; not lux/photon counts; stable head-worn motion/serial processing remain confounded'
    arrays.update(noise_x=x,noise_y=y,noise_difference_y=diff,noise_sid=labels,noise_block=bis,noise_cells=np.array(cells).reshape(-1,2))
    # Fixed patterns: held-out conditions used only as diagnostics, never parameters.
    fixed={}
    for sid,d in all_d.items():
        ids=np.flatnonzero(d['stable'])
        if not len(ids):fixed[sid]=dict(status='NOT_EVALUABLE');continue
        med=np.median(d['H'][ids],0);residual=med-np.median(med,0)[None,:]
        spread=1.4826*np.median(abs(med-np.median(med,0)[None,:]),0)
        peak=np.floor(d['D'][ids]/1000/WIDTH).astype(int)
        no_peak=(~good(d)[ids])[:,:,None] | (abs(np.arange(16)[None,None,:]-peak[:,:,None])>2)
        arrays[sid+'_stable_median']=med;arrays[sid+'_fixed_residual']=residual
        fixed[sid]=dict(stable_frames=len(ids),row4_col2_bins12=dict(median=med[34,1:3].tolist(),spatial_residual=residual[34,1:3].tolist(),
                         spatial_MAD=spread[1:3].tolist(),above_3_spatial_MAD=(residual[34,1:3]>3*spread[1:3]).tolist(),
                         outside_valid_firmware_peak_fraction=no_peak[:,34,1:3].mean(0).tolist()),
                         all_cell_residual=stats(residual),limit='Spatial residual is not uniquely identified instrument bias')
    # Frozen nominal simulator analytic expectation; no synthetic scenes generated.
    f,source=_sensor();p=_nominal(f);rays,_=f.angular_rays(1);cos=rays.reshape(64,3)[:,2]
    plane={};kernels=[];gain_x=[];gain_y=[]
    for sid,low,high in [('2',.3,1.),('3',1.,2.)]:
        d=fit_d[sid];ix=np.flatnonzero(d['stable']);items=[]
        if len(ix):
            valid=good(d)[ix];ranges=d['D'][ix];h=np.median(d['H'][ix],0)-bias
            dist=np.median(ranges,0)/1000
            zone=valid.all(0)&(np.ptp(ranges,axis=0)<=100)&(dist>=low)&(dist<=high)
            axial=float(np.median(dist[zone]*cos[zone])) if zone.any() else None
            expected,_=_expected_plane(f,p,axial) if axial is not None else (np.zeros((64,16)),None)
            for z in np.flatnonzero(zone):
                center=int(np.floor(dist[z]/WIDTH));bs=np.arange(max(0,center-2),min(16,center+3));obs=np.maximum(h[z,bs],0)
                if obs.sum()<=0 or expected[z,bs].sum()<=0:continue
                probs=np.zeros(5);probs[bs-center+2]=obs/obs.sum()
                items.append(dict(zone=int(z),distance_m=float(dist[z]),center=center,observed_energy=float(obs.sum()),nominal_energy=float(expected[z,bs].sum()),
                                  profile=probs.tolist(),observed_local=h[z,bs].tolist(),bins=bs.tolist()))
                if sid=='2':kernels.append(probs);gain_x.append(expected[z,bs].sum());gain_y.append(obs.sum())
            arrays[sid+'_plane_observed']=h;arrays[sid+'_plane_nominal']=expected
            plane[sid]=dict(status='WEAK_PLANE_PROXY' if items else 'NOT_EVALUABLE',axial_m=axial,zones=len(items),items=items)
        else:plane[sid]=dict(status='NOT_EVALUABLE',zones=0,items=[])
    xx=np.array(gain_x);yy=np.array(gain_y)
    gain=float(xx@yy/(xx@xx)) if len(xx) else None
    kernel=np.median(kernels,0) if kernels else None
    if kernel is not None:kernel/=kernel.sum()
    amplitude=dict(status='EFFECTIVE_ONLY' if gain is not None else 'NOT_EVALUABLE',fit_condition='2',check_condition='3',gain_real_over_nominal=gain,
                   rho_assumed=.3,pulse_kernel_offsets=[-2,-1,0,1,2],pulse_kernel=kernel.tolist() if kernel is not None else None,
                   kernel_sigma_bins=float(np.sqrt(kernel@((np.arange(-2,3)-kernel@np.arange(-2,3))**2))) if kernel is not None else None,
                   gain_application='Signal and noise unit conversion cancels in Z; no alert gain claim, deconvolution or peak removal')
    for sid in ('2','3'):
        items=plane[sid]['items']
        if gain is not None and items:
            obs=np.array([v['observed_energy'] for v in items]);pred=gain*np.array([v['nominal_energy'] for v in items]);res=obs-pred
            raw=[];predbins=[]
            for v in items:
                raw.extend(v['observed_local']);predbins.extend((gain*v['nominal_energy']*kernel[np.array(v['bins'])-v['center']+2]).tolist())
            raw=np.array(raw);predbins=np.array(predbins)
            plane[sid].update(energy_relative_l2_residual=float(np.linalg.norm(res)/max(np.linalg.norm(obs),1e-12)),
                              observed_over_predicted_energy=stats(obs/np.maximum(pred,1e-12)),
                              fixed_kernel_relative_l2_residual=float(np.linalg.norm(raw-predbins)/max(np.linalg.norm(raw),1e-12)),
                              mean_energy_residual=float(res.mean()))
    report=dict(plan_sha256=sha(PLAN),fit_conditions=list(FIT),noise=noise,
                bias=dict(covered_cells=int(available.sum()),total_cells=1024,coverage_by_bin=available.sum(0).tolist(),excluded_possible_railing_raw_rows=[0,1,2,3],
                          source='SCENE_INFORMED_WEAK: upper balcony view, invalid range does not prove no surface',
                          not_evaluable_cells=int((~available).sum()),policy='uncovered subtract0; no scene-reference fallback'),
                fixed_pattern=fixed,amplitude=amplitude,plane=plane,
                simulator_source_sha256=sha(source),v1_report_sha256=sha(V1/'instrument_report.json'),v1_arrays_sha256=sha(V1/'instrument_arrays.npz'))
    np.savez_compressed(OUT/'instrument_arrays.npz',**arrays);save('instrument_parameters.json',report)
    print('ESTIMATE',json.dumps(dict(bias=report['bias'],noise=noise,amplitude=amplitude,plane={s:{k:v for k,v in plane[s].items() if k!='items'} for s in plane}),ensure_ascii=False),flush=True)


def adapt_v2(h,a,bias,coeff,floor):
    """Training-floor analogue: keep surface signal and signed residuals."""
    variance=coeff[0]*np.maximum(bias,0)+coeff[1]*np.maximum(a,0)[...,None]+coeff[2]
    return (h-bias)/np.sqrt(np.maximum(variance,floor))


def inputs(d):
    from cnh_sim2real_input_contract import adapt
    old=json.loads((V1/'instrument_report.json').read_text());n=old['noise_parameters']
    with np.load(V1/'instrument_arrays.npz') as p:b1=p['bias']
    p=read('instrument_parameters.json');n2=p['noise']
    with np.load(OUT/'instrument_arrays.npz') as a:b2=a['bias']
    return dict(v1=adapt(d['H'],d['A'],b1,np.array([n['intercept'],n['ambient_coefficient'],n['positive_bias_coefficient']])),
                v2=adapt_v2(d['H'],d['A'],b2,np.array(n2['adapter_coefficients_a_b_c']),n2['numerical_variance_floor']))


def expected(sid,q):
    if sid=='1' or (sid=='8' and q==1):return 'UNKNOWN'
    if sid in ('4','4b','9'):return 'SILENT'
    if sid=='7':return 'NO_STRONG'
    return 'ALARM'


def summarize(sid,r):
    ready=r['ready'];n=int(ready.sum());out={}
    for arm in ('m3','old5','Sensemble'):
        grades=r['S_grade'] if arm=='Sensemble' else np.where(r[arm],2,0)
        queries=[]
        for q in range(2):
            g=grades[ready,q];exp=expected(sid,q)
            match=np.zeros(len(g),bool) if exp=='UNKNOWN' else (g==0 if exp=='SILENT' else g<2 if exp=='NO_STRONG' else g>0)
            queries.append(dict(query=('HEAD','BODY')[q],expectation=exp,strong=int((g==2).sum()),light=int((g==1).sum()),
                                strong_rate=float((g==2).mean()) if n else None,light_rate=float((g==1).mean()) if n else None,
                                conform=int(match.sum()),nonconform=0 if exp=='UNKNOWN' else int((~match).sum()),
                                unknown=n if exp=='UNKNOWN' else 0))
        out[arm]=queries
    return dict(ready=n,warmup=int((~ready).sum()),resets=int(r['reset'].sum()),arms=out)


def replay(cap):
    from cnh_sim2real_frozen import FrozenReplay
    started=time.monotonic();engine=None;report=dict(status='RUNNING',segments={},plan_sha256=sha(PLAN),parameter_sha256=sha(OUT/'instrument_parameters.json'))
    def check():
        if time.monotonic()-started>=cap:raise TimeoutError('GPU command cap reached')
    try:
        engine=FrozenReplay(batch_size=32)
        report.update(models=engine.used_hashes,thresholds=dict(m3=.8557642486787612,raised=.9404184587540165,local=4.625390338985158,S=engine.s_tau),
                      backend=dict(device=str(engine.device),gpu=engine.torch.cuda.get_device_name(),feature_precision=engine.precision))
        for sid,d in data().items():
            report['segments'][sid]={}
            for path,z in inputs(d).items():
                check();result=engine.infer(z.reshape(-1,8,8,16),d['seq'],d['ms'],check)
                np.savez_compressed(OUT/(sid+'_'+path+'_scores.npz'),**result,seq=d['seq'],ms=d['ms'])
                report['segments'][sid][path]=summarize(sid,result)
                save('replay_partial.json',report)
                print('REPLAY',sid,path,report['segments'][sid][path]['ready'],[(a,[v['strong']+v['light'] for v in report['segments'][sid][path]['arms'][a]]) for a in ('m3','old5','Sensemble')],round(time.monotonic()-started,2),flush=True)
        report['status']='COMPLETE'
    except BaseException as err:
        report['error']=repr(err);report['status']='PARTIAL';raise
    finally:
        if engine is not None:engine.close();report['resources_released']=True
        report['elapsed_seconds']=time.monotonic()-started;save('replay_report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','estimate','replay']);parser.add_argument('--cap',type=float,default=230)
    args=parser.parse_args();began=time.monotonic()
    if args.stage=='prepare':prepare()
    elif args.stage=='estimate':estimate()
    else:replay(args.cap)
    print('STAGE_SECONDS',round(time.monotonic()-began,3),flush=True)
