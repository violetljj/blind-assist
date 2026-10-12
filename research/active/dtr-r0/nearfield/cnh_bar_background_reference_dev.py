"""Observable neighbor-background precheck; no projection, inference or photons."""
import json
from pathlib import Path
import sys,time
import numpy as np
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S

OUT=B.ROOT/'artifacts.local/work/cnh-bar-background-reference-dev-20261009'
BASE=B.ROOT/'artifacts.local/work/cnh-bar-cached-diagnostic-dev-20261008'
WIDTH=8*.0375348
FAR=np.arange(10,16)
CENTERS=(FAR+.5)*WIDTH
DIRECTIONS=('vertical','horizontal')

def save(name,value):B.save(OUT/name,value)
def quantile(x,p=.95):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return float(np.quantile(x,p)) if len(x) else None
def summary(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(n=int(len(x)),median=float(np.median(x)) if len(x) else None,q95=quantile(x))
def neighbors(x,direction):
    axis=-3 if direction=='vertical' else -2
    return np.roll(x,1,axis),np.roll(x,-1,axis)
def interior(direction):
    ok=np.ones((8,8),bool)
    if direction=='vertical':ok[[0,7],:]=False
    else:ok[:,[0,7]]=False
    return ok

def peak(h,v,threshold):
    snr=h/np.sqrt(v);n=h.shape[-1];index=h.argmax(-1)
    left=np.concatenate((np.full((*h.shape[:-1],1),-np.inf),h[...,:-1]),-1)
    right=np.concatenate((h[...,1:],np.full((*h.shape[:-1],1),-np.inf)),-1)
    # Ties assigned to the first local maximum. Separated above-noise maxima => ambiguous.
    maxima=(h>left)&(h>=right)&(snr>=threshold)
    multi=maxima.sum(-1)>1
    strong=np.take_along_axis(snr,index[...,None],-1)[...,0]>=threshold
    weight=np.maximum(h,0)*(np.abs(np.arange(n)-index[...,None])<=1)
    denom=weight.sum(-1);location=np.sum(weight*CENTERS,-1)/np.maximum(denom,1e-30)
    width=np.sqrt(np.sum(weight*(CENTERS-location[...,None])**2,-1)/np.maximum(denom,1e-30))
    valid=strong&~multi&(denom>0)
    return dict(valid=valid,strong=strong,multi=multi,position=location,width=width)

def prepare():
    t=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PLAN.json').exists():raise FileExistsError('Preserve background-reference PLAN')
    rows=json.loads((S.OUT/'PLAN.json').read_text(encoding='utf-8-sig'))['scene_rows']
    ids=[r['id'] for r in rows if r['family']=='horizontal' and r['rho']==.25 and 'thick0.04' in r['variant']]
    assert len(ids)==44 and all(rows[i]['background']==S.BASE_BG for i in ids)
    clear=[i for i in ids if rows[i]['placement']=='clear']
    group=lambda i:(rows[i]['side'],round(rows[i]['hi'][0]-rows[i]['lo'][0],2))
    cal=[i for i in clear if group(i) in ((-1,.3),(1,.9))];val=[i for i in clear if i not in cal]
    assert len(cal)==len(val)==4 and len({group(i) for i in cal})==len({group(i) for i in val})==2
    inputs=[S.OUT/'PLAN.json',S.OUT/'geometry.npz',S.OUT/'physical.npz',BASE/'PLAN.json',BASE/'background_reference.npz']
    sources=[Path(__file__),Path(B.__file__),Path(S.__file__),Path(__file__).with_name('cnh_displacement_ceiling_render.py')]
    plan=dict(task='CNH_BAR_BACKGROUND_REFERENCE_DEV_20261009',base_commit='d937fcda',lane='EXPLORE consumed Development precheck',
        authorization='User supplied corrected roundtable nextstep background-reference precheck; preserve M3/5slot and do not claim verticalquery cost solved',
        budgets_CPU_stage_wall_seconds=dict(prepare=180,observable_fit_extraction=300,quality_summary_checks=180,independent_audit=120,geometry_evaluator=120),
        scope='44dark4cm horizontal scenes(original28contact,8pass,8clear),K4,all16rawframes/all64zones;contact56events perheight. Vertical±1rows priority,horizontal±1columns control;two-neighbor requirement,edgesUNKNOWN. Originalbg only, no inference/projection/newphotons/training.',
        scene_ids=ids,clear_cal_ids=cal,clear_validation_ids=val,cal_geometry_groups=[[-1,.3],[1,.9]],val_geometry_groups=[[-1,.9],[1,.3]],
        split='HEAD/BODY+K4+allframes stay together per(side,length)geometry; each half2geometrygroups,not independentframe cohort. No observation/outcome selection.',
        observable='Signedhist already background-subtracted. Fixedfarbins10..15=[3.002784,4.8044544)m,allcounts included for residual. Neighborprofile weights1/2 each, no oracle/shape/targettruth/backgroundabsence/truequeryerror in fit oravailability.',
        peak='Peak largest signed farbin;positive3bin neighborhood centroid/RMSwidth. Above-calibrated-noise localmax count>1 => ambiguous. Weak/multi/edge/no calibration candidates UNKNOWN. Targetpeak absent doesnot invalidate reference/negative residual.',
        calibration='Cal onlysignedhist+ambient: one-sided95% quantile max6bin centeredK4noise SNR sets peak floor;withinreplica centering corrected sqrt(1−1/4). Bothidentified referencepairs set95% distance difference,widthdifference,maxwidth andjointnoise-standardized farcount agreement tolerances. Targetprediction distance/width tolerances from cal where both references andtargetpeak identifiable. No truthfilter/noabsentmean parameter fit.',
        intensity='FarR=(target_sum−.5neighbor1_sum−.5neighbor2_sum)/sqrt(Vtarget+.25Vneighbor1+.25Vneighbor2);V=sum(max(signedbin,0)+16ambient). Positive-partμ plugin biased,not exactnoise distribution;cal noise-band95% ofabsK4centered targetprediction residual corrected sqrt.75. Thisnoise band constrains backgroundbias validation only,never filters contactnegativeR.',
        availability='Require bothref peaks andpredictedprofile peak identified; pairdistance/width/maxwidth/intensityagreement within cal95bounds. Only reference-side agreement,not targetR. Unknown supplemental evidence doesnot change system3state orbasealarm.',
        validation='Afterfit fixed: heldoutclear pergeometry report availableframe fractions, observed targetprediction distance/width errors on evaluatorunpollutedtriples, andmean backgroundpredictionbias using cachedabsenttargetmean/presentrefmeans withtruejointnoise. Oracle used ONLY evaluation,notcal/stat/availability. Separate contamination ofref subrays andtimelytargetzone eventcoverage.',
        decision_check='Onecontactevent1/56,height56each. Prereference support: verticaleachheldoutgeometry must haveavailable samples andq95absbackgroundmeanbias<=calK4noise95band; bothcontactheights need nonzero timelytargetzoneavailable events. Thisconditionalnoise-scale rule is not a globalpracticalbias tolerance or realworldvalidation. Ifcondition fails/no calref, do not run jointreadout; partial coverage/pollution remain descriptive. Passedprecheck only supports separately declared jointincrement test,not automaticpolicyupgrade.',
        adjustable_scope='Implement/repair fixedtwo-direction precheck, geometry-only pollution labels,focusedchecks/report/current/RUNS/directpush;no otherreference fallback/thresholdsweep/M3/policy change.',
        stop='Complete fixedprecheck orstagecaps includingfailures. Jointreadout/±3/strongbg increment not executed inprecheck; needs nextdeclaredcandidatequeue afterreferenceevidence.',
        inputs_sha256={B.logical_path(p):B.sha(p) for p in inputs},source_sha256={B.logical_path(p):B.sha(p) for p in sources})
    plan['prepare_seconds']=time.monotonic()-t;assert plan['prepare_seconds']<180
    save('PLAN.json',plan);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print('PREPARED',plan['prepare_seconds'],flush=True)

def compute(h,a,parameters,direction):
    v=np.maximum(h,0)+16*a[None,None,...,None];h1,h2=neighbors(h,direction);v1,v2=neighbors(v,direction)
    c=h.sum(-1);c1=h1.sum(-1);c2=h2.sum(-1);vt=v.sum(-1);vr=.25*(v1+v2).sum(-1)
    predicted=.5*(h1+h2);r=(c-.5*(c1+c2))/np.sqrt(vt+vr)
    p=peak(h,v,parameters['peak_noise_floor']);p1=peak(h1,v1,parameters['peak_noise_floor']);p2=peak(h2,v2,parameters['peak_noise_floor'])
    pp=peak(predicted,.25*(v1+v2),parameters['peak_noise_floor'])
    distance=np.abs(p1['position']-p2['position']);width=np.abs(p1['width']-p2['width'])
    agreement=np.abs(c1-c2)/np.sqrt((v1+v2).sum(-1));reason=np.zeros(r.shape,np.uint16)
    reason|=np.where(interior(direction),0,1).astype(np.uint16)
    reason|=np.where(p1['strong']&p2['strong'],0,2).astype(np.uint16)
    reason|=np.where(p1['multi']|p2['multi'],4,0).astype(np.uint16)
    for field,values,bit in [('pair_distance',distance,8),('pair_width',width,16),('pair_intensity',agreement,32),('max_width',np.maximum(p1['width'],p2['width']),256)]:
        tolerance=parameters[field];reason|=np.where(values<=tolerance,0,bit).astype(np.uint16) if tolerance is not None else bit
    reason|=np.where(pp['strong'],0,64).astype(np.uint16);reason|=np.where(pp['multi'],128,0).astype(np.uint16)
    return dict(available=reason==0,reason=reason,residual=r,observed_variance=vt+vr,
        pair_distance=distance,pair_width=width,pair_intensity=agreement,predicted_position=pp['position'],predicted_width=pp['width'],
        target_peak_valid=p['valid'],target_position=p['position'],target_width=p['width'],target_peak_strong=p['strong'],target_peak_multi=p['multi'])

def fit(h,a,direction,peak_floor):
    v=np.maximum(h,0)+16*a[None,None,...,None];h1,h2=neighbors(h,direction);v1,v2=neighbors(v,direction)
    p1=peak(h1,v1,peak_floor);p2=peak(h2,v2,peak_floor)
    base=p1['valid']&p2['valid']&interior(direction)
    parameters=dict(peak_noise_floor=peak_floor,cal_identified_pairs=int(base.sum()),
        pair_distance=quantile(np.abs(p1['position']-p2['position'])[base]),pair_width=quantile(np.abs(p1['width']-p2['width'])[base]),
        max_width=quantile(np.maximum(p1['width'],p2['width'])[base]),pair_intensity=quantile((np.abs(h1.sum(-1)-h2.sum(-1))/np.sqrt((v1+v2).sum(-1)))[base]))
    residual=h.sum(-1)-.5*(h1+h2).sum(-1);variance=(v+.25*(v1+v2)).sum(-1)
    centered=residual-residual.mean(1,keepdims=True)
    parameters['background_noise_band']=quantile((np.abs(centered)/np.sqrt(.75*variance))[np.broadcast_to(interior(direction),residual.shape)])
    z=compute(h,a,parameters,direction);used=z['available']&z['target_peak_valid']
    parameters['target_distance']=quantile(np.abs(z['target_position']-z['predicted_position'])[used])
    parameters['target_width']=quantile(np.abs(z['target_width']-z['predicted_width'])[used]);parameters['cal_available']=int(z['available'].sum())
    return parameters

def run():
    t=time.monotonic();previous=sum(json.loads(f.read_text(encoding='utf-8-sig'))['seconds'] for f in OUT.glob('failure_run*.json'))
    try:
        if (OUT/'observations.npz').exists():raise FileExistsError('Preserve observed reference scores')
        plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf-8-sig'))
        for p,h in {**plan['inputs_sha256'],**plan['source_sha256']}.items():assert B.sha(B.ROOT/p)==h,p
        ids=np.asarray(plan['scene_ids']);indices=[list(ids).index(i) for i in plan['clear_cal_ids']]
        with np.load(S.OUT/'physical.npz') as d:hist=d['hist'][ids].astype(float)[...,FAR];ambient=d['ambient']
        assert hist.shape==(44,4,16,8,8,6)
        calibration=hist[indices];mean=calibration.mean(1,keepdims=True)
        # Learned from K4 centered shot noise, not absent truth or actual inference scores.
        peak_floor=quantile(((calibration-mean)/np.sqrt(.75*(np.maximum(mean,0)+16*ambient[None,None,...,None]))).max(-1))
        assert peak_floor is not None and peak_floor>0
        parameters={d:fit(calibration,ambient,d,peak_floor) for d in DIRECTIONS}
        save('calibration.json',dict(parameters=parameters,clear_cal_ids=plan['clear_cal_ids'],uses_only='cal signedhist+ambient;noabsenttruth/targetgeometry/scoreoutcomes',quantiles=.95))
        values=dict(scene_ids=ids,ambient=ambient)
        for direction in DIRECTIONS:
            z=compute(hist,ambient,parameters[direction],direction)
            for key,value in z.items():values[direction+'_'+key]=value
        np.savez_compressed(OUT/'observations.npz',**values)
        if previous+time.monotonic()-t>300:raise TimeoutError('cumulative extraction300s')
        save('result_run.json',dict(status='COMPLETE',seconds=time.monotonic()-t,cumulative_seconds=previous+time.monotonic()-t,parameters=parameters,observed_zone_direction_records=44*4*16*8*8*2,new_photons=0,new_inference=0,new_projection=0,training=0))
        print('COMPLETE OBSERVABLE PRECHECK',time.monotonic()-t,parameters,flush=True)
    except BaseException as e:save(f'failure_run_{time.time_ns()}.json',dict(seconds=time.monotonic()-t,error=repr(e)));raise

def analyze():
    t=time.monotonic()
    if (OUT/'result_analysis.json').exists():raise FileExistsError('Preserve reference analysis')
    plan=json.loads((OUT/'PLAN.json').read_text(encoding='utf-8-sig'));par=json.loads((OUT/'calibration.json').read_text(encoding='utf-8-sig'))['parameters'];rows=json.loads((S.OUT/'PLAN.json').read_text(encoding='utf-8-sig'))['scene_rows']
    with np.load(OUT/'observations.npz') as d:obs={k:d[k] for k in d.files}
    ids=obs['scene_ids'];a=obs['ambient'];ix={int(i):j for j,i in enumerate(ids)}
    with np.load(OUT/'geometry/coverage.npz') as d:
        np.testing.assert_array_equal(ids,d['scene_ids']);hit=d['coverage']
    with np.load(S.OUT/'physical.npz') as d:present=d['expectation'][ids][...,FAR]
    with np.load(BASE/'background_reference.npz') as d:bg=d['expectation'][...,FAR];np.testing.assert_array_equal(a,d['ambient'])
    with np.load(S.OUT/'geometry.npz') as d:cat=d['category'][ids]
    records=[];events=[];quality=[]
    for direction in DIRECTIONS:
        available=obs[direction+'_available'];res=obs[direction+'_residual'];h1,h2=neighbors(present,direction)
        hf1,hf2=neighbors(hit[...,None],direction);ref_fraction=.5*(hf1[...,0]+hf2[...,0])
        bias=(bg[None]-.5*(h1+h2)).sum(-1)
        ambient_field=np.broadcast_to(a[None,...,None],present.shape)
        a1,a2=neighbors(ambient_field,direction)
        actual_var=(bg[None]+16*ambient_field+.25*(h1+16*a1+h2+16*a2)).sum(-1)
        standardized=bias/np.sqrt(actual_var);band=par[direction]['background_noise_band']
        for role,selected in [('calibration',plan['clear_cal_ids']),('validation',plan['clear_validation_ids']),('contact',[int(i) for i,c in zip(ids,cat) if (c=='contact').any()]),('pass',[int(i) for i,c in zip(ids,cat) if (c=='pass').any() and not(c=='contact').any()])]:
            indices=np.array([ix[i] for i in selected]);av=available[indices];geom=hit[indices]>0
            target=np.broadcast_to(geom[:,None],av.shape);rf=np.broadcast_to(ref_fraction[indices,None],av.shape)
            bval=np.broadcast_to(standardized[indices,None],av.shape)
            clean=~target&(rf==0)
            dist=np.abs(obs[direction+'_target_position'][indices]-obs[direction+'_predicted_position'][indices]);wd=np.abs(obs[direction+'_target_width'][indices]-obs[direction+'_predicted_width'][indices]);pv=obs[direction+'_target_peak_valid'][indices]
            row=dict(direction=direction,role=role,scenes=len(indices),all_zone_frames=int(av.size),available=int(av.sum()),available_fraction=float(av.mean()),
                target_zone_frames=int(target.sum()),available_target_zone_frames=int((av&target).sum()),target_coverage=float((av&target).sum()/target.sum()) if target.any() else None,
                available_reference_polluted_fraction=float((av&(rf>0)).sum()/av.sum()) if av.any() else None,reference_subray_fraction_mean=float(rf[av].mean()) if av.any() else None,
                clean_background_bias=summary(np.abs(bval[av&clean])),all_available_background_bias=summary(np.abs(bval[av])),observed_clean_target_distance=summary(dist[av&clean&pv]),observed_clean_target_width=summary(wd[av&clean&pv]),
                candidate_target_residual=summary(res[indices][av&target]),failure_bits={str(bit):int(((obs[direction+'_reason'][indices]&bit)>0).sum()) for bit in (1,2,4,8,16,32,64,128,256)})
            records.append(row)
        for side,length in ((-1,.9),(1,.3)):
            indices=np.array([ix[i] for i in plan['clear_validation_ids'] if rows[i]['side']==side and round(rows[i]['hi'][0]-rows[i]['lo'][0],2)==length])
            av=available[indices];bv=np.broadcast_to(np.abs(standardized[indices,None]),av.shape)[av]
            q=quantile(bv);quality.append(dict(direction=direction,side=side,length=length,available=int(av.sum()),background_bias_q95=q,cal_noise_band=band,noise_scale_condition=bool(q is not None and band is not None and q<=band)))
        for j,i in enumerate(ids):
            for k in range(4):
                for q in (0,1):
                    if cat[j,q]!='contact':continue
                    target=hit[j,3:14]>0;av=available[j,k,3:14]&target
                    events.append(dict(direction=direction,scene=int(i),replica=k,height=('HEAD','BODY')[q],has_timely_reference=bool(av.any()),target_timely_zone_frames=int(target.sum()),available_timely_target_zone_frames=int(av.sum())))
    coverage=[dict(direction=d,height=h,denominator=sum(e['direction']==d and e['height']==h for e in events),available=sum(e['has_timely_reference'] for e in events if e['direction']==d and e['height']==h)) for d in DIRECTIONS for h in ('HEAD','BODY')]
    primary=all(q['noise_scale_condition'] for q in quality if q['direction']=='vertical') and all(r['available']>0 for r in coverage if r['direction']=='vertical')
    save('result_analysis.json',dict(status='COMPLETE',seconds=time.monotonic()-t,reference_quality=quality,coverage=coverage,records=records,event_rows=len(events),precheck_condition=primary,
        decision='VERTICAL_REFERENCE_CONDITION_SUPPORTED_PENDING_INCREMENT_TEST' if primary else 'VERTICAL_REFERENCE_CONDITION_NOT_MET_NO_JOINT_READOUT',
        limitation='Conditionaloraclemean backgroundbias only evaluation,finite4noise draws and2heldoutgeometry groups. Thresholdsobservedcal only;negativecontactresidualnot gated. No fullshape policy/realpose or hardware validation.'))
    save('event_coverage.json',events);assert time.monotonic()-t<180
    print('QUALITY',quality,'COVERAGE',coverage,'PRECHECK',primary,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
