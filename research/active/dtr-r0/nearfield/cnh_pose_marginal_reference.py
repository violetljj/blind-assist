"""Fixed causal speed/three-axis bias marginal on frozen signed CNH.

Known scene/current anchor are privileged. Step-noise is not marginalized.
No fitting, new photons, or nuisance choice using true pose.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from itertools import product
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
import cnh_displacement_ceiling_render as R
import cnh_displacement_ceiling_evaluate as E
import cnh_pose_expected_only as Q

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'artifacts.local/work/cnh-displacement-ceiling-20261003'
FACTORIAL=ROOT/'artifacts.local/work/cnh-pose-factorial-20261004'
OUT=ROOT/'artifacts.local/work/cnh-pose-marginal-reference-20261004'
RUN='CNH_POSE_MARGINAL_REFERENCE_20261004'
SPEED_NODES=(-np.sqrt(3/5)*.2,0.,np.sqrt(3/5)*.2)
SPEED_WEIGHTS=(5/18,4/9,5/18)
GRID=[(s,*bias) for s in SPEED_NODES for bias in product((-1.,1.),repeat=3)]
LOG_PRIOR=np.log(np.repeat(SPEED_WEIGHTS,8)/8)
ZERO=24
HYPOTHESES=GRID+[(0.,0.,0.,0.)]
TIDS=[0,1,4,5,6]
FRAMES=np.arange(3,16)
WINDOWS=(8,12)
PAST=np.concatenate([np.arange(max(0,f-11),f+1) for f in FRAMES])
OFFSETS=np.cumsum([0]+[min(int(f)+1,12) for f in FRAMES])
BINS={'primary1.2-2.1m':(1.2,2.1),'near0.9-1.2m':(.9,1.2),'combined0.9-2.1m':(.9,2.1)}
ARMS=('marginal8','marginal12','plugin8','plugin12','true8','true12','M3_smooth')
read,save,sha=E.read,E.save,E.sha


def corrected_sequence(noisy,scale,bx,by,bz):
    """Cancel a fixed hypothetical bias from estimated causal increments."""
    poses=np.asarray(noisy,float)
    corrected=poses.copy()
    cancel=Rotation.from_rotvec(-np.deg2rad([bx,by,bz])*.2).as_matrix()
    for t in range(1,len(poses)):
        delta=np.linalg.inv(poses[t-1])@poses[t]
        delta[:3,3]/=1+scale
        delta[:3,:3]=delta[:3,:3]@cancel
        corrected[t]=corrected[t-1]@delta
    return corrected


def anchored_poses(sensor,noisy,hypothesis):
    # Every queried prefix independently excludes future noisy poses.
    result=[]
    for f in FRAMES:
        corrected=corrected_sequence(noisy[:f+1],*hypothesis)
        if hypothesis==(0.,0.,0.,0.):
            corrected=noisy[:f+1]  # Frozen plug-in path, no integration roundoff.
        result.append(sensor[f]@np.linalg.inv(corrected[f])@corrected[max(0,f-11):f+1])
    return np.concatenate(result)


def mixture(candidate_loglike):
    return logsumexp(candidate_loglike[:2],axis=0)-np.log(2)-logsumexp(candidate_loglike[2:],axis=0)+np.log(3)


def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'PLAN.json'
    if path.exists():return read(path)
    prior=read(SOURCE/'result.json');factor=read(FACTORIAL/'render_lambda1.json')
    assert prior['status']=='COMPLETE' and factor['status']=='COMPLETE' and len(prior['units'])==48
    hashes={str(p):sha(p) for p in (SOURCE/'result.json',FACTORIAL/'result.json',FACTORIAL/'render_lambda1.json',FACTORIAL/'scores/all_scores.npz')}
    sources=R.source_sha256();sources.update({str(Path(__file__).resolve()):sha(__file__),str(Path(E.__file__).resolve()):sha(E.__file__)})
    record=dict(run=RUN,units=prior['units'],status='FIXED_BEFORE_FIRST_SCENE',input_sha256=hashes,source_sha256=sources,
        role='Consumed synthetic Development known-scene pose nuisance diagnostic; no training/new photons',
        grid=[dict(scale=h[0],bias_deg_per_s=list(h[1:]),prior=float(np.exp(LOG_PRIOR[i]))) for i,h in enumerate(GRID)],windows=list(WINDOWS),K=4,
        pose_rule='Causal estimated increments: translation/(1+s), rotation right multiply exp(-threeaxisbias*.2seconds); true sensor[f] current anchor; zero correction separate',
        marginal_order='Each nuisance trajectory shared by whole causal window: sum exposure logPMF, weighted logsumexp24 with GL3 speed prior times binary3axisbias1/8, then equal-class candidate mixture (2inside vs3outside)',
        candidate_indices=TIDS,observed_delta_indices=list(range(7)),frames=FRAMES.tolist(),past=PAST.tolist(),offsets=OFFSETS.tolist(),bins=BINS,
        primary='Low-visible12scene combined0.9-2.1m marginal12 minus plugin12 macroAUC; primary1.2-2.1 and near0.9-1.2 also mandatory',
        visibility='Original primary five-branch target-return frame fraction>=.5;36high/12low; verify factorial subgroup agrees',
        raw_likelihood_axes=['nuisance24','candidate5','observed_delta7','replica4','anchor13','window8_12'],
        zero_reference='Zero scale/bias reference independent of24nodeprior; raw_plugin and plugin8/12',
        reuse='Zero hypothesis window8 frozen belief-lambda1 expectation; window12 only supplement missing earlier exposures; true expected reused',
        scene_bootstrap=dict(n=1000,seed=2026100403),budget_seconds=3600,initial_workers=4,BLAS_threads=1,
        compute='TASK_NOT_GPU_SUITABLE CPU expected-only raytrace plus exact signed Skellam',
        limits=['Current true query anchor/full scene remain privileged','GL3 speed approximates continuous uniformscale; binary3axisbias uses declared signs; stepnoise remains unmodeled','Grid is fixed finite approximate nuisance prior, not physical information upper bound','Outside10cm belongs pass0-10; nominal class discrimination is not full all-object alert truth','No automatic old C entry or training'])
    save(path,record);print('PLAN',sha(path),flush=True);return record


def check_sources(p):
    for path,digest in p['source_sha256'].items():
        assert sha(path)==digest,('source changed',path)


def effective_plan():
    record=read(OUT/'PLAN.json')
    if (OUT/'PLAN_AMENDMENT.json').exists():
        amendment=read(OUT/'PLAN_AMENDMENT.json')
        assert amendment['original_plan_sha256']==sha(OUT/'PLAN.json')
        record.update(amendment['replacement_fields'])
    if (OUT/'PERFORMANCE_AMENDMENT.json').exists():
        performance=read(OUT/'PERFORMANCE_AMENDMENT.json')
        assert performance['original_plan_sha256']==sha(OUT/'PLAN.json')
        record['source_sha256']=performance['source_sha256']
    return record


def amendment():
    """Preserve original9 PLAN; bind root's revised grid before24 outputs."""
    if (OUT/'PLAN_AMENDMENT.json').exists():return read(OUT/'PLAN_AMENDMENT.json')
    if len(read(OUT/'PLAN.json')['grid'])==24:return read(OUT/'PLAN.json')
    assert not list((OUT/'scores').glob('*.npz')) and not list((OUT/'units').glob('*.json'))
    sources=R.source_sha256();sources.update({str(Path(__file__).resolve()):sha(__file__),str(Path(E.__file__).resolve()):sha(E.__file__)})
    record=dict(original_plan_sha256=sha(OUT/'PLAN.json'),reason='Root changed recipe before first unit completed: abandoned9 speed/yaw grid, fixed24 GaussLegendre speed times8 binary three-axis bias prior',
        abandoned_run=dict(status='INTERRUPTED_BEFORE_ANY_UNIT_RESULT',sealed_scientific_arrays=0,reserved_elapsed_s=120),
        replacement_fields=dict(source_sha256=sources,
            grid=[dict(scale=h[0],bias_deg_per_s=list(h[1:]),prior=float(np.exp(LOG_PRIOR[i]))) for i,h in enumerate(GRID)],
            pose_rule='Causally integrate estimated increments: translation/(1+s), rotation right multiply exp(-bias*.2seconds); current true sensor[f] anchor retained; zero reference independent',
            marginal_order='Sum full causal window logPMF per candidate and nuisance; weighted logsumexp24 pose with GL3 speed weights times uniform8 binarygyro; then equal-class mixture',
            raw_likelihood_axes=['nuisance24','candidate5','observed_delta7','replica4','anchor13','window8_12'],
            zero_reference='Zero scale/bias not added to24pose prior; separate raw_plugin and plugin8/12',
            initial_workers=8,limits=['Known scene/current true query anchor remain privileged','GaussLegendre3 approximates uniform scale[-.2,.2]; binarythreeaxisbias matches declared signs; stepnoise remains unmodeled','Finite prior diagnostic is not allpose information ceiling; no model family rejection or old C entry']))
    save(OUT/'PLAN_AMENDMENT.json',record);print('AMENDMENT',sha(OUT/'PLAN_AMENDMENT.json'),flush=True);return record


def performance_amendment():
    path=OUT/'PERFORMANCE_AMENDMENT.json'
    if path.exists():return read(path)
    assert not list((OUT/'units').glob('*.json'))
    verified=read(OUT/'performance/verification.json');assert verified['status']=='PASS' and verified['expectation_array_equal']
    sources=R.source_sha256();sources.update({str(Path(module.__file__).resolve()):sha(module.__file__) for module in (E,Q)})
    sources[str(Path(__file__).resolve())]=sha(__file__)
    record=dict(original_plan_sha256=sha(OUT/'PLAN.json'),scientific_amendment_sha256=sha(OUT/'PLAN_AMENDMENT.json') if (OUT/'PLAN_AMENDMENT.json').exists() else None,
        source_sha256=sources,verification_sha256=sha(OUT/'performance/verification.json'),
        change='Exact expected-only electronics removes discarded Poisson RNG; internal16pose blocks; all currentt=f expectations reuse original true_expected',
        unchanged='24nodes/weightedprior, sub16, seven frozen observations,K4, all13anchors,8/12wholewindowLL and classmix',
        interrupted_benchmarks_sha256=sha(OUT/'interrupted_benchmarks.json'),charged_aborted_seconds=300,
        new_plan_path_repaired='Fresh PLAN correctly declares24four-coordinate nodes; original9 PLAN remains unchanged')
    save(path,record);print('PERFORMANCE AMENDMENT',sha(path),flush=True);return record


def run_scene(unit,smoke=False):
    started=time.monotonic();p=effective_plan();check_sources(p)
    receipt_path=OUT/'units'/f'unit{unit}.json'
    if receipt_path.exists():
        receipt=read(receipt_path);assert receipt['plan_sha256']==sha(OUT/'PLAN.json') and receipt['score_sha256']==sha(OUT/receipt['score_file']);return receipt
    prior=read(SOURCE/'result.json');old=next(s for s in prior['per_scene'] if s['unit']==unit)
    truth=read(SOURCE/'truth'/f'unit{unit}.json')
    for folder,ext in [('observations','npz'),('templates','npz'),('truth','json'),('scores','npz')]:
        path=SOURCE/folder/f'unit{unit}.{ext}';assert sha(path)==prior['provenance']['input_sha256'][str(path)]
    with np.load(SOURCE/'observations'/f'unit{unit}.npz') as z:hist,ambient,sensor,noisy=z['hist'],z['ambient'],z['sensor'],z['noisy']
    with np.load(SOURCE/'templates'/f'unit{unit}.npz') as z:true_expected=z['expected']
    source_record=next(s for s in read(FACTORIAL/'render_lambda1.json')['units'] if s['unit']==unit)
    assert sha(source_record['path'])==source_record['sha256']
    with np.load(source_record['path']) as z:cached=z['expected'];cached_past=z['past'];cached_offsets=z['offsets']
    assert cached.shape==(4,5,len(cached_past),8,8,16)
    totals=np.empty((25,5,7,4,13,2),float)
    render_seconds=0.;likelihood_seconds=0.;zero_cache_max_error=0.
    for hi,hyp in enumerate(HYPOTHESES):
        for k in range(4):
            poses=anchored_poses(sensor,noisy[k],hyp)
            for ci,di in enumerate(TIDS):
                tick=time.monotonic()
                if hi==ZERO:
                    mean=np.empty((len(PAST),8,8,16),float);missing=[]
                    for ai,f in enumerate(FRAMES):
                        past=PAST[OFFSETS[ai]:OFFSETS[ai+1]];cp=cached_past[cached_offsets[ai]:cached_offsets[ai+1]]
                        for j,t in enumerate(past):
                            ix=OFFSETS[ai]+j
                            match=np.flatnonzero(cp==t)
                            if len(match):mean[ix]=cached[k,ci,cached_offsets[ai]+match[0]]
                            else:missing.append(ix)
                    if missing:
                        rendered=Q.expected(dict(poses=poses[missing],boxes=truth['boxes'][di]),pose_batch=16);mean[missing]=rendered['expectation']
                        np.testing.assert_array_equal(rendered['ambient'],ambient[PAST[missing]])
                else:
                    # At t=f every anchored trajectory equals the actual S_f;
                    # use its already frozen expectation instead of reraycast.
                    current=OFFSETS[1:]-1
                    past_only=np.ones(len(PAST),bool);past_only[current]=False
                    rendered=Q.expected(dict(poses=poses[past_only],boxes=truth['boxes'][di]),pose_batch=16)
                    mean=np.empty((len(PAST),8,8,16),float)
                    mean[past_only]=rendered['expectation'];mean[current]=true_expected[di,FRAMES]
                    np.testing.assert_array_equal(rendered['ambient'],ambient[PAST[past_only]])
                render_seconds+=time.monotonic()-tick;tick=time.monotonic()
                ll=E.skellam_logpmf_signed(hist[:,k,PAST],mean[None],8*ambient[PAST][None,...,None]).sum(axis=(-3,-2,-1))
                for ai,f in enumerate(FRAMES):
                    end=OFFSETS[ai+1]
                    for wi,w in enumerate(WINDOWS):totals[hi,ci,:,k,ai,wi]=ll[:,max(OFFSETS[ai],end-w):end].sum(-1)
                likelihood_seconds+=time.monotonic()-tick
        print('HYP',unit,hi+1,'/25',round(time.monotonic()-started,1),flush=True)
    candidate_marginal=logsumexp(totals[:24]+LOG_PRIOR[:,None,None,None,None,None],axis=0)
    scores={f'marginal{w}':mixture(candidate_marginal[...,wi]) for wi,w in enumerate(WINDOWS)}
    scores.update({f'plugin{w}':mixture(totals[ZERO,...,wi]) for wi,w in enumerate(WINDOWS)})
    true_ll=np.empty((5,7,4,16),float)
    for ci,di in enumerate(TIDS):true_ll[ci]=E.skellam_logpmf_signed(hist,true_expected[di][None,None],8*ambient[None,None,...,None]).sum(axis=(-3,-2,-1))
    for w in WINDOWS:scores[f'true{w}']=mixture(E.rolling_sum(true_ll,w))[...,FRAMES]
    with np.load(SOURCE/'scores'/f'unit{unit}.npz') as z:scores['M3_smooth']=E.m3_smooth(z['raw'])[...,truth['group']]
    with np.load(FACTORIAL/'scores/all_scores.npz') as z:
        i=z['units'].tolist().index(unit)
        zero_error=float(np.max(np.abs(scores['plugin8']-z['D_oracle8_belief'][i])))
        true_error=float(np.max(np.abs(scores['true8']-z['C_oracle8_true'][i])))
    assert zero_error<1e-8 and true_error<1e-8,(zero_error,true_error)
    checks=dict(zero8_score_max_error=zero_error,true8_score_max_error=true_error)
    if smoke:
        changed=noisy[0].copy();changed[10:,:3,3]+=100
        for hyp in HYPOTHESES:
            np.testing.assert_array_equal(corrected_sequence(noisy[0,:10],*hyp),corrected_sequence(changed[:10],*hyp))
        # Pose marginal is invariant to hypothesis order; zero-only recovers plugin.
        weighted=totals[:24]+LOG_PRIOR[:,None,None,None,None,None]
        np.testing.assert_allclose(logsumexp(weighted[::-1],axis=0),logsumexp(weighted,axis=0),atol=1e-9,rtol=0)
        np.testing.assert_array_equal(mixture(totals[ZERO,...,1]),scores['plugin12'])
        checks.update(no_future=True,marginal_hypothesis_permutation=True,zero_only_plugin_identity=True)
    visibility=old['target_visibility']['primary1.2-2.1m']
    vf=sum(visibility[str(d)]['frames_with_target_return'] for d in (1,2,-10,-15,-20))/sum(visibility[str(d)]['frames'] for d in (1,2,-10,-15,-20))
    fs=next(s for s in read(FACTORIAL/'scenes.json') if s['unit']==unit)
    assert (vf>=.5)==(fs['visibility_fraction']>=.5)
    score_path=OUT/'scores'/f'unit{unit}.npz';score_path.parent.mkdir(parents=True,exist_ok=True)
    with score_path.open('xb') as stream:np.savez_compressed(stream,raw_loglike=totals[:24],raw_plugin=totals[ZERO],log_pose_prior=LOG_PRIOR,frames=FRAMES,windows=np.array(WINDOWS),grid=np.asarray(GRID),**scores)
    receipt=dict(unit=unit,status='COMPLETE',plan_sha256=sha(OUT/'PLAN.json'),amendment_sha256=sha(OUT/'PLAN_AMENDMENT.json') if (OUT/'PLAN_AMENDMENT.json').exists() else None,
        performance_amendment_sha256=sha(OUT/'PERFORMANCE_AMENDMENT.json'),score_file=str(score_path.relative_to(OUT)).replace('\\','/'),score_sha256=sha(score_path),
        elapsed_s=time.monotonic()-started,render_s=render_seconds,likelihood_s=likelihood_seconds,checks=checks,
        memory=Q.memory(),
        front_range_m=np.asarray(truth['front_range_m'])[FRAMES].tolist(),visibility_fraction=vf,stratum='high_visible' if vf>=.5 else 'low_visible')
    receipt_path.parent.mkdir(parents=True,exist_ok=True);save(receipt_path,receipt)
    print('SCENE',unit,'seconds',round(receipt['elapsed_s'],2),flush=True);return receipt


def smoke():
    plan();amendment();performance_amendment();p=effective_plan();tick=time.monotonic();r=run_scene(p['units'][0],True)
    eta4=300+r['elapsed_s']*(1+47/4);eta8=300+r['elapsed_s']*(1+47/8)
    record=dict(status='PASS',plan_sha256=sha(OUT/'PLAN.json'),amendment_sha256=sha(OUT/'PLAN_AMENDMENT.json'),unit=r['unit'],elapsed_s=r['elapsed_s'],render_s=r['render_s'],likelihood_s=r['likelihood_s'],
        forecast4workers_s=eta4,forecast8workers_s=eta8,checks=r['checks'],memory=r['memory'],charged_aborted_s=300,
        complete_sequence_frames=FRAMES.tolist(),grid24=True,scientific_seconds_spent=time.monotonic()-tick)
    save(OUT/'smoke.json',record);print(record,flush=True)


def aggregate():
    p=read(OUT/'PLAN.json');units=[read(OUT/'units'/f'unit{u}.json') for u in p['units']]
    assert len(units)==48
    strata={'all':np.ones(48,bool),'high_visible':np.array([u['stratum']=='high_visible' for u in units]),'low_visible':np.array([u['stratum']=='low_visible' for u in units])}
    assert strata['high_visible'].sum()==36 and strata['low_visible'].sum()==12
    arrays={a:[] for a in ARMS}
    for u in units:
        assert u['score_sha256']==sha(OUT/u['score_file'])
        with np.load(OUT/u['score_file']) as z:
            for a in arrays:arrays[a].append(z[a])
    arrays={a:np.stack(v) for a,v in arrays.items()}
    cells={};rng=np.random.default_rng(2026100403)
    for stratum,selection in strata.items():
        cells[stratum]={}
        for b,(lo,hi) in BINS.items():
            values={a:[] for a in ARMS};pooled={a:[[],[]] for a in ARMS}
            for i,u in enumerate(units):
                if not selection[i]:continue
                keep=(np.asarray(u['front_range_m'])>=lo)&(np.asarray(u['front_range_m'])<hi)
                for a in ARMS:
                    pos=arrays[a][i,:2,:,keep].ravel();neg=arrays[a][i,4:,:,keep].ravel()
                    values[a].append(E.binary_auc(pos,neg));pooled[a][0].append(pos);pooled[a][1].append(neg)
            n=int(selection.sum());boot=rng.integers(n,size=(1000,n));cells[stratum][b]={}
            for a,v in values.items():
                v=np.asarray(v);pos,neg=(np.concatenate(q) for q in pooled[a]);sample=v[boot].mean(1)
                cells[stratum][b][a]=dict(scene_macro_AUC=float(v.mean()),ci95=np.percentile(sample,[2.5,97.5]).tolist(),pooled_AUC=E.binary_auc(pos,neg),scenes=n,positive_n=len(pos),negative_n=len(neg))
            diff=np.asarray(values['marginal12'])-np.asarray(values['plugin12'])
            cells[stratum][b]['marginal12_minus_plugin12']=dict(scene_macro_difference=float(diff.mean()),paired_ci95=np.percentile(diff[boot].mean(1),[2.5,97.5]).tolist())
    result=dict(status='COMPLETE',run=RUN,plan_sha256=sha(OUT/'PLAN.json'),cells=cells,per_scene=units,interpretation='Conditional finite pose prior diagnostic only; no training decision')
    save(OUT/'result.json',result)
    lines=['# 固定速度/三轴bias位姿边缘化参照','', '24格先整段累积精确Skellam似然，再按GL速度×gyro符号先验对位姿边缘化，再对候选类别混合。原观测冻结；完整f3–15序列。真当前anchor、场景、候选尺寸/rho仍是特权；步噪声未边缘化。本轮不自动进入训练。','', '| 分层/域 | 场景 | marginal8 | marginal12 | plugin8 | plugin12 | true8 | true12 | M3 | 12−plugin12 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s,bs in cells.items():
        for b,c in bs.items():lines.append('| '+' | '.join([s+'/'+b,str(c['marginal12']['scenes'])]+[f"{c[a]['scene_macro_AUC']:.4f}" for a in ARMS]+[f"{c['marginal12_minus_plugin12']['scene_macro_difference']:+.4f}"])+' |')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');return result


def run(workers):
    assert 1<=workers<=8
    p=effective_plan();assert read(OUT/'smoke.json')['status']=='PASS'
    tick=time.monotonic();smoke=read(OUT/'smoke.json');remaining=3600-300-smoke['elapsed_s']
    pending=[u for u in p['units'] if not (OUT/'units'/f'unit{u}.json').exists()]
    forecast=smoke['elapsed_s']*int(np.ceil(len(pending)/workers))
    if forecast>remaining:raise RuntimeError(f'Forecast {forecast:.1f}s exceeds remaining scientific budget {remaining:.1f}s; no full run started')
    pool=ProcessPoolExecutor(max_workers=workers);jobs={};waiting=iter(pending);completed=0
    try:
        for _ in range(min(workers,len(pending))):
            u=next(waiting);jobs[pool.submit(run_scene,u)]=u
        while jobs:
            if time.monotonic()-tick>=remaining:raise TimeoutError('Scientific run budget exhausted; preserve partial receipts')
            done,_=wait(jobs,timeout=min(30,max(.1,remaining-(time.monotonic()-tick))),return_when=FIRST_COMPLETED)
            for future in done:
                future.result();jobs.pop(future);completed+=1;print('COMPLETE',completed,'/',len(pending),flush=True)
                u=next(waiting,None)
                if u is not None:jobs[pool.submit(run_scene,u)]=u
    except BaseException:
        for future in jobs:future.cancel()
        for process in list(pool._processes.values()):
            if process.is_alive():process.terminate()
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    save(OUT/'runtime.json',dict(wall_s=time.monotonic()-tick,workers=workers,smoke_s=smoke['elapsed_s'],charged_aborted_s=300))
    aggregate()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['plan','amendment','performance-amendment','smoke','run','aggregate'],required=True);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    {'plan':plan,'amendment':amendment,'performance-amendment':performance_amendment,'smoke':smoke,'run':lambda:run(args.workers),'aggregate':aggregate}[args.stage]()
