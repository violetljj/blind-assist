"""Recomputable nonlinear readout diagnostics, no alarm policy or model changes."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import cnh_aligned_boundary_dev as B

READOUT=B.ROOT/'artifacts.local/work/cnh-bar-representation-dev-20261008/readout'
OUT=READOUT/'analysis'
SEED=2026100843
NBOOT=1000


def save(name,value):B.save(OUT/name,value)


def prepare():
    save('PLAN.json',dict(task='CNH_BAR_READOUT_ANALYSIS_DEV_20261008',lane='EXPLORE consumed simulation',
        authorization='User explicitly continued third readout sensitivity diagnostic',budget_wall_seconds=180,
        scope='Analyze parent-generated frozen M3 mean/jacobian/64 existing-plan Poisson noise scores; no training/threshold/EMA changes',
        expected_inputs=['../PLAN.json','../predictions.npz','../draws.npy'],
        endpoints=37,ends=[12,13],scales=[.25,1.],samples_per_endpoint=64,
        pairs_per_end=52,paired_reports=104,independent_unit='Endpoint/draw, reused pass dependencies retained; 104 reports are not 104 independent trials',
        bootstrap=dict(seed=SEED,resamples=NBOOT,percentiles=[2.5,97.5],method='Endpoint-independent draw resampling; same endpoint draw indices reused over all heights/ends/scales and shared references; ddof1 sample variance'),
        variance_classification='95% CI excluding1 => VARIANCE_MISMATCH; including1 => UNRESOLVED, not linearity validation; zero/degenerate denominators => NOT_EVALUABLE',
        pair_formula='signed d_J=delta mean probe/sqrt(mean endpoint analytic J variance); d_J_scaled=d_J/alpha. signed d_MC=delta sampled score means/sqrt(mean endpoint sampled variance).',
        limitations='64 noise draws do not resolve rare tails, nonsmooth regions or deployment distribution; bootstrap intervals are diagnostic and unadjusted for multiple comparisons. Local Jacobian scores are not information-loss stages or alarm-gain evidence.',
        stop='Complete analysis and focused archive recomputation within cumulative 180s or report incomplete; preserve failures',
        source_sha256=B.sha(Path(__file__))))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('ANALYSIS_PREPARED',flush=True)


def csv_write(name,rows):
    with (OUT/name).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def ratio(a,b):
    return float(a/b) if b>0 and np.isfinite(a) and np.isfinite(b) else None


def ci(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x,[2.5,97.5])] if len(x)==NBOOT else None


def classify(interval):
    if interval is None:return 'NOT_EVALUABLE'
    return 'VARIANCE_MISMATCH' if interval[0]>1 or interval[1]<1 else 'UNRESOLVED'


def dist(vals):
    vals=[v for v in vals if v is not None and np.isfinite(v)]
    return dict(n=len(vals),minimum=float(min(vals)),median=float(np.median(vals)),maximum=float(max(vals))) if vals else None


def load():
    p=json.loads((READOUT/'PLAN.json').read_text(encoding='utf8'))
    with np.load(READOUT/'predictions.npz') as z:arrays={k:z[k] for k in z.files}
    return p,arrays


def calculate(plan,z,indices,check=lambda:None):
    ids=z['endpoint_ids'].tolist();ends=z['ends'].tolist();scales=z['scales'].tolist();index={int(i):j for j,i in enumerate(ids)}
    score=z['mean_scores'].astype(np.float64);jvar=z['jacobian_variance'].astype(np.float64)
    mc=z['mc_scores'].astype(np.float64);linear=z['linear_noise'].astype(np.float64)
    assert mc.shape==(37,2,2,64,2) and linear.shape==(37,2,64,2)
    assert score.shape==jvar.shape==(37,2,2)
    assert indices.shape==(37,NBOOT,64)
    assert np.isfinite(mc).all() and np.isfinite(linear).all() and np.isfinite(jvar).all() and (jvar>=0).all()
    bm=np.empty((37,2,2,NBOOT,2));bv=np.empty_like(bm);blv=np.empty((37,2,NBOOT,2))
    endpoint_rows=[]
    for i in range(37):
        check();ix=indices[i]
        for e,end in enumerate(ends):
            blv[i,e]=np.var(linear[i,e][ix],axis=1,ddof=1)
            for s,alpha in enumerate(scales):
                draw=mc[i,e,s][ix];bm[i,e,s]=draw.mean(1);bv[i,e,s]=draw.var(1,ddof=1)
                for h,height in enumerate(('HEAD','BODY')):
                    actual=mc[i,e,s,:,h];noise=alpha*linear[i,e,:,h]
                    predvar=alpha*alpha*jvar[i,e,h];av=float(actual.var(ddof=1));lv=float(noise.var(ddof=1))
                    residual=actual-score[i,e,h]-noise;shift=float(actual.mean()-score[i,e,h])
                    with np.errstate(divide='ignore',invalid='ignore'):
                        cj=ci(bv[i,e,s,:,h]/predvar) if predvar>0 else None
                        cl=ci(bv[i,e,s,:,h]/(alpha*alpha*blv[i,e,:,h]))
                    corr=float(np.corrcoef(actual,noise)[0,1]) if av>0 and lv>0 else None
                    gaps=z['max_pool_gaps'][i,e,:,h]
                    endpoint_rows.append(dict(endpoint=int(ids[i]),end=int(end),alpha=float(alpha),height=height,
                        mean_probe=float(score[i,e,h]),mc_mean=float(actual.mean()),mean_shift=shift,
                        jacobian_variance=float(jvar[i,e,h]),predicted_variance=float(predvar),mc_variance=av,
                        sampled_linear_variance=lv,mc_sigma=float(np.sqrt(av)),predicted_sigma=float(np.sqrt(predvar)),
                        mc_over_J_variance=ratio(av,predvar),mc_over_sample_linear_variance=ratio(av,lv),
                        mc_over_J_ci_low=cj[0] if cj else None,mc_over_J_ci_high=cj[1] if cj else None,
                        mc_over_sample_linear_ci_low=cl[0] if cl else None,mc_over_sample_linear_ci_high=cl[1] if cl else None,
                        J_variance_status=classify(cj),sample_linear_variance_status=classify(cl),correlation=corr,
                        residual_rms=float(np.sqrt(np.mean(residual**2))),
                        centered_residual_rms=float(np.sqrt(np.mean((residual-residual.mean())**2))),
                        residual_rms_over_predicted_sigma=ratio(np.sqrt(np.mean(residual**2)),np.sqrt(predvar)),
                        centered_residual_rms_over_predicted_sigma=ratio(np.sqrt(np.mean((residual-residual.mean())**2)),np.sqrt(predvar)),
                        mean_shift_over_predicted_sigma=ratio(shift,np.sqrt(predvar)),
                        max_pool_gap_min=float(gaps.min()),max_pool_gap_median=float(np.median(gaps)),
                        max_pool_gap_exact_zero=int(np.count_nonzero(gaps==0))))
    pair_rows=[]
    for pair in plan['pairs']:
        check();one=index[int(pair['scene'])];zero=index[int(pair['reference'])];h=int(pair['height_idx'])
        for e,end in enumerate(ends):
            delta=float(score[one,e,h]-score[zero,e,h]);varj=float(.5*(jvar[one,e,h]+jvar[zero,e,h]))
            dj=ratio(delta,np.sqrt(varj))
            for s,alpha in enumerate(scales):
                a=mc[one,e,s,:,h];b=mc[zero,e,s,:,h];dm=float(a.mean()-b.mean());vm=float(.5*(a.var(ddof=1)+b.var(ddof=1)))
                dmc=ratio(dm,np.sqrt(vm))
                db=bm[one,e,s,:,h]-bm[zero,e,s,:,h];vb=.5*(bv[one,e,s,:,h]+bv[zero,e,s,:,h])
                with np.errstate(divide='ignore',invalid='ignore'):dc=ci(db/np.sqrt(vb))
                pair_rows.append(dict(scene=int(pair['scene']),reference=int(pair['reference']),contrast=pair['contrast'],
                    height=('HEAD','BODY')[h],end=int(end),alpha=float(alpha),delta_mean_probe=delta,
                    probe_mean_1=float(score[one,e,h]),probe_mean_0=float(score[zero,e,h]),
                    average_J_variance=varj,d_J=dj,d_J_scaled=dj/alpha if dj is not None else None,
                    mc_mean_1=float(a.mean()),mc_mean_0=float(b.mean()),delta_mc_mean=dm,
                    mc_sigma_1=float(a.std(ddof=1)),mc_sigma_0=float(b.std(ddof=1)),average_mc_variance=vm,
                    d_MC=dmc,d_MC_ci_low=dc[0] if dc else None,d_MC_ci_high=dc[1] if dc else None,
                    mean_sign_reversal=bool(delta*dm<0),d_status='EVALUABLE' if dj is not None and dmc is not None else 'NOT_EVALUABLE'))
    summary=[]
    for contrast in ('A_absent','B_pass'):
        for height in ('HEAD','BODY'):
            for end in ends:
                for alpha in scales:
                    rs=[r for r in pair_rows if r['contrast']==contrast and r['height']==height and r['end']==end and r['alpha']==alpha]
                    summary.append(dict(contrast=contrast,height=height,end=int(end),alpha=float(alpha),pair_reports=len(rs),
                        sign_reversals=sum(r['mean_sign_reversal'] for r in rs),negative_d_J=sum(r['d_J'] is not None and r['d_J']<0 for r in rs),
                        negative_d_MC=sum(r['d_MC'] is not None and r['d_MC']<0 for r in rs),
                        d_J=dist([r['d_J'] for r in rs]),d_J_scaled=dist([r['d_J_scaled'] for r in rs]),d_MC=dist([r['d_MC'] for r in rs]),
                        delta_mean_probe=dist([r['delta_mean_probe'] for r in rs]),delta_mc_mean=dist([r['delta_mc_mean'] for r in rs]),
                        mc_sigma_1=dist([r['mc_sigma_1'] for r in rs]),mc_sigma_0=dist([r['mc_sigma_0'] for r in rs])))
    endpoint_summary=[]
    for height in ('HEAD','BODY'):
        for end in ends:
            for alpha in scales:
                rs=[r for r in endpoint_rows if r['height']==height and r['end']==end and r['alpha']==alpha]
                endpoint_summary.append(dict(height=height,end=int(end),alpha=float(alpha),endpoint_reports=len(rs),
                    J_variance_mismatch=sum(r['J_variance_status']=='VARIANCE_MISMATCH' for r in rs),
                    sample_linear_variance_mismatch=sum(r['sample_linear_variance_status']=='VARIANCE_MISMATCH' for r in rs),
                    J_variance_unevaluable=sum(r['J_variance_status']=='NOT_EVALUABLE' for r in rs),
                    sample_linear_variance_unevaluable=sum(r['sample_linear_variance_status']=='NOT_EVALUABLE' for r in rs),
                    correlation=dist([r['correlation'] for r in rs]),mean_shift_over_predicted_sigma=dist([r['mean_shift_over_predicted_sigma'] for r in rs]),
                    centered_residual_rms_over_predicted_sigma=dist([r['centered_residual_rms_over_predicted_sigma'] for r in rs]),
                    mc_over_J_variance=dist([r['mc_over_J_variance'] for r in rs]),
                    mc_over_sample_linear_variance=dist([r['mc_over_sample_linear_variance'] for r in rs])))
    return endpoint_rows,pair_rows,summary,endpoint_summary


def run():
    began=time.monotonic();p=json.loads((OUT/'PLAN.json').read_text(encoding='utf8'))
    spent=sum(json.loads(f.read_text())['seconds'] for f in OUT.glob('run_failure_*.json'))
    def check():
        if spent+time.monotonic()-began>p['budget_wall_seconds']:raise TimeoutError('180s cumulative analysis cap')
    try:
        assert B.sha(Path(__file__))==p['source_sha256']
        plan,z=load()
        assert len(plan['pairs'])==52
        # Per-endpoint seed spawning fixes independent endpoint resampling without losing reuse.
        indices=np.stack([np.random.default_rng(seed).integers(0,64,(NBOOT,64),dtype=np.int16)
                          for seed in np.random.SeedSequence(SEED).spawn(37)])
        er,pr,summary,es=calculate(plan,z,indices,check)
        assert len(er)==296 and len(pr)==208
        check();np.savez_compressed(OUT/'bootstrap_indices.npz',indices=indices)
        csv_write('endpoint_diagnostics.csv',er);csv_write('pair_diagnostics.csv',pr)
        save('result.json',dict(status='COMPLETE',seconds=time.monotonic()-began,prior_failed_seconds=spent,
            endpoint_rows=len(er),pair_rows=len(pr),pair_scene_count=52,pair_frame_reports=104,
            bootstrap=NBOOT,seed=SEED,pair_groups=summary,endpoint_groups=es,limitations=p['limitations'],
            input_sha256={name:B.sha(READOUT/name) for name in ('PLAN.json','predictions.npz','draws.npy')},
            csv_sha256={name:B.sha(OUT/name) for name in ('endpoint_diagnostics.csv','pair_diagnostics.csv','bootstrap_indices.npz')}))
        print('READOUT_ANALYSIS_COMPLETE',round(time.monotonic()-began,3),flush=True)
    except BaseException as e:
        save('run_failure_'+str(time.time_ns())+'.json',dict(status='FAILED',error=repr(e),seconds=time.monotonic()-began));raise


def focused_check():
    began=time.monotonic();r=json.loads((OUT/'result.json').read_text(encoding='utf8'))
    p=json.loads((OUT/'PLAN.json').read_text(encoding='utf8'))
    def check():
        if r['seconds']+r['prior_failed_seconds']+time.monotonic()-began>p['budget_wall_seconds']:
            raise TimeoutError('Analysis+focused recomputation180s cap')
    for name,digest in r['input_sha256'].items():assert B.sha(READOUT/name)==digest
    for name,digest in r['csv_sha256'].items():assert B.sha(OUT/name)==digest
    plan,z=load()
    with np.load(OUT/'bootstrap_indices.npz') as a:indices=a['indices']
    er,pr,summary,es=calculate(plan,z,indices,check)
    assert summary==r['pair_groups'] and es==r['endpoint_groups']
    count=0
    for name,expected in (('endpoint_diagnostics.csv',er),('pair_diagnostics.csv',pr)):
        with (OUT/name).open(encoding='utf8') as f:actual=list(csv.DictReader(f))
        assert len(actual)==len(expected)
        for row,want in zip(actual,expected):
            for key,value in want.items():
                if value is None:assert row[key]==''
                elif isinstance(value,(str,bool)):assert row[key]==str(value)
                else:np.testing.assert_allclose(float(row[key]),value,rtol=1e-12,atol=1e-15)
            count+=1
    save('focused_check.json',dict(status='PASS',rows_recomputed=count,seconds=time.monotonic()-began,
        total_analysis_seconds=r['seconds']+r['prior_failed_seconds']+time.monotonic()-began,
        bootstrap_endpoint_reuse='Indices reused across all ends/heights/scales; same pass endpoints reuse identical indices'))
    print('READOUT_ANALYSIS_CHECK_PASS',count,round(time.monotonic()-began,3),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','check'))
    {'prepare':prepare,'run':run,'check':focused_check}[p.parse_args().action]()
