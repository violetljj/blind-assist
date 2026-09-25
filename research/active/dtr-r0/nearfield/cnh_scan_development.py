"""Frozen scanner reproduction on consumed Development inputs, never protected data."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from cnh_route_sensor import RAW_BIN_M
from cnh_h3_geometry_diagnostic import query_weights, summarize, train_threshold
from cnh_r1_readout_ops import zone_cosines

TAUS = (.25, .5, .75)
WINDOWS = ((1,1,1),(2,1,1),(3,1,1),(1,2,1),(1,3,1),(2,2,1),(1,1,2))
SEED = 20260925


def shift_matrix(bins, delta):
    """Matrix A[out,in] for scratch's fractional shift toward near."""
    lo = int(np.floor(delta)); fraction = delta-lo
    a = np.zeros((bins, bins))
    for coefficient, offset in ((1-fraction,lo),(fraction,lo+1)):
        src = np.arange(bins); target = src-offset
        ok = (target >= 0) & (target < bins)
        a[target[ok], src[ok]] += coefficient
    return a


def accumulate(residual, variance, data, k=4, *, bin_width=RAW_BIN_M*8, corrected=False):
    """Return residual sum, diagonal variance, adjacent-bin covariance.

    Independent original bins/frames are the declared null assumption. Linear
    interpolation introduces lag-one covariance, retained in corrected mode.
    """
    residual, variance = np.asarray(residual, float), np.asarray(variance, float)
    if residual.ndim != 3 or residual.shape != variance.shape or residual.shape[1] != 64:
        raise ValueError('Equal [N,64,bins] residual/variance required')
    if not np.isfinite(residual).all() or not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError('Finite values and nonnegative variances required')
    clips, steps = np.asarray(data['clip_ids']), np.asarray(data['steps'])
    index = {(c,int(s)): i for i,(c,s) in enumerate(zip(clips,steps))}
    if len(index) != len(residual):
        raise ValueError('Duplicate clip/step')
    r, v = residual.copy(), variance.copy()
    cov = np.zeros(residual.shape[:-1]+(residual.shape[-1]-1,))
    for lag in range(1,k):
        past = np.array([index.get((c,int(s)-lag),-1) for c,s in zip(clips,steps)])
        ok = past >= 0
        if not ok.any():
            continue
        for zone, cosine in enumerate(zone_cosines()):
            a = shift_matrix(residual.shape[-1],lag*.1*cosine/bin_width)
            r[ok,zone] += residual[past[ok],zone] @ a.T
            v[ok,zone] += variance[past[ok],zone] @ ((a*a).T if corrected else a.T)
            if corrected:
                cov[ok,zone] += variance[past[ok],zone] @ (a[:-1]*a[1:]).T
    return r,v,cov


def box_sum(x, shape):
    """Valid sliding box sums over final three dimensions."""
    result = np.asarray(x)
    for axis,size in zip(range(result.ndim-3,result.ndim),shape):
        cs = np.cumsum(result,axis=axis)
        padding = [(0,0)]*result.ndim; padding[axis] = (1,0)
        cs = np.pad(cs,padding)
        lo,hi = [slice(None)]*result.ndim,[slice(None)]*result.ndim
        lo[axis],hi[axis] = slice(None,-size),slice(size,None)
        result = cs[tuple(hi)]-cs[tuple(lo)]
    return result


def scan(residual, variance, weights, tau, *, covariance=None):
    n,_,bins = residual.shape
    r,v = residual.reshape(n,8,8,bins),variance.reshape(n,8,8,bins)
    support = weights.reshape(6,8,8,bins) >= tau
    out = np.full((n,6),-50.)
    c = None if covariance is None else covariance.reshape(n,8,8,bins-1)
    for height,width,depth in WINDOWS:
        shape = (height,width,depth)
        total, var = box_sum(r,shape),box_sum(v,shape)
        if depth > 1 and c is not None:
            var += 2*box_sum(c,(height,width,depth-1))
        admissible = box_sum(support.astype(int),shape) == height*width*depth
        z = total/np.sqrt(np.maximum(var,1e-9))
        for q in range(6):
            if admissible[q].any():
                out[:,q] = np.maximum(out[:,q],z[:,admissible[q]].max(1))
    return out


def fit_bias(hist,train,kind):
    if kind == 'zero':
        return np.zeros(hist.shape[1:])
    if kind != 'train_median' or np.asarray(train).dtype != bool or not np.any(train):
        raise ValueError('Nonempty boolean train-only selection required')
    return np.median(np.asarray(hist,dtype=float)[train],axis=0)


def arms(data,hist,bias,*,raw=False):
    weights = query_weights()
    residual = np.asarray(hist,float)-bias
    variance = 2*(1 if raw else 8)*data['ambient'][:,:,None]+np.maximum(bias,0)[None]
    variance = np.broadcast_to(variance,residual.shape).copy()
    width = RAW_BIN_M*(1 if raw else 8)
    r,v,c = accumulate(residual,variance,data,bin_width=width)
    rc,vc,cc = accumulate(residual,variance,data,bin_width=width,corrected=True)
    if not np.array_equal(r,rc):
        raise AssertionError('Variance correction changed residual')
    if raw:
        residual,variance,_=aggregate_h3(residual,variance,np.zeros(residual.shape[:-1]+(127,)))
        r,v,c=aggregate_h3(r,v,c)
        rc,vc,cc=aggregate_h3(rc,vc,cc)
    scores = {'CENTERED_SUM_K1':np.einsum('nzb,qzb->nq',residual,weights),
              'B1_SUM_K4_SCRATCH':np.einsum('nzb,qzb->nq',r,weights)}
    for tau in TAUS:
        scores[f'S1_K1_tau{tau}'] = scan(residual,variance,weights,tau)
        scores[f'S2_K4_LITERAL_tau{tau}'] = scan(r,v,weights,tau)
        scores[f'S2_K4_CORRECTED_tau{tau}'] = scan(rc,vc,weights,tau,covariance=cc)
    return scores


def aggregate_h3(residual,variance,covariance):
    """Aggregate raw128, including within-H3 variance and cross-H3 covariance."""
    r=residual.reshape(*residual.shape[:-1],16,8).sum(-1)
    v=variance.reshape(*variance.shape[:-1],16,8).sum(-1)
    for b in range(16):
        v[...,b] += 2*covariance[...,b*8:b*8+7].sum(-1)
    return r,v,covariance[...,7:120:8].copy()


def select_taus(y,scores,train):
    selected = {}
    for family in ('S1_K1','S2_K4_LITERAL','S2_K4_CORRECTED'):
        aps = {tau:float(average_precision_score(y[train].ravel(),scores[f'{family}_tau{tau}'][train].ravel())) for tau in TAUS}
        tau = max(TAUS,key=lambda t:(aps[t],-t))
        selected[family] = dict(tau=tau,train_aps=aps,key=f'{family}_tau{tau}')
    return selected


def metrics(y,s,threshold):
    return dict(**summarize(y,s,threshold),per_query=[summarize(y[:,q],s[:,q],threshold) for q in range(6)])


def paired_bootstrap(y,s,baseline,clips,draws=2000,deadline=None):
    """Clip cluster bootstrap; not independent-layout/generalization inference."""
    unique,which = np.unique(clips,return_inverse=True)
    rng = np.random.default_rng(SEED)
    def prepare(scores):
        values=scores.ravel();order=np.argsort(-values,kind='stable')
        starts=np.r_[0,np.flatnonzero(np.diff(values[order]))+1]
        return order,starts
    prepared=[prepare(x) for x in (s,baseline)]
    truth=y.ravel();frame_groups=np.repeat(which,y.shape[1])
    deltas=[]
    for _ in range(draws):
        if deadline is not None and _ % 100 == 0 and time.monotonic()>deadline:
            raise TimeoutError('Frozen 1800s CPU budget during bootstrap')
        mult=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[frame_groups]
        if np.sum(truth*mult)==0:
            continue
        aps=[]
        for order,starts in prepared:
            pos=np.add.reduceat((truth*mult)[order],starts)
            total=np.add.reduceat(mult[order],starts)
            cp,ct=np.cumsum(pos),np.cumsum(total)
            precision=np.divide(cp,ct,out=np.zeros_like(cp,dtype=float),where=ct>0)
            aps.append(float(np.sum(precision*pos)/cp[-1]))
        deltas.append(aps[0]-aps[1])
    return dict(unit='Development clip; layouts/geometries dependent',clusters=len(unique),requested=draws,
                evaluable=len(deltas),delta_ap=float(average_precision_score(truth,s.ravel())-average_precision_score(truth,baseline.ravel())),
                ci95=np.quantile(deltas,[.025,.975]).tolist())


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def null_diagnostic():
    rng=np.random.default_rng(SEED)
    residual=(rng.poisson(32,size=(2048,64,16))-rng.poisson(32,size=(2048,64,16))).astype(float)
    fixed=residual[:,0,:4].sum(-1)/16
    maxima=scan(residual,np.full_like(residual,64),query_weights(),.5)
    def describe(x):
        return dict(mean=float(x.mean()),std=float(x.std()),quantiles=np.quantile(x,[.5,.9,.95,.99]).tolist())
    return dict(seed=SEED,frames=2048,null='Independent Poisson32 minus Poisson32 per H3 cell; zero template',
                fixed_window=describe(fixed),max_scan=[describe(maxima[:,q]) for q in range(6)],
                warning='Fixed-window normal approximation only; query MAX distribution is selection-biased and not N(0,1).')


def worker_cache(path,street):
    path=Path(path)
    identity=json.loads(path.with_name('street-worker-identity.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=identity['raw_sha256']:
        raise ValueError('Worker raw cache hash differs')
    if identity['frame_keys']!=[r['frame_key'] for r in street['rows'][960:]]:
        raise ValueError('Worker frame order differs')
    raw=np.load(path,allow_pickle=False)
    if raw.shape!=(960,64,128) or not np.isfinite(raw).all():
        raise ValueError('Worker raw cache shape/numerics')
    if not np.array_equal(raw.reshape(960,64,16,8).sum(-1).astype(np.float32),street['histogram'][960:]):
        raise ValueError('Worker cache stored H3 parity failure')
    return raw,identity


def run(output,worker_raw=None):
    from cnh_r1_readout_data import load_alley,load_street,load_frame
    from cnh_r1_readout_run import synthesize
    started=time.monotonic();deadline=started+1800
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','protocol_sha256':hashlib.sha256(Path(__file__).with_name('CNH_SCAN_DEVELOPMENT_PROTOCOL_20260925.md').read_bytes()).hexdigest(),'results':{}}
    try:
        write(output/'null-diagnostic.json',null_diagnostic())
        alley=load_alley(require_depth=True)
        raw,_,_,seeds=synthesize(alley,load_frame,deadline)
        np.save(output/'alley-raw128.npy',raw)
        write(output/'original-seeds.json',dict(frame_keys=[r['frame_key'] for r in alley['rows']],seeds=seeds,stored_float32_h3_parity=len(raw)))
        street=load_street(require_depth=False)
        street_raw=None
        if worker_raw is not None:
            worker,identity=worker_cache(worker_raw,street)
            local={**street,'rows':street['rows'][:960],'frames':street['frames'][:960],
                   'histogram':street['histogram'][:960]}
            local_raw,_,_,local_seeds=synthesize(local,load_frame,deadline)
            street_raw=np.concatenate([local_raw,worker])
            write(output/'street-raw-identity.json',dict(worker_identity=identity,local_seeds=local_seeds,
                                                       stored_float32_h3_parity=1920))
        datasets={'alley':alley,'street':street}
        baseline={name:np.einsum('nzb,qzb->nq',data['histogram'].astype(float),query_weights()) for name,data in datasets.items()}
        previous=Path(__file__).resolve().parents[4]/'artifacts.local/evidence/cnh-r1-readout-20260925-v1/alley-scores.npz'
        with np.load(previous,allow_pickle=False) as old:
            if not np.array_equal(old['frame_key'],[r['frame_key'] for r in alley['rows']]):
                raise AssertionError('Prior R1 frame-key order differs')
            if not np.array_equal(old['R0'],baseline['alley']):
                raise AssertionError('B0 scores differ bitwise from prior R1 R0')
        report['B0_prior_R1_score_parity']='BITWISE_ALL_960x6'
        y,tr,dv=alley['labels'],alley['train'],alley['dev']
        threshold=train_threshold(y[tr],baseline['alley'][tr])
        report['results']['B0_EXACT_STORED_H3_UNCENTERED']={
            'threshold':threshold,'train':metrics(y[tr],baseline['alley'][tr],threshold),
            'dev':metrics(y[dv],baseline['alley'][dv],threshold),
            'street':metrics(street['labels'],baseline['street'],threshold)}
        if abs(report['results']['B0_EXACT_STORED_H3_UNCENTERED']['dev']['auprc']-.21011599820183707)>1e-12:
            raise AssertionError('Frozen B0 parity failed')
        saved={'labels':y,'train':tr,'dev':dv,'B0':baseline['alley']}
        street_saved={'labels':street['labels'],'B0':baseline['street']}
        for resolution,hist in [('H3_CONTROL',alley['histogram']),('RAW128_PRIMARY',raw)]:
            is_raw=resolution.startswith('RAW')
            for template in ('zero','train_median'):
                if time.monotonic()>deadline:raise TimeoutError('Frozen 1800s CPU budget')
                bias=fit_bias(hist,tr,template)
                np.save(output/f'bias-{resolution}-{template}.npy',bias)
                scores=arms(alley,hist,bias,raw=is_raw)
                selected=select_taus(y,scores,tr)
                ss=(None if street_raw is None else arms(street,street_raw,bias,raw=True)) if is_raw else arms(street,street['histogram'],bias)
                key=f'{resolution}_{template}'
                entry={'selection':selected,'arms':{},'street_input':('NOT_RUN_MISSING_WORKER_SOURCE' if street_raw is None else 'RAW128_ALL_1920_SOURCE_LOCAL_PARITY') if is_raw else 'H3_ALL_1920'}
                primary={'CENTERED_SUM_K1','B1_SUM_K4_SCRATCH'}|{v['key'] for v in selected.values()}
                for name,s in scores.items():
                    if time.monotonic()>deadline:raise TimeoutError('Frozen 1800s CPU budget')
                    threshold=train_threshold(y[tr],s[tr])
                    row={'threshold':threshold,'train':metrics(y[tr],s[tr],threshold),'dev':metrics(y[dv],s[dv],threshold),'selected':name in primary}
                    if name in primary:
                        row['paired_delta_vs_B0']=paired_bootstrap(y[dv],s[dv],baseline['alley'][dv],alley['clip_ids'][dv],deadline=deadline)
                    if ss is not None:
                        row['street']=metrics(street['labels'],ss[name],threshold)
                        street_saved[key+'_'+name]=ss[name]
                    saved[key+'_'+name]=s;entry['arms'][name]=row
                report['results'][key]=entry
                write(output/'result.json',report)
        np.savez_compressed(output/'alley-scores.npz',**saved)
        np.savez_compressed(output/'street-scores.npz',**street_saved)
        report.update(status='COMPLETE',elapsed_s=time.monotonic()-started,
                      evidence_limit='Consumed degenerate Development; clip CIs do not establish independent generalization; MAX scan null is not N(0,1).')
        write(output/'result.json',report)
        return report
    except Exception as exc:
        report.update(status='FAILED',error=repr(exc),elapsed_s=time.monotonic()-started)
        write(output/'result.json',report)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--worker-raw')
    args=parser.parse_args();run(args.output,args.worker_raw)
