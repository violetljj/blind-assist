"""Read-only probability/location/width diagnostics; never selects a predictor."""
import numpy as np


def quantile(mu,scale,p):
    mu,scale,p=np.broadcast_arrays(np.asarray(mu,float),np.asarray(scale,float),np.asarray(p,float))
    if not (np.isfinite(mu).all() and np.isfinite(scale).all() and (scale>0).all() and np.isfinite(p).all() and ((p>0)&(p<=1)).all()):
        raise ValueError('Expected finite location, positive scale and quantile in(0,1]')
    logf=-np.logaddexp(0.,-(3.-mu)/scale)+np.log(p)
    logf=np.minimum(logf,-np.finfo(float).eps)
    z=mu+scale*(logf-np.log(-np.expm1(logf)))
    return np.where(p==1.,3.,np.clip(z,.3,3.))


def crossing(q,mu,scale,cutoff):
    q=np.asarray(q,float)
    if not (np.isfinite(q).all() and ((q>=0)&(q<=1)).all() and np.isfinite(cutoff) and cutoff>0):
        raise ValueError('Expected probability q and finite positive cutoff')
    admitted=q>=cutoff
    ratio=np.minimum(cutoff/np.maximum(q,np.finfo(float).tiny),1.)
    return np.where(admitted,quantile(mu,scale,ratio),np.inf)


def errors(prediction,truth,mask):
    mask=np.asarray(mask,bool);resolved=mask&np.isfinite(prediction)
    err=np.abs(prediction[resolved]-truth[resolved]);n=int(mask.sum())
    return dict(truth_count=n,resolved=int(resolved.sum()),missing=int((mask&~resolved).sum()),
        within5cm=int((err<=.050001).sum()),hit_rate=float((err<=.050001).sum()/n) if n else None,
        conditional_MAE_m=float(err.mean()) if len(err) else None,conditional_p90_m=float(np.quantile(err,.9)) if len(err) else None)


def distribution(x,mask):
    a=np.asarray(x)[mask]
    return dict(count=int(a.size),median=float(np.median(a)) if a.size else None,p10=float(np.quantile(a,.1)) if a.size else None,p90=float(np.quantile(a,.9)) if a.size else None)


def focus(q,mu,scale,truth,cutoff):
    finite=np.isfinite(truth)&(truth>.300001)&(truth<=3.000001)
    left=np.isfinite(truth)&(truth<=.300001);right=truth>3.000001
    admitted=q>=cutoff;median=quantile(mu,scale,.5);actual=crossing(q,mu,scale,cutoff)
    span=quantile(mu,scale,.9)-quantile(mu,scale,.1)
    median_good=np.zeros(truth.shape,bool);median_good[finite]=np.abs(median[finite]-truth[finite])<=.050001
    actual_good=np.zeros(truth.shape,bool);resolved=finite&np.isfinite(actual)
    actual_good[resolved]=np.abs(actual[resolved]-truth[resolved])<=.050001
    qone=quantile(mu,scale,min(cutoff,1.)) if cutoff<=1 else np.full_like(mu,np.inf)
    groups={name:mask for name,mask in [('all_finite',finite),('blocked_finite',finite&~admitted),('admitted_finite',finite&admitted),('right_censored',right)]}
    return dict(actual=errors(actual,truth,finite),conditional_median=errors(median,truth,finite),raw_mu=errors(mu,truth,finite),
        q_unity_same_cutoff_diagnostic=errors(qone,truth,finite),
        left_censored_truth=int(left.sum()),right_censored_truth=int(right.sum()),
        false_admissions=int((right&admitted).sum()),q_unity_false_crossings=int((right&np.isfinite(qone)).sum()),
        partition=dict(blocked_median_accurate=int((finite&~admitted&median_good).sum()),blocked_median_inaccurate=int((finite&~admitted&~median_good).sum()),admitted_median_accurate=int((finite&admitted&median_good).sum()),admitted_median_inaccurate=int((finite&admitted&~median_good).sum())),
        admitted_median_good_actual_bad=int((finite&admitted&median_good&~actual_good).sum()),
        admitted_median_bad_actual_good=int((finite&admitted&~median_good&actual_good).sum()),
        narrow_but_wrong=int((finite&(span<=.1)&~median_good).sum()),
        q_distribution={k:distribution(q,m) for k,m in groups.items()},
        scale_distribution={k:distribution(scale,m) for k,m in groups.items()},
        conditional_10_90_span={k:distribution(span,m) for k,m in groups.items()})


def decompose(q,mu,scale,truth,widths,cutoff):
    q,mu,scale,truth=[np.asarray(a,float) for a in (q,mu,scale,truth)]
    if q.shape!=mu.shape or q.shape!=scale.shape or q.shape!=truth.shape or q.ndim!=3 or q.shape[1]!=2 or q.shape[2]!=len(widths):
        raise ValueError('Expected matching N,2,W arrays')
    col=np.flatnonzero(np.isclose(widths,.6,atol=1e-7,rtol=0))
    if len(col)!=1:raise ValueError('Need one width.6 anchor')
    finite=np.isfinite(truth)&(truth<=3.000001);pair=finite[...,:-1]&finite[...,1:]
    med=quantile(mu,scale,.5);actual=crossing(q,mu,scale,cutoff);admit=q>=cutoff
    dq=np.diff(q,axis=-1);dm=np.diff(med,axis=-1)
    truth_up=np.zeros(pair.shape,bool);truth_up[pair]=(truth[...,1:][pair]-truth[...,:-1][pair])>1e-8
    zpair=np.isfinite(actual[...,:-1])&np.isfinite(actual[...,1:]);dz=np.zeros(pair.shape)
    dz[zpair]=actual[...,1:][zpair]-actual[...,:-1][zpair]
    shape=dict(adjacent_pairs=int(pair.size),both_truth_finite_pairs=int(pair.sum()),
        truth_wrong_direction_pairs=int(truth_up.sum()),truth_finite_disappears=int((finite[...,:-1]&~finite[...,1:]).sum()),
        q_decrease_pairs=int((dq< -1e-6).sum()),q_decrease_curves=int((dq< -1e-6).any(-1).sum()),maximum_q_drop=float(max(0.,-dq.min())),
        conditional_median_increases_over1cm=int((pair&(dm>.01)).sum()),conditional_median_wrong_direction_curves=int((pair&(dm>.01)).any(-1).sum()),
        maximum_conditional_median_increase_m=float(max(0.,dm[pair].max())) if pair.any() else None,
        admitted_contact_disappears=int((admit[...,:-1]&~admit[...,1:]).sum()),
        actual_crossing_increases_over1cm=int((zpair&(dz>.01)).sum()),
        actual_crossing_wrong_direction_curves=int((zpair&(dz>.01)).any(-1).sum()))
    return dict(anchor=focus(q[:,:,col[0]],mu[:,:,col[0]],scale[:,:,col[0]],truth[:,:,col[0]],cutoff),
        all_widths=focus(q,mu,scale,truth,cutoff),width_curve=shape,
        note='Conditional positions and q-unity are evaluator-partitioned diagnostics, never deployed predictions or causal separation.')
