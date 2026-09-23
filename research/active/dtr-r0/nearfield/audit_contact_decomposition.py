"""Independent bisection-CDF diagnostic arithmetic and world-coordinate audit."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from audit_exact_contact import world_targets
from audit_metric_contact import read,sha,equal
from metric_contact_model import MetricContactModel

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];ART=ROOT/'artifacts.local/evidence'
DEST=ART/'ba-contact-decomposition-20260923-v3';OUT=ART/'ba-contact-decomposition-20260923-audit'
BASE=ART/'ba-contact-boundary-20260923'
MODELS={'binary':ART/'ba-metric-contact-20260923','exact':ART/'ba-exact-contact-20260923'}


def inverse(mu,s,p):
    mu,s,p=np.broadcast_arrays(mu,s,p)
    # Bisection of the normalized logistic CDF, not the producer inverse formula.
    lo=np.full(mu.shape,.3);hi=np.full(mu.shape,3.)
    lognormal=-np.logaddexp(0.,-(3.-mu)/s)
    for _ in range(60):
        middle=(lo+hi)/2
        cdf=np.exp(-np.logaddexp(0.,-(middle-mu)/s)-lognormal)
        below=cdf<p;lo=np.where(below,middle,lo);hi=np.where(below,hi,middle)
    return np.where(p==1.,3.,(lo+hi)/2)


def crossing(q,mu,s,cut):
    ratio=np.divide(cut,q,out=np.ones_like(q,dtype=float),where=q>0)
    return np.where(q>=cut,inverse(mu,s,np.minimum(ratio,1.)),np.inf)


def error(pred,truth,mask):
    resolved=mask&np.isfinite(pred);err=abs(pred[resolved]-truth[resolved]);n=int(mask.sum())
    return dict(truth_count=n,resolved=int(resolved.sum()),missing=int((mask&~resolved).sum()),within5cm=int((err<=.050001).sum()),
        hit_rate=float((err<=.050001).sum()/n) if n else None,conditional_MAE_m=float(err.mean()) if len(err) else None,
        conditional_p90_m=float(np.percentile(err,90)) if len(err) else None)


def dist(x,mask):
    v=x[mask]
    return dict(count=int(v.size),median=float(np.median(v)) if len(v) else None,
        p10=float(np.percentile(v,10)) if len(v) else None,p90=float(np.percentile(v,90)) if len(v) else None)


def focus(q,mu,s,z,cut):
    finite=np.isfinite(z)&(z>.300001)&(z<=3.000001);right=z>3.000001;admit=q>=cut
    med=inverse(mu,s,.5);actual=crossing(q,mu,s,cut);span=inverse(mu,s,.9)-inverse(mu,s,.1)
    good=np.zeros(z.shape,bool);good[finite]=abs(med[finite]-z[finite])<=.050001
    agood=np.zeros(z.shape,bool);resolved=finite&np.isfinite(actual);agood[resolved]=abs(actual[resolved]-z[resolved])<=.050001
    qone=inverse(mu,s,cut) if cut<=1 else np.full_like(mu,np.inf)
    masks={'all_finite':finite,'blocked_finite':finite&~admit,'admitted_finite':finite&admit,'right_censored':right}
    return dict(actual=error(actual,z,finite),conditional_median=error(med,z,finite),raw_mu=error(mu,z,finite),
        q_unity_same_cutoff_diagnostic=error(qone,z,finite),left_censored_truth=int((np.isfinite(z)&(z<=.300001)).sum()),right_censored_truth=int(right.sum()),
        false_admissions=int((right&admit).sum()),q_unity_false_crossings=int((right&np.isfinite(qone)).sum()),
        partition=dict(blocked_median_accurate=int((finite&~admit&good).sum()),blocked_median_inaccurate=int((finite&~admit&~good).sum()),
            admitted_median_accurate=int((finite&admit&good).sum()),admitted_median_inaccurate=int((finite&admit&~good).sum())),
        admitted_median_good_actual_bad=int((finite&admit&good&~agood).sum()),admitted_median_bad_actual_good=int((finite&admit&~good&agood).sum()),
        narrow_but_wrong=int((finite&(span<=.1)&~good).sum()),q_distribution={k:dist(q,m) for k,m in masks.items()},
        scale_distribution={k:dist(s,m) for k,m in masks.items()},conditional_10_90_span={k:dist(span,m) for k,m in masks.items()})


def width_metrics(q,mu,s,z,cut):
    finite=np.isfinite(z)&(z<=3.000001);pair=finite[...,:-1]&finite[...,1:]
    med=inverse(mu,s,.5);actual=crossing(q,mu,s,cut);admit=q>=cut
    qdelta=q[...,1:]-q[...,:-1];mdelta=med[...,1:]-med[...,:-1]
    truthbad=np.zeros(pair.shape,bool);truthbad[pair]=z[...,1:][pair]-z[...,:-1][pair]>1e-8
    apair=np.isfinite(actual[...,:-1])&np.isfinite(actual[...,1:]);adelta=np.zeros(pair.shape)
    adelta[apair]=actual[...,1:][apair]-actual[...,:-1][apair]
    return dict(adjacent_pairs=int(pair.size),both_truth_finite_pairs=int(pair.sum()),truth_wrong_direction_pairs=int(truthbad.sum()),
        truth_finite_disappears=int((finite[...,:-1]&~finite[...,1:]).sum()),q_decrease_pairs=int((qdelta< -1e-6).sum()),
        q_decrease_curves=int((qdelta< -1e-6).any(-1).sum()),maximum_q_drop=float(max(0.,-qdelta.min())),
        conditional_median_increases_over1cm=int((pair&(mdelta>.01)).sum()),conditional_median_wrong_direction_curves=int((pair&(mdelta>.01)).any(-1).sum()),
        maximum_conditional_median_increase_m=float(max(0.,mdelta[pair].max())) if pair.any() else None,
        admitted_contact_disappears=int((admit[...,:-1]&~admit[...,1:]).sum()),actual_crossing_increases_over1cm=int((apair&(adelta>.01)).sum()),
        actual_crossing_wrong_direction_curves=int((apair&(adelta>.01)).any(-1).sum()))


def main():
    assert os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL'),'Use governed research-ue'
    assert not (OUT/'result.json').exists(),'Preserve completed audit'
    seal=read(DEST/'source-seal.json');cs=read(DEST/'component-seal.json');result=read(DEST/'result.json')
    for path,h in seal['inputs'].items():assert sha(Path(path))==h,path
    for name,h in seal['code'].items():
        assert sha(HERE/name)==h,name
        assert sha(DEST/'source-snapshot'/name)==h,name
    for name,h in cs['files'].items():assert sha(DEST/name)==h,name
    original=ART/'ba-contact-decomposition-20260923'
    original_seal=read(original/'component-seal.json')
    for name,h in cs['files'].items():
        assert sha(original/name)==h==original_seal['files'][name],name
    assert sha(DEST/'queries.npy')==cs['queries_sha256']
    assert cs['geometry_targets_joined'] is False and result['trained'] is False
    query=np.load(DEST/'queries.npy');widths=query[:51,0];anchor=int(np.flatnonzero(np.isclose(widths,.6,atol=1e-7,rtol=0))[0])
    np.testing.assert_allclose(widths,np.linspace(.2,1.2,51),atol=1e-7,rtol=0)
    np.testing.assert_array_equal(query[:,1],3.)
    ids=read(ART/'ba-query-occupancy-20260922-prepared/observations/identities.json')
    geometry=read(ART/'ba-query-occupancy-20260922-capture/evaluator/geometry.json')
    cohort={s:v for s,v in read(BASE/'cohort.json')['indices'].items() if s in ('train','evaluation')};truths={};count=0
    for split,indices in cohort.items():
        values=[]
        for index in indices:
            assert geometry[index]['id']==ids[index]['id'] and ids[index]['split']==split
            values.append(world_targets(geometry[index],query).reshape(2,51))
        z=np.array(values);truths[split]=z;count+=z.size
        np.testing.assert_allclose(z,np.load(DEST/f'{split}-truth.npy'),rtol=0,atol=1e-10)
        target=np.load(BASE/('training-targets.npz' if split=='train' else 'evaluation-targets.npz'))['horizon_boundary']
        target=np.where(target<=3.,target,np.inf)
        np.testing.assert_allclose(z[:,:,anchor],target,rtol=0,atol=1e-9)
    norm=np.load(BASE/'normalization.npz');features=((np.load(BASE/'features.npy')-norm['mean'])/norm['std']).astype(np.float32)
    torch.set_num_threads(4);replay=0;maximum=0.
    for name,root in MODELS.items():
        oldseal=read(root/'prediction-seal.json');selection=read(root/'metric-selection.json');cut=selection['threshold']
        assert sha(root/'metric.pt')==oldseal['checkpoint_sha256']==cs['checkpoint_hashes'][name]
        assert sha(root/'metric-selection.json')==oldseal['selection_sha256']
        equal(cut,cs['cutoffs'][name]);equal(cut,result['cutoffs'][name])
        assert read(root/'result.json')['actual_device']=='cpu'
        model=MetricContactModel().eval();model.load_state_dict(torch.load(root/'metric.pt',map_location='cpu',weights_only=True))
        for split,indices in cohort.items():
            values=np.load(DEST/f'{name}-{split}-components.npz');q,mu,s=[values[k].astype(float) for k in ('q','mu','scale')];z=truths[split]
            report=result['reports'][name][split]
            equal(focus(q[:,:,anchor],mu[:,:,anchor],s[:,:,anchor],z[:,:,anchor],cut),report['anchor'])
            equal(focus(q,mu,s,z,cut),report['all_widths']);equal(width_metrics(q,mu,s,z,cut),report['width_curve'])
            for layer,label in enumerate(('BODY','HEAD')):
                equal(focus(q[:,layer,anchor],mu[:,layer,anchor],s[:,layer,anchor],z[:,layer,anchor],cut),report['layers'][label])
            assert report['images']==len(indices) and report['layouts']==len({ids[i]['base_group_id'] for i in indices})
            assert report['unknown_sensor_frames']==sum(bool(ids[i]['baseline']['unknown']) for i in indices)
            oldfile=f'metric-{split}-predictions.npz';continuous=f'continuous-{split}.npy'
            for file in (oldfile,continuous):assert sha(root/file)==oldseal['files'][file]
            oldq=np.load(root/oldfile)['width_curve'].reshape(q.shape);np.testing.assert_allclose(q,oldq,atol=2e-7,rtol=2e-6)
            actual=crossing(q[:,:,anchor],mu[:,:,anchor],s[:,:,anchor],cut);saved=np.load(root/continuous);finite=np.isfinite(saved)
            np.testing.assert_array_equal(np.isfinite(actual),finite)
            max_crossing=float(abs(actual[finite]-saved[finite]).max()) if finite.any() else 0.
            assert max_crossing<=.001
            anchor_truth=z[:,:,anchor]
            observed=finite&np.isfinite(anchor_truth)&(anchor_truth>.300001)&(anchor_truth<=3.000001)
            np.testing.assert_array_equal(abs(actual[observed]-anchor_truth[observed])<=.050001,
                abs(saved[observed]-anchor_truth[observed])<=.050001)
            equal(dict(max_q_error=float(abs(q-oldq).max()),max_crossing_error_m=max_crossing,
                crossing_numerical_bound_m=.001,finite_masks_and_5cm_decisions_identical=True),result['parity'][name][split])
            with torch.inference_mode():
                components=model.components(torch.tensor(features[np.asarray(indices[:32])]),torch.tensor(query))
            for key,v in zip(('q','mu','scale'),components):
                a=v.numpy().reshape(32,2,51);b=values[key][:32]
                np.testing.assert_allclose(a,b,atol=1e-7,rtol=1e-6);maximum=max(maximum,float(abs(a-b).max()));replay+=a.size
    assert count==146880
    # Confirm the audit has not changed any sealed input or output payload.
    for path,h in seal['inputs'].items():assert sha(Path(path))==h,path
    for name,h in cs['files'].items():assert sha(DEST/name)==h,name
    output=dict(status='PASS',world_coordinate_targets=count,independent_quantile_method='60-step normalized-CDF bisection',
        all_focus_partitions_distributions_layers_width_counts_verified=True,old_q_and_continuous_parity_verified=True,
        representative_component_scalars_replayed=replay,maximum_component_replay_error=maximum,replay_device='cpu',
        source_inputs_components_unchanged=True,audit_code_sha256=sha(Path(__file__)),
        resumed_components_byte_identical=True,
        limitations='First32rows per model/split component replay only; remaining rows arithmetic and saved-q parity checked; no retraining or causal attribution.')
    OUT.mkdir(exist_ok=True);(OUT/'result.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8');print(json.dumps(output))


if __name__=='__main__':main()
