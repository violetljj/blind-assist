"""Independent saved-output arithmetic plus selected-checkpoint replay audit."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from metric_contact_model import MetricContactModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEST = ROOT / 'artifacts.local/evidence/ba-metric-contact-20260923'
BASE = DEST.with_name('ba-contact-boundary-20260923')
SAMP = DEST.with_name('ba-contact-sampling-20260923')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(actual, expected):
    if isinstance(actual, dict):
        for key, value in actual.items():
            equal(value, expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected): equal(value, reference)
    elif actual is None or isinstance(actual, (bool, str)):
        assert actual == expected, (actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)


def counts(score, truth, cut):
    on, y = score >= cut, truth.astype(bool)
    tp = int(np.count_nonzero(on & y)); fn = int(np.count_nonzero(~on & y))
    fp = int(np.count_nonzero(on & ~y)); tn = int(np.count_nonzero(~on & ~y))
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None, FPR=fp/(fp+tn) if fp+tn else None,
                Brier=float(np.mean((score.astype(float)-y)**2)))


def boundary(score, truth, kind, cut):
    axis = np.linspace(.2, 1.2, 51) if kind == 'width' else np.linspace(.3, 3, 55)
    p = score.astype(float).reshape(-1, 2, len(axis)); on = p >= cut
    crossing = np.full(truth.shape, np.inf)
    for row, layer in np.ndindex(truth.shape):
        hits = np.flatnonzero(on[row, layer])
        if len(hits): crossing[row, layer] = axis[hits[0]]
    finite = np.isfinite(truth) & (truth > axis[0]+1e-6) & (truth <= axis[-1]+1e-6)
    resolved = finite & np.isfinite(crossing)
    error = np.full(truth.shape, np.inf); error[resolved] = abs(crossing[resolved]-truth[resolved])
    n = int(finite.sum()); right = truth > axis[-1]+1e-6
    metrics = dict(interior_true_boundaries=n, left_censored_truth=int((truth <= axis[0]+1e-6).sum()),
        right_censored_truth=int(right.sum()), predicted_boundary_coverage=float(resolved.sum()/n) if n else None,
        conditional_MAE_m=float(error[resolved].mean()) if resolved.any() else None,
        within_5cm=int((finite & (error <= .050001)).sum()),
        joint_within_5cm=float((finite & (error <= .050001)).sum()/n) if n else None,
        wrong_crossings_on_right_censored=int((right & np.isfinite(crossing)).sum()),
        probability_monotonic_violations=int((np.diff(p, axis=-1) < -1e-6).sum()),
        binary_reversals=int((on[..., :-1] & ~on[..., 1:]).sum()))
    tail = dict(resolved=int(resolved.sum()), missing=int((finite & ~resolved).sum()),
                conditional_p90_m=float(np.quantile(error[resolved], .9)) if resolved.any() else None)
    return metrics, tail


def summary(pred, targets, metadata, cut):
    metrics = {k:counts(pred[k], targets[k], cut) for k in ('seen','width','horizon','both','extrapolation')}
    metrics['boundaries'] = {k:boundary(pred[k+'_curve'], targets[k+'_boundary'], k, cut)[0]
                             for k in ('width','horizon')}
    for field in ('type_id','layout_relation'):
        metrics[field] = {}
        for value in sorted({m[field] for m in metadata}):
            rows = np.array([m[field] == value for m in metadata])
            metrics[field][value] = counts(pred['both'][rows], targets['both'][rows], cut)
    groups = {}
    for i, row in enumerate(metadata):
        groups.setdefault((row['base_group_id'], row['frame_in_clip']), {})[row['layout_relation']] = i
    margins=[]; joint=[]
    for group in groups.values():
        for a,b in [('INSIDE','BOUNDARY'),('BOUNDARY','OUTSIDE'),('INSIDE','OUTSIDE')]:
            a,b = group[a],group[b]
            ya,yb=targets['both'][a],targets['both'][b]; change=ya != yb
            margins.extend(((pred['both'][a].astype(float)-pred['both'][b])*np.where(ya,1.,-1.))[change])
            joint.extend((((pred['both'][a]>=cut)==ya)&((pred['both'][b]>=cut)==yb))[change])
    margins=np.array(margins)
    metrics['lateral_pairs']=dict(changed_pairs=len(margins), strict_order_correct=int((margins>0).sum()),
        ties=int((margins==0).sum()), order_accuracy=float(((margins>0)+.5*(margins==0)).mean()) if len(margins) else None,
        both_decisions_correct=int(np.sum(joint)), joint_accuracy=float(np.mean(joint)) if joint else None)
    metrics['unknown_sensor_frames']=sum(bool(m['baseline']['unknown']) for m in metadata)
    metrics['component_gate']=bool(metrics['both']['recall']>=.8 and metrics['both']['FPR']<=.05 and
        all((metrics['boundaries'][k]['joint_within_5cm'] or 0)>=.8 for k in ('width','horizon')))
    return metrics


def threshold(score, truth):
    values,inverse=np.unique(score.ravel().astype(float), return_inverse=True)
    pos=np.bincount(inverse,weights=truth.ravel(),minlength=len(values))
    neg=np.bincount(inverse,weights=~truth.ravel(),minlength=len(values))
    tp=np.cumsum(pos[::-1])[::-1]; fp=np.cumsum(neg[::-1])[::-1]
    ok=np.flatnonzero(fp <= .05*np.count_nonzero(~truth)+1e-10)
    if not len(ok): return float(np.nextafter(values[-1],np.inf))
    best=max(ok,key=lambda i:(tp[i],-fp[i],values[i]))
    return float(values[best]) if tp[best]>0 else float(np.nextafter(values[-1],np.inf))


def main():
    assert os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL'), 'Use governed research-ue'
    seal=read(DEST/'source-seal.json'); prediction_seal=read(DEST/'prediction-seal.json')
    for path,digest in seal['inputs'].items(): assert sha(Path(path))==digest,path
    for name,digest in seal['code'].items():
        assert sha(HERE/name)==digest,name
        assert sha(DEST/'source-snapshot'/name)==digest,name
    for name,digest in prediction_seal['files'].items(): assert sha(DEST/name)==digest,name
    assert sha(DEST/'metric-selection.json')==prediction_seal['selection_sha256']
    assert sha(DEST/'metric.pt')==prediction_seal['checkpoint_sha256']
    assert prediction_seal['evaluation_targets_joined'] is False
    selection=read(DEST/'metric-selection.json'); result=read(DEST/'result.json')
    equal(selection,result['selection'])
    cohort=read(BASE/'cohort.json')['indices']; norm=np.load(BASE/'normalization.npz')
    features=((np.load(BASE/'features.npy')-norm['mean'])/norm['std']).astype(np.float32)
    identities=read(ROOT/'artifacts.local/evidence/ba-query-occupancy-20260922-prepared/observations/identities.json')
    dev_y=np.load(BASE/'dev-selection-targets.npz')['seen']
    checkpoints=np.load(DEST/'dev-checkpoint-logits.npz'); losses=[]
    assert np.array_equal(checkpoints['epochs'],np.arange(10,101,10))
    for i,logits in enumerate(checkpoints['logits']):
        z=logits.astype(float); loss=float((np.logaddexp(0,z)-dev_y*z).mean()); losses.append(loss)
        equal(loss,selection['history'][i]['dev_BCE'])
    assert selection['epoch']==int(checkpoints['epochs'][np.argmin(losses)])
    dev_pred=dict(np.load(DEST/'metric-dev-predictions.npz'))
    cut=threshold(dev_pred['seen'],dev_y); equal(cut,selection['threshold'])
    equal(counts(dev_pred['seen'],dev_y,cut),selection['dev'])
    selected_logits=checkpoints['logits'][np.argmin(losses)]
    np.testing.assert_allclose(1/(1+np.exp(-selected_logits.astype(float))),dev_pred['seen'],atol=2e-7,rtol=2e-6)
    control_result=read(SAMP/'result.json'); control_cut=read(SAMP/'geometry-selection.json')['threshold']
    for split,targetname in [('train','training-targets.npz'),('evaluation','evaluation-targets.npz')]:
        target=dict(np.load(BASE/targetname)); pred=dict(np.load(DEST/f'metric-{split}-predictions.npz'))
        meta=[identities[i] for i in cohort[split]]; report=result['reports'][split]
        metrics=summary(pred,target,meta,cut); equal(metrics,report['metrics'])
        control=dict(np.load(SAMP/f'geometry-{split}-predictions.npz'))
        base=summary(control,target,meta,control_cut); equal(base,report['control'])
        equal(base,control_result['arms']['geometry'][split]['metrics'])
        nb,bb=metrics['boundaries'],base['boundaries']
        decision=dict(horizon_gain=nb['horizon']['joint_within_5cm']-bb['horizon']['joint_within_5cm'],
            horizon_false_crossing_change=nb['horizon']['wrong_crossings_on_right_censored']-bb['horizon']['wrong_crossings_on_right_censored'],
            width_gain=nb['width']['joint_within_5cm']-bb['width']['joint_within_5cm'],
            recall_change=metrics['both']['recall']-base['both']['recall'], FPR_change=metrics['both']['FPR']-base['both']['FPR'])
        decision['joint_pass']=bool(decision['horizon_gain']>=.10 and decision['horizon_false_crossing_change']<=0 and
            decision['width_gain']>=-.03 and decision['recall_change']>=-.03 and decision['FPR_change']<=.02)
        equal(decision,report['decision'])
        for kind in ('width','horizon'):
            equal(boundary(pred[kind+'_curve'],target[kind+'_boundary'],kind,cut)[1],report['tails'][kind]['metric'])
            equal(boundary(control[kind+'_curve'],target[kind+'_boundary'],kind,control_cut)[1],report['tails'][kind]['control'])
        continuous=np.load(DEST/f'continuous-{split}.npy'); truth=target['horizon_boundary']
        finite=np.isfinite(truth)&(truth>.300001)&(truth<=3.000001); resolved=finite&np.isfinite(continuous)
        errors=abs(continuous[resolved]-truth[resolved])
        equal(dict(finite_truth=int(finite.sum()),resolved=int(resolved.sum()),within5cm=int((errors<=.050001).sum()),
            conditional_MAE_m=float(errors.mean()) if len(errors) else None),report['continuous_horizon'])
    # Reproduce the actual selected backend; cross-device float32 differences
    # are not a checkpoint-integrity failure. No tolerance is relaxed.
    torch.set_num_threads(4); device=result['actual_device']
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=MetricContactModel().to(device);model.load_state_dict(torch.load(DEST/'metric.pt',map_location=device,weights_only=True));model.eval()
    query_sets={k:np.asarray(v,np.float32) for k,v in read(BASE/'queries.json').items()}
    replayed=0; maximum=0.
    with torch.inference_mode():
        for split,indices in cohort.items():
            saved=dict(np.load(DEST/f'metric-{split}-predictions.npz'))
            for name,query in query_sets.items():
                chunks=[];qq=torch.tensor(query,device=device)
                for start in range(0,len(indices),32):
                    x=torch.tensor(features[np.array(indices[start:start+32])],device=device)
                    chunks.append(model(x,qq).sigmoid().cpu().numpy())
                actual=np.concatenate(chunks);maximum=max(maximum,float(abs(actual-saved[name]).max()))
                np.testing.assert_allclose(actual,saved[name],atol=2e-6,rtol=2e-5);replayed+=actual.size
            chunks=[];qq=torch.tensor([[.6,3.,0],[.6,3.,1]],device=device)
            for start in range(0,len(indices),32):
                x=torch.tensor(features[np.array(indices[start:start+32])],device=device)
                chunks.append(model.continuous_crossing(x,qq,cut).cpu().numpy())
            np.testing.assert_allclose(np.concatenate(chunks),np.load(DEST/f'continuous-{split}.npy'),atol=2e-6,rtol=2e-5)
    output=dict(status='PASS',audit_code_sha256=sha(Path(__file__)),checkpoint_replay_device=device,
        probabilities_replayed=replayed,maximum_probability_replay_error=maximum,checkpoint_selection_epochs=10,
        input_source_prediction_seals_verified=True,independent_arithmetic=True,
        limitations='Verifies recorded seal claims and indexed code path, not historical process isolation; no retraining or bootstrap audit.')
    path=DEST.with_name(DEST.name+'-audit')/'result.json'; assert not path.exists(), 'Preserve existing audit'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8'); print(json.dumps(output))


if __name__=='__main__': main()
