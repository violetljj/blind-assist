"""Independent world-AABB target audit and saved exact-supervision replay."""
import os
from pathlib import Path

import numpy as np
import torch

from audit_metric_contact import read, sha, equal, counts, boundary, summary, threshold
from metric_contact_model import MetricContactModel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DEST=ROOT/'artifacts.local/evidence/ba-exact-contact-20260923'
BASE=DEST.with_name('ba-contact-boundary-20260923')
PRIOR=DEST.with_name('ba-metric-contact-20260923')
SAMP=DEST.with_name('ba-contact-sampling-20260923')
OUT=DEST.with_name(DEST.name+'-audit')


def world_targets(geometry, queries):
    c=geometry['declared_camera']
    assert all(abs(c[k])<1e-8 for k in ('pitch','yaw','roll'))
    bands=((.42,.9),(-.2,.42))
    answers=[]
    for width,horizon,layer in queries:
        down_lo,down_hi=bands[int(layer)]
        body_y=(c['y']-float(width)/2,c['y']+float(width)/2)
        body_z=(c['z']-down_hi,c['z']-down_lo)
        first=float('inf')
        for obj in geometry['objects']:
            center=obj['render_bounds_center_m'];extent=obj['render_bounds_extent_m']
            lo=[center[j]-extent[j] for j in range(3)]
            hi=[center[j]+extent[j] for j in range(3)]
            if (hi[1]>=body_y[0] and lo[1]<=body_y[1] and hi[2]>=body_z[0] and lo[2]<=body_z[1]
                    and hi[0]>=c['x']+.3 and lo[0]<=c['x']+3.):
                first=min(first,max(.3,lo[0]-c['x']))
        answers.append(first)
    return np.array(answers,dtype=float)


def comparison(new,old):
    n,b=new['boundaries'],old['boundaries']
    d=dict(horizon_gain=n['horizon']['joint_within_5cm']-b['horizon']['joint_within_5cm'],
        horizon_false_crossing_change=n['horizon']['wrong_crossings_on_right_censored']-b['horizon']['wrong_crossings_on_right_censored'],
        width_gain=n['width']['joint_within_5cm']-b['width']['joint_within_5cm'],
        recall_change=new['both']['recall']-old['both']['recall'],FPR_change=new['both']['FPR']-old['both']['FPR'])
    d['horizon_false_rate_change']=d['horizon_false_crossing_change']/n['horizon']['right_censored_truth']
    d['joint_pass']=bool(d['horizon_gain']>=.1 and d['horizon_false_rate_change']<=.02 and
        d['width_gain']>=-.03 and d['recall_change']>=-.03 and d['FPR_change']<=.02)
    return d


def main():
    import json
    assert os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL'),'Use governed research-ue'
    assert not (OUT/'result.json').exists(),'Preserve completed audit'
    seal=read(DEST/'source-seal.json'); ps=read(DEST/'prediction-seal.json')
    for path,digest in seal['inputs'].items():assert sha(Path(path))==digest,path
    for name,digest in seal['code'].items():
        assert sha(HERE/name)==digest,name
        assert sha(DEST/'source-snapshot'/name)==digest,name
    for name,digest in ps['files'].items():assert sha(DEST/name)==digest,name
    assert sha(DEST/'metric.pt')==ps['checkpoint_sha256']
    assert sha(DEST/'metric-selection.json')==ps['selection_sha256']
    assert ps['evaluation_targets_joined'] is False
    initial=torch.load(DEST/'initial.pt',map_location='cpu',weights_only=True)
    oldinitial=torch.load(PRIOR/'initial.pt',map_location='cpu',weights_only=True)
    assert initial.keys()==oldinitial.keys()
    for key in initial:assert torch.equal(initial[key],oldinitial[key]),key
    cohort=read(BASE/'cohort.json')['indices']
    identities=read(ROOT/'artifacts.local/evidence/ba-query-occupancy-20260922-prepared/observations/identities.json')
    geometry=read(ROOT/'artifacts.local/evidence/ba-query-occupancy-20260922-capture/evaluator/geometry.json')
    targets=np.load(DEST/'exact-training-targets.npz'); sampled=np.load(SAMP/'training-queries.npz')
    for key in ('queries','indices'):np.testing.assert_array_equal(targets[key],sampled[key])
    np.testing.assert_array_equal(targets['binary'],sampled['labels'])
    np.testing.assert_array_equal(targets['indices'],cohort['train'])
    zs=[]
    for row,index in enumerate(targets['indices']):
        assert identities[int(index)]['split']=='train'
        assert geometry[int(index)]['id']==identities[int(index)]['id']
        z=world_targets(geometry[int(index)],targets['queries'][row]);zs.append(z)
        np.testing.assert_allclose(z,targets['z'][row],rtol=0,atol=1e-10)
        np.testing.assert_array_equal(z<=targets['queries'][row,:,1],targets['binary'][row])
    zs=np.array(zs);assert zs.size==62208
    equal(dict(rows=int(zs.size),finite=int(np.isfinite(zs).sum()),right_censored=int(np.isinf(zs).sum()),
        left_censored=int((zs<=.300001).sum()),half_bin_m=.025,metric_weight=1.),read(DEST/'label-receipt.json'))
    selection=read(DEST/'metric-selection.json'); result=read(DEST/'result.json');equal(selection,result['selection'])
    dev_y=np.load(BASE/'dev-selection-targets.npz')['seen'];history=np.load(DEST/'dev-checkpoint-logits.npz')
    assert np.array_equal(history['epochs'],np.arange(10,101,10))
    train_targets=dict(np.load(BASE/'training-targets.npz'));losses=[]
    for i,logits in enumerate(history['logits']):
        z=logits.astype(float);loss=float((np.logaddexp(0,z)-dev_y*z).mean());losses.append(loss)
        item=selection['history'][i];equal(loss,item['dev_BCE'])
        equal(item['train_objective'],item['train_balanced_BCE']+item['train_interval_NLL'])
        curves=dict(np.load(DEST/f'train-curves-epoch{int(history["epochs"][i])}.npz'))
        for kind in ('width','horizon'):
            equal(boundary(curves[kind+'_curve'],train_targets[kind+'_boundary'],kind,.5)[0],item['train_curves_at_half'][kind])
    assert selection['epoch']==int(history['epochs'][np.argmin(losses)])
    dev_pred=dict(np.load(DEST/'metric-dev-predictions.npz'));cut=threshold(dev_pred['seen'],dev_y)
    equal(cut,selection['threshold']);equal(counts(dev_pred['seen'],dev_y,cut),selection['dev'])
    np.testing.assert_allclose(1/(1+np.exp(-history['logits'][np.argmin(losses)].astype(float))),dev_pred['seen'],atol=2e-7,rtol=2e-6)
    prior_result=read(PRIOR/'result.json');sample_result=read(SAMP/'result.json')
    oldcut=read(PRIOR/'metric-selection.json')['threshold'];samplecut=read(SAMP/'geometry-selection.json')['threshold']
    decisions={}
    for split,filename in [('train','training-targets.npz'),('evaluation','evaluation-targets.npz')]:
        truth=dict(np.load(BASE/filename));meta=[identities[i] for i in cohort[split]]
        pred=dict(np.load(DEST/f'metric-{split}-predictions.npz'));old=dict(np.load(PRIOR/f'metric-{split}-predictions.npz'))
        sampled_pred=dict(np.load(SAMP/f'geometry-{split}-predictions.npz'))
        new_metrics=summary(pred,truth,meta,cut);old_metrics=summary(old,truth,meta,oldcut)
        sampled_metrics=summary(sampled_pred,truth,meta,samplecut);report=result['reports'][split]
        equal(new_metrics,report['metrics']);equal(old_metrics,report['control']);equal(old_metrics,prior_result['reports'][split]['metrics'])
        equal(sampled_metrics,sample_result['arms']['geometry'][split]['metrics'])
        decisions[split]=comparison(new_metrics,old_metrics);equal(decisions[split],report['decision'])
        equal(comparison(new_metrics,sampled_metrics),report['sampled_geometry_comparison'])
        for kind in ('width','horizon'):
            equal(boundary(pred[kind+'_curve'],truth[kind+'_boundary'],kind,cut)[1],report['tails'][kind]['metric'])
            equal(boundary(old[kind+'_curve'],truth[kind+'_boundary'],kind,oldcut)[1],report['tails'][kind]['control'])
        continuous=np.load(DEST/f'continuous-{split}.npy');z=truth['horizon_boundary']
        finite=np.isfinite(z)&(z>.300001)&(z<=3.000001);resolved=finite&np.isfinite(continuous);error=abs(continuous[resolved]-z[resolved])
        equal(dict(finite_truth=int(finite.sum()),resolved=int(resolved.sum()),within5cm=int((error<=.050001).sum()),
            conditional_MAE_m=float(error.mean()) if len(error) else None),report['continuous_horizon'])
    gain=decisions['train']['horizon_gain']>=.1 and decisions['evaluation']['horizon_gain']>=.1
    assert result['supervision_effect_train_and_eval']==gain
    assert result['joint_supervision_pass']==(gain and decisions['evaluation']['joint_pass'])
    device=result['actual_device'];assert device==read(DEST/'fit-started.json')['device']
    if device=='cuda':assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=MetricContactModel().to(device);model.load_state_dict(torch.load(DEST/'metric.pt',map_location=device,weights_only=True));model.eval()
    norm=np.load(BASE/'normalization.npz');features=((np.load(BASE/'features.npy')-norm['mean'])/norm['std']).astype(np.float32)
    query_sets=read(BASE/'queries.json');maximum=0.;replayed=0
    with torch.inference_mode():
        for split,indices in cohort.items():
            pred=dict(np.load(DEST/f'metric-{split}-predictions.npz'))
            for name,q in query_sets.items():
                query=torch.tensor(q,device=device,dtype=torch.float32);chunks=[]
                for start in range(0,len(indices),32):
                    x=torch.tensor(features[np.asarray(indices[start:start+32])],device=device)
                    chunks.append(model(x,query).sigmoid().cpu().numpy())
                actual=np.concatenate(chunks);maximum=max(maximum,float(abs(actual-pred[name]).max()))
                np.testing.assert_allclose(actual,pred[name],atol=2e-6,rtol=2e-5);replayed+=actual.size
            query=torch.tensor([[.6,3.,0],[.6,3.,1]],device=device);chunks=[]
            for start in range(0,len(indices),32):
                x=torch.tensor(features[np.asarray(indices[start:start+32])],device=device)
                chunks.append(model.continuous_crossing(x,query,cut).cpu().numpy())
            np.testing.assert_allclose(np.concatenate(chunks),np.load(DEST/f'continuous-{split}.npy'),atol=2e-6,rtol=2e-5)
    output=dict(status='PASS',world_coordinate_targets=int(zs.size),sampled_binary_parity=True,initial_tensors_identical=True,
        ten_checkpoint_selections_verified=True,train_trajectory_boundary_summaries_verified=True,
        source_input_prediction_hashes_verified=True,actual_replay_device=device,probabilities_replayed=replayed,
        maximum_probability_replay_error=maximum,audit_code_sha256=sha(Path(__file__)),
        limitations='No retraining, bootstrap recomputation, optimizer-loss replay, historical process-isolation proof, or intermediate-checkpoint replay.')
    OUT.mkdir(exist_ok=True);(OUT/'result.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8');print(json.dumps(output))


if __name__=='__main__':main()
