"""Independent regional sampling, paired-input, selection and metric audit."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from audit_metric_contact import read,sha,equal,counts,threshold,summary,boundary
from audit_contact_decomposition import inverse,error
from regional_contact_model import RegionContactModel

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];ART=ROOT/'artifacts.local/evidence'
DEST=ART/'ba-regional-contact-20260923';OUT=DEST.with_name(DEST.name+'-audit')
BASE=ART/'ba-contact-boundary-20260923';EXACT=ART/'ba-exact-contact-20260923';SAMP=ART/'ba-contact-sampling-20260923'


def independent_pool(raw):
    visual=raw[:,:4480].reshape(-1,40,8,14).transpose(0,2,3,1).astype(float)
    boxes=raw[:,4480:].reshape(-1,64,6)[:,:,2:].astype(float)
    output=np.zeros((len(raw),64,40),float);batch=np.arange(len(raw))[:,None]
    for fy in (1/6,3/6,5/6):
        for fx in (1/6,3/6,5/6):
            y=np.clip((boxes[:,:,0]+fy*(boxes[:,:,2]-boxes[:,:,0]))*8-.5,0,7)
            x=np.clip((boxes[:,:,1]+fx*(boxes[:,:,3]-boxes[:,:,1]))*14-.5,0,13)
            yl=np.floor(y).astype(int);xl=np.floor(x).astype(int);yh=np.minimum(yl+1,7);xh=np.minimum(xl+1,13)
            wy=(y-yl)[...,None];wx=(x-xl)[...,None]
            output+=(visual[batch,yl,xl]*(1-wx)*(1-wy)+visual[batch,yl,xh]*wx*(1-wy)+
                visual[batch,yh,xl]*(1-wx)*wy+visual[batch,yh,xh]*wx*wy)/9
    return output


def expected_permutations(raw):
    rows=[]
    for frame in np.asarray(raw,np.float32):
        digest=hashlib.sha256(b'regional-contact-20260923-v1'+frame.tobytes()).digest()
        generator=np.random.default_rng(int.from_bytes(digest[:8],'little'))
        for attempt in range(128):
            order=generator.permutation(64)
            if not any(order==np.arange(64)):
                rows.append(order);break
        else:raise AssertionError('Derangement unavailable')
    return np.asarray(rows)


def main():
    from audit_exact_contact import comparison
    assert os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL'),'Use governed research-ue'
    assert not (OUT/'result.json').exists(),'Preserve completed audit'
    seal=read(DEST/'source-seal.json');ps=read(DEST/'prediction-seal.json');result=read(DEST/'result.json')
    for path,digest in seal['inputs'].items():assert sha(Path(path))==digest,path
    for name,digest in seal['code'].items():
        assert sha(HERE/name)==digest,name
        assert sha(DEST/'source-snapshot'/name)==digest,name
    for name,digest in ps['files'].items():assert sha(DEST/name)==digest,name
    assert ps['evaluation_targets_joined'] is False
    indices=read(BASE/'cohort.json')['indices'];raw=np.load(BASE/'features.npy')
    rgb=np.load(DEST/'regional-rgb.npy');pooled=independent_pool(raw)
    np.testing.assert_allclose(pooled,rgb,atol=2e-5,rtol=2e-5)
    pool_error=float(abs(pooled-rgb).max())
    permutations=np.load(DEST/'permutations.npy');np.testing.assert_array_equal(permutations,expected_permutations(raw))
    assert np.all(permutations!=np.arange(64))
    np.testing.assert_array_equal(np.sort(permutations,axis=1),np.broadcast_to(np.arange(64),permutations.shape))
    norm=np.load(DEST/'normalization.npz');train=rgb[np.asarray(indices['train'])]
    # Saved pooling is serialized in C order; float32 reduction order can change
    # from the original strided tensor view. Independently check float64 moments.
    mean=train.astype(float).mean(axis=(0,1));std=np.maximum(train.astype(float).std(axis=(0,1)),.01)
    np.testing.assert_allclose(mean,norm['mean'],atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(std,norm['std'],atol=2e-6,rtol=2e-6)
    norm_error=max(float(abs(mean-norm['mean']).max()),float(abs(std-norm['std']).max()))
    normalized=(rgb-norm['mean'])/norm['std'];sensor=raw[:,4480:].reshape(-1,64,6)
    assert np.isin(sensor[:,:,1],[0,1]).all()
    assert ((sensor[:,:,2:]>=0)&(sensor[:,:,2:]<=1)).all()
    assert (sensor[:,:,4:]>sensor[:,:,2:4]).all()
    assert (sensor[:,:,0][sensor[:,:,1]==0]==0).all()
    assert ((sensor[:,:,0][sensor[:,:,1]==1]>0)&(sensor[:,:,0][sensor[:,:,1]==1]<1)).all()
    features={mode:np.load(DEST/f'{mode}-features.npy') for mode in ('aligned','misaligned')}
    np.testing.assert_array_equal(features['aligned'][:,:,:40],normalized)
    np.testing.assert_array_equal(features['misaligned'][:,:,:40],normalized[np.arange(len(raw))[:,None],permutations])
    for value in features.values():np.testing.assert_array_equal(value[:,:,40:],sensor)
    initial=torch.load(DEST/'initial.pt',map_location='cpu',weights_only=True)
    expected=RegionContactModel();assert initial.keys()==expected.state_dict().keys()
    assert sum(t.numel() for t in initial.values())==20163
    device=result['actual_device'];torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda':assert torch.cuda.is_available()
    dev_y=np.load(BASE/'dev-selection-targets.npz')['seen']
    identities=read(ART/'ba-query-occupancy-20260922-prepared/observations/identities.json')
    targets={split:dict(np.load(BASE/name)) for split,name in [('train','training-targets.npz'),('evaluation','evaluation-targets.npz')]}
    query_sets=read(BASE/'queries.json');computed={};tails={};replayed=0;max_replay=0.
    for mode,x in features.items():
        start=read(DEST/f'{mode}-fit-started.json')
        assert start['initial_sha256']==sha(DEST/'initial.pt') and start['schedule_sha256']==sha(BASE/'batch-schedule.npy')
        assert start['device']==device and start['updates']==2700 and start['epochs']==100
        selection=read(DEST/f'{mode}-selection.json');equal(selection,result['selections'][mode])
        assert selection['checkpoint_sha256']==sha(DEST/f'{mode}.pt')
        saved=np.load(DEST/f'{mode}-dev-checkpoint-logits.npz');np.testing.assert_array_equal(saved['epochs'],np.arange(10,101,10))
        losses=[]
        for i,logits in enumerate(saved['logits']):
            z=logits.astype(float);loss=float((np.logaddexp(0,z)-dev_y*z).mean());losses.append(loss)
            row=selection['history'][i];equal(loss,row['dev_BCE']);equal(row['train_objective'],row['train_balanced_BCE']+row['train_interval_NLL'])
        assert selection['epoch']==int(saved['epochs'][np.argmin(losses)])
        dev_pred=dict(np.load(DEST/f'{mode}-dev-predictions.npz'));cut=threshold(dev_pred['seen'],dev_y)
        equal(cut,selection['threshold']);equal(counts(dev_pred['seen'],dev_y,cut),selection['dev'])
        np.testing.assert_allclose(1/(1+np.exp(-saved['logits'][np.argmin(losses)].astype(float))),dev_pred['seen'],atol=2e-7,rtol=2e-6)
        computed[mode]={};tails[mode]={}
        model=RegionContactModel().to(device).eval();model.load_state_dict(torch.load(DEST/f'{mode}.pt',map_location=device,weights_only=True))
        for split,rows in indices.items():
            pred=dict(np.load(DEST/f'{mode}-{split}-predictions.npz'));parts=np.load(DEST/f'{mode}-{split}-components.npz')
            if split in targets:
                truth=targets[split];meta=[identities[i] for i in rows]
                metrics=summary(pred,truth,meta,cut);equal(metrics,result['arms'][mode][split]['metrics'])
                z=truth['horizon_boundary'];finite=np.isfinite(z)&(z>.300001)&(z<=3.000001)
                median=inverse(parts['mu'].astype(float),parts['scale'].astype(float),.5)
                m=error(median,z,finite);equal(m,result['arms'][mode][split]['conditional_median'])
                computed[mode][split]=dict(metrics=metrics,median=m)
                tails[mode][split]={kind:boundary(pred[kind+'_curve'],truth[kind+'_boundary'],kind,cut)[1]
                    for kind in ('width','horizon')}
            with torch.inference_mode():
                replay_rows=rows[:32]
                for name,query in query_sets.items():
                    qq=torch.tensor(query,device=device,dtype=torch.float32);chunks=[]
                    for first in range(0,len(replay_rows),32):
                        xx=torch.tensor(x[np.asarray(replay_rows[first:first+32])],device=device)
                        chunks.append(model(xx,qq).sigmoid().cpu().numpy())
                    actual=np.concatenate(chunks);np.testing.assert_allclose(actual,pred[name][:32],atol=2e-6,rtol=2e-5)
                    max_replay=max(max_replay,float(abs(actual-pred[name][:32]).max()));replayed+=actual.size
                qq=torch.tensor([[.6,3.,0],[.6,3.,1]],device=device);chunks=[[],[],[]]
                for first in range(0,len(replay_rows),32):
                    xx=torch.tensor(x[np.asarray(replay_rows[first:first+32])],device=device)
                    for group,value in zip(chunks,model.components(xx,qq)):group.append(value.cpu().numpy())
                for key,group in zip(('q','mu','scale'),chunks):np.testing.assert_allclose(np.concatenate(group),parts[key][:32],atol=2e-6,rtol=2e-5)
        del model
    for split in targets:
        a,b=computed['aligned'][split],computed['misaligned'][split]
        d=comparison(a['metrics'],b['metrics']);d['conditional_median_gain']=a['median']['hit_rate']-b['median']['hit_rate']
        d['joint_pass']=bool(d['joint_pass'] and d['conditional_median_gain']>=.10)
        equal(d,result['reports'][split]['decision'])
    assert result['joint_alignment_pass']==result['reports']['evaluation']['decision']['joint_pass']
    for path,digest in seal['inputs'].items():assert sha(Path(path))==digest,path
    for name,digest in ps['files'].items():assert sha(DEST/name)==digest,name
    output=dict(status='PASS',regional_values_verified=int(rgb.size),maximum_independent_pooling_error=pool_error,
        derangements_verified=len(permutations),train_only_channel_normalization_verified=True,sensor_fields_unchanged=True,
        maximum_float64_normalization_error=norm_error,
        shared_initial_and_schedule_receipts_verified=True,checkpoint_selection_epochs_per_arm=10,
        conditional_median_method='independent60stepCDFbisection',representative_probability_values_replayed=replayed,
        independent_boundary_tails=tails,
        maximum_probability_replay_error=max_replay,actual_replay_device=device,input_source_prediction_seals_verified=True,
        audit_code_sha256=sha(Path(__file__)),limitations='Checkpoint replay first32rows per arm/split only; arithmetic all rows. No training replay, optimizer-loss verification, bootstrap audit or causal attribution.')
    OUT.mkdir(exist_ok=True);(OUT/'result.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8');print(json.dumps(output))


if __name__=='__main__':main()
