"""Independent MZ35 exact-prefix, schedule, selector and task audit."""
import argparse
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import read,write,sha,load_npz
from mz11_audit import scalar_metrics
from mz30_audit import forward_numpy


def restored_case(root,run):
    work=root/'artifacts.local/work';name='relation10000';frame=621;q=0;values={}
    for label,folder in [('MZ30',work/'mz30-branch-responsibility-20260910/run-v1'),('MZ32',work/'mz32-expanded-responsibility-20260910/run-v1'),('MZ35',run)]:
        pred=load_npz(folder/'predictions.npz');assert pred[name+'/global_ids'][frame]==11821 and pred[name+'/truth'][frame,q]
        rgb=float(pred[name+'/rgb'][frame,q]);tof=float(pred[name+'/tof'][frame,q]);z=float(pred[name+'/selector_logits'][frame,q])
        confidence=float(pred[name+'/confidence'][frame,q]);cutoff=float(np.load(folder/'cutoff.npy')[q]);removed=bool(pred[name+'/remove'][frame,q])
        assert rgb>0 and tof<0 and pred[name+'/eligible'][frame,q]
        np.testing.assert_allclose(confidence,1/(1+np.exp(z)),atol=1e-7,rtol=1e-6)
        assert removed==(confidence>=cutoff)
        values[label]=dict(selector_logit=z,negative_branch_confidence=confidence,cutoff=cutoff,removed=removed,raw_responsibility_correct=z>=0)
    assert values['MZ30']['removed'] and not values['MZ32']['removed'] and not values['MZ35']['removed']
    return dict(global_id=11821,cohort=name,frame=frame,query='BODY_NEAR',values=values,
        interpretation=('MZ35 raw responsibility now favors the correct RGB branch.' if values['MZ35']['raw_responsibility_correct'] else 'MZ35 raw responsibility still favors the incorrect ToF branch; alert retention does not establish corrected classification.'))


def prefix_optimizer(run,parent,initial,final,loss):
    prefix=torch.load(run/'prefix1200.pt',map_location='cpu',weights_only=True)
    reference=torch.load(parent/'selector.pt',map_location='cpu',weights_only=True)
    parity=read(run/'prefix-parity.json');assert parity['status']=='PASS'
    assert set(prefix)==set(reference)==set(initial)==set(final)
    errors={}
    for key in prefix:
        assert torch.equal(prefix[key],reference[key]),key
        assert prefix[key].dtype==reference[key].dtype and prefix[key].numpy().tobytes()==reference[key].numpy().tobytes(),key
        errors[key]=float((prefix[key]-reference[key]).abs().max())
    original_loss=np.load(parent/'training-loss.npy')
    np.testing.assert_array_equal(np.load(run/'prefix1200-loss.npy'),original_loss)
    np.testing.assert_array_equal(loss[:1200],original_loss)
    assert loss[:1200].dtype==original_loss.dtype and loss[:1200].tobytes()==original_loss.tobytes()
    assert parity['tensor_bit_exact'] and parity['loss_rows_bit_exact'] and parity['max_abs_by_tensor']==errors
    optimizers={}
    for step in [1200,4800]:
        optimizer=torch.load(run/f'optimizer{step}.pt',map_location='cpu',weights_only=True)
        assert len(optimizer['param_groups'])==1
        group=optimizer['param_groups'][0];ids=group['params']
        assert ids==list(range(len(initial))) and group['lr']==.001 and tuple(group['betas'])==(.9,.999)
        assert group['eps']==1e-8 and group['weight_decay']==0 and not group['amsgrad'] and not group['maximize']
        assert set(optimizer['state'])==set(ids)
        tensors={}
        for parameter_id,(name,parameter) in zip(ids,initial.items()):
            state=optimizer['state'][parameter_id]
            assert set(state)=={'step','exp_avg','exp_avg_sq'} and float(state['step'])==step
            assert state['exp_avg'].shape==state['exp_avg_sq'].shape==parameter.shape
            assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all() and (state['exp_avg_sq']>=0).all()
            tensors[name]=dict(step=int(state['step']),first_moment_norm=float(state['exp_avg'].norm()),second_moment_norm=float(state['exp_avg_sq'].norm()))
        optimizers[str(step)]=dict(sha256=sha(run/f'optimizer{step}.pt'),parameters=tensors,lr=group['lr'])
    return dict(tensor_bit_exact=True,loss_rows_bit_exact=True,max_abs_by_tensor=errors,optimizers=optimizers,
        limitation='Saved moments/step counters and frozen runner establish continuous optimizer configuration; no optimizer trajectory replay.')


def main(root,run):
    assert not (run/'audit.json').exists();receipt=read(run/'receipt.json');assert receipt['status']=='PASS' and receipt['training_steps']==4800
    for name,h in receipt['outputs'].items():assert sha(run/name)==h,name
    for name,h in receipt['code_sha256'].items():assert sha(Path(__file__).with_name(name))==h,name
    work=root/'artifacts.local/work';prior=work/'mz30-branch-responsibility-20260910/run-v1'
    coverage=work/'mz31-responsibility-coverage-20260910/run-v1';coverage_result=read(coverage/'result.json')
    assert read(prior/'audit.json')['status']=='PASS' and read(coverage/'audit.json')['status']=='PASS' and coverage_result['coverage_pass']
    data=load_npz(prior/'branches.npz');lookup=np.full(14200,-1,int);lookup[data['global_ids']]=np.arange(len(data['global_ids']))
    normal={k.removeprefix('normal/'):v for k,v in data.items() if k.startswith('normal/')}
    stress={k.removeprefix('stress/'):v for k,v in data.items() if k.startswith('stress/')}
    cohorts={k.removeprefix('ids/'):v for k,v in data.items() if k.startswith('ids/')}
    parent=work/'mz32-expanded-responsibility-20260910/run-v1';assert read(parent/'audit.json')['status']=='PASS'
    p32=load_npz(parent/'predictions.npz')
    previous=load_npz(work/'mz28-packet-availability-20260910/run-v1/predictions.npz');p30=load_npz(prior/'predictions.npz')
    preparation=dict(status='PASS',mz30_audit_sha256=sha(prior/'audit.json'),mz31_audit_sha256=sha(coverage/'audit.json'),coverage_examples=coverage_result['all_disagreement_examples'],mz32_audit_sha256=sha(parent/'audit.json'))
    for path,h in receipt['inputs'].items():assert sha(path)==h,path
    start=read(run/'start.json');result=read(run/'result.json');norm=load_npz(prior/'normalization.npz')
    assert start['inputs']==receipt['inputs']
    assert start['seed']==123 and start['steps']==4800 and start['batch_size']==16 and len(data['train_ids'])==7562
    assert start['initial_state_exact'] and start['normalization_exact']
    assert start['code_sha256']==receipt['code_sha256'] and start['runtime']['deterministic'] and not start['runtime']['matmul_tf32']
    assert start['runtime']['cublas_workspace']==':4096:8'
    ti=lookup[data['train_ids']];x=normal['features'];mean=x[ti].mean(0,keepdims=True);std=x[ti].std(0,keepdims=True).clip(.1)
    np.testing.assert_array_equal(norm['mean'],mean);np.testing.assert_array_equal(norm['std'],std)
    supervision=(normal['rgb']>=0)!=(normal['tof']>=0)
    np.testing.assert_array_equal(normal['target'],(normal['rgb']>=0)==normal['truth'])
    np.testing.assert_array_equal(normal['eligible'],supervision&(normal['baseline']>=0)&~normal['support'])
    mask=supervision[ti];target=normal['target'][ti];counts=[int((mask&target).sum()),int((mask&~target).sum())]
    weights=[sum(counts)/(2*n) for n in counts];assert start['class_counts_rgb_tof']==counts==[478,687] and start['class_weights_rgb_tof']==weights
    assert result['class_counts_rgb_tof']==counts and result['class_weights_rgb_tof']==weights and sum(counts)==coverage_result['all_disagreement_examples']==1165
    supervised=load_npz(run/'supervision.npz')
    for key,value in dict(global_ids=data['train_ids'],mask=mask,target=target,batches=np.tile(data['batches'],(4,1))).items():np.testing.assert_array_equal(supervised[key],value)
    np.testing.assert_array_equal(np.unique(supervised['batches']),data['train_ids'])
    assert supervised['batches'].shape==(4800,16)
    for cycle in range(4):np.testing.assert_array_equal(supervised['batches'][cycle*1200:(cycle+1)*1200],data['batches'])
    for q,name in enumerate(['BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR']):
        for desired,label in [(True,'RGB_correct'),(False,'ToF_correct')]:assert int((mask[:,q]&(target[:,q]==desired)).sum())==coverage_result['all_disagreements'][name][label]['examples']
    initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=True);final=torch.load(run/'selector.pt',map_location='cpu',weights_only=True)
    old_initial=torch.load(prior/'initial.pt',map_location='cpu',weights_only=True)
    for key in initial:torch.testing.assert_close(initial[key],old_initial[key],atol=0,rtol=0)
    torch.set_num_threads(1);torch.manual_seed(123)
    expected=torch.nn.Sequential(torch.nn.Linear(268,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    for key,value in expected.state_dict().items():torch.testing.assert_close(initial['net.'+key],value,atol=0,rtol=0)
    assert set(initial)==set(final) and sum(v.numel() for v in final.values())==start['parameters']==result['fit']['parameters']==8641
    scores=load_npz(run/'selector-scores.npz');replay={};initial_logits=None;final_train_logits=None
    for prefix,a in [('normal',normal),('stress',stress)]:
        feature=(a['features']-mean)/std;z0=forward_numpy(feature,initial);z=forward_numpy(feature,final)
        if prefix=='normal':initial_logits=z0;final_train_logits=z[ti]
        np.testing.assert_allclose(z,scores[prefix+'/logits'],atol=2e-4,rtol=2e-5)
        probability=np.exp(-np.logaddexp(0.,-scores[prefix+'/logits'].astype(float)))
        confidence=np.where(a['rgb']<0,probability,1-probability)
        np.testing.assert_allclose(confidence,scores[prefix+'/confidence'],atol=1e-7,rtol=1e-6)
        assert np.isfinite(z0).all()
        replay[prefix]=dict(frames=len(feature),raw_logit_values=z.size,final_max_abs_error=float(np.abs(z-scores[prefix+'/logits']).max()),initial_min=float(z0.min()),initial_max=float(z0.max()),confidence_max_abs_error=float(np.abs(confidence-scores[prefix+'/confidence']).max()))
    loss=np.load(run/'training-loss.npy');assert loss.shape==(4800,3);np.testing.assert_array_equal(loss[:,0],np.arange(1,4801))
    exposure=supervision[lookup[supervised['batches']]].sum((1,2));np.testing.assert_array_equal(loss[:,2],exposure)
    assert np.isfinite(loss).all() and (loss[:,1]>=0).all() and (loss[exposure==0,1]==0).all()
    assert result['fit']['steps']==4800 and result['fit']['active_batches']==int((exposure>0).sum()) and result['fit']['eligible_query_exposures']==int(exposure.sum())
    ix=lookup[data['batches'][0]];yy=normal['target'][ix];mm=supervision[ix];zz=initial_logits[ix]
    first_loss=(np.logaddexp(0.,np.where(yy,-zz,zz))*np.where(yy,weights[0],weights[1])*mm).sum()/max(1,mm.sum())
    np.testing.assert_allclose(first_loss,loss[0,1],atol=1e-6,rtol=1e-5)
    assert int(exposure.sum())==16120 and int((exposure>0).sum())==4472
    prefix_check=prefix_optimizer(run,parent,initial,final,loss)
    train_saved=load_npz(run/'train-scores.npz')
    for field,value in dict(global_ids=data['train_ids'],target=target,mask=mask,logits=scores['normal/logits'][ti]).items():np.testing.assert_array_equal(train_saved[field],value)
    train_error=((final_train_logits>=0)!=target)&mask
    saved_error=((train_saved['logits']>=0)!=target)&mask
    np.testing.assert_array_equal(train_error,saved_error)
    train_bce=np.logaddexp(0.,np.where(target,-final_train_logits,final_train_logits))*np.where(target,weights[0],weights[1])
    train_metrics=dict(examples=int(mask.sum()),responsibility_errors=int(train_error.sum()),errors_by_query=train_error.sum(0).tolist(),
        weighted_bce=float(train_bce[mask].mean()),weighted_bce_by_query=[float(train_bce[:,q][mask[:,q]].mean()) for q in range(4)])
    for k in ['examples','responsibility_errors','errors_by_query']:assert train_metrics[k]==result['train'][k]
    for k in ['weighted_bce','weighted_bce_by_query']:np.testing.assert_allclose(train_metrics[k],result['train'][k],atol=1e-6,rtol=1e-5)
    cut=np.load(run/'cutoff.npy');np.testing.assert_array_equal(cut,result['cutoff']);devix=lookup[cohorts['DEV']];calibration=[]
    for q in range(4):
        conf=scores['normal/confidence'][devix,q].astype(float);eligible=normal['eligible'][devix,q];truth=normal['truth'][devix,q]
        values=conf[eligible];disable=np.nextafter(max(1.,values.max() if len(values) else 1.),np.inf)
        candidates=np.unique(np.r_[.5,values[values>=.5],disable]);records=[]
        for threshold in candidates:
            removed=eligible&(conf>=threshold);tp=int((removed&truth).sum());fp=int((removed&~truth).sum())
            if tp==0:records.append((fp,-int(removed.sum()),float(threshold)))
        optimal=max(records);assert cut[q]==optimal[2]
        detail=dict(candidates=len(candidates),removed_fp=optimal[0],removed_tp=0,threshold=optimal[2],disabled=optimal[2]>1)
        assert detail==result['calibration'][q];calibration.append(detail)
    saved=load_npz(run/'predictions.npz');bits=0;changes={}
    for name,ii in cohorts.items():
        prefix='stress' if name=='stress' else 'normal';source=stress if name=='stress' else normal;ix=np.arange(len(ii)) if name=='stress' else lookup[ii]
        eligibility=source['eligible'][ix];confidence=scores[prefix+'/confidence'][ix];remove=eligibility&(confidence.astype(float)>=cut)
        negative=np.where(source['rgb'][ix]<0,source['rgb'][ix],source['tof'][ix]);assert (negative[remove]<0).all()
        for wrong in ([False] if name=='DEV' else [False,True]):
            key=name+('_wrong' if wrong else '');get=lambda k:saved[key+'/'+k]
            base,truth,old=(previous[key+'/'+k] for k in ['baseline','truth','candidate']);candidate=np.where(remove,negative,old)
            for field,value in dict(global_ids=ii,baseline=base,truth=truth,mz28=old,candidate=candidate,eligible=eligibility,remove=remove,confidence=confidence,
                selector_logits=scores[prefix+'/logits'][ix],rgb=source['rgb'][ix],tof=source['tof'][ix],support=source['support'][ix]).items():np.testing.assert_array_equal(get(field),value)
            assert not ((candidate>=0)&(old<0)).any() and not (remove&((base<0)|source['support'][ix])).any()
            agreement=(source['rgb'][ix]>=0)==(source['tof'][ix]>=0);np.testing.assert_array_equal(candidate[agreement],old[agreement])
            np.testing.assert_array_equal(candidate[base<0],old[base<0]);np.testing.assert_array_equal(candidate[~eligibility],old[~eligibility])
            score30=p30[key+'/candidate'];np.testing.assert_array_equal(get('mz30'),score30)
            actual=dict(baseline=scalar_metrics(base,truth),mz28=scalar_metrics(old,truth),mz30=scalar_metrics(score30,truth),candidate=scalar_metrics(candidate,truth),
                removed_fp=(remove&~truth).sum(0).tolist(),lost_tp=(remove&truth).sum(0).tolist(),
                remaining_baseline_fp=((candidate>=0)&(base>=0)&~truth).sum(0).tolist(),removed_baseline_fp=(remove&(base>=0)&~truth).sum(0).tolist(),
                added_tp=((candidate>=0)&(base<0)&truth).sum(0).tolist(),added_fp=((candidate>=0)&(base<0)&~truth).sum(0).tolist(),
                versus_mz30=dict(tp_gained=((candidate>=0)&(score30<0)&truth).sum(0).tolist(),tp_lost=((candidate<0)&(score30>=0)&truth).sum(0).tolist(),fp_removed=((candidate<0)&(score30>=0)&~truth).sum(0).tolist(),fp_added=((candidate>=0)&(score30<0)&~truth).sum(0).tolist()))
            score32=p32[key+'/candidate'];np.testing.assert_array_equal(get('mz32'),score32)
            actual['mz32']=scalar_metrics(score32,truth)
            actual['versus_mz32']=dict(tp_gained=((candidate>=0)&(score32<0)&truth).sum(0).tolist(),tp_lost=((candidate<0)&(score32>=0)&truth).sum(0).tolist(),
                fp_removed=((candidate<0)&(score32>=0)&~truth).sum(0).tolist(),fp_added=((candidate>=0)&(score32<0)&~truth).sum(0).tolist())
            assert actual==result['metrics'][key];changes[key]=actual;bits+=candidate.size
            if key in result['thin']:assert result['thin'][key]==scalar_metrics(candidate[100:125],truth[100:125])
    rows=read(root/'artifacts.local/work/mz15-shared-support-20260910/cache-v1/selected.json')
    for name in ['relation10000','distance5000']:
        rr=[r for r in rows if r['dataset']==name]
        for field,units in result['groups'][name].items():
            assert set(units)=={r[field] for r in rr}
            for unit,reported in units.items():
                ix=[i for i,r in enumerate(rr) if r[field]==unit];truth=saved[name+'/truth'][ix]
                assert reported==dict(frames=len(ix),mz32=scalar_metrics(saved[name+'/mz32'][ix],truth),candidate=scalar_metrics(saved[name+'/candidate'][ix],truth))
    removed=sum(sum(changes[n]['removed_baseline_fp']) for n in ['relation10000','distance5000'])
    totalfp=sum(sum(changes[n]['candidate']['fp']) for n in ['relation10000','distance5000'])
    gates=dict(placement_total_fp_le20=totalfp<=20,no_new_fp=all(not ((saved[n+'/candidate']>=0)&(saved[n+'/mz28']<0)).any() for n in cohorts),
        mz28_tp_retained=all(sum(changes[n]['lost_tp'])==0 for n in cohorts),additions_preserved=all(np.array_equal(saved[n+'/candidate'][saved[n+'/baseline']<0],saved[n+'/mz28'][saved[n+'/baseline']<0]) for n in cohorts),
        pole_clean=sum(result['thin']['clean']['tp'][1::2])>=48,pole_stress=sum(result['thin']['stress']['tp'][1::2])>=48)
    gates['decisive_improvement']=all(gates.values());assert gates==result['gates'] and totalfp==result['placement_total_fp'];assert bits==31200
    audit=dict(status='PASS',code_sha256=sha(Path(__file__)),receipt_sha256=sha(run/'receipt.json'),inputs=receipt['inputs'],preparation=preparation,
        training=dict(steps=4800,batch_cycles=4,class_counts_rgb_tof=counts,class_weights_rgb_tof=weights,initial_seed_exact=True,normalization_exact=True,
            active_batches=int((exposure>0).sum()),eligible_query_exposures=int(exposure.sum()),first_loss_error=abs(float(first_loss)-float(loss[0,1]))),
        prefix_check=prefix_check,train_metrics_numpy=train_metrics,paired_mz32={k:v['versus_mz32'] for k,v in changes.items()},
        selector_replay=replay,calibration=calibration,task_bits=bits,gates=gates,placement_baseline_fp_removed=removed,placement_total_fp=totalfp,
        restored_case=restored_case(root,run),
        dependencies={n:sha(Path(__file__).with_name(n)) for n in ['mz30_audit.py','mz11_audit.py','mz5_ensemble_readout.py']},
        limits=['MZ35 adds only optimization duration after exact MZ32 prefix; runtime eligibility remains MZ30.',
                'Frozen branch preparation is inherited through its bound MZ30 audit; no backbone or branch reconstruction rerun.',
                'Selector forward uses independent NumPy float64; frozen saved float32 confidence defines audited cutoff decisions.',
                'TRAIN branch predictions are in-sample; consumed DEV calibration does not establish independent generalization.'])
    write(run/'audit.json',audit);print('PASS',bits,'task bits',gates,'placement baseline FP removed',removed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    try:main(a.root,a.run)
    except Exception:
        import traceback
        if not (a.run/'audit.json').exists():write(a.run/'audit.json',dict(status='FAIL',code_sha256=sha(Path(__file__)),error=traceback.format_exc()))
        raise
