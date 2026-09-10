"""Independent MZ30 opportunity, selector and preservation audit; no fit."""
import argparse
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import read,write,sha,load_npz
from mz11_audit import scalar_metrics
from mz29_audit import numpy_geometry


def audit_prepare(root,run):
    work=root/'artifacts.local/work';rank=work/'mz20-rank-objective-20260910/run-v1'
    receipt,start,gate=(read(run/name) for name in ['prepare-receipt.json','start.json','gate.json'])
    assert receipt['status']=='PASS'
    for name,h in receipt['outputs'].items():assert sha(run/name)==h,name
    for path,h in start['inputs'].items():assert sha(path)==h,path
    assert start['code_sha256']==sha(Path(__file__).with_name('mz30_prepare.py'))
    assert start['protocol_sha256']==sha(Path(__file__).with_name('MZ30_BRANCH_RESPONSIBILITY_PROTOCOL_20260910.md'))
    data=load_npz(run/'branches.npz');ids=data['global_ids'];train=data['train_ids'];batches=data['batches']
    np.testing.assert_array_equal(batches,np.load(rank/'batches.npy'));np.testing.assert_array_equal(train,np.unique(batches));assert len(train)==7562
    oldpath=work/'mz8-attribution-20260910/cache-v5';newpath=work/'mz15-shared-support-20260910/cache-v1'
    old,new=load_npz(oldpath/'observations.npz'),load_npz(newpath/'observations.npz')
    truth=np.concatenate([load_npz(oldpath/'evaluator.npz')['truth'],load_npz(newpath/'evaluator.npz')['truth']])
    ranges=np.concatenate([old['ranges'],new['ranges']]);valid=np.concatenate([old['valid'],new['valid']])
    cohorts=dict(DEV=np.flatnonzero(old['role']=='DEV_ONLY'),clean=np.arange(3500,3700),stress=np.arange(3500,3700),relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    np.testing.assert_array_equal(ids,np.unique(np.r_[train,*cohorts.values()]));assert len(ids)==11562
    assert np.isin(train,np.r_[np.flatnonzero(old['role']=='TRAIN_ONLY'),np.arange(3500,11200)]).all()
    for name,ii in cohorts.items():np.testing.assert_array_equal(data['ids/'+name],ii)
    lookup=np.full(14200,-1,int);lookup[ids]=np.arange(len(ids));reference=load_npz(rank/'predictions.npz')
    p28=load_npz(work/'mz28-packet-availability-20260910/run-v1/predictions.npz')
    p11=load_npz(work/'mz11-selective-addition-20260910/run-v1/predictions.npz');p13=load_npz(work/'mz13-training-coverage-20260910/features-v1/features.npz')
    base=np.full((14200,4),np.nan);base[np.flatnonzero(old['role']=='TRAIN_ONLY')]=p11['TRAIN/baseline']
    base[3700:8700]=p13['baseline'][p13['dataset']=='relation'];base[8700:11200]=p13['baseline'][p13['dataset']=='distance']
    for name,ii in cohorts.items():
        if name!='stress':base[ii]=reference['BODY_RANK/'+name+'/baseline']
    rays=torch.load(rank/'BODY_RANK.pt',map_location='cpu',weights_only=True)['rays'].numpy()
    frozen=torch.load(work/'mz5-fixed-ensemble-20260910/compact-v1/compact.pt',map_location='cpu',weights_only=True)
    summaries={};maxerr=0.;normal={k.removeprefix('normal/'):v for k,v in data.items() if k.startswith('normal/')}
    stress={k.removeprefix('stress/'):v for k,v in data.items() if k.startswith('stress/')}
    for name,a,ii in [('normal',normal,ids),('stress',stress,cohorts['stress'])]:
        assert a['features'].shape==(len(ii),264) and np.isfinite(a['features']).all()
        np.testing.assert_array_equal(a['truth'],truth[ii])
        np.testing.assert_array_equal(a['baseline'],reference['BODY_RANK/stress/baseline'] if name=='stress' else base[ii])
        np.testing.assert_array_equal(a['features'][:,256:260],a['rgb']);np.testing.assert_array_equal(a['features'][:,260:],a['tof'])
        for branch,offset in [('rgb',0),('tof',128)]:
            hidden=a['features'][:,offset:offset+128];assert (hidden>=0).all()
            reconstructed=hidden.astype(float)@frozen[branch+'.2.weight'].numpy().astype(float).T+frozen[branch+'.2.bias'].numpy().astype(float)
            np.testing.assert_allclose(reconstructed,a[branch],atol=1e-4,rtol=1e-5)
        average=.5*(a['rgb']+a['tof']);np.testing.assert_allclose(average,a['baseline'],atol=1e-4,rtol=1e-5)
        np.testing.assert_array_equal(average>=0,a['baseline']>=0);maxerr=max(maxerr,float(np.abs(average-a['baseline']).max()))
        rr,vv=(old['stress_ranges'],old['stress_valid']) if name=='stress' else (ranges[ii],valid[ii])
        for begin in range(0,len(ii),64):
            end=min(begin+64,len(ii));g=numpy_geometry(rr[begin:end],vv[begin:end],rays).any((1,2,3));np.testing.assert_array_equal(g,a['support'][begin:end])
        disagree=(a['rgb']>=0)!=(a['tof']>=0);eligible=(a['baseline']>=0)&~a['support']&disagree
        target=(a['rgb']>=0)==a['truth'];np.testing.assert_array_equal(eligible,a['eligible']);np.testing.assert_array_equal(target,a['target'])
        assert ((a['rgb']>=0)!=a['truth'])[eligible].sum()==target[eligible].size-target[eligible].sum()
    assert maxerr==gate['baseline_parity_max_abs']
    for name,ii in cohorts.items():
        a=stress if name=='stress' else {k:v[lookup[ii]] for k,v in normal.items()}
        np.testing.assert_array_equal(a['support'],p28[name+'/original_support']);summaries[name]={}
        for label,title in [(False,'baseline_false'),(True,'baseline_true')]:
            selected=(a['baseline']>=0)&(a['truth']==label);disagree=(a['rgb']>=0)!=(a['tof']>=0)
            summary=dict(total=selected.sum(0).tolist(),unsupported=(selected&~a['support']).sum(0).tolist(),eligible=(selected&a['eligible']).sum(0).tolist(),
                unsupported_both_positive=(selected&~a['support']&~disagree).sum(0).tolist(),eligible_rgb_correct=(selected&a['eligible']&a['target']).sum(0).tolist(),eligible_tof_correct=(selected&a['eligible']&~a['target']).sum(0).tolist())
            assert summary==gate['summaries'][name][title];summaries[name][title]=summary
    mask=normal['eligible'][lookup[train]];target=normal['target'][lookup[train]]
    counts=[int((mask&target).sum()),int((mask&~target).sum())];opportunity=sum(sum(summaries[n]['baseline_false']['eligible']) for n in ['relation10000','distance5000'])
    expected=dict(placement_opportunity=opportunity,required=41,train_rgb_tof_correct=counts,passed=opportunity>=41 and min(counts)>=20)
    assert gate['gate']==expected and gate['status']=='PASS' and expected['passed']
    return data,normal,stress,lookup,cohorts,p28,dict(status='PASS',gate=expected,baseline_parity_max_abs=maxerr,prepare_receipt_sha256=sha(run/'prepare-receipt.json'),inputs=start['inputs'])


def forward_numpy(features,state):
    n=len(features);x=np.concatenate([np.broadcast_to(features[:,None,:],(n,4,264)),np.broadcast_to(np.eye(4),(n,4,4))],axis=-1)
    weights={k:v.numpy().astype(float) for k,v in state.items()}
    hidden=np.maximum(x@weights['net.0.weight'].T+weights['net.0.bias'],0)
    return (hidden@weights['net.2.weight'].T+weights['net.2.bias'])[...,0]


def main(root,run):
    assert not (run/'audit.json').exists();receipt=read(run/'receipt.json');assert receipt['status']=='PASS' and receipt['training_steps']==1200
    for name,h in receipt['outputs'].items():assert sha(run/name)==h,name
    for name,h in receipt['code_sha256'].items():assert sha(Path(__file__).with_name(name))==h,name
    data,normal,stress,lookup,cohorts,previous,preparation=audit_prepare(root,run)
    start=read(run/'train-start.json');result=read(run/'result.json');norm=load_npz(run/'normalization.npz')
    assert start['prepare_receipt_sha256']==sha(run/'prepare-receipt.json') and start['seed']==123 and start['batches']==1200 and start['batch_size']==16 and start['train_unique']==7562
    assert start['code_sha256']==receipt['code_sha256'] and start['runtime']['deterministic'] and not start['runtime']['matmul_tf32']
    assert start['runtime']['cublas_workspace']==':4096:8'
    ti=lookup[data['train_ids']];x=normal['features'];mean=x[ti].mean(0,keepdims=True);std=x[ti].std(0,keepdims=True).clip(.1)
    np.testing.assert_array_equal(norm['mean'],mean);np.testing.assert_array_equal(norm['std'],std)
    mask=normal['eligible'][ti];target=normal['target'][ti];counts=[int((mask&target).sum()),int((mask&~target).sum())]
    weights=[sum(counts)/(2*n) for n in counts];assert start['class_counts_rgb_tof']==counts and start['class_weights_rgb_tof']==weights
    initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=True);final=torch.load(run/'selector.pt',map_location='cpu',weights_only=True)
    torch.set_num_threads(1);torch.manual_seed(123)
    expected=torch.nn.Sequential(torch.nn.Linear(268,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    for key,value in expected.state_dict().items():torch.testing.assert_close(initial['net.'+key],value,atol=0,rtol=0)
    assert set(initial)==set(final) and sum(v.numel() for v in final.values())==start['parameters']==result['fit']['parameters']==8641
    scores=load_npz(run/'selector-scores.npz');replay={};initial_logits=None
    for prefix,a in [('normal',normal),('stress',stress)]:
        feature=(a['features']-mean)/std;z0=forward_numpy(feature,initial);z=forward_numpy(feature,final)
        if prefix=='normal':initial_logits=z0
        np.testing.assert_allclose(z,scores[prefix+'/logits'],atol=2e-4,rtol=2e-5)
        probability=np.exp(-np.logaddexp(0.,-scores[prefix+'/logits'].astype(float)))
        confidence=np.where(a['rgb']<0,probability,1-probability)
        np.testing.assert_allclose(confidence,scores[prefix+'/confidence'],atol=1e-7,rtol=1e-6)
        assert np.isfinite(z0).all()
        replay[prefix]=dict(frames=len(feature),raw_logit_values=z.size,final_max_abs_error=float(np.abs(z-scores[prefix+'/logits']).max()),initial_min=float(z0.min()),initial_max=float(z0.max()),confidence_max_abs_error=float(np.abs(confidence-scores[prefix+'/confidence']).max()))
    loss=np.load(run/'training-loss.npy');assert loss.shape==(1200,3);np.testing.assert_array_equal(loss[:,0],np.arange(1,1201))
    exposure=normal['eligible'][lookup[data['batches']]].sum((1,2));np.testing.assert_array_equal(loss[:,2],exposure)
    assert np.isfinite(loss).all() and (loss[:,1]>=0).all() and (loss[exposure==0,1]==0).all()
    assert result['fit']['steps']==1200 and result['fit']['active_batches']==int((exposure>0).sum()) and result['fit']['eligible_query_exposures']==int(exposure.sum())
    ix=lookup[data['batches'][0]];yy=normal['target'][ix];mm=normal['eligible'][ix];zz=initial_logits[ix]
    first_loss=(np.logaddexp(0.,np.where(yy,-zz,zz))*np.where(yy,weights[0],weights[1])*mm).sum()/max(1,mm.sum())
    np.testing.assert_allclose(first_loss,loss[0,1],atol=1e-6,rtol=1e-5)
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
            actual=dict(baseline=scalar_metrics(base,truth),mz28=scalar_metrics(old,truth),candidate=scalar_metrics(candidate,truth),
                removed_fp=(remove&~truth).sum(0).tolist(),lost_tp=(remove&truth).sum(0).tolist(),new_fp=((candidate>=0)&(old<0)&~truth).sum(0).tolist(),
                remaining_baseline_fp=((candidate>=0)&(base>=0)&~truth).sum(0).tolist(),removed_baseline_fp=(remove&(base>=0)&~truth).sum(0).tolist(),
                added_tp=((candidate>=0)&(base<0)&truth).sum(0).tolist(),added_fp=((candidate>=0)&(base<0)&~truth).sum(0).tolist())
            assert actual==result['metrics'][key];changes[key]=actual;bits+=candidate.size
            if key in result['thin']:assert result['thin'][key]==scalar_metrics(candidate[100:125],truth[100:125])
    rows=read(root/'artifacts.local/work/mz15-shared-support-20260910/cache-v1/selected.json')
    for name in ['relation10000','distance5000']:
        rr=[r for r in rows if r['dataset']==name]
        for field,units in result['groups'][name].items():
            assert set(units)=={r[field] for r in rr}
            for unit,reported in units.items():
                ix=[i for i,r in enumerate(rr) if r[field]==unit];truth=saved[name+'/truth'][ix]
                assert reported==dict(frames=len(ix),mz28=scalar_metrics(saved[name+'/mz28'][ix],truth),candidate=scalar_metrics(saved[name+'/candidate'][ix],truth))
    removed=sum(sum(changes[n]['removed_baseline_fp']) for n in ['relation10000','distance5000'])
    gates=dict(placement_baseline_fp_removed=removed>=41,no_new_fp=all(sum(changes[n]['new_fp'])==0 for n in cohorts),
        baseline_tp_retained=all(sum(changes[n]['lost_tp'])==0 for n in cohorts),additions_preserved=all(np.array_equal(saved[n+'/candidate'][saved[n+'/baseline']<0],saved[n+'/mz28'][saved[n+'/baseline']<0]) for n in cohorts),
        pole_clean=sum(result['thin']['clean']['tp'][1::2])>=48,pole_stress=sum(result['thin']['stress']['tp'][1::2])>=48)
    gates['useful_effect']=all(gates.values());assert gates==result['gates'] and removed==result['placement_baseline_fp_removed'];assert bits==31200
    assert result['gate']==preparation['gate']
    audit=dict(status='PASS',code_sha256=sha(Path(__file__)),receipt_sha256=sha(run/'receipt.json'),preparation=preparation,
        training=dict(steps=1200,class_counts_rgb_tof=counts,class_weights_rgb_tof=weights,initial_seed_exact=True,normalization_exact=True,
            active_batches=int((exposure>0).sum()),eligible_query_exposures=int(exposure.sum()),first_loss_error=abs(float(first_loss)-float(loss[0,1]))),
        selector_replay=replay,calibration=calibration,task_bits=bits,gates=gates,placement_baseline_fp_removed=removed,
        dependencies={n:sha(Path(__file__).with_name(n)) for n in ['mz29_audit.py','mz11_audit.py','mz5_ensemble_readout.py','contact_retina_spec.py']},
        limits=['Preparation opportunity and TRAIN class sufficiency are separately verified from selector benefit.',
                'Frozen branch final layers replay on saved hidden features; visual backbone is not rerun.',
                'Selector forward uses independent NumPy float64; frozen saved float32 confidence defines audited cutoff decisions.',
                'TRAIN branch predictions are in-sample; consumed DEV calibration does not establish independent generalization.'])
    write(run/'audit.json',audit);print('PASS',bits,'task bits',gates,'placement baseline FP removed',removed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.root,a.run)
