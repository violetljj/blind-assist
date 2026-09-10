"""One matched fit with all TRAIN branch disagreements as supervision."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mz5_ensemble_readout import read,write,sha,load_npz
from mz8_train import metrics
from mz30_select import BranchSelector,negative_branch_confidence
from mz30_train import calibrate

def main(root,run):
    run.mkdir(parents=True,exist_ok=False);started=time.perf_counter();work=root/'artifacts.local/work'
    previous=work/'mz30-branch-responsibility-20260910/run-v1';coverage=work/'mz31-responsibility-coverage-20260910/run-v1';inputs={}
    def bind(path,expected=None):
        h=sha(path);assert expected is None or h==expected,str(path);inputs[str(path)]=h
    receipt=read(previous/'receipt.json');assert receipt['status']=='PASS';bind(previous/'receipt.json')
    for name in ['branches.npz','normalization.npz','initial.pt','predictions.npz','result.json']:
        bind(previous/name,receipt['outputs'][name])
    bind(previous/'audit.json');assert read(previous/'audit.json')['status']=='PASS'
    cr=read(coverage/'receipt.json');assert cr['status']=='PASS';bind(coverage/'receipt.json');bind(coverage/'result.json',cr['outputs']['result.json'])
    covered=read(coverage/'result.json');assert covered['coverage_pass']
    oldpath=work/'mz28-packet-availability-20260910/run-v1';bind(oldpath/'receipt.json')
    bind(oldpath/'predictions.npz',read(oldpath/'receipt.json')['outputs']['predictions.npz'])
    metadata=work/'mz15-shared-support-20260910/cache-v1';bind(metadata/'receipt.json');bind(metadata/'selected.json',read(metadata/'receipt.json')['files']['selected.json'])
    protocol=Path(__file__).with_name('MZ32_EXPANDED_RESPONSIBILITY_PROTOCOL_20260910.md');bind(protocol)
    code={n:sha(Path(__file__).with_name(n)) for n in ['mz32_train.py','mz30_select.py','mz30_train.py']}
    data=load_npz(previous/'branches.npz');normalization=load_npz(previous/'normalization.npz');mean=normalization['mean'];std=normalization['std']
    ids=data['global_ids'];lookup=np.full(14200,-1,int);lookup[ids]=np.arange(len(ids));train=data['train_ids'];batches=data['batches'];ti=lookup[train]
    assert len(train)==7562 and np.array_equal(train,np.unique(batches)) and batches.shape==(1200,16)
    x=data['normal/features'];np.testing.assert_array_equal(mean,x[ti].mean(0,keepdims=True));np.testing.assert_array_equal(std,x[ti].std(0,keepdims=True).clip(.1))
    supervision=(data['normal/rgb']>=0)!=(data['normal/tof']>=0);target=data['normal/target']
    counts=[int((supervision[ti]&target[ti]).sum()),int((supervision[ti]&~target[ti]).sum())];assert counts==[478,687]
    assert sum(counts)==covered['all_disagreement_examples'];weights=[sum(counts)/(2*n) for n in counts]
    np.savez_compressed(run/'supervision.npz',global_ids=train,mask=supervision[ti],target=target[ti],batches=batches)
    torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);torch.manual_seed(123)
    model=BranchSelector();initial=torch.load(previous/'initial.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items());torch.save(model.state_dict(),run/'initial.pt');model.cuda()
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    write(run/'start.json',dict(status='STARTED',inputs=inputs,code_sha256=code,seed=123,steps=1200,batch_size=16,parameters=sum(p.numel() for p in model.parameters()),
        class_counts_rgb_tof=counts,class_weights_rgb_tof=weights,initial_state_exact=True,normalization_exact=True,supervision='All original TRAIN RGB/ToF sign disagreements',runtime_mask='Unchanged MZ30 base-positive unsupported disagreement',
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),deterministic=torch.are_deterministic_algorithms_enabled(),matmul_tf32=False,cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'])))
    xx=torch.from_numpy((x-mean)/std).cuda();yy=torch.from_numpy(target).cuda();mask=torch.from_numpy(supervision).cuda();losses=[];active=0;exposures=0;tick=time.perf_counter()
    try:
        for step,batch in enumerate(batches):
            ix=lookup[batch];z=model(xx[ix]);valid=mask[ix];w=torch.where(yy[ix],weights[0],weights[1])
            loss=(F.binary_cross_entropy_with_logits(z,yy[ix].float(),reduction='none')*w*valid).sum()/valid.sum().clamp_min(1)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();n=int(valid.sum());active+=int(n>0);exposures+=n
            losses.append([step+1,float(loss.detach()),n])
            if step%400==0 or step==1199:print('FIT',losses[-1],flush=True)
        torch.cuda.synchronize();fitseconds=time.perf_counter()-tick;model.eval();torch.save(model.cpu().state_dict(),run/'selector.pt');model.cuda();np.save(run/'training-loss.npy',np.array(losses))
        scored={}
        with torch.inference_mode():
            for prefix in ['normal','stress']:
                feature=torch.from_numpy((data[prefix+'/features']-mean)/std).cuda();rgb=torch.from_numpy(data[prefix+'/rgb']).cuda();zz=model(feature)
                scored[prefix+'/logits']=zz.cpu().numpy();scored[prefix+'/confidence']=negative_branch_confidence(rgb,zz).cpu().numpy()
        np.savez_compressed(run/'selector-scores.npz',**scored)
        devix=lookup[data['ids/DEV']];cut,calibration=calibrate(scored['normal/confidence'][devix],data['normal/eligible'][devix],data['normal/truth'][devix]);np.save(run/'cutoff.npy',cut)
        p28=load_npz(oldpath/'predictions.npz');p30=load_npz(previous/'predictions.npz');rows=read(metadata/'selected.json')
        result=dict(metrics={},thin={},groups={},calibration=calibration,cutoff=cut.tolist(),fit=dict(steps=1200,active_batches=active,eligible_query_exposures=exposures,seconds=fitseconds,parameters=sum(p.numel() for p in model.parameters())),class_counts_rgb_tof=counts,class_weights_rgb_tof=weights);arrays={}
        for name in ['DEV','clean','stress','relation10000','distance5000']:
            ii=data['ids/'+name];ix=np.arange(len(ii)) if name=='stress' else lookup[ii];prefix='stress' if name=='stress' else 'normal'
            eligible=data[prefix+'/eligible'][ix];rgb=data[prefix+'/rgb'][ix];tof=data[prefix+'/tof'][ix];confidence=scored[prefix+'/confidence'][ix]
            remove=eligible&(confidence.astype(float)>=cut);negative=np.where(rgb<0,rgb,tof);assert (negative[remove]<0).all()
            for wrong in ([False] if name=='DEV' else [False,True]):
                key=name+('_wrong' if wrong else '');base=p28[key+'/baseline'];truth=p28[key+'/truth'];oldscore=p28[key+'/candidate'];score30=p30[key+'/candidate'];candidate=np.where(remove,negative,oldscore)
                assert not (remove&~(base>=0)).any();assert not ((candidate>=0)&(oldscore<0)).any()
                m=dict(baseline=metrics(base,truth),mz28=metrics(oldscore,truth),mz30=metrics(score30,truth),candidate=metrics(candidate,truth),
                    removed_fp=(remove&~truth).sum(0).tolist(),lost_tp=(remove&truth).sum(0).tolist(),
                    removed_baseline_fp=(remove&(base>=0)&~truth).sum(0).tolist(),remaining_baseline_fp=((candidate>=0)&(base>=0)&~truth).sum(0).tolist(),
                    added_tp=((candidate>=0)&(base<0)&truth).sum(0).tolist(),added_fp=((candidate>=0)&(base<0)&~truth).sum(0).tolist(),
                    versus_mz30=dict(tp_gained=((candidate>=0)&(score30<0)&truth).sum(0).tolist(),tp_lost=((candidate<0)&(score30>=0)&truth).sum(0).tolist(),fp_removed=((candidate<0)&(score30>=0)&~truth).sum(0).tolist(),fp_added=((candidate>=0)&(score30<0)&~truth).sum(0).tolist()))
                result['metrics'][key]=m
                if name in ['clean','stress']:result['thin'][key]=metrics(candidate[100:125],truth[100:125])
                for k,v in dict(global_ids=ii,baseline=base,truth=truth,mz28=oldscore,mz30=score30,candidate=candidate,eligible=eligible,remove=remove,confidence=confidence,
                    selector_logits=scored[prefix+'/logits'][ix],rgb=rgb,tof=tof,support=data[prefix+'/support'][ix]).items():arrays[key+'/'+k]=v
                print(key,'exact',m['candidate']['exact'],'FP',m['candidate']['fp'],'lostTP',m['lost_tp'],flush=True)
        for name in ['relation10000','distance5000']:
            rr=[r for r in rows if r['dataset']==name];result['groups'][name]={}
            for field in ['family','group','site']:
                units={}
                for i,r in enumerate(rr):units.setdefault(r[field],[]).append(i)
                result['groups'][name][field]={u:dict(frames=len(ix),mz30=metrics(arrays[name+'/mz30'][ix],arrays[name+'/truth'][ix]),candidate=metrics(arrays[name+'/candidate'][ix],arrays[name+'/truth'][ix])) for u,ix in units.items()}
        normal=['DEV','clean','stress','relation10000','distance5000'];totalfp=sum(sum(result['metrics'][n]['candidate']['fp']) for n in ['relation10000','distance5000'])
        gates=dict(placement_total_fp_le20=totalfp<=20,mz28_tp_retained=all(sum(result['metrics'][n]['lost_tp'])==0 for n in normal),
            no_new_fp=all(not ((arrays[n+'/candidate']>=0)&(arrays[n+'/mz28']<0)).any() for n in normal),
            additions_preserved=all(np.array_equal(arrays[n+'/candidate'][arrays[n+'/baseline']<0],arrays[n+'/mz28'][arrays[n+'/baseline']<0]) for n in normal),
            pole_clean=sum(result['thin']['clean']['tp'][1::2])>=48,pole_stress=sum(result['thin']['stress']['tp'][1::2])>=48)
        gates['decisive_improvement']=all(gates.values());result.update(gates=gates,placement_total_fp=totalfp)
        np.savez_compressed(run/'predictions.npz',**arrays);write(run/'result.json',result)
        write(run/'receipt.json',dict(status='PASS',backend='CUDA branch-selector fit and cached feature inference',training_steps=1200,seconds=time.perf_counter()-started,inputs=inputs,code_sha256=code,outputs={p.name:sha(p) for p in run.iterdir() if p.is_file()}))
        print('PASS',gates,flush=True)
    finally:model.cpu();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.root,a.run)
