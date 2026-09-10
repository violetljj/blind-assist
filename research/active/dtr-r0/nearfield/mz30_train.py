"""One bounded branch selector fit after the fixed MZ30 opportunity gate."""
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


def calibrate(confidence,eligible,truth):
    cut=[];detail=[]
    for q in range(4):
        values=confidence[eligible[:,q],q].astype(float)
        disable=float(np.nextafter(max(1.,float(values.max()) if len(values) else 1.),np.inf))
        candidates=np.unique(np.r_[.5,values[values>=.5],disable]);best=None
        for threshold in candidates:
            remove=eligible[:,q]&(confidence[:,q].astype(float)>=threshold)
            if (remove&truth[:,q]).any():continue
            removed=int((remove&~truth[:,q]).sum());key=(removed,-int(remove.sum()),float(threshold))
            if best is None or key>best[0]:best=(key,float(threshold))
        assert best is not None;cut.append(best[1]);detail.append(dict(candidates=len(candidates),removed_fp=best[0][0],removed_tp=0,threshold=best[1],disabled=best[1]>1))
    return np.array(cut),detail


def main(root,run):
    assert not (run/'train-start.json').exists();started=time.perf_counter();prep=read(run/'prepare-receipt.json');assert prep['status']=='PASS'
    for name,digest in prep['outputs'].items():assert sha(run/name)==digest,name
    gate=read(run/'gate.json');assert gate['gate']['passed'] and gate['gate']['placement_opportunity']>=41
    data=load_npz(run/'branches.npz');ids=data['global_ids'];lookup=np.full(14200,-1,int);lookup[ids]=np.arange(len(ids));train=data['train_ids'];batches=data['batches']
    ti=lookup[train];x=data['normal/features'];mean=x[ti].mean(0,keepdims=True);std=x[ti].std(0,keepdims=True).clip(.1)
    np.savez(run/'normalization.npz',mean=mean,std=std)
    eligible=data['normal/eligible'];target=data['normal/target'];counts=[int((eligible[ti]&target[ti]).sum()),int((eligible[ti]&~target[ti]).sum())]
    assert counts==gate['gate']['train_rgb_tof_correct'] and min(counts)>=20;weights=[sum(counts)/(2*n) for n in counts]
    torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);torch.manual_seed(123)
    model=BranchSelector().cuda();torch.save(model.cpu().state_dict(),run/'initial.pt');model.cuda();opt=torch.optim.Adam(model.parameters(),lr=.001)
    code={n:sha(Path(__file__).with_name(n)) for n in [Path(__file__).name,'mz30_select.py','mz30_prepare.py']}
    write(run/'train-start.json',dict(status='STARTED',seed=123,batches=1200,batch_size=16,train_unique=len(train),normalization_scope='264 frozen features across exact unique TRAIN frames; onehot4 unnormalized',
        class_counts_rgb_tof=counts,class_weights_rgb_tof=weights,prepare_receipt_sha256=sha(run/'prepare-receipt.json'),code_sha256=code,
        parameters=sum(p.numel() for p in model.parameters()),runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),deterministic=torch.are_deterministic_algorithms_enabled(),matmul_tf32=False,cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'])))
    xx=torch.from_numpy((x-mean)/std).cuda();yy=torch.from_numpy(target).cuda();mask=torch.from_numpy(eligible).cuda();losses=[];active=0;exposures=0;tick=time.perf_counter()
    try:
        for step,batch in enumerate(batches):
            ix=lookup[batch];z=model(xx[ix]);valid=mask[ix];target_batch=yy[ix].float();w=torch.where(yy[ix],weights[0],weights[1])
            loss=(F.binary_cross_entropy_with_logits(z,target_batch,reduction='none')*w*valid).sum()/valid.sum().clamp_min(1)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();n=int(valid.sum());active+=int(n>0);exposures+=n
            losses.append([step+1,float(loss.detach()),n])
            if step%200==0 or step==1199:print('FIT',losses[-1],flush=True)
        torch.cuda.synchronize();fitseconds=time.perf_counter()-tick;model.eval();torch.save(model.cpu().state_dict(),run/'selector.pt');model.cuda();np.save(run/'training-loss.npy',np.array(losses))
        scored={}
        with torch.inference_mode():
            for prefix in ['normal','stress']:
                feature=torch.from_numpy((data[prefix+'/features']-mean)/std).cuda();rgb=torch.from_numpy(data[prefix+'/rgb']).cuda();zz=model(feature)
                scored[prefix+'/logits']=zz.cpu().numpy();scored[prefix+'/confidence']=negative_branch_confidence(rgb,zz).cpu().numpy()
        np.savez_compressed(run/'selector-scores.npz',**scored)
        devix=lookup[data['ids/DEV']];cut,calibration=calibrate(scored['normal/confidence'][devix],eligible[devix],data['normal/truth'][devix]);np.save(run/'cutoff.npy',cut)
        work=root/'artifacts.local/work';previous_path=work/'mz28-packet-availability-20260910/run-v1/predictions.npz';previous=load_npz(previous_path)
        assert sha(previous_path)==read(work/'mz28-packet-availability-20260910/run-v1/receipt.json')['outputs']['predictions.npz']
        rows=read(work/'mz15-shared-support-20260910/cache-v1/selected.json');result=dict(metrics={},thin={},groups={},calibration=calibration,
            fit=dict(steps=1200,active_batches=active,eligible_query_exposures=exposures,seconds=fitseconds,parameters=sum(p.numel() for p in model.parameters())),gate=gate['gate']);arrays={}
        for name in ['DEV','clean','stress','relation10000','distance5000']:
            ii=data['ids/'+name];ix=np.arange(len(ii)) if name=='stress' else lookup[ii];prefix='stress' if name=='stress' else 'normal'
            mask_np=data[prefix+'/eligible'][ix];rgb_np=data[prefix+'/rgb'][ix];tof_np=data[prefix+'/tof'][ix];confidence=scored[prefix+'/confidence'][ix]
            remove=mask_np&(confidence.astype(float)>=cut);negative=np.where(rgb_np<0,rgb_np,tof_np);assert (negative[remove]<0).all()
            for wrong in ([False] if name=='DEV' else [False,True]):
                key=name+('_wrong' if wrong else '');base=previous[key+'/baseline'];truth=previous[key+'/truth'];oldscore=previous[key+'/candidate'];candidate=np.where(remove,negative,oldscore)
                assert not (remove&~(base>=0)).any();assert not ((candidate>=0)&(oldscore<0)).any()
                m=dict(baseline=metrics(base,truth),mz28=metrics(oldscore,truth),candidate=metrics(candidate,truth),
                    removed_fp=(remove&~truth).sum(0).tolist(),lost_tp=(remove&truth).sum(0).tolist(),new_fp=[0]*4,
                    remaining_baseline_fp=((candidate>=0)&(base>=0)&~truth).sum(0).tolist(),removed_baseline_fp=(remove&(base>=0)&~truth).sum(0).tolist(),
                    added_tp=((candidate>=0)&(base<0)&truth).sum(0).tolist(),added_fp=((candidate>=0)&(base<0)&~truth).sum(0).tolist())
                result['metrics'][key]=m
                if name in ['clean','stress']:result['thin'][key]=metrics(candidate[100:125],truth[100:125])
                for k,v in dict(global_ids=ii,baseline=base,truth=truth,mz28=oldscore,candidate=candidate,eligible=mask_np,remove=remove,confidence=confidence,
                    selector_logits=scored[prefix+'/logits'][ix],rgb=rgb_np,tof=tof_np,support=data[prefix+'/support'][ix]).items():arrays[key+'/'+k]=v
                print(key,'exact',m['candidate']['exact'],'removedFP',m['removed_fp'],'lostTP',m['lost_tp'],flush=True)
        for name in ['relation10000','distance5000']:
            rr=[r for r in rows if r['dataset']==name];result['groups'][name]={}
            for field in ['family','group','site']:
                units={}
                for i,r in enumerate(rr):units.setdefault(r[field],[]).append(i)
                result['groups'][name][field]={u:dict(frames=len(ix),mz28=metrics(arrays[name+'/mz28'][ix],arrays[name+'/truth'][ix]),candidate=metrics(arrays[name+'/candidate'][ix],arrays[name+'/truth'][ix])) for u,ix in units.items()}
        normal=['DEV','clean','stress','relation10000','distance5000'];removed=sum(sum(result['metrics'][n]['removed_baseline_fp']) for n in ['relation10000','distance5000'])
        gates=dict(placement_baseline_fp_removed=removed>=41,no_new_fp=all(not ((arrays[n+'/candidate']>=0)&(arrays[n+'/mz28']<0)).any() for n in normal),
            baseline_tp_retained=all(sum(result['metrics'][n]['lost_tp'])==0 for n in normal),
            additions_preserved=all(np.array_equal(arrays[n+'/candidate'][arrays[n+'/baseline']<0],arrays[n+'/mz28'][arrays[n+'/baseline']<0]) for n in normal),
            pole_clean=sum(result['thin']['clean']['tp'][1::2])>=48,pole_stress=sum(result['thin']['stress']['tp'][1::2])>=48)
        gates['useful_effect']=all(gates.values());result.update(gates=gates,placement_baseline_fp_removed=removed,cutoff=cut.tolist())
        np.savez_compressed(run/'predictions.npz',**arrays);write(run/'result.json',result)
        write(run/'receipt.json',dict(status='PASS',backend='CUDA branch selector fit and cached inference',training_steps=1200,seconds=time.perf_counter()-started,code_sha256=code,outputs={p.name:sha(p) for p in run.iterdir() if p.is_file()}))
        print('PASS',gates,'placement FP removed',removed,flush=True)
    finally:model.cpu();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.root,a.run)
