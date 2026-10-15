"""Full-data official PLSP training adapter; no test-set/GT access.

Preserves official architecture, losses, trainable CLIP, LR and 40-epoch recipe.
Differences: portable paths, explicit trusted official pickle loading, fixed-final
checkpoint, correct accumulation tail, persistent exact epoch permutation/resume.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.util, json, math, os, random, sys, time, uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

def atomic_json(path,value):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8');p.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def set_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

def load_dataset_module(vendor,out):
    src=vendor/'codes/dataset/data.py';s=src.read_text()
    # Dataset assets are official author's pickle dictionaries, explicitly scoped.
    s=s.replace('torch.load(exo_obj_file)','torch.load(exo_obj_file, map_location="cpu", weights_only=False)')
    s=s.replace('torch.load(ego_obj_file)','torch.load(ego_obj_file, map_location="cpu", weights_only=False)')
    for name in ['Noun','Verb','Part']:
        old=f'torch.load(os.path.join(data_dir, "sentenceFeat{name}AGD.pth"))'
        new=f'torch.load(os.path.join(data_dir, "sentenceFeat{name}AGD.pth"), map_location="cpu", weights_only=False)'
        s=s.replace(old,new)
    s=s.replace('input_p = self.input_paths[idx]','input_p = self.input_paths[idx].replace(chr(92), "/")')
    line="            exo_p = os.path.join('/'.join(input_p.split('/')[:-1]).replace('egocentric', 'exocentric'), exo_p)"
    assert line in s;s=s.replace(line,line+'.replace(chr(92), "/")')
    # No test dataset class is instantiated by this runner.
    out.mkdir(parents=True,exist_ok=True);p=out/'data_compat.py';p.write_text(s,encoding='utf8')
    atomic_json(out/'adapter_receipt.json',{'original_sha256':sha(src),'adapted_sha256':sha(p),'changes':['Windows path normalization','explicit cpu trusted official asset torch.load'],'test_class_used':False})
    spec=importlib.util.spec_from_file_location('plsp_data_compat',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def make_dataset(args,out):
    d=load_dataset_module(args.vendor,out/'compat')
    root=args.data.resolve().as_posix()
    ds=d.DatasetAGD_train(root,'Seen',224,f'{root}/Seen/trainset/det_wholeobj_exo.pth',f'{root}/Seen/trainset/det_wholeobj_ego.pth',num_exo=1,PL_mode='refined',aug4imgRatio=0.5)
    if len(ds)!=6929: raise RuntimeError(f'Full Seen training required: expected6929 rows, received{len(ds)}')
    return ds,d.collate_fn_train

def main(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();cpu=time.process_time();deadline=started+args.wall_seconds
    torch.set_num_threads(args.cpu_threads)
    import cv2
    cv2.setNumThreads(args.cpu_threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    sys.path.insert(0,str(args.vendor.resolve()/'codes'))
    from models.full_model import ModelAGDsup
    from models.metric import KL_loss
    config=yaml.safe_load((args.vendor/'codes/configs/seen.yaml').read_text())
    ds,collate=make_dataset(args,out)
    if args.audit_only:
        samples=[ds[i] for i in [0,len(ds)//2,len(ds)-1]]
        batch=collate(samples)
        atomic_json(out/'cpu_dataset_audit.json',{'status':'PASS','full_rows':len(ds),'sample_indices':[0,len(ds)//2,len(ds)-1],'batch_shapes':{k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)},'no_test_GT':True})
        return
    if not torch.cuda.is_available(): raise RuntimeError('GPU backend required for full train; no silent CPU fallback')
    torch.cuda.set_device(args.device);torch.cuda.reset_peak_memory_stats()
    model=ModelAGDsup(**config['model'])
    clip_state=torch.jit.load(str(args.clip),map_location='cpu').float().state_dict()
    visual={k.split('visual.')[1]:v for k,v in clip_state.items() if k.startswith('visual.')}
    result=model.encoder.load_state_dict(visual,strict=True);del clip_state,visual
    model=model.cuda();model.train()
    enc_ids={id(p) for p in model.encoder.parameters()}
    optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if id(p) not in enc_ids and p.requires_grad],'lr':1e-4},{'params':[p for p in model.encoder.parameters() if p.requires_grad],'lr':1e-5}],betas=(.9,.95),weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5000000,gamma=.1)
    source_hashes={p.relative_to(args.vendor).as_posix():sha(p) for p in (args.vendor/'codes').rglob('*.py')}
    metadata={'official_commit':'71a12b2cc941c8eda85176b98c220c2cb3d55f14','official_source_sha256':source_hashes,'rows':len(ds),'data_order_sha256':hashlib.sha256('\n'.join(ds.input_paths).encode()).hexdigest(),'epochs':40,'effective_batch':20,'microbatch':args.microbatch,'seed':args.seed,'precision':'fp32','trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'trainable_encoder_parameters':sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),'clip_sha256':sha(args.clip),'original_config':config,'no_test_GT':True,'checkpoint_selection':'fixed epoch40 final; no evaluation access','differences':['portable dataset paths','trusted official torch.load cpu','correct accumulation-tail normalization and flush','no unused frozen encoder allocation','sequential microbatches matching effectivebatch20','fixedfinal instead of best-test-KLD'],'device':torch.cuda.get_device_name(args.device)}
    atomic_json(out/'configuration.json',metadata)
    epoch=0;next_group=0;steps=0;history=[];resume_wall_s=0;losses=[]
    ckpt=out/'last.pt'
    if args.resume:
        state=torch.load(ckpt,map_location='cpu',weights_only=False)
        for k in ['seed','rows','epochs','effective_batch','microbatch','clip_sha256','official_source_sha256','data_order_sha256']:
            if state['metadata'][k]!=metadata[k]:raise RuntimeError(f'Resume configuration mismatch {k}')
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        epoch=state['epoch'];next_group=state['next_group'];steps=state['steps'];history=state['history'];resume_wall_s=state['training_process_wall_s'];losses=state['epoch_losses'];set_rng(state['rng']);del state
    last_save=time.monotonic();step_durations=[];status='RUNNING'
    def checkpoint(ep,group):
        nonlocal last_save
        torch.cuda.synchronize();temp=out/'last.tmp'
        torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':ep,'next_group':group,'steps':steps,'history':history,'rng':rng_state(),'metadata':metadata,'epoch_losses':losses,'training_process_wall_s':resume_wall_s+time.monotonic()-started},temp)
        temp.replace(ckpt);last_save=time.monotonic()
    try:
        while epoch<40:
            perm=np.random.default_rng(args.seed+epoch).permutation(len(ds)).tolist()
            groups=[perm[i:i+20] for i in range(0,len(ds),20)]
            while next_group<len(groups):
                if time.monotonic()>deadline or (args.max_steps and steps>=args.max_steps):
                    status='LIMIT_REACHED';checkpoint(epoch,next_group);return
                ids=groups[next_group];t=time.monotonic();optimizer.zero_grad(set_to_none=True);parts=[]
                for begin in range(0,len(ids),args.microbatch):
                    mids=ids[begin:begin+args.microbatch]
                    batch=collate([ds[i] for i in mids])
                    for k,v in batch.items():
                        if isinstance(v,torch.Tensor):batch[k]=v.cuda()
                    aff,sim,exo,pnoun,ppart=model(batch['input_image'],batch['sent_feats'],batch['exo_image'],batch['exo_objbox_mask_patch'],num_exo=1)
                    pred=F.interpolate(aff,size=batch['gt_mask'].shape[-2:],mode='bilinear')
                    kl=KL_loss(pred,batch['gt_mask_prob'],batch['valid_input'])
                    ns=(1-F.cosine_similarity(pnoun,batch['noun_feats'],dim=2)).mean()
                    ps=(1-F.cosine_similarity(ppart,batch['part_feats'],dim=2)).mean()
                    ce=F.cross_entropy(exo,batch['vids'].long())
                    total=kl+10*sim.mean()+10*ce+ns+.1*ps
                    if not torch.isfinite(total):raise RuntimeError('Nonfinite full-training loss')
                    (total*(len(mids)/len(ids))).backward();parts.append(float(total.detach())*len(mids)/len(ids))
                    del batch,aff,sim,exo,pnoun,ppart,pred,total,kl,ns,ps,ce
                optimizer.step();torch.cuda.synchronize();steps+=1;next_group+=1
                duration=time.monotonic()-t;step_durations.append(duration);losses.append(sum(parts))
                progress={'status':'RUNNING','epoch':epoch,'next_group':next_group,'groups_per_epoch':len(groups),'steps':steps,'loss':sum(parts),'step_wall_s':duration,'elapsed_s':time.monotonic()-started,'cpu_s':time.process_time()-cpu,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'remaining_estimate_s':float(np.median(step_durations[-20:]))*((40-epoch)*len(groups)-next_group)}
                if steps<=3 or steps%10==0:atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
                if time.monotonic()-last_save>=args.checkpoint_seconds:checkpoint(epoch,next_group)
            scheduler.step();history.append({'epoch':epoch,'loss_mean':float(np.mean(losses)),'steps':steps});losses=[]
            epoch+=1;next_group=0;checkpoint(epoch,next_group)
        torch.save({'state_dict':model.state_dict(),'metadata':metadata},out/'final_epoch40.pt');status='COMPLETE'
    except BaseException as error:
        status='FAILED';atomic_json(out/'error.json',{'type':type(error).__name__,'message':str(error),'epoch':epoch,'next_group':next_group,'steps':steps});raise
    finally:
        atomic_json(out/'terminal.json',{'status':status,'epochs_completed':epoch,'next_group':next_group,'steps':steps,'wall_s':time.monotonic()-started,'cpu_s':time.process_time()-cpu,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'peak_cuda_reserved':torch.cuda.max_memory_reserved(),'no_test_GT':True,'last_checkpoint_exists':ckpt.exists()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--vendor',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--clip',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=10000);p.add_argument('--microbatch',type=int,default=10);p.add_argument('--device',type=int,default=0);p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--wall-seconds',type=int,default=21600);p.add_argument('--checkpoint-seconds',type=int,default=600);p.add_argument('--max-steps',type=int,default=0);p.add_argument('--resume',action='store_true');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if not 1<=a.microbatch<=20:raise ValueError('microbatch must be1..20')
    a.out.mkdir(parents=True,exist_ok=True)
    lock=a.out/'owner.lock';identity=str(uuid.uuid4());entered=time.time()
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Job already owned; inspect {lock} and owner PID before recovering a stale lock')
    with os.fdopen(fd,'w',encoding='utf8') as f:
        json.dump({'pid':os.getpid(),'identity':identity,'started_unix':entered,'argv':sys.argv},f)
    try:
        main(a)
    except BaseException as error:
        terminal=a.out/'terminal.json'
        if not terminal.exists() or terminal.stat().st_mtime<entered:
            atomic_json(terminal,{'status':'FAILED','stage':'preflight_or_initialization','error_type':type(error).__name__,'error':str(error),'pid':os.getpid(),'no_test_GT':True})
        raise
    finally:
        if lock.exists() and json.loads(lock.read_text())['identity']==identity:lock.unlink()
