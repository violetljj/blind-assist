"""Fixed-final, complete official Seen PLSP evaluation; never used for selection."""
import argparse, hashlib, json, os, sys, time, uuid
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from full_plsp_train import load_dataset_module, sha, atomic_json

def require_complete(a):
    t=json.loads((a.train/'terminal.json').read_text())
    if t.get('status')!='COMPLETE' or t.get('epochs_completed')!=40:raise RuntimeError('Final evaluation requires completed epoch40 training')
    if (a.train/'owner.lock').exists():raise RuntimeError('Training owner still active')
    if not (a.train/'final_epoch40.pt').is_file():raise RuntimeError('Final epoch40 checkpoint absent')
    q=json.loads(a.training_queue_terminal.read_text())
    if q.get('status')!='SUCCESS':raise RuntimeError('Original GPU training queue must report SUCCESS first')

def groups_for(rows,consumed):
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(rows):
        for key in [('sha',r['sha256']),('source',Path(r['name']).stem)]:
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
    clusters=defaultdict(list)
    for i,r in enumerate(rows):clusters[find(i)].append(r)
    for rr in clusters.values():
        group=min(r['sha256'] for r in rr)
        exposed=any(Path(r['name']).stem in consumed for r in rr)
        for r in rr:r.update(source_group=group,stratum='consumed' if exposed else 'fresh_unconsumed')
    return rows

def summarize(rows):
    result={}
    for name in ['all','fresh_unconsumed','consumed','fresh_without_train_overlap']:
        subset=[r for r in rows if name=='all' or r['stratum']==name or (name=='fresh_without_train_overlap' and r['stratum']=='fresh_unconsumed' and not r.get('train_RGB_overlap',False))]
        valid=[r for r in subset if r['status']=='EVALUATED']
        groups=defaultdict(list)
        for r in valid:groups[r['source_group']].append(r)
        out={'rows':len(subset),'evaluated_rows':len(valid),'NOT_EVAL':len(subset)-len(valid),'independent_source_groups':len(groups),'CI_method':'2000 source-group bootstrap replicates; row-weighted mean, fixed seed10000'}
        rng=np.random.default_rng(10000)
        counts=np.array([len(rr) for rr in groups.values()])
        if len(counts):
            draws=rng.integers(0,len(counts),size=(2000,len(counts)))
            for metric in ['KLD','SIM','NSS']:
                sums=np.array([sum(r[metric] for r in rr) for rr in groups.values()])
                boot=sums[draws].sum(1)/counts[draws].sum(1)
                out[metric]={'mean':float(sums.sum()/counts.sum()),'CI95':np.quantile(boot,[.025,.975]).tolist()}
        result[name]=out
    return result

def main(a):
    require_complete(a)  # Before model loading, image decoding, GT access or CUDA initialization.
    started=time.monotonic();torch.set_num_threads(4)
    content=[json.loads(l) for l in (a.test_preparation/'test_content_manifest.jsonl').read_text().splitlines()]
    rgb=[r for r in content if '/egocentric/' in r['name']]
    assert len(rgb)==1710
    consumed=set(json.loads(a.consumed.read_text())['source_ids'])
    rows=groups_for(rgb,consumed)
    train_manifest=a.test_preparation.parent/'training_rgb_content_manifest.jsonl'
    train_hashes={json.loads(l)['sha256'] for l in train_manifest.read_text().splitlines()}
    overlap_groups={r['source_group'] for r in rows if r['sha256'] in train_hashes}
    for r in rows:r['train_RGB_overlap']=r['source_group'] in overlap_groups
    sys.path.insert(0,str(a.vendor.resolve()/'codes'))
    from models.full_model import ModelAGDsup
    from models.metric import KLD,SIM,NSS
    d=load_dataset_module(a.vendor,a.out/'compat')
    ds=d.DatasetAGD_test(a.data.resolve().as_posix(),'Seen',224,no_pad_gt=True)
    index={Path(p).resolve():i for i,p in enumerate(ds.input_paths)}
    state=torch.load(a.train/'final_epoch40.pt',map_location='cpu',weights_only=False)
    if state['metadata']['epochs']!=40 or state['metadata']['checkpoint_selection']!='fixed epoch40 final; no evaluation access':raise RuntimeError('Unexpected final checkpoint metadata')
    for name,digest in state['metadata']['official_source_sha256'].items():
        if sha(a.vendor/name)!=digest:raise RuntimeError('Official source changed since training: '+name)
    model=ModelAGDsup(**state['metadata']['original_config']['model'])
    model.load_state_dict(state['state_dict'],strict=True);del state
    torch.cuda.set_device(a.device);model.cuda().eval();torch.cuda.reset_peak_memory_stats()
    evidence={'checkpoint_sha256':sha(a.train/'final_epoch40.pt'),'consumed_manifest_sha256':sha(a.consumed),'script_sha256':sha(Path(__file__)),'official_sources':{p.relative_to(a.vendor).as_posix():sha(p) for p in (a.vendor/'codes').rglob('*.py')},'differences':['Windows path and trusted torch.load adapter','GT0 excluded as NOT_EVAL instead of upstream uniform replacement','group-bootstrap CI and exposure strata','fixedfinal epoch40, no test-based selection'],'protocol':'Official eval.py batch1 RGB224 GT224 bilinear and spatial softmax; original KLD/SIM/NSS functions','GT_first_decode_after_completed_training':True,'train_RGB_overlap_rows':sum(r['train_RGB_overlap'] for r in rows),'test_content_manifest_sha256':sha(a.test_preparation/'test_content_manifest.jsonl')}
    atomic_json(a.out/'protocol.json',evidence)
    logits={}
    with (a.out/'per_row.jsonl').open('w',encoding='utf8') as f,torch.inference_mode():
        for n,r in enumerate(rows):
            p=a.data.parent/r['name'];gt=p.as_posix().replace('/egocentric/','/GT/').replace('.jpg','.png')
            r.update(action=Path(r['name']).parts[-3],object=Path(r['name']).parts[-2],source_id=p.stem)
            if not Path(gt).is_file():r.update(status='NOT_EVAL',reason='MISSING_GT')
            else:
                raw=np.asarray(Image.open(gt))
                if raw.size==0 or not np.isfinite(raw).all() or raw.max()==0:r.update(status='NOT_EVAL',reason='GT0_OR_INVALID')
                else:
                    batch=d.collate_fn_test([ds[index[p.resolve()]]])
                    aff,_,_=model(batch['input_image'].cuda(),batch['sent_feats'].cuda())
                    pred=F.interpolate(aff,size=batch['gt_mask'].shape[-2:],mode='bilinear')
                    prob=F.softmax(pred.flatten(1),dim=1);target=batch['gt_mask_prob'].cuda().flatten(1)
                    metrics={'KLD':float(KLD(prob.clone(),target.clone())),'SIM':float(SIM(prob.clone(),target.clone())),'NSS':float(NSS(prob.clone(),target.clone()))}
                    if all(np.isfinite(v) for v in metrics.values()):r.update(status='EVALUATED',**metrics)
                    else:r.update(status='NOT_EVAL',reason='NONFINITE_METRIC',metric_values={k:v if np.isfinite(v) else None for k,v in metrics.items()})
                    logits[str(n)]=aff[0,0].cpu().numpy()
            f.write(json.dumps(r)+'\n');f.flush()
            if (n+1)%100==0:atomic_json(a.out/'progress.json',{'rows':n+1,'total':len(rows),'wall_s':time.monotonic()-started})
    np.savez_compressed(a.out/'native_logits.npz',**logits)
    report=summarize(rows);atomic_json(a.out/'metrics.json',report)
    for stratum in ['fresh_unconsumed','consumed']:
        (a.out/f'{stratum}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['stratum']==stratum),encoding='utf8')
    atomic_json(a.out/'terminal.json',{'status':'COMPLETE','rows':len(rows),'wall_s':time.monotonic()-started,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'fixed_final':True})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['vendor','data','train','test-preparation','consumed','training-queue-terminal','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',type=int,default=0);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    lock=a.out/'owner.lock';identity=str(uuid.uuid4());fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'identity':identity},f)
    try:main(a)
    except BaseException as error:
        atomic_json(a.out/'terminal.json',{'status':'FAILED','error':str(error),'type':type(error).__name__});raise
    finally:
        if lock.exists() and json.loads(lock.read_text())['identity']==identity:lock.unlink()
