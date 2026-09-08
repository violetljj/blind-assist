"""HEAD-X0: fixed zero-training counterfactual and within-model feature probes."""
import argparse
import json
import time
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from city_data import CityRGBDataset, _array
from PIL import Image
from city_dev_baseline import sha, read, write
from decoupled_model import DecoupledModel
from city_finetune_pilot import predict


def load(cache, split, tag):
    data=CityRGBDataset(cache,split)
    record=read(cache/('supervision' if split=='train' else 'evaluator')/(split+'.json'))
    assert record['sample_indices']==data.ids
    y=np.array(_array(cache.resolve(),record['near']),copy=True)
    masks=np.array(_array(cache.resolve(),record['support']),copy=True)
    if (cache/'evaluator/groups.json').exists():
        meta={r['sample_index']:r for r in read(cache/'evaluator/groups.json')}
    else:meta={}
    rows=[dict(domain=tag,index=i,sample_id=f'{tag}:{sid}',sample_index=sid,group_id=meta.get(sid,{}).get('group_id',f'{tag}:triplet{sid//3}'),family=meta.get(sid,{}).get('family','legacy_family_unavailable')) for i,sid in enumerate(data.ids)]
    return dict(data=data,y=y,masks=masks,rows=rows)


def diverse(pool,n):
    result=[];groups=set()
    for r in pool:
        if r['group_id'] not in groups:
            result.append(r);groups.add(r['group_id'])
            if len(result)==n:break
    assert len(result)==n
    return result


def sham(mask):
    """Nearest nonoverlapping translation of same binary shape, without wrap."""
    ys,xs=np.where(mask);assert len(ys)
    options=[]
    for dy in range(-int(ys.min()),18-int(ys.max())):
        for dx in range(-int(xs.min()),32-int(xs.max())):
            if not dx and not dy:continue
            overlap=int(mask[ys+dy,xs+dx].sum())
            options.append((overlap,dx*dx+dy*dy,dy,dx))
    _,_,dy,dx=min(options)
    other=np.zeros_like(mask);other[ys+dy,xs+dx]=True
    return other,dict(dx=dx,dy=dy,overlap_cells=int((mask&other).sum()),same_area=True,location_match='nearest admissible shape translation; not identical location')


@torch.inference_mode()
def forward(model,x):
    count=len(x);assert count<=32
    x=torch.cat([x,x[:1].expand(32-count,-1,-1,-1)],0) if count<32 else x
    deep,shallow=model.extract((x-model.image_mean)/model.image_std)
    projected=F.interpolate(model.deep_projection(deep),size=(18,32),mode='bilinear',align_corners=False)
    support=model.support(projected+model.detail(shallow))
    gated=projected[:,None]*support.sigmoid()[:,:,None]
    b,h,c,y,z=gated.shape
    pooled=model.near[0](gated.reshape(b*h,c,y,z)).reshape(b,h,-1)
    near=(pooled*model.near[2].weight[None]).sum(-1)+model.near[2].bias
    return near[:count],support[:count],deep[:count]


def modified(x,mask):
    mean=x.mean((1,2),keepdim=True).expand_as(x)
    blur=gaussian_blur(x,[17,17],[5.,5.])
    m=torch.from_numpy(np.repeat(np.repeat(mask,8,0),8,1)).to(x.device)[None]
    return {f'{op}_{name}':torch.where(m,x,replacement) if op=='retain' else torch.where(m,replacement,x) for name,replacement in [('mean',mean),('blur',blur)] for op in ['retain','erase']}


@torch.inference_mode()
def run(a):
    start=time.perf_counter();assert torch.cuda.is_available() and torch.__version__=='2.9.1+cu128'
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    backend=dict(deterministic=torch.are_deterministic_algorithms_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    write(a.output.parent/'recovery-backend.json',backend)
    out=a.output.resolve();assert not out.exists();out.mkdir(parents=True)
    checkpoints={'original':a.initial,'B':a.frozen_run/'B-step2000.pt','F':a.full_run/'F-step2000.pt'}
    inputfiles=list(checkpoints.values())+[a.full_run/'selection.json',a.full_run/'plaza-predictions.npz']
    for cache in [a.old_cache,a.new_cache,a.dev_cache]:inputfiles.append(cache/'manifest.json')
    for runpath in [a.full_run,a.frozen_run]:
        receipt=read(runpath/'receipt.json');assert sha(runpath/'result.json')==receipt['result_sha256']
        assert sha(checkpoints['F' if runpath==a.full_run else 'B'])==read(runpath/'result.json')['fit']['checkpoint_sha256']
    assert sha(a.initial)=='0c14179486102993508dd2385a3b6f9e17d51a6d53e657c0240325e5f78d5c7b'
    states={k:torch.load(p,map_location='cpu',weights_only=True) for k,p in checkpoints.items()}
    backbone_keys=[k for k in states['original'] if k.startswith('backbone.')]
    identity=all(torch.equal(states['original'][k],states['B'][k]) for k in backbone_keys)
    assert identity,'B backbone must equal original including buffers'
    domains={'old':load(a.old_cache,'train','old'),'new':load(a.new_cache,'train','new'),'DEV':load(a.dev_cache,'dev','DEV'),'plaza':load(a.old_cache,'test','plaza')}
    plaza=domains['plaza'];threshold=read(a.full_run/'selection.json')['thresholds'][1]
    with np.load(a.full_run/'plaza-predictions.npz') as z:
        assert np.array_equal(z['sample_indices'],plaza['data'].ids)
        cached=z['near'].copy();cachedsupport=z['support'].copy()
    pos=[r for r in plaza['rows'] if plaza['y'][r['index'],1]==1]
    fps=[r for r in plaza['rows'] if plaza['y'][r['index'],1]==0 and float(cached[r['index'],1])>=threshold]
    assert len(pos)==15 and len(fps)==171
    train=diverse([r for r in domains['old']['rows'] if domains['old']['y'][r['index'],1]==1],3)
    families=sorted(set(r['family'] for r in domains['new']['rows']));assert len(families)==4
    for family in families:train+=diverse([r for r in domains['new']['rows'] if r['family']==family and domains['new']['y'][r['index'],1]==1],3)
    dev=[]
    for family in families:dev+=diverse([r for r in domains['DEV']['rows'] if r['family']==family and domains['DEV']['y'][r['index'],1]==1],2)
    selected=pos+fps+train+dev;assert len(selected)==209 and len({r['sample_id'] for r in selected})==209
    k=max(1,int(np.median([int((domains[r['domain']]['masks'][r['index'],1]==1).sum()) for r in pos+train+dev])))
    roi=[];shams=[]
    for r in selected:
        d=domains[r['domain']];mask=d['masks'][r['index'],1]==1
        if not mask.any():
            assert r in fps
            mask=np.zeros((18,32),dtype=bool);scores=cachedsupport[r['index'],1].reshape(-1)
            mask.reshape(-1)[np.argsort(-scores,kind='stable')[:k]]=True
            r['ROI_kind']='F_prediction_conditioned_topK_not_GT'
        else:r['ROI_kind']='GT_visible_HEAD_support_not_whole_object'
        other,info=sham(mask);roi.append(mask);shams.append(other)
        r.update(roi_cells=int(mask.sum()),sham=info,near_GT=d['y'][r['index']].tolist(),unknown_cells=int((d['masks'][r['index'],1]==-1).sum()))
    pairs=[]
    for i,(tr,pl) in enumerate(zip(train,pos)):
        for donor,dest in [(tr,pl),(pl,tr)]:
            pairs.append(dict(pair=i,donor=donor['sample_id'],destination=dest['sample_id'],scope='Appearance only, unchanged donor pixel coordinates/scale; mixed scene, no inferred geometry label'))
    for d in domains.values():
        data=d['data'];inputfiles.append(data.root/data.entry['rgb']['path'])
        split={'old':'train','new':'train','DEV':'dev','plaza':'test'}[d['rows'][0]['domain']]
        recordfile=data.root/('supervision' if split=='train' else 'evaluator')/(split+'.json')
        rec=read(recordfile);inputfiles+=[recordfile,data.root/rec['near']['path'],data.root/rec['support']['path']]
        if (data.root/'evaluator/groups.json').exists():inputfiles.append(data.root/'evaluator/groups.json')
    inputfiles=list(dict.fromkeys(inputfiles))
    provenance=dict(input_sha256={str(p):sha(p) for p in inputfiles},source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},protocol_sha256=sha(Path(__file__).with_name('HEAD_X0_AUDIT_20260908.md')),original_B_backbone_tensor_identical=identity,backbone_tensor_count=len(backbone_keys),threshold=threshold,FP_ROI_cells=k,selected=selected,paste_pairs=pairs,scope='ZERO_TRAINING; no threshold selection; ROI visible support not object silhouette')
    write(out/'inputs.json',provenance)
    np.savez_compressed(out/'ROIs.npz',roi=np.array(roi),sham=np.array(shams),GT_support=np.array([domains[r['domain']]['masks'][r['index']] for r in selected]))
    # All references are TRAIN; both classes and same-group exclusions are explicit.
    refs=domains['old']['rows']+domains['new']['rows'];refy=np.array([domains[r['domain']]['y'][r['index'],1] for r in refs])
    write(out/'reference-identities.json',refs)
    cached_domains={}
    for tag,key in [('plaza','plaza'),('DEV','dev'),('old','old750'),('new','new384')]:
        with np.load(a.full_run/(key+'-predictions.npz')) as z:
            assert np.array_equal(z['sample_indices'],domains[tag]['data'].ids)
            cached_domains[tag]={'near':z['near'].copy(),'support':z['support'].copy()}
    preflight_model=DecoupledModel(a.pretrained).cuda().eval();preflight_model.load_state_dict(states['F'],strict=True)
    parity={}
    for tag,data in domains.items():
        pn,ps=predict(preflight_model,data['data'])
        parity[tag]=dict(near_max_abs=float(np.abs(pn-cached_domains[tag]['near']).max()),support_max_abs=float(np.abs(ps-cached_domains[tag]['support']).max()))
    write(out/'batch32-parity.json',dict(backend=backend,domains=parity,tolerance=1e-5))
    assert all(max(v.values())<=1e-5 for v in parity.values()),'F original full-domain batch32 cached drift'
    padded=[]
    for r in selected:
        x=domains[r['domain']]['data'][r['index']]['rgb'].cuda()[None]
        n,ss,_=forward(preflight_model,x)
        nn,sss=preflight_model(x.expand(32,-1,-1,-1).contiguous())
        assert torch.equal(n,nn[:1]) and torch.equal(ss,sss[:1]),'Padded32 custom/API mismatch'
        cachedrow=cached_domains[r['domain']]
        padded.append(dict(sample_id=r['sample_id'],near_max_abs=float(np.abs(n[0].sigmoid().cpu().numpy()-cachedrow['near'][r['index']]).max()),support_max_abs=float(np.abs(ss[0].sigmoid().cpu().numpy()-cachedrow['support'][r['index']]).max())))
    write(out/'padded32-preflight.json',dict(rows=padded,tolerance=1e-5,custom_API_exact=True,near_max_abs=max(r['near_max_abs'] for r in padded),support_max_abs=max(r['support_max_abs'] for r in padded)))
    assert all(max(r['near_max_abs'],r['support_max_abs'])<=1e-5 for r in padded),'Padded32 original/cache mismatch; stop before full dispatch'
    del preflight_model
    feature_rows=domains['plaza']['rows']+domains['DEV']['rows']+train
    feature_masks=[]
    for r in feature_rows:
        d=domains[r['domain']];mask=d['masks'][r['index'],1]==1
        if mask.any():kind='GT_visible_HEAD_support'
        else:
            peak=int(cached_domains[r['domain']]['support'][r['index'],1].argmax())
            mask=np.zeros((18,32),dtype=bool);mask.reshape(-1)[peak]=True;kind='F_prediction_conditioned_peak_cell_not_GT'
        feature_masks.append(mask)
    write(out/'feature-query-identities.json',[dict(r,feature_ROI_kind=('GT_visible_HEAD_support' if domains[r['domain']]['y'][r['index'],1]==1 else 'F_prediction_conditioned_peak_cell_not_GT')) for r in feature_rows])
    np.save(out/'feature-ROIs.npy',np.array(feature_masks),allow_pickle=False)
    outcomes=[];features={}
    for arm,state in states.items():
        model=DecoupledModel(a.pretrained).cuda().eval();model.load_state_dict(state,strict=True)
        original_features=[];base_logits={};base_prob={};predrows=[];logits=[];supports=[]
        for j,r in enumerate(selected):
            x=domains[r['domain']]['data'][r['index']]['rgb'].cuda()
            variants={'original':x}
            for kind,mask in [('roi',roi[j]),('sham',shams[j])]:
                variants.update({kind+'_'+n:v for n,v in modified(x,mask).items()})
            if arm=='F' and (j<3 or r in train or r in dev):
                folder=out/'previews';folder.mkdir(exist_ok=True)
                columns=[variants[name] for name in ['original','roi_retain_mean','roi_erase_mean','sham_erase_mean']]
                image=torch.cat(columns,dim=2).permute(1,2,0).cpu().numpy()
                Image.fromarray(np.rint(image*255).clip(0,255).astype(np.uint8)).save(folder/f'case{j:03d}.png')
            names=list(variants);xbatch=torch.stack(list(variants.values()));n,s,d=forward(model,xbatch)
            if j==0:
                nn,ss=model(torch.cat([xbatch,xbatch[:1].expand(32-len(xbatch),-1,-1,-1)],0));assert torch.equal(nn[:len(xbatch)],n) and torch.equal(ss[:len(xbatch)],s)
            original_features.append(d[0].cpu().numpy());base_logits[r['sample_id']]=n[0].cpu().numpy();base_prob[r['sample_id']]=n[0].sigmoid().cpu().numpy()
            for z,name in enumerate(names):
                predrows.append(dict(sample_id=r['sample_id'],variant=name,delta_HEAD_logit=float(n[z,1]-n[0,1]),delta_HEAD_probability=float(n[z,1].sigmoid()-n[0,1].sigmoid())))
            logits.extend(n.cpu().numpy());supports.extend(s.sigmoid().cpu().numpy())
        for pair in pairs:
            donor=next(i for i,r in enumerate(selected) if r['sample_id']==pair['donor']);dest=next(i for i,r in enumerate(selected) if r['sample_id']==pair['destination'])
            dr=selected[donor];rr=selected[dest]
            dx=domains[dr['domain']]['data'][dr['index']]['rgb'].cuda();rx=domains[rr['domain']]['data'][rr['index']]['rgb'].cuda()
            batch=[]
            for mask in [roi[donor],shams[donor]]:
                m=torch.from_numpy(np.repeat(np.repeat(mask,8,0),8,1)).cuda()[None];batch.append(torch.where(m,dx,rx))
            if arm=='F':
                folder=out/'paste-previews';folder.mkdir(exist_ok=True)
                image=torch.cat([rx,dx,*batch],dim=2).permute(1,2,0).cpu().numpy()
                Image.fromarray(np.rint(image*255).clip(0,255).astype(np.uint8)).save(folder/f'pair{pair["pair"]:02d}-{dr["domain"]}-to-{rr["domain"]}.png')
            n,s,_=forward(model,torch.stack(batch))
            for z,name in enumerate(['donor_roi','donor_sham']):
                predrows.append(dict(sample_id=rr['sample_id'],donor=dr['sample_id'],variant=name,pair=pair['pair'],delta_HEAD_logit=float(n[z,1])-float(base_logits[rr['sample_id']][1]),delta_HEAD_probability=float(n[z,1].sigmoid())-float(base_prob[rr['sample_id']][1])))
            logits.extend(n.cpu().numpy());supports.extend(s.sigmoid().cpu().numpy())
        if arm=='original':features['original_selected']=np.array(original_features)
        if arm=='B':assert np.array_equal(features['original_selected'],np.array(original_features))
        if arm=='F':
            drift=max(float(np.abs(base_prob[r['sample_id']]-cached_domains[r['domain']]['near'][r['index']]).max()) for r in selected)
            write(out/'cached-parity.json',dict(near_max_abs=drift,tolerance=1e-5,comparison='F original variants vs immutable F2000 cached near; padded32 original vs original cached batch'))
            assert drift<=1e-5,'F cached normal prediction drift'
        np.savez_compressed(out/f'{arm}-counterfactual.npz',logits=np.array(logits),near=torch.from_numpy(np.array(logits)).sigmoid().numpy(),support=np.array(supports))
        write(out/f'{arm}-rows.json',predrows)
        if arm!='B':
            allfeatures=[]
            for startidx in range(0,len(refs),32):
                batch=refs[startidx:startidx+32]
                x=torch.stack([domains[r['domain']]['data'][r['index']]['rgb'] for r in batch]).cuda()
                d,_=model.extract((x-model.image_mean)/model.image_std);allfeatures.extend(d.cpu().numpy())
            query_features=[]
            for startidx in range(0,len(feature_rows),32):
                batch=feature_rows[startidx:startidx+32]
                x=torch.stack([domains[r['domain']]['data'][r['index']]['rgb'] for r in batch]).cuda()
                d,_=model.extract((x-model.image_mean)/model.image_std);query_features.extend(d.cpu().numpy())
            q=torch.from_numpy(np.array(query_features)).cuda();ref=torch.from_numpy(np.array(allfeatures)).cuda()
            # Bilinear-upsample feature then masked pool, expressed as an exact
            # linear interpolation basis to avoid allocating >1GB repeated maps.
            feature_h,feature_w=q.shape[-2:]
            basis=F.interpolate(torch.eye(feature_h*feature_w,device='cuda').reshape(feature_h*feature_w,1,feature_h,feature_w),size=(18,32),mode='bilinear',align_corners=False)[:,0]
            np.savez_compressed(out/f'{arm}-features.npz',selected=q.cpu().numpy(),reference=ref.cpu().numpy(),reference_HEAD=refy)
            fr=[]
            for i,r in enumerate(feature_rows):
                allowed=np.array([r['group_id']!=rr['group_id'] for rr in refs])
                mask=torch.from_numpy(feature_masks[i].astype(np.float32)).cuda()
                mask=(basis*mask).sum((1,2)).reshape(feature_h,feature_w)/mask.sum()
                for mode,weights in [('whole',torch.ones_like(mask)),('query_ROI_same_coordinates',mask)]:
                    qv=(q[i]*weights).sum((1,2))/weights.sum();rv=(ref*weights).sum((2,3))/weights.sum()
                    cos=F.cosine_similarity(qv[None],rv).cpu().numpy();classes={}
                    for label in [0,1]:
                        inds=np.flatnonzero(allowed&(refy==label));order=inds[np.argsort(-cos[inds],kind='stable')]
                        classes[str(label)]=dict(count=len(inds),mean=float(cos[inds].mean()),max=float(cos[order[0]]),top5=[refs[t]['sample_id'] for t in order[:5]])
                    fr.append(dict(sample_id=r['sample_id'],mode=mode,reference=classes,positive_minus_negative_mean=classes['1']['mean']-classes['0']['mean'],positive_minus_negative_max=classes['1']['max']-classes['0']['max']))
            write(out/f'{arm}-similarity.json',fr)
        outcomes.append(dict(arm=arm,rows=len(predrows),seconds=time.perf_counter()-start))
        del model
        torch.cuda.empty_cache()
    for p,h in provenance['input_sha256'].items():assert sha(p)==h
    write(out/'receipt.json',dict(status='PASS',backend_flags=backend,training_steps=0,models=list(states),selected=209,paste_variants=60,rows_per_model=1941,feature_models=['original','F'],feature_queries=len(feature_rows),reference_count=len(refs),feature_scope='Raw deep5x8 receptive fields include context; bilinear18x32 support weighted pooling; same query coordinate ROI applied to both positive and negative references',original_B_backbone_identical=identity,device=torch.cuda.get_device_name(),torch_version=torch.__version__,seconds=time.perf_counter()-start,outcomes=outcomes))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['initial','frozen-run','full-run','pretrained','old-cache','new-cache','dev-cache','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    try:run(args)
    except BaseException as exc:
        if args.output.exists():write(args.output/'failure.json',dict(status='FAIL',error=str(exc),training_steps=0))
        raise
