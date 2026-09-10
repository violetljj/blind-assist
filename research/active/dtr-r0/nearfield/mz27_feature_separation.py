"""Frozen, label-blind packet matching and feature-neighbor diagnostic."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mz5_ensemble_readout import read,write,sha,load_npz
from mz16_detail_readout import make_model
from mz25_fixed_sampler import StableAngularAvailability


def packet(r,v):
    good=v&np.isfinite(r)&(r>0)&(r<=4)
    return np.concatenate([np.where(good,r,0)/4,good.astype(np.float32)],-1).astype(np.float32)


def summary(rows):
    names=['packet4','raw64','raw3x3_576','learned40'];out={}
    for name in names:
        per={str(label):[r['retrieval'][name]['prediction']==label for r in rows if r['known']==bool(label)] for label in [0,1]}
        agreement={k:float(np.mean(v)) if v else None for k,v in per.items()}
        out[name]=dict(anchors=len(rows),label_counts={k:len(v) for k,v in per.items()},agreement=agreement,
            balanced_agreement=float(np.mean(list(agreement.values()))) if all(v is not None for v in agreement.values()) else None)
    paired={}
    for name in ['raw64','raw3x3_576','learned40']:
        for control in (['packet4','learned40'] if name!='learned40' else ['packet4']):
            a=np.array([r['retrieval'][name]['prediction']==r['known'] for r in rows],bool)
            b=np.array([r['retrieval'][control]['prediction']==r['known'] for r in rows],bool)
            paired[name+'_vs_'+control]=dict(n=len(rows),improved=int((a&~b).sum()),worsened=int((~a&b).sum()),both_correct=int((a&b).sum()),both_wrong=int((~a&~b).sum()))
    return dict(descriptors=out,paired=paired)


def main(root,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();work=root/'artifacts.local/work';inputs={}
    def bind(path,expected=None):
        h=sha(path);assert expected is None or h==expected,str(path);inputs[str(path)]=h
    cache=work/'mz16-visual-detail-20260910/cache-v2';oldpath=work/'mz8-attribution-20260910/cache-v5'
    newpath=work/'mz15-shared-support-20260910/cache-v1';labelpath=work/'mz9-source-supervision-20260910/labels-v1'
    run=work/'mz26-deterministic-convergence-20260910/run-v1';extpath=run.parent/'extrema-v1'
    rank=work/'mz20-rank-objective-20260910/run-v1';prior=work/'mz15-shared-support-20260910/run-v1'
    normpath=work/'mz9-source-supervision-20260910/run-v1/normalization.npz'
    indexpath=work/'body-query-5000-20260909/dataset-v1/index.json'
    for folder,names in [(cache,['dense_HIGH_DETAIL.npy','ids.npy','selected.json']),
                         (oldpath,['observations.npz']),(newpath,['observations.npz','evaluator.npz','selected.json'])]:
        receipt=read(folder/'receipt.json');assert receipt['status']=='PASS';bind(folder/'receipt.json')
        for name in names:bind(folder/name,receipt['files'][name])
    for folder,names in [(run,['availability-step4800.pt','step4800-predictions.npz','batches.npy','start.json']),
                         (extpath,['extrema.npz','result.json']),
                         (rank,['BODY_RANK.pt','BODY_RANK-cutoff.npy','predictions.npz','batches.npy']),
                         (prior,['QUERY-initial.pt'])]:
        receipt=read(folder/'receipt.json');assert receipt['status']=='PASS';bind(folder/'receipt.json')
        for name in names:bind(folder/name,receipt['outputs'][name])
    bind(labelpath/'receipt.json');bind(labelpath/'evaluator.npz',read(labelpath/'receipt.json')['labels_sha256'])
    bind(normpath,read(normpath.parent/'receipt.json')['outputs']['normalization.npz']);bind(normpath.parent/'receipt.json')
    bind(indexpath,read(work/'mz1-tiny-fusion-20260910/cache-v1/features-receipt.json')['source_index_sha256'])
    bind(work/'mz1-tiny-fusion-20260910/cache-v1/features-receipt.json')
    old,new=load_npz(oldpath/'observations.npz'),load_npz(newpath/'observations.npz')
    ol,nl=load_npz(labelpath/'evaluator.npz'),load_npz(newpath/'evaluator.npz')
    known=np.concatenate([ol['cell_known_counts']>0,nl['known']]);query=np.concatenate([ol['query_counts']>0,nl['query_presence']]);del ol,nl
    ranges=np.concatenate([old['ranges'],new['ranges']]);valid=np.concatenate([old['valid'],new['valid']]);packets=packet(ranges,valid)
    ex=load_npz(extpath/'extrema.npz');lookup_ex={int(g):i for i,g in enumerate(ex['global_ids'])}
    saved=load_npz(run/'step4800-predictions.npz');ref=load_npz(rank/'predictions.npz');cut=np.load(rank/'BODY_RANK-cutoff.npy')
    train=np.unique(np.load(rank/'batches.npy'));np.testing.assert_array_equal(train,ex['ids/TRAIN_UNIQUE']);assert len(train)==7562
    cacheids=np.load(cache/'ids.npy');lookup=np.full(14200,-1,int);lookup[cacheids]=np.arange(len(cacheids))
    rows=read(cache/'selected.json');metadata={r['global_id']:r for r in rows};sourceindex=read(indexpath)['frames'];newrows=read(newpath/'selected.json')
    old_identity_bindings=read(oldpath/'receipt.json')['inputs']
    identities={}
    for gid in cacheids:
        gid=int(gid);row=metadata[gid]
        if gid<3500:
            s=sourceindex[int(old['old_index'][gid])];assert s['rgb_sha256']==row['rgb_sha']
            assert old_identity_bindings[gid]['rgb_sha']==row['rgb_sha'] and s['native_sha256']==row['native_sha']==old_identity_bindings[gid]['native_sha']
            identity=dict(site=s.get('site_id'),group=s.get('group_id'),origin='source_index[observations.old_index]',source='old5000')
        elif gid>=3700:
            s=newrows[gid-3700];assert s['cache_index']==gid-3700 and s['rgb_sha']==row['rgb_sha']
            identity=dict(site=s.get('site'),group=s.get('group'),origin='placement cache_index/global_id',source=s['dataset'])
        else:identity=dict(site=None,group=None,origin='MZ6 no explicit site/group mapping',source='MZ6_sequence')
        identity['status']='KNOWN' if identity['site'] and identity['group'] else 'UNKNOWN'
        identities[gid]=identity
    selections=[];missing=[]
    placement={'relation10000':np.arange(11200,13200),'distance5000':np.arange(13200,14200)}
    canonical={'relation10000':'relationDEV','distance5000':'distanceDEV'}
    for name,ids in placement.items():
        for i,q in np.argwhere(saved[name+'/accepted']&~saved[name+'/truth']):
            selections.append(dict(kind='residual_false_addition',cohort=canonical[name],global_id=int(ids[i]),query=int(q)))
    assert len(selections)==3
    for cohort in ['TRAIN_UNIQUE','oldDEV','relationDEV','distanceDEV']:
        choices=[]
        for gid in ex['ids/'+cohort]:
            pos=lookup_ex[int(gid)]
            for q in np.flatnonzero(ex['negative_valid'][pos]):choices.append((float(ex['step4800/negative_max'][pos,q]),int(gid),int(q)))
        choices.sort(key=lambda t:(-t[0],t[1],t[2]))
        assert len(choices)>=8
        for score,gid,q in choices[:8]:selections.append(dict(kind='negative_tail',cohort=cohort,global_id=gid,query=q,saved_availability=score))
    far_count=0
    for name,ids in placement.items():
        prefix='BODY_RANK/'+name+'/'
        take=ref[prefix+'accepted']&ref[prefix+'truth'];take[:,[0,2]]=False
        for i,q in np.argwhere(take):
            gid=int(ids[i]);pos=lookup_ex[gid];far_count+=1
            s=dict(kind='far_true_witness',cohort=canonical[name],global_id=gid,query=int(q))
            if not ex['positive_valid'][pos,q]:missing.append(dict(s,status='NO_ELIGIBLE_ACTUAL_QUERY_WITNESS'))
            else:selections.append(dict(s,flat_location=int(ex['positive_teacher_index'][pos,q]),saved_availability=float(ex['step4800/positive_witness'][pos,q])))
    assert far_count==75
    protocol=Path(__file__).with_name('MZ27_FEATURE_SEPARATION_PROTOCOL_20260910.md')
    code={name:sha(Path(__file__).with_name(name)) for name in [Path(__file__).name,'mz25_fixed_sampler.py','mz23_availability.py','mz16_detail_readout.py']}
    write(output/'start.json',dict(status='STARTED',inputs=inputs,code_sha256=code,protocol_sha256=sha(protocol),training_steps=0,
        anchor_requests=selections,missing_witnesses=missing,reference_frames=len(train),reference_identity_unknown=sum(identities[int(i)]['status']=='UNKNOWN' for i in train),
        fixed=dict(packet_references=128,min_per_class=5,neighbors=5,raw_patch_offsets=[-1,0,1]),
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda),scope='Consumed Development; retrieval labels are native availability, never CLEAR.'))
    print('START',len(selections),'anchor requests',len(missing),'missing witnesses',flush=True)
    torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    frozen=make_model(torch.load(prior/'QUERY-initial.pt',map_location='cpu',weights_only=True))
    frozen.load_state_dict(torch.load(rank/'BODY_RANK.pt',map_location='cpu',weights_only=True));frozen.cuda().eval().requires_grad_(False)
    head=StableAngularAvailability(frozen.grid.cpu());head.load_state_dict(torch.load(run/'availability-step4800.pt',map_location='cpu',weights_only=True));head.cuda().eval().requires_grad_(False)
    norm=load_npz(normpath);maps=np.load(cache/'dense_HIGH_DETAIL.npy',mmap_mode='r');anchors={};features_saved={}
    def features(ii):return torch.from_numpy((np.array(maps[lookup[ii]])-norm['mean'])/norm['std']).cuda()
    try:
        # Locate all anchors once using frozen models; labels only choose diagnostic masks.
        unique=sorted({s['global_id'] for s in selections})
        with torch.inference_mode():
            for begin in range(0,len(unique),16):
                ii=np.array(unique[begin:begin+16]);x=features(ii);r=torch.from_numpy(ranges[ii].astype(np.float32)).cuda();v=torch.from_numpy(valid[ii]).cuda()
                o=frozen.inspect(x,r,v);av=head(x,r,v).cpu().numpy();e=o['eligible'].cpu().numpy();scores=o['candidate_logits'].cpu().numpy()
                for j,gid in enumerate(ii):
                    for s in [s for s in selections if s['global_id']==gid]:
                        q=s['query'];kind=s['kind']
                        if kind=='far_true_witness':
                            flat=s['flat_location'];mask=e[j,...,q]&query[gid,...,q]
                            assert flat==int(np.where(mask,scores[j,...,q],-np.inf).argmax())
                        else:
                            mask=e[j,...,q]&((scores[j,...,q].astype(float)>=cut[q]) if kind=='residual_false_addition' else ~known[gid,:,None,:])
                            assert mask.any();flat=int(np.where(mask,av[j,:,None,:],-np.inf).argmax())
                        z,echo,c=map(int,np.unravel_index(flat,(64,2,49)));value=float(av[j,z,c])
                        if 'saved_availability' in s:np.testing.assert_allclose(value,s['saved_availability'],atol=2e-5,rtol=1e-5)
                        assert bool(known[gid,z,c])==(kind=='far_true_witness')
                        key=(int(gid),z,c)
                        if key not in anchors:anchors[key]=dict(global_id=int(gid),zone=z,cell=c,known=bool(known[gid,z,c]),availability=value,identity=identities[int(gid)],memberships=[])
                        anchors[key]['memberships'].append(dict(s,echo=echo))
        print('LOCATED',len(anchors),'unique locations',flush=True)
        def descriptors(ii,z,c):
            gathered={k:[] for k in ['packet4','raw64','raw3x3_576','learned40']}
            for begin in range(0,len(ii),16):
                jj=np.array(ii[begin:begin+16]);x=features(jj);n=len(jj)
                with torch.inference_mode():
                    center=head.grid[z,c];offsets=torch.tensor([[dx,dy] for dy in [-1,0,1] for dx in [-1,0,1]],device='cuda')*(2/28)
                    grid=(center[None]+offsets)[None,None].expand(n,1,9,2)
                    patch=F.grid_sample(x,grid,align_corners=False,padding_mode='zeros').squeeze(2).permute(0,2,1)
                    local=head.local(x);zone=(local.flatten(2)@head.sampler.matrix[z*49:(z+1)*49].t()).permute(0,2,1)
                    p=torch.from_numpy(packets[jj,z]).cuda()
                    learned=torch.cat([zone[:,c],zone.mean(1),center[None].expand(n,-1),head.relative[c][None].expand(n,-1),p],-1)
                for name,tensor in [('packet4',p),('raw64',patch[:,4]),('raw3x3_576',patch.flatten(1)),('learned40',learned)]:gathered[name].append(tensor.cpu().numpy())
            return {k:np.concatenate(v) for k,v in gathered.items()}
        results=[];train_known=np.array([int(i) for i in train if identities[int(i)]['status']=='KNOWN'])
        for ai,anchor in enumerate(anchors.values()):
            gid,z,c=anchor['global_id'],anchor['zone'],anchor['cell'];identity=anchor['identity'];anchor['anchor_id']=ai
            if identity['status']=='UNKNOWN':anchor['status']='UNKNOWN_IDENTITY';results.append(anchor);continue
            eligible=np.array([int(i) for i in train_known if i!=gid and identities[int(i)]['site']!=identity['site'] and identities[int(i)]['group']!=identity['group']])
            distance=np.linalg.norm(packets[eligible,z].astype(float)-packets[gid,z],axis=1)
            order=np.lexsort((eligible,distance))[:128];refs=eligible[order];dist=distance[order]
            assert len(refs)==128
            # Reference IDs are finalized before accessing their evaluator labels.
            labels=known[refs,z,c];anchor.update(references=[dict(global_id=int(i),identity=identities[int(i)],known=bool(k),packet_distance=float(d)) for i,k,d in zip(refs,labels,dist)],
                reference_class_counts=[int((~labels).sum()),int(labels.sum())],eligible_reference_frames=len(eligible))
            if min(anchor['reference_class_counts'])<5:anchor['status']='INSUFFICIENT_CLASS_SUPPORT';results.append(anchor);continue
            ds=descriptors(np.r_[gid,refs],z,c);anchor['status']='SCORED';anchor['retrieval']={}
            with torch.inference_mode():reproduced=float(head.head(torch.from_numpy(ds['learned40'][:1]).cuda()).cpu().item())
            np.testing.assert_allclose(reproduced,anchor['availability'],atol=2e-5,rtol=1e-5)
            anchor['descriptor_head_abs_error']=abs(reproduced-anchor['availability'])
            for name,values in ds.items():
                distances=np.sqrt(np.mean((values[1:].astype(float)-values[0].astype(float))**2,axis=1));ranked=np.lexsort((refs,distances));nearest=ranked[:5]
                prediction=bool(labels[nearest].sum()>=3)
                anchor['retrieval'][name]=dict(prediction=prediction,agreement=prediction==anchor['known'],distances=distances.tolist(),
                    top5_reference_indices=nearest.tolist(),top5_labels=labels[nearest].tolist(),
                    closest5_by_class={str(k):np.flatnonzero(labels==bool(k))[np.argsort(distances[labels==bool(k)],kind='stable')[:5]].tolist() for k in [0,1]})
                features_saved[f'anchor{ai}/{name}']=values
            results.append(anchor)
            if ai%10==0:print('RETRIEVAL',ai+1,'/',len(anchors),flush=True)
        cohorts=sorted({m['cohort'] for r in results for m in r['memberships']});scored=[r for r in results if r['status']=='SCORED']
        slices={name:[r for r in scored if any(m['cohort']==name for m in r['memberships'])] for name in cohorts}
        slices.update({'kind/'+name:[r for r in scored if any(m['kind']==name for m in r['memberships'])] for name in ['residual_false_addition','negative_tail','far_true_witness']})
        coverage={name:{status:sum(r['status']==status and any(m['cohort']==name for m in r['memberships']) for r in results) for status in sorted({r['status'] for r in results})} for name in cohorts}
        result=dict(status='PASS',training_steps=0,anchor_requests=len(selections),unique_anchor_locations=len(anchors),collapsed_duplicates=len(selections)-len(anchors),coverage_by_cohort=coverage,
            mz20_far_addition_bags=far_count,missing_witnesses=missing,status_counts={s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})},
            anchors=results,summaries={name:summary(rr) for name,rr in slices.items()},inputs=inputs,code_sha256=code,protocol_sha256=sha(protocol),
            limits=['Known0 is unavailable native evidence, not occupancy/CLEAR.','Diagnostic distance retrieval has no fitted parameters or deployed cutoff.',
                    'Anchors/cells/queries are correlated; no independent-trial or significance claim.','Failure of these fixed descriptors/metrics cannot prove information absence.',
                    'Missing identity/class support is unscored, never silently replaced.'])
        np.savez_compressed(output/'descriptors.npz',**features_saved);write(output/'result.json',result)
        write(output/'receipt.json',dict(status='PASS',training_steps=0,backend='CUDA frozen model/descriptor sampling; NumPy CPU retrieval',device=torch.cuda.get_device_name(),
            seconds=time.perf_counter()-started,inputs=inputs,code_sha256=code,protocol_sha256=sha(protocol),
            outputs={name:sha(output/name) for name in ['start.json','result.json','descriptors.npz']}))
        print('PASS',result['status_counts'],'unique anchors',len(anchors),'missing witnesses',len(missing),flush=True)
    finally:
        frozen.cpu();head.cpu();del maps;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();preexisting=a.output.exists()
    try:main(a.root,a.output)
    except Exception as exc:
        if not preexisting and a.output.exists():write(a.output/'failure.json',dict(status='FAIL',training_steps=0,error_type=type(exc).__name__,error=str(exc),code_sha256=sha(Path(__file__))))
        raise
