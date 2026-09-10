"""Independent fixed TRAIN-bank membership, retrieval and task-output audit."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from mz5_ensemble_readout import read,write,sha,load_npz
from mz11_audit import scalar_metrics
from mz26_audit import audit_arm
from mz24_audit import geometry
from mz15_evaluator import align_stress_labels


def observable_packet(ranges,valid):
    good=valid&np.isfinite(ranges)&(ranges>0)&(ranges<=4)
    return np.concatenate([np.where(good,ranges,0)/4,good.astype(np.float32)],-1).astype(np.float32)


def main(root,run):
    assert not (run/'audit.json').exists()
    receipt,start,result=(read(run/name) for name in ['receipt.json','start.json','result.json'])
    assert receipt['status']=='PASS'
    for name,digest in receipt['outputs'].items():assert sha(run/name)==digest,name
    for path,digest in start['inputs'].items():assert sha(path)==digest,path
    for name,digest in start['code_sha256'].items():assert sha(Path(__file__).with_name(name))==digest,name
    protocols=list(Path(__file__).parent.glob('MZ28*PROTOCOL*.md'));assert len(protocols)==1
    assert sha(protocols[0])==start['protocol_sha256']
    work=root/'artifacts.local/work';rank=work/'mz20-rank-objective-20260910/run-v1'
    oldpath=work/'mz8-attribution-20260910/cache-v5';newpath=work/'mz15-shared-support-20260910/cache-v1'
    old,new=load_npz(oldpath/'observations.npz'),load_npz(newpath/'observations.npz');rows=read(newpath/'selected.json')
    ol=load_npz(work/'mz9-source-supervision-20260910/labels-v1/evaluator.npz');nl=load_npz(newpath/'evaluator.npz')
    known=np.concatenate([ol['cell_known_counts']>0,nl['known']]);source=np.concatenate([ol['source_counts']>0,nl['source_presence']])
    query=np.concatenate([ol['query_counts']>0,nl['query_presence']]);del ol,nl
    ranges=np.concatenate([old['ranges'],new['ranges']]);valid=np.concatenate([old['valid'],new['valid']]);packets=observable_packet(ranges,valid)
    ids=dict(DEV=np.flatnonzero(old['role']=='DEV_ONLY'),clean=np.arange(3500,3700),stress=np.arange(3500,3700),
        relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    bank=load_npz(run/'bank.npz');original_train=np.unique(np.load(rank/'batches.npy'))
    expected_ids=original_train[(original_train<3500)|(original_train>=3700)]
    np.testing.assert_array_equal(bank['ids'],expected_ids);assert len(expected_ids)==7362 and len(original_train)==7562
    assert bank['packet'].shape==(7362,64,4) and bank['packet'].dtype==np.float32
    np.testing.assert_array_equal(bank['packet'],packets[expected_ids]);np.testing.assert_array_equal(bank['known'],known[expected_ids])
    assert bank['known'].dtype==bool and bank['known'].shape==(7362,64,49)
    allowed=np.r_[np.flatnonzero(old['role']=='TRAIN_ONLY'),np.arange(3700,11200)]
    assert np.isin(expected_ids,allowed).all()
    assert all(not np.isin(expected_ids,v).any() for v in ids.values())
    # Explicit site/group identity is an audit of bank provenance, never an inference input.
    indexpath=work/'body-query-5000-20260909/dataset-v1/index.json'
    sourceindex=read(indexpath)['frames'];assert sha(indexpath)==read(work/'mz1-tiny-fusion-20260910/cache-v1/features-receipt.json')['source_index_sha256']
    oldbindings=read(oldpath/'receipt.json')['inputs']
    identity={}
    for gid in np.r_[np.arange(3500),np.arange(3700,14200)]:
        if gid<3500:
            row=sourceindex[int(old['old_index'][gid])]
            assert row['rgb_sha256']==oldbindings[gid]['rgb_sha'] and row['native_sha256']==oldbindings[gid]['native_sha']
            site,group=row['site_id'],row['group_id']
        else:
            row=rows[gid-3700];assert row['cache_index']==gid-3700;site,group=row['site'],row['group']
        assert site and group;identity[int(gid)]=(site,group)
    train_sites={identity[int(i)][0] for i in expected_ids};train_groups={identity[int(i)][1] for i in expected_ids}
    assert len(train_sites)==375
    assert start['reference_frames']==7362 and start['excluded_sequence_frames']==200 and start['bank_sites']==375
    assert start['training_steps']==result['training_steps']==0 and start['fixed']==dict(neighbors=5,minimum_votes=3)
    assert start['runtime']['deterministic'] and start['runtime']['distance_dtype']=='float64' and start['runtime']['packet_dtype']=='float32'
    assert start['bank_array_bytes']==sum(x.nbytes for x in bank.values())
    assert start['runtime_buffer_bytes']==bank['ids'].size*8+bank['packet'].size*8+bank['known'].size
    np.testing.assert_array_equal(read(run/'bank-identities.json'),[dict(site=identity[int(i)][0],group=identity[int(i)][1]) for i in expected_ids])
    disjoint={}
    for name in ['DEV','relation10000','distance5000']:
        sites={identity[int(i)][0] for i in ids[name]};groups={identity[int(i)][1] for i in ids[name]}
        assert not sites&train_sites and not groups&train_groups
        disjoint[name]=dict(frames=len(ids[name]),sites=len(sites),site_overlap=0,group_overlap=0)
        assert start['dev_site_overlap'][name]==0
    assert sha(run/'cutoff.npy')==sha(rank/'BODY_RANK-cutoff.npy');cut=np.load(run/'cutoff.npy')
    saved=load_npz(run/'predictions.npz');reference=load_npz(rank/'predictions.npz')
    mz26=load_npz(work/'mz26-deterministic-convergence-20260910/run-v1/step4800-predictions.npz')
    state=torch.load(rank/'BODY_RANK.pt',map_location='cpu',weights_only=True);torch.set_num_threads(1)
    extpath=work/'mz26-deterministic-convergence-20260910/extrema-v1';extr=read(extpath/'receipt.json');assert extr['status']=='PASS'
    assert sha(extpath/'extrema.npz')==extr['outputs']['extrema.npz'];ext=load_npz(extpath/'extrema.npz')
    extlookup=np.full(14200,-1,int);extlookup[ext['global_ids']]=np.arange(len(ext['global_ids']))
    witness_checks={}
    adapted=copy.deepcopy(result)
    for name,value in adapted['availability'].items():adapted['availability'][name]={k:v for k,v in value.items() if k not in ['vote_histogram','nearest_distance_quantiles']}
    task=audit_arm('MZ28',saved,adapted,reference,ids,known,source,query,ranges,valid,old,state['rays'],rows,cut)
    banklookup=np.full(14200,-1,int);banklookup[bank['ids']]=np.arange(len(bank['ids']));neighbor_bits=0;comparisons={}
    for name in result['metrics']:
        normal=name.removesuffix('_wrong');ii=ids[normal];get=lambda k:saved[name+'/'+k]
        np.testing.assert_array_equal(get('global_ids'),ii)
        np.testing.assert_array_equal(get('mz26'),mz26[name+'/candidate'])
        neighborids=get('neighbor_ids');distances=get('neighbor_distances');votes=get('votes')
        assert neighborids.shape==(len(ii),64,5) and distances.shape==neighborids.shape and votes.shape==(len(ii),64,49)
        assert np.isin(neighborids,bank['ids']).all() and (distances>=0).all() and np.isfinite(distances).all()
        assert ((np.diff(distances,axis=-1)>0)|((np.diff(distances,axis=-1)==0)&(np.diff(neighborids,axis=-1)>0))).all()
        np.testing.assert_array_equal(get('availability'),votes.astype(np.float32)-2.5)
        rr,vv=(old['stress_ranges'][ii-3500],old['stress_valid'][ii-3500]) if normal=='stress' else (ranges[ii],valid[ii])
        current=observable_packet(rr,vv)
        for begin in range(0,len(ii),64):
            end=min(begin+64,len(ii));which=banklookup[neighborids[begin:end]];zz=np.arange(64)[None,:,None]
            recounted=bank['known'][which,zz].sum(2)
            np.testing.assert_array_equal(recounted,votes[begin:end]);neighbor_bits+=recounted.size
            exact=np.sqrt(((bank['packet'][which,zz].astype(np.float64)-current[begin:end,:,None,:].astype(np.float64))**2).sum(-1))
            np.testing.assert_allclose(exact,distances[begin:end],rtol=1e-13,atol=1e-14)
        if name!=normal:
            for key in ['votes','neighbor_ids','neighbor_distances','availability']:np.testing.assert_array_equal(get(key),saved[normal+'/'+key])
        before=mz26[name+'/candidate'];after=get('candidate');truth=get('truth')
        if 'mz26' in result['metrics'][name]:assert result['metrics'][name]['mz26']==scalar_metrics(before,truth)
        comparisons[name]=dict(mz26=scalar_metrics(before,truth),candidate=scalar_metrics(after,truth),
            gained_tp=((after>=0)&(before<0)&truth).sum(0).tolist(),lost_tp=((after<0)&(before>=0)&truth).sum(0).tolist(),
            added_fp=((after>=0)&(before<0)&~truth).sum(0).tolist(),removed_fp=((after<0)&(before>=0)&~truth).sum(0).tolist())
        for reported,computed in [('mz26_tp_lost','lost_tp'),('mz26_tp_recovered','gained_tp'),('mz26_fp_added','added_fp'),('mz26_fp_removed','removed_fp')]:assert result['metrics'][name][reported]==comparisons[name][computed]
        if name==normal:
            assert result['availability'][name]['vote_histogram']==np.bincount(votes.flatten(),minlength=6).tolist()
            np.testing.assert_array_equal(result['availability'][name]['nearest_distance_quantiles'],np.quantile(distances[:,:,0],[0,.5,.9,.99,1]))
            if name!='stress':
                np.testing.assert_array_equal(get('teacher_witness_valid'),ext['positive_valid'][extlookup[ii]])
                present=get('teacher_witness_valid');positions=get('teacher_witness_index')
                np.testing.assert_array_equal(positions[present],ext['positive_teacher_index'][extlookup[ii]][present])
                zz,ee,cc=np.unravel_index(positions,(64,2,49));selected_votes=votes[np.arange(len(ii))[:,None],zz,cc].astype(np.int16)
                selected_votes[~present]=-1;np.testing.assert_array_equal(selected_votes,get('teacher_witness_votes'))
                for begin in range(0,len(ii),64):
                    end=min(begin+64,len(ii));positive=geometry(rr[begin:end],vv[begin:end],state['rays'])&query[ii[begin:end]]
                    np.testing.assert_array_equal(positive.sum((1,2,3)),get('actual_contributors')[begin:end])
                    np.testing.assert_array_equal((positive&(votes[begin:end,:,None,:,None]<3)).sum((1,2,3)),get('rejected_contributors')[begin:end])
                recounted=dict(valid=present.sum(0).tolist(),retained=((selected_votes>=3)&present).sum(0).tolist(),actual_contributors=get('actual_contributors').sum(0).tolist(),rejected_contributors=get('rejected_contributors').sum(0).tolist())
                assert result['witnesses'][name]==recounted;witness_checks[name]=recounted
    # Full bank search on only the predeclared 58 frames, independently in NumPy.
    selections={name:np.unique(np.argwhere(reference['BODY_RANK/'+name+'/accepted']&~reference['BODY_RANK/'+name+'/truth'])[:,0]) for name in ['relation10000','distance5000']}
    selections.update(clean=np.arange(100,125),stress=np.arange(100,125));assert sum(len(v) for v in selections.values())==58
    selected={}
    for name,frames in selections.items():
        ii=ids[name][frames];rr,vv=(old['stress_ranges'][ii-3500],old['stress_valid'][ii-3500]) if name=='stress' else (ranges[ii],valid[ii])
        current=observable_packet(rr,vv)
        for i,frame in enumerate(frames):
            for z in range(64):
                distance=np.sqrt(((bank['packet'][:,z].astype(float)-current[i,z].astype(float))**2).sum(-1))
                nearest=np.lexsort((bank['ids'],distance))[:5]
                np.testing.assert_array_equal(bank['ids'][nearest],saved[name+'/neighbor_ids'][frame,z])
                np.testing.assert_allclose(distance[nearest],saved[name+'/neighbor_distances'][frame,z],atol=1e-14,rtol=1e-13)
                np.testing.assert_array_equal(bank['known'][nearest,z].sum(0),saved[name+'/votes'][frame,z])
        selected[name]=dict(frames=len(frames),zones=len(frames)*64,bank_candidates_per_zone=7362)
    correction=run.parent/'correction-v1';cr=read(correction/'receipt.json');cv=read(correction/'result.json');assert cr['status']=='PASS'
    for name,digest in cr['outputs'].items():assert sha(correction/name)==digest,name
    for path,digest in cr['inputs'].items():assert sha(path)==digest,path
    assert cr['code_sha256']==sha(Path(__file__).with_name('mz28_stress_witness_correction.py'))
    assert cr['alignment_sha256']==sha(Path(__file__).with_name('mz15_evaluator.py'))
    corrected=load_npz(correction/'corrected.npz');ii=ids['stress'];np.testing.assert_array_equal(corrected['global_ids'],ii)
    _,qq,moved=align_stress_labels(source[ii],query[ii],ranges[ii],valid[ii],old['stress_ranges'],old['stress_valid'])
    np.testing.assert_array_equal(corrected['moved_echo_zones'],moved)
    ee=geometry(old['stress_ranges'],old['stress_valid'],state['rays']);positive=ee&qq
    np.testing.assert_array_equal(corrected['actual_contributors'],positive.sum((1,2,3)))
    np.testing.assert_array_equal(corrected['rejected_contributors'],(positive&(saved['stress/votes'][:,:,None,:,None]<3)).sum((1,2,3)))
    localpath=rank/'local_samples.npz';assert sha(localpath)==read(rank/'receipt.json')['outputs']['local_samples.npz'];local=load_npz(localpath)
    positions=np.zeros((200,4),int);present=positive.any((1,2,3))
    for q in range(4):
        prefix=f'BODY_RANK/stress/{q}/';scores=local[prefix+'score'];labels_q=local[prefix+'label'];frames=local[prefix+'frame']
        for frame in range(200):
            mask=ee[frame,...,q]&known[ii[frame],:,None,:]&old['stress_valid'][frame,...,None]
            coords=np.argwhere(mask);take=frames==frame;yy=qq[frame,...,q][mask]
            np.testing.assert_array_equal(labels_q[take],yy);assert len(scores[take])==len(coords)
            if present[frame,q]:
                winner=coords[np.where(yy,scores[take],-np.inf).argmax()]
                positions[frame,q]=np.ravel_multi_index(tuple(winner),(64,2,49))
    np.testing.assert_array_equal(corrected['teacher_witness_index'],positions)
    np.testing.assert_array_equal(corrected['teacher_witness_valid'],present)
    zz,unused,cc=np.unravel_index(positions,(64,2,49));votes=saved['stress/votes'][np.arange(200)[:,None],zz,cc].astype(np.int16);votes[~present]=-1
    np.testing.assert_array_equal(corrected['teacher_witness_votes'],votes)
    stress_summary=dict(valid=present.sum(0).tolist(),retained=((votes>=3)&present).sum(0).tolist(),actual_contributors=corrected['actual_contributors'].sum(0).tolist(),rejected_contributors=corrected['rejected_contributors'].sum(0).tolist())
    assert cv['corrected_stress_witnesses']==stress_summary and cv['moved_echo_zones']==int(moved.sum())
    for key,count in cv['changed_elements'].items():assert count==int((corrected[key]!=saved['stress/'+key]).sum())
    assert cv['original_receipt_sha256']==sha(run/'receipt.json')
    audit=dict(status='PASS',training_steps=0,code_sha256=sha(Path(__file__)),receipt_sha256=sha(run/'receipt.json'),
        inputs_sha256=start['inputs'],additional_inputs_sha256={str(path):sha(path) for path in [extpath/'receipt.json',extpath/'extrema.npz',correction/'receipt.json',correction/'corrected.npz',correction/'result.json',localpath]},
        audit_dependencies={name:sha(Path(__file__).with_name(name)) for name in ['mz26_audit.py','mz24_audit.py','mz11_audit.py']},
        bank=dict(frames=7362,excluded_sequence=200,train_sites=375,source_index_sha256=sha(indexpath),heldout_identity=disjoint),
        task=task,all_neighbor_votes_checked=neighbor_bits,selected_global_neighbor_search=selected,mz26_comparison=comparisons,normal_witness_checks=witness_checks,
        stress_witness_correction=dict(status='PASS',summary=stress_summary,changed_elements=cv['changed_elements'],supersedes=cv['supersedes'],independent_teacher='Reconstructed from saved frozen MZ20 local scores with aligned labels; no new inference'),
        scope='All saved scalar decisions, votes, selected-neighbor distances and geometry audited; exact global top5 search checked on58frames. No training or duplicate frozen visual inference.')
    write(run/'audit.json',audit);print('PASS',task['scalar_bits'],'task bits;58 independent full-bank neighbor frames')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.root,a.run)
