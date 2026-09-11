"""Independent CPU MZ69 fixed-checkpoint transfer score; saved outputs only.

python -B mz69_score.py --task <mz69-topology-transfer-20260911>
Run only after root seals run-v1 and gives SCORE GO. No fit, inference or cuts.
"""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback

import numpy as np
import mz58_score as scalar_source
import mz63_score as transfer_source
from mz58_score import read,write,sha,load,metrics,scalar_metrics,paired
from mz63_score import check_packets,check_geometry,exchange_masks

TASK='mz69-topology-transfer-20260911'
SOURCE_INDEX_SHA='23aa6f552e38a0534052d2723b9a447ee91574bacacd503ecc64957c81b6560b'
PROFILES=('IDEAL','MERGE_CLOSE','DROP_CLOSE','ALL_INVALID')
QUERIES=('BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR')
HEADS=('MZ64/GEOMETRY','MZ68/NULL_COVERAGE')
LOCAL=('OLD_NEG/OPEN','OLD_NEG/GATED')+HEADS
BASE='OLD_NEG/UNION'
METHODS=('MZ37',BASE,'OLD_NEG/OPEN/candidate','OLD_NEG/GATED/candidate')+tuple(h+'/candidate' for h in HEADS)+tuple(h+'/UNION' for h in HEADS)
FIELDS=('role','family','range','support_context','site_id','relation')
ROLES={'TRAIN_CANDIDATE':2048,'CALIBRATION':1024,'HELDOUT_GEOMETRY':1024}
COMPARISONS={HEADS[0]+'/candidate':('MZ37',BASE),HEADS[0]+'/UNION':(BASE,),
             HEADS[1]+'/candidate':('MZ37',BASE,HEADS[0]+'/candidate'),HEADS[1]+'/UNION':(BASE,HEADS[0]+'/UNION')}


def reconstruct(saved,cuts):
    """Scalar original-cut/MZ37 retention and union reconstruction, no recut."""
    base=saved['MZ37'];n=len(base)
    assert base.shape==(n,4) and np.isfinite(base).all()
    for key in LOCAL:
        raw,support,winner,candidate=(saved[key+'/'+field] for field in ('raw','support','winner','candidate'))
        assert all(a.shape==(n,4) for a in (raw,support,winner,candidate))
        assert support.dtype==bool and np.issubdtype(winner.dtype,np.integer)
        assert np.isfinite(raw).all() and np.isfinite(candidate).all()
        assert ((winner>=0)&(winner<(3136 if key.startswith('OLD_NEG') else 3600))).all()
        assert cuts[key].shape==(4,) and np.isfinite(cuts[key]).all()
        expected=np.empty((n,4),np.float64)
        for i in range(n):
            for q in range(4):
                delta=float(raw[i,q])-float(cuts[key][q])
                expected[i,q]=delta if float(base[i,q])<0 and bool(support[i,q]) and delta>=0 else base[i,q]
        np.testing.assert_array_equal(candidate,expected)
        np.testing.assert_array_equal(candidate[base>=0],base[base>=0])
    members={BASE:('OLD_NEG/OPEN/candidate','OLD_NEG/GATED/candidate')}
    members.update({head+'/UNION':(BASE,head+'/candidate') for head in HEADS})
    for name,(left,right) in members.items():
        expected=np.empty((n,4),np.float64)
        for i in range(n):
            for q in range(4):expected[i,q]=max(float(saved[left][i,q]),float(saved[right][i,q]))
        np.testing.assert_array_equal(saved[name],expected)
        np.testing.assert_array_equal(saved[name]>=0,(saved[left]>=0)|(saved[right]>=0))
    available=saved['valid'].any((1,2))
    for head in HEADS:
        assert saved[head+'/anchor_available'].dtype==bool
        np.testing.assert_array_equal(saved[head+'/anchor_available'],available)
        vector=saved[head+'/anchor_vector']
        assert vector.shape==(n,8) and np.isfinite(vector).all() and (vector[~available]==0).all()
    for method in METHODS:assert not ((base>=0)&(saved[method]<0)).any()
    return n*4*(len(LOCAL)+len(members))


def native_winners(saved,counts,valid_counts,angular,angular_known):
    """Vector lookup checked by independent row/column scalar indexing."""
    n=len(counts);winners={};checked=0
    for key in LOCAL:
        w=saved[key+'/winner'].astype(int)
        if key.startswith('OLD_NEG'):
            present=angular.reshape(n,3136,4);known=angular_known.reshape(n,3136)
        else:present=(counts>0).reshape(n,3600,4);known=(valid_counts>0).reshape(n,3600)
        hit=present[np.arange(n)[:,None],w,np.arange(4)[None,:]]
        kn=known[np.arange(n)[:,None],w]
        for i in range(n):
            for q in range(4):
                a,b=divmod(int(w[i,q]),49 if key.startswith('OLD_NEG') else 80)
                value=angular[i,a,b,q] if key.startswith('OLD_NEG') else counts[i,a,b,q]>0
                kval=angular_known[i,a,b] if key.startswith('OLD_NEG') else valid_counts[i,a,b]>0
                assert bool(hit[i,q])==bool(value) and bool(kn[i,q])==bool(kval)
                checked+=1
        assert not (hit&~kn).any()
        winners[key]=dict(native=hit,known=kn)
    return winners,checked


def split(mask,win):
    native=mask&win['native'];wrong=mask&win['known']&~win['native'];unknown=mask&~win['known']
    np.testing.assert_array_equal(native.astype(int)+wrong.astype(int)+unknown.astype(int),mask.astype(int))
    return dict(events=mask.sum(0).tolist(),native=native.sum(0).tolist(),
        known_wrong=wrong.sum(0).tolist(),unknown=unknown.sum(0).tolist())


def accepted_new(saved,cuts,head):
    return saved[head+'/support'] & (saved[head+'/raw'].astype(np.float64)-cuts[head]>=0) & (saved['MZ37']<0)


def scalar_exchange(after,before,truth,known):
    counts={key:[0]*4 for key in ('TP_GAIN','TP_LOST','FP_ADDED','FP_REMOVED','UNKNOWN_DECISION_ADDED','UNKNOWN_DECISION_REMOVED')}
    for i in range(len(truth)):
        for q in range(4):
            a,b=float(after[i,q])>=0,float(before[i,q])>=0
            if a==b:continue
            key=('UNKNOWN_DECISION_ADDED' if a else 'UNKNOWN_DECISION_REMOVED') if not known[i,q] else (
                ('TP_GAIN' if a else 'TP_LOST') if truth[i,q] else ('FP_ADDED' if a else 'FP_REMOVED'))
            counts[key][q]+=1
    return counts


def score_arrays(p,packets,truth,known,records,counts,valid_counts,angular,angular_known,cuts,pairs):
    crop,sensor=check_geometry(p);n=len(truth)
    groups={'all':np.arange(n)}
    for field in FIELDS:
        for value in sorted({str(r[field]) for r in records}):
            groups[field+'/'+value]=np.array([i for i,r in enumerate(records) if str(r[field])==value])
    results={};coverage={};events={};contexts={};flags={}
    scalar_decisions=scalar_winners=scalar_reconstruction=0
    for profile in PROFILES:
        saved={key[len(profile)+1:]:value for key,value in p.items() if key.startswith(profile+'/')}
        coverage[profile]=check_packets(packets,saved,profile)
        scalar_reconstruction+=reconstruct(saved,cuts)
        winners,checked=native_winners(saved,counts,valid_counts,angular,angular_known);scalar_winners+=checked
        new={head:accepted_new(saved,cuts,head) for head in HEADS}
        results[profile]={}
        for group,ii in groups.items():
            t,k=truth[ii],known[ii]
            row=dict(methods={method:metrics(saved[method][ii],t,k) for method in METHODS},paired={},local={})
            if group=='all' or group.startswith('role/'):
                for method in METHODS:
                    assert row['methods'][method]==scalar_metrics(saved[method][ii],t,k)
                    scalar_decisions+=int(k.sum())
            for method,baselines in COMPARISONS.items():
                row['paired'][method]={baseline:paired(saved[method][ii],saved[baseline][ii],t,k) for baseline in baselines}
            for head in HEADS:
                win={key:value[ii] for key,value in winners[head].items()}
                positive=t&k;negative=~t&k
                added=positive&(saved[head+'/UNION'][ii]>=0)&(saved[BASE][ii]<0)&new[head][ii]
                row['local'][head]=dict(native_added_beyond_old=split(added,win),
                    accepted_new_true_over_mz37=split(positive&new[head][ii],win),
                    accepted_new_false_over_mz37=split(negative&new[head][ii],win),
                    final_true_winner_lookup=split(positive&(saved[head+'/UNION'][ii]>=0),win),
                    no_support_true=(positive&~saved[head+'/support'][ii]).sum(0).tolist(),
                    supported_below_cutoff_true=(positive&saved[head+'/support'][ii]&~(saved[head+'/raw'][ii].astype(float)-cuts[head]>=0)).sum(0).tolist())
            results[profile][group]=row
        events[profile]={}
        for method,baselines in COMPARISONS.items():
            head=method.rsplit('/',1)[0];events[profile][method]={}
            for baseline in baselines:
                masks=exchange_masks(saved[method],saved[baseline],truth,known)
                scalar=scalar_exchange(saved[method],saved[baseline],truth,known)
                assert {key:mask.sum(0).tolist() for key,mask in masks.items()}==scalar
                rows=[]
                for change,mask in masks.items():
                    for i,q in zip(*np.where(mask)):
                        def at(key):
                            w=int(saved[key+'/winner'][i,q]);kn=bool(winners[key]['known'][i,q]);native=bool(winners[key]['native'][i,q])
                            return dict(winner=w,native=native,known_wrong=kn and not native,unknown=not kn,
                                support=bool(saved[key+'/support'][i,q]),raw=float(saved[key+'/raw'][i,q]),cutoff=float(cuts[key][q]),
                                candidate_positive=bool(saved[key+'/candidate'][i,q]>=0))
                        before_heads=LOCAL[:2] if baseline==BASE else (baseline.rsplit('/',1)[0],) if baseline!='MZ37' else ()
                        w=int(saved[head+'/winner'][i,q])
                        rows.append(dict(index=int(i),frame_id=str(p['frame_ids'][i]),query=QUERIES[q],change=change,
                            **{field:records[i][field] for field in FIELDS},pair_id=records[i]['pair_id'],geometry_id=records[i]['geometry_id'],
                            truth=bool(truth[i,q]),known=bool(known[i,q]),before=float(saved[baseline][i,q]),after=float(saved[method][i,q]),
                            after_head=at(head),before_heads={key:at(key) for key in before_heads},
                            raster_row=w//80,raster_col=w%80,in_crop=bool(crop[w]),in_sensor=bool(sensor[w]),
                            accepted_new_branch=bool(new[head][i,q]),native_gain_credit=bool(change=='TP_GAIN' and new[head][i,q] and winners[head]['native'][i,q])))
                events[profile][method][baseline]=dict(counts=scalar,rows=rows)
        contexts[profile]={}
        for method in METHODS:
            rows=[];counts_by_role={}
            for pair_id,ii in pairs.items():
                a=next(i for i in ii if records[i]['support_context']=='unsupported')
                b=next(i for i in ii if records[i]['support_context']=='supported')
                change=(saved[method][a]>=0)!=(saved[method][b]>=0)
                role=records[a]['role'];entry=counts_by_role.setdefault(role,dict(pairs=0,changed_pairs=0,changed_queries=[0]*4))
                entry['pairs']+=1;entry['changed_pairs']+=int(change.any())
                entry['changed_queries']=(np.array(entry['changed_queries'])+change).tolist()
                if change.any():rows.append(dict(pair_id=pair_id,role=role,family=records[a]['family'],range=records[a]['range'],
                    frame_ids=[str(p['frame_ids'][i]) for i in (a,b)],contexts=['unsupported','supported'],
                    changed_queries=change.tolist(),decisions=[(saved[method][i]>=0).tolist() for i in (a,b)],truth=truth[a].tolist(),known=known[[a,b]].tolist()))
            contexts[profile][method]=dict(total_pairs=len(pairs),changed_pairs=len(rows),by_role=counts_by_role,rows=rows)
        ii=groups['role/HELDOUT_GEOMETRY'];g,h=HEADS
        masks=exchange_masks(saved[h+'/UNION'][ii],saved[g+'/UNION'][ii],truth[ii],known[ii])
        gain=masks['TP_GAIN']&new[h][ii];loss=masks['TP_LOST']&new[g][ii]
        gain_split=split(gain,{k:v[ii] for k,v in winners[h].items()})
        loss_split=split(loss,{k:v[ii] for k,v in winners[g].items()})
        flags[profile]=dict(held_frames=len(ii),comparison=h+'/UNION versus '+g+'/UNION',
            at_least_one_new_true_native_winner=sum(gain_split['native'])>=1,
            no_new_false_bits=int(masks['FP_ADDED'].sum())==0,
            descriptive_transfer_flag=sum(gain_split['native'])>=1 and int(masks['FP_ADDED'].sum())==0,
            per_query_exchanges={key:value.sum(0).tolist() for key,value in masks.items()},
            gained_true_null_winner=gain_split,lost_true_geometry_winner=loss_split,
            native_added_beyond_old={head:results[profile]['role/HELDOUT_GEOMETRY']['local'][head]['native_added_beyond_old'] for head in HEADS},
            interpretation='Both components and their descriptive conjunction are reported; not a replacement gate, ranking or winner selection. FP additions are distinct from net FP change.')
    return dict(profiles=results,packet_coverage=coverage,descriptive_flags=flags),dict(exchanges=events,support_context_pairs=contexts),dict(
        scalar_known_decisions=scalar_decisions,scalar_native_winner_lookups=scalar_winners,
        scalar_candidate_and_union_values=scalar_reconstruction,scalar_exchange_counts=True)


def validate_score_go(task):
    """Reject missing/stale root admission before creating scientific outputs."""
    rd=task/'run-v1';go_path=task/'score-go-v1.json';receipt_path=rd/'receipt.json'
    go=read(go_path)
    assert go['status']=='ROOT_SCORE_GO' and go['run_exit_code']==0
    assert go['run_receipt_sha256']==sha(receipt_path)
    assert go['scorer_sha256']==sha(Path(__file__))
    rr=read(receipt_path)
    assert rr['status']=='PASS' and rr['frames']==4096 and rr['fixed_checkpoints']
    assert tuple(rr['methods'])==METHODS and tuple(rr['profiles'])==PROFILES
    return go,rr


def run(task):
    task=task.resolve(strict=True);assert task.name==TASK
    go,admitted=validate_score_go(task)
    rd=task/'run-v1';out=task/'score-v1';out.mkdir(exist_ok=False);inputs={};start=time.perf_counter()
    try:
        def bind(path,expected=None):
            path=Path(path).resolve(strict=True);digest=sha(path);assert expected is None or digest==expected,str(path)
            inputs[str(path)]=digest;return path
        for path in (__file__,scalar_source.__file__,transfer_source.__file__,task/'score-go-v1.json'):bind(path)
        rr=read(bind(rd/'receipt.json',go['run_receipt_sha256']));assert rr==admitted
        assert tuple(rr['methods'])==METHODS and tuple(rr['profiles'])==PROFILES
        assert rr['training_steps']==rr['new_cutoffs']==rr['source_calibration_rows_used']==rr['old_cohort_replay_frames']==0
        assert not rr['evaluator_labels_read'] and rr['native_depth_reads']==0 and rr['existing_cutoff_vectors']==4
        for key in ('design_input','protocol'):bind(rr[key]['path'],rr[key]['sha256'])
        design=read(rr['design_input']['path']);p=load(bind(rd/'predictions.npz',rr['outputs']['predictions.npz']))
        source=Path(rr['source_task']).resolve(strict=True)
        assert rr['source_index_sha256']==SOURCE_INDEX_SHA
        index=read(bind(source/'source-index.json',SOURCE_INDEX_SHA))
        assert index['status']=='COMPLETE' and index['schema']=='mz67-topology-source-v1'
        assert index['frames']==4096 and index['pairs']==2048 and index['roles']==ROLES and len(index['shards'])==10
        assert index['source_role']=='CONSUMED_DEVELOPMENT'
        def combined(name):
            ref=index['combined'][name];path=(source/ref['path']).resolve(strict=True);assert path.is_relative_to(source)
            return bind(path,ref['sha256'])
        sr=read(combined('receipt.json'));assert sr['status']=='PASS' and sr['original_sources_unchanged']
        names=('packets.npz','evaluator.npz','fullframe-cells.npz','angular-auxiliary.npz','metadata.json','result.json','native-audit.json')
        for name in names:assert sr['outputs'][name]==index['combined'][name]['sha256']
        packets,evaluator,full,aux=(load(combined(name)) for name in names[:4])
        meta,result,audit=(read(combined(name)) for name in names[4:])
        assert result['frames']==4096 and result['independent_native_frames']==result['visual_reviewed_frames']==80
        assert audit['status']=='PASS' and audit['frames']==80
        records,pairs=meta['records'],meta['pairs'];ids=p['frame_ids'];n=len(ids)
        assert len(records)==len(set(ids))==n==4096
        np.testing.assert_array_equal(ids,[r['frame_id'] for r in records])
        np.testing.assert_array_equal([r['index'] for r in records],np.arange(n))
        for obj in (packets,evaluator,full,aux):np.testing.assert_array_equal(ids,obj['frame_ids'])
        np.testing.assert_array_equal(full['global_indices'],np.arange(n))
        truth,known=evaluator['truth'],evaluator['known']
        assert truth.dtype==known.dtype==bool and truth.shape==known.shape==(n,4)
        np.testing.assert_array_equal(known,np.repeat(np.array([r['source_valid'] for r in records],bool)[:,None],4,1))
        counts,vc=full['fullframe_event_counts'],full['valid_counts']
        assert counts.shape==(n,45,80,4) and vc.shape==(n,45,80) and counts.dtype==vc.dtype==np.uint8
        assert (vc<=64).all() and (counts<=vc[...,None]).all()
        np.testing.assert_array_equal(counts.sum((1,2))>=3,truth)
        assert int((~known).sum())==result['unknown_query_bits'] and int((vc==0).sum())==result['unknown_fullframe_cells']
        angular,ak=aux['cell_event_presence'],aux['cell_known']
        assert angular.shape==(n,64,49,4) and ak.shape==(n,64,49) and angular.dtype==ak.dtype==bool
        assert not (angular&~ak[...,None]).any()
        assert Counter(r['role'] for r in records)==ROLES
        assert len(pairs)==2048 and sorted(i for ii in pairs.values() for i in ii)==list(range(n))
        for pair_id,ii in pairs.items():
            assert len(ii)==2 and {records[i]['pair_id'] for i in ii}=={pair_id}
            assert len({records[i]['role'] for i in ii})==1
            assert {records[i]['support_context'] for i in ii}=={'unsupported','supported'}
            np.testing.assert_array_equal(truth[ii[0]],truth[ii[1]])
            np.testing.assert_array_equal(counts[ii[0]],counts[ii[1]])
        cuts={}
        assert set(rr['frozen']['cutoffs'])==set(LOCAL)
        for key in LOCAL:
            ref=rr['frozen']['cutoffs'][key];original=design['cutoffs'][key]
            assert ref['sha256']==original['sha256'] and ref['values']==original['values']
            path=bind(rd/ref['filename'],rr['outputs'][ref['filename']]);assert sha(path)==ref['sha256']
            bind(ref['path'],ref['sha256']);cuts[key]=np.load(path,allow_pickle=False)
            np.testing.assert_array_equal(cuts[key],ref['values'])
        assert rr['frozen']['checkpoints']==design['checkpoints']
        for ref in rr['frozen']['checkpoints'].values():bind(ref['path'],ref['sha256'])
        for path,digest in rr['inputs'].items():
            if path.endswith('.py'):bind(path,digest)
        scored,events,checks=score_arrays(p,packets,truth,known,records,counts,vc,angular,ak,cuts,pairs)
        answer=dict(status='PASS',frames=n,methods=METHODS,queries=QUERIES,profiles_order=PROFILES,
            primary_profiles=['DROP_CLOSE','ALL_INVALID'],source_role='CONSUMED_DEVELOPMENT',all_roles_descriptive=True,
            role_counts=dict(Counter(r['role'] for r in records)),known_query_bits=known.sum(0).tolist(),unknown_query_bits=(~known).sum(0).tolist(),
            unknown_fullframe_cells=int((vc==0).sum()),unknown_angular_cells=int((~ak).sum()),original_cutoffs={k:v.tolist() for k,v in cuts.items()},
            source_intent_matching_frames=result['intent_matching_frames'],source_geometry_generalization_claim=result['geometry_generalization_claim'],
            collision_audit=index['combined']['tensor-collision-audit.json'],old_train_collision_audit=index['combined']['old-train-collision-audit.json'],
            native_attribution='Native means query presence at the saved winner; known_wrong means local known but this query absent. Global false bits and local winner categories are separate. Inherited MZ37/OLD positives receive no new-branch credit.',
            limitation='Frozen transfer on consumed controlled MZ67. No fit, recalibration, source-role selection or replacement gate. Native lookup verifies a saved winner, not causal use or recomputed dense-logit argmax. UNKNOWN preserved. No natural-scene, hardware, Android/default-app or safety claim.',**scored)
        mechanical=dict(status='PASS',frames=n,**checks,source_pairs=len(pairs),original_cutoffs_exact=True,packet_scalar_parity=True,
            native_winner_argmax_recomputed=False,MZ37_retained_by_all_methods=True,all_groups_retained=True,
            unknown_separate=True,fits=0,inference_frames=0,cutoff_searches=0,RGB_reads=0,raw_native_reads=0)
        for path,digest in inputs.items():assert sha(path)==digest,path
        write(out/'result.json',answer);write(out/'paired-events.json',events);write(out/'audit.json',mechanical)
        lines=['# MZ69 fixed topology transfer','',answer['limitation'],'',
            '| Profile / HELD query | G TP/FP/FN/TN | NULL TP/FP/FN/TN | TP gained/lost | FP added/removed | Native NULL gains |',
            '| --- | --- | --- | --- | --- | --- |']
        for profile in ('DROP_CLOSE','ALL_INVALID'):
            held=answer['profiles'][profile]['role/HELDOUT_GEOMETRY'];flags=answer['descriptive_flags'][profile]
            a=held['methods'][HEADS[0]+'/UNION'];b=held['methods'][HEADS[1]+'/UNION'];ex=flags['per_query_exchanges']
            for q,name in enumerate(QUERIES):
                lines.append('| '+profile+'/'+name+' | '+' / '.join(str(a[k][q]) for k in ('tp','fp','fn','tn'))+' | '+
                    ' / '.join(str(b[k][q]) for k in ('tp','fp','fn','tn'))+f" | {ex['TP_GAIN'][q]} / {ex['TP_LOST'][q]} | {ex['FP_ADDED'][q]} / {ex['FP_REMOVED'][q]} | {flags['gained_true_null_winner']['native'][q]} |")
        lines+=['','Two descriptive flags per profile and all eight methods, four profiles, roles/families/ranges/support contexts/sites/relations are retained in [result.json](result.json). No conjunction selects a winner. Known-wrong and UNKNOWN winner partitions, both sides of each exchange, exact changed-bit IDs and supported/unsupported pairs are in [paired-events.json](paired-events.json).','']
        (out/'report.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
        write(out/'receipt.json',dict(status='PASS',inputs=inputs,source_index_sha256=SOURCE_INDEX_SHA,frozen=rr['frozen'],seconds=time.perf_counter()-start,
            outputs={path.name:sha(path) for path in out.iterdir() if path.is_file()},backend='FROZEN_PROTOCOL_CPU_ONLY',new_inferences=0,new_fits=0,new_cutoffs=0))
        print(dict(status='PASS',seconds=time.perf_counter()-start,descriptive={profile:answer['descriptive_flags'][profile] for profile in ('DROP_CLOSE','ALL_INVALID')}))
    except BaseException:
        write(out/'failure.json',dict(status='FAIL_PRESERVE_OUTPUT',inputs=inputs,error=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task',type=Path,required=True)
    run(parser.parse_args().task)
