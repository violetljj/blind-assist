"""CPU candidate scorer: sealed fixed MZ61 outputs only; no cutoff search.
--self-test is synthetic. --saved-parity reads four sealed MZ62 rows including inherited MZ58 outputs.
Scientific scoring is a separate --task invocation after root SCORE GO.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(sys.argv[sys.argv.index('--root') + 1]) if '--root' in sys.argv else Path('E:/linnan/linnan')
sys.path.insert(0, str(ROOT / 'research/active/dtr-r0/nearfield'))
import numpy as np
import mz58_score as prior
from mz58_score import read, write, sha, load, metrics, scalar_metrics, paired, check_geometry, check_packets as original_check_packets

PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE', 'ALL_INVALID')
QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
HEADS = ('MZ56/GLOBAL_ANCHOR', 'MZ62/CONTROL', 'MZ62/COVERAGE')
LOCAL = ('OLD_NEG/OPEN', 'OLD_NEG/GATED') + HEADS
BASE = 'OLD_NEG/UNION'
UNIONS = ('MZ57/GLOBAL_ANCHOR/UNION', 'MZ62/CONTROL/UNION', 'MZ62/COVERAGE/UNION')
METHODS = ('MZ37', BASE) + tuple(k + '/candidate' for k in HEADS) + UNIONS
COMPARATORS = (BASE, UNIONS[0])
CHALLENGERS = tuple(k + suffix for k in HEADS[1:] for suffix in ('/candidate', '/UNION'))
FIELDS = ('role', 'site_id', 'family', 'relation', 'range', 'setting', 'support_context', 'geometry_recipe_id', 'site_replica')


def check_packets(original, saved, profile):
    if profile != 'ALL_INVALID':
        assert profile in PROFILES[:3]
        return original_check_packets(original, saved, profile)
    np.testing.assert_array_equal(saved['ranges'], np.zeros_like(original['ranges']))
    np.testing.assert_array_equal(saved['valid'], np.zeros_like(original['valid'], dtype=bool))
    assert saved['ranges'].dtype == np.float32 and saved['valid'].dtype == bool
    return dict(valid_slots=0,zone_return_counts=[len(original['ranges'])*64,0,0],
        all_tof_missing_frames=len(original['ranges']),stress_boundary='ALL_INVALID, no residual true range; no hardware failure probability claimed')


def check_candidates(saved, cuts, truth, known):
    n = len(truth)
    base = saved['MZ37']
    assert base.shape == (n, 4) and np.isfinite(base).all()
    for key in LOCAL:
        raw, support, winner, candidate = (saved[key + '/' + x] for x in ('raw', 'support', 'winner', 'candidate'))
        assert all(v.shape == (n, 4) for v in (raw, support, winner, candidate))
        assert support.dtype == bool and np.issubdtype(winner.dtype, np.integer)
        assert np.isfinite(raw).all() and np.isfinite(candidate).all()
        assert ((winner >= 0) & (winner < (3136 if key.startswith('OLD_NEG') else 3600))).all()
        expected = np.empty((n, 4), float)
        for i in range(n):
            for q in range(4):
                margin = float(raw[i, q]) - float(cuts[key][q])
                expected[i, q] = margin if base[i, q] < 0 and support[i, q] and margin >= 0 else base[i, q]
        np.testing.assert_array_equal(candidate, expected)
    members = {BASE: (LOCAL[0] + '/candidate', LOCAL[1] + '/candidate')}
    members.update({u: (BASE, h + '/candidate') for h, u in zip(HEADS, UNIONS)})
    for key, (left, right) in members.items():
        np.testing.assert_array_equal(saved[key], np.maximum(saved[left], saved[right]))
        np.testing.assert_array_equal(saved[key] >= 0, (saved[left] >= 0) | (saved[right] >= 0))
    available = saved['valid'].any((1, 2))
    for key in HEADS:
        np.testing.assert_array_equal(saved[key + '/anchor_available'], available)
        v = saved[key + '/anchor_vector']
        assert v.shape == (n, 8) and np.isfinite(v).all() and (v[~available] == 0).all()
    for method in METHODS:
        assert scalar_metrics(saved[method], truth, known) == metrics(saved[method], truth, known)
    return int(known.sum()) * len(METHODS)


def native_winners(saved, full, full_known, angular, angular_known):
    result = {}
    n = len(full)
    for key in LOCAL:
        pres, kn = (angular, angular_known) if key.startswith('OLD_NEG') else (full, full_known)
        w = saved[key + '/winner'].astype(int)
        native = pres[np.arange(n)[:, None], w, np.arange(4)[None, :]]
        known = kn[np.arange(n)[:, None], w]
        assert not (native & ~known).any()
        result[key] = dict(native=native, known=known)
    return result


def exchange_masks(after, before, truth, known):
    a, b = after >= 0, before >= 0
    return dict(TP_GAIN=a & ~b & truth & known, TP_LOST=~a & b & truth & known,
                FP_ADDED=a & ~b & ~truth & known, FP_REMOVED=~a & b & ~truth & known,
                UNKNOWN_DECISION_ADDED=a & ~b & ~known, UNKNOWN_DECISION_REMOVED=~a & b & ~known)


def winner_split(mask, win):
    return dict(events=mask.sum(0).tolist(), native=(mask & win['native']).sum(0).tolist(),
        known_non_native=(mask & win['known'] & ~win['native']).sum(0).tolist(),
        unknown=(mask & ~win['known']).sum(0).tolist())


def distribution(values):
    return dict(n=len(values), quantiles=np.quantile(values, [0, .25, .5, .75, 1]).tolist() if len(values) else [])


def score_arrays(p, packets, truth, known, records, full, full_known, angular, angular_known, cuts, pairs):
    crop, sensor = check_geometry(p)
    n = len(truth)
    groups = {'all': np.arange(n)}
    for field in FIELDS:
        for value in sorted({str(r[field]) for r in records}):
            groups[field + '/' + value] = np.array([i for i,r in enumerate(records) if str(r[field]) == value])
    results, events, contexts, coverage, descriptive = {}, {}, {}, {}, {}
    scalar = 0
    for profile in PROFILES:
        saved = {k[len(profile)+1:]: v for k,v in p.items() if k.startswith(profile + '/')}
        coverage[profile] = check_packets(packets, saved, profile)
        coverage[profile]['anchors'] = {h:dict(available_frames=int(saved[h+'/anchor_available'].sum()),zero_vectors=int((saved[h+'/anchor_vector']==0).all(1).sum())) for h in HEADS}
        scalar += check_candidates(saved, cuts, truth, known)
        winners = native_winners(saved, full, full_known, angular, angular_known)
        results[profile] = {}
        for group, ids in groups.items():
            row = dict(methods={m: metrics(saved[m][ids], truth[ids], known[ids]) for m in METHODS},
                paired={m: {b: paired(saved[m][ids], saved[b][ids], truth[ids], known[ids]) for b in COMPARATORS} for m in CHALLENGERS}, local={})
            for key in LOCAL:
                native = {k:v[ids] for k,v in winners[key].items()}
                positive = truth[ids] & known[ids]
                negative = ~truth[ids] & known[ids]
                margin = saved[key+'/raw'][ids].astype(float) - cuts[key]
                support = saved[key+'/support'][ids]
                row['local'][key] = dict(positive=winner_split(positive,native), negative=winner_split(negative,native),
                    above_cutoff_supported_positive=winner_split(positive & support & (margin >= 0),native),
                    candidate_accepted_positive=winner_split(positive & (saved[key+'/candidate'][ids] >= 0),native),
                    inherited_mz37_positive=(positive & (saved['MZ37'][ids] >= 0)).sum(0).tolist(),
                    supported_below_cutoff_positive=winner_split(positive & support & (margin < 0),native),
                    no_support_positive=winner_split(positive & ~support,native),
                    branch_added_over_mz37=winner_split(positive & support & (margin >= 0) & (saved['MZ37'][ids] < 0),native),
                    margins={q:dict(positive=distribution(margin[positive[:,j],j]),negative=distribution(margin[negative[:,j],j]),
                        native_positive=distribution(margin[(positive & native['native'])[:,j],j])) for j,q in enumerate(QUERIES)})
            results[profile][group] = row
        events[profile] = {}
        for method in CHALLENGERS:
            head = method.rsplit('/',1)[0]
            win = winners[head]; ww = saved[head+'/winner'].astype(int)
            events[profile][method] = {}
            for baseline in COMPARATORS:
                masks = exchange_masks(saved[method], saved[baseline], truth, known)
                rows=[]
                for change, mask in masks.items():
                    for i,q in zip(*np.where(mask)):
                        w=int(ww[i,q])
                        responsible=bool(saved[head+'/candidate'][i,q]>=0 and saved['MZ37'][i,q]<0 and saved[head+'/support'][i,q])
                        rows.append(dict(index=int(i),frame_id=str(p['frame_ids'][i]),query=QUERIES[q],change=change,
                            **{field:records[i][field] for field in FIELDS},pair_id=records[i]['pair_id'],geometry_id=records[i]['geometry_id'],
                            truth=bool(truth[i,q]),known=bool(known[i,q]),winner=w,raster_row=w//80,raster_col=w%80,
                            native=bool(win['native'][i,q]),local_known=bool(win['known'][i,q]),in_crop=bool(crop[w]),in_sensor=bool(sensor[w]),
                            support=bool(saved[head+'/support'][i,q]),raw=float(saved[head+'/raw'][i,q]),cutoff=float(cuts[head][q]),
                            before=float(saved[baseline][i,q]),after=float(saved[method][i,q]),
                            new_branch_positive=responsible,native_added_credit=change=='TP_GAIN' and responsible and bool(win['native'][i,q])))
                events[profile][method][baseline]=rows
        contexts[profile]={}
        for method in METHODS:
            rows=[]
            for pair_id,ii in pairs.items():
                a,b=ii
                change=(saved[method][a]>=0)!=(saved[method][b]>=0)
                if change.any():
                    rows.append(dict(pair_id=pair_id,frame_ids=[str(p['frame_ids'][i]) for i in ii],
                        contexts=[records[i]['support_context'] for i in ii],changed_queries=change.tolist(),
                        decisions=[(saved[method][i]>=0).tolist() for i in ii],truth=truth[a].tolist(),known=known[ii].tolist()))
            contexts[profile][method]=dict(total_pairs=len(pairs),changed_pairs=len(rows),rows=rows)
        descriptive[profile]={}
        held=groups['role/HELDOUT_GEOMETRY']
        for head,union in zip(HEADS[1:],UNIONS[1:]):
            change=exchange_masks(saved[union][held],saved[BASE][held],truth[held],known[held])
            native_bn=int((change['TP_GAIN'] & winners[head]['native'][held])[:,0].sum())
            fp=int(change['FP_ADDED'].sum())
            descriptive[profile][union]=dict(held_frames=len(held),native_body_near_added=native_bn,fp_added=fp,
                native_gain_and_no_added_fp=native_bn>0 and fp==0,
                interpretation='Descriptive strict flag only; all sensitivity gains and false-positive costs remain reported. No automatic winner selection.')
    return dict(profiles=results,packet_coverage=coverage,descriptive_flags=descriptive),dict(exchanges=events,support_context_pairs=contexts),scalar


def run(task):
    out=task/'score-v1'; out.mkdir(exist_ok=False)
    inputs={}; started=time.perf_counter()
    try:
        def bind(path,expected=None):
            path=Path(path).resolve(strict=True); digest=sha(path)
            assert expected is None or digest==expected,str(path)
            inputs[str(path)]=digest; return path
        bind(__file__); bind(prior.__file__)
        rr=read(bind(task/'run-v1/receipt.json'))
        assert rr['status']=='PASS' and rr['frames']==4096 and rr['fixed_checkpoints']
        assert tuple(rr['methods'])==METHODS and tuple(rr['profiles'])==PROFILES
        assert rr['training_steps']==rr['new_cutoffs']==rr['source_calibration_rows_used']==rr['old_cohort_replay_frames']==0
        assert not rr['evaluator_labels_read'] and rr['native_depth_reads']==0 and rr['existing_cutoff_vectors']==5
        for key in ('design_input','protocol'):
            bind(rr[key]['path'],rr[key]['sha256'])
        design=read(rr['design_input']['path'])
        p=load(bind(task/'run-v1/predictions.npz',rr['outputs']['predictions.npz']))
        source=Path(rr['source_task']).resolve(strict=True)
        index=read(bind(source/'source-index.json',rr['source_index_sha256']))
        assert index['schema']=='mz61-geometry-source-v1' and index['status']=='COMPLETE'
        assert index['frames']==4096 and len(index['shards'])==10 and index['source_role']=='CONSUMED_DEVELOPMENT'
        def combined(name):
            ref=index['combined'][name]; path=(source/ref['path']).resolve(strict=True)
            assert path.is_relative_to(source)
            return bind(path,ref['sha256'])
        sr=read(combined('receipt.json'))
        assert sr['status']=='PASS' and sr['original_sources_unchanged']
        names=('packets.npz','evaluator.npz','fullframe-cells.npz','angular-auxiliary.npz','metadata.json','result.json','tensor-collision-audit.json','native-audit.json')
        for name in names: assert sr['outputs'][name]==index['combined'][name]['sha256']
        packets,evaluator,full,aux=(load(combined(name)) for name in names[:4])
        meta,result,collision,native_audit=(read(combined(name)) for name in names[4:])
        assert result['frames']==4096 and result['independent_native_frames']==result['visual_reviewed_frames']==80
        assert native_audit['status']=='PASS' and native_audit['frames']==80
        records,pairs=meta['records'],meta['pairs']; ids=p['frame_ids']; n=len(ids)
        assert len(records)==len(set(ids))==n==4096
        np.testing.assert_array_equal(ids,[r['frame_id'] for r in records])
        for obj in (packets,evaluator,full,aux): np.testing.assert_array_equal(ids,obj['frame_ids'])
        np.testing.assert_array_equal([r['index'] for r in records],np.arange(n))
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
        assert not (angular & ~ak[...,None]).any()
        assert Counter(r['role'] for r in records)==dict(TRAIN_CANDIDATE=2048,CALIBRATION=1024,HELDOUT_GEOMETRY=1024)
        assert len(pairs)==2048 and sorted(i for ii in pairs.values() for i in ii)==list(range(n))
        for pair_id,ii in pairs.items():
            assert len(ii)==2 and {records[i]['pair_id'] for i in ii}=={pair_id}
            assert len({records[i]['role'] for i in ii})==1 and len({records[i]['support_context'] for i in ii})==2
            np.testing.assert_array_equal(truth[ii[0]],truth[ii[1]])
            np.testing.assert_array_equal(counts[ii[0]],counts[ii[1]])
        cuts={}
        for key in LOCAL:
            ref=rr['frozen']['cutoffs'][key]; target=design['cutoffs'][key]
            assert ref['sha256']==target['sha256'] and ref['values']==target['values']
            path=bind(task/'run-v1'/ref['filename'],rr['outputs'][ref['filename']]); assert sha(path)==ref['sha256']
            bind(ref['path'],ref['sha256']); cuts[key]=np.load(path,allow_pickle=False)
            np.testing.assert_array_equal(cuts[key],ref['values'])
        assert rr['frozen']['checkpoints']==design['checkpoints']
        for ref in rr['frozen']['checkpoints'].values(): bind(ref['path'],ref['sha256'])
        for path,digest in rr['inputs'].items():
            if path.endswith('.py'): bind(path,digest)
        scored,events,scalar=score_arrays(p,packets,truth,known,records,(counts>0).reshape(n,3600,4),(vc>0).reshape(n,3600),angular.reshape(n,3136,4),ak.reshape(n,3136),cuts,pairs)
        answer=dict(status='PASS',frames=n,methods=METHODS,queries=QUERIES,profiles_order=PROFILES,
            source_role='CONSUMED_DEVELOPMENT',all_roles_descriptive=True,
            known_query_bits=known.sum(0).tolist(),unknown_query_bits=(~known).sum(0).tolist(),
            unknown_fullframe_cells=int((vc==0).sum()),unknown_angular_cells=int((~ak).sum()),
            source_intent_matching_frames=sum(bool(r['intent_matches']) for r in records),
            source_geometry_generalization_claim=result['geometry_generalization_claim'],
            collision_audit=index['combined']['tensor-collision-audit.json'],
            collision_summary=result['tensor_collision_summary'],original_cutoffs={k:v.tolist() for k,v in cuts.items()},**scored,
            limitation='Consumed controlled Development. No new fit/calibration or mode selection. Native winner lookup does not establish causal surface use; dense logits were not saved to rederive argmax. UNKNOWN preserved. No hardware/natural-scene/safety claim. Older 600-step versus MZ62 1200-step comparison does not isolate budget from schedule.')
        audit=dict(status='PASS',frames=n,scalar_known_decisions=scalar,independent_candidate_checks=n*4*len(LOCAL)*len(PROFILES),
            source_pairs=2048,all_methods_profiles=True,original_cutoffs_exact=True,packet_scalar_parity=True,
            query_local_unknown_separate=True,native_argmax_recomputed=False,source_arrays_only=True,
            fits=0,inference_frames=0,cutoff_searches=0,RGB_reads=0,raw_native_reads=0)
        for path,digest in inputs.items(): assert sha(path)==digest,path
        write(out/'result.json',answer); write(out/'paired-events.json',events);write(out/'audit.json',audit)
        lines=['# Fixed MZ61 transfer (candidate scorer)','', 'All 4096 frames and roles are descriptive; frozen weights and original cutoffs.','',
            '| DROP method | TP | FP | FN | Exact |','| --- | ---: | ---: | ---: | ---: |']
        for method,row in answer['profiles']['DROP_CLOSE']['all']['methods'].items():
            lines.append('| '+method+' | '+' | '.join(str(sum(row[k])) for k in ('tp','fp','fn'))+' | '+str(row['exact_frames'])+' |')
        lines+=['','Complete per-query/group TP/FP exchanges, UNKNOWN changes, winner partitions, margins and support pairs are in result.json and paired-events.json. Strict no-added-FP flags do not replace the sensitivity/cost tradeoff or select a winner.','',answer['limitation']]
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        write(out/'receipt.json',dict(status='PASS',inputs=inputs,source_index_sha256=rr['source_index_sha256'],frozen=rr['frozen'],seconds=time.perf_counter()-started,
            outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},backend='FROZEN_PROTOCOL_CPU_ONLY',new_inferences=0,new_fits=0,new_cutoffs=0))
        print('PASS',time.perf_counter()-started,answer['descriptive_flags']['DROP_CLOSE'])
    except BaseException:
        write(out/'failure.json',dict(status='FAIL',inputs=inputs,error=traceback.format_exc()));raise


def self_test():
    truth=np.array([[1,0,1,0],[1,0,1,0],[0,1,0,1],[0,1,0,1]],bool)
    known=np.ones_like(truth);known[2:,2]=False
    cuts={k:np.full(4,.2) for k in LOCAL}
    rr=np.zeros((4,64,2),np.float32); vv=np.zeros_like(rr,bool)
    rr[:2,0]=[1,1.5];vv[:2,0]=True
    packets=dict(ranges=rr,valid=vv)
    saved=dict(MZ37=np.full((4,4),-1.));saved['MZ37'][0,0]=.1
    for j,k in enumerate(LOCAL):
        raw=np.array([[.2,.5,-.4,.1],[.3,-.3,.6,.2],[.4,.9,.8,-.2],[.1,.7,-.1,.9]])+j*.03
        if j==4:raw[:,1]-=.9
        support=np.ones((4,4),bool);support[1,2]=False
        saved[k+'/raw']=raw;saved[k+'/support']=support
        saved[k+'/winner']=np.full((4,4),j,np.int16)
        margin=raw-cuts[k]
        saved[k+'/candidate']=np.where((saved['MZ37']<0)&support&(margin>=0),margin,saved['MZ37'])
    saved[BASE]=np.maximum(saved[LOCAL[0]+'/candidate'],saved[LOCAL[1]+'/candidate'])
    for h,u in zip(HEADS,UNIONS):
        saved[u]=np.maximum(saved[BASE],saved[h+'/candidate'])
        saved[h+'/anchor_available']=vv.any((1,2));saved[h+'/anchor_vector']=np.zeros((4,8),np.float32)
    saved.update(packets)
    check_candidates(saved,cuts,truth,known)
    assert saved[LOCAL[0]+'/candidate'][1,2]==-1
    full=np.zeros((4,3600,4),bool);fk=np.ones((4,3600),bool)
    angular=np.zeros((4,3136,4),bool);ak=np.ones((4,3136),bool)
    full[:,2,0]=True;full[:,3,1]=True;fk[:,4]=False
    win=native_winners(saved,full,fk,angular,ak)
    assert win[HEADS[0]]['native'][:,0].all() and not win[HEADS[2]]['known'].any()
    # Equal net FP still preserves distinct added/removed IDs; UNKNOWN separate.
    t=np.zeros((2,4),bool);k=np.ones_like(t);k[1,2]=False
    before=np.full((2,4),-1.);after=before.copy();before[0,0]=1;after[1,0]=1;after[1,2]=1
    exchange=exchange_masks(after,before,t,k)
    assert exchange['FP_ADDED'].sum()==exchange['FP_REMOVED'].sum()==1
    assert exchange['UNKNOWN_DECISION_ADDED'].sum()==1
    for change in ('union','candidate','winner','anchor'):
        broken={k:v.copy() for k,v in saved.items()}
        if change=='union':broken[BASE][0,0]=-1
        elif change=='candidate':broken[HEADS[0]+'/candidate'][0,0]=99
        elif change=='winner':broken[HEADS[0]+'/winner'][0,0]=3600
        else:broken[HEADS[0]+'/anchor_vector'][2,0]=1
        try:check_candidates(broken,cuts,truth,known)
        except AssertionError:pass
        else:raise AssertionError('Corruption accepted: '+change)
    # Execute full result grouping/event accounting with synthetic geometry.
    row,col=np.indices((45,80),dtype=float);focal=320/np.tan(np.deg2rad(50))
    right=(col*8+3.5-319.5)/focal;up=(179.5-row*8-3.5)/focal
    rays=np.stack([np.ones_like(right),right,up],-1);rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
    zr,zc=np.indices((8,8),dtype=np.float32);cr,cc=np.indices((7,7),dtype=float)
    az=np.deg2rad(-22.5+(zc.reshape(64,1)+((cc+.5)/7).reshape(1,49))*5.625)
    el=np.deg2rad(22.5-(zr.reshape(64,1)+((cr+.5)/7).reshape(1,49))*5.625)
    ar=np.stack([np.ones_like(az),np.tan(az),np.tan(el)],-1);ar/=np.linalg.norm(ar,axis=-1,keepdims=True)
    p={'frame_ids':np.array(['synthetic'+str(i) for i in range(4)]),
       'geometry/crop_mask':(row>=9)&(row<=35)&(col>=26)&(col<=53),
       'geometry/sensor_coverage':(np.abs(np.rad2deg(np.arctan(right)))<=22.5)&(np.abs(np.rad2deg(np.arctan(up)))<=22.5),
       'geometry/rays':rays,'geometry/angular_rays':ar,
       'geometry/zone_angles':np.stack([-22.5+(zc+.5)*5.625,22.5-(zr+.5)*5.625],-1).reshape(64,2)/22.5}
    for profile in PROFILES:
        ss={k:v.copy() for k,v in saved.items()}
        if profile=='MERGE_CLOSE':ss['ranges'][:2,0]=[1.25,0];ss['valid'][:2,0]=[True,False]
        if profile=='DROP_CLOSE':ss['ranges'][:2,0]=0;ss['valid'][:2,0]=False
        if profile=='ALL_INVALID':ss['ranges'][:]=0;ss['valid'][:]=False
        for h in HEADS:ss[h+'/anchor_available']=ss['valid'].any((1,2))
        p.update({profile+'/'+k:v for k,v in ss.items()})
    records=[dict(frame_id=str(p['frame_ids'][i]),role='HELDOUT_GEOMETRY',site_id='synthetic',family='synthetic',relation='synthetic',range='near',setting=0,support_context='unsupported' if i%2==0 else 'supported',geometry_recipe_id='g0',site_replica=0,pair_id='pair'+str(i//2),geometry_id='g'+str(i//2)) for i in range(4)]
    answer,events,scalar=score_arrays(p,packets,truth,known,records,full,fk,angular,ak,cuts,dict(pair0=[0,1],pair1=[2,3]))
    assert scalar==int(known.sum())*8*4 and len(answer['profiles'])==4
    for profile in PROFILES:
        for method in CHALLENGERS:
            for baseline in COMPARATORS:
                counts=Counter(x['change'] for x in events['exchanges'][profile][method][baseline])
                got=answer['profiles'][profile]['all']['paired'][method][baseline]
                for key,event in [('tp_gained','TP_GAIN'),('tp_lost','TP_LOST'),('fp_added','FP_ADDED'),('fp_removed','FP_REMOVED')]:assert sum(got[key])==counts[event]
    return dict(status='PASS',synthetic_frames=4,profiles=4,methods=8,corruptions_rejected=4,full_group_event_flow=True,unknown_separate=True,equal_net_fp_exchange_ids=True)


def saved_parity(root, design_path):
    # Already sealed MZ62 contains the MZ58 baselines plus both actual MZ62 heads.
    # Only four historical rows; no source/RGB/native decode or new inference.
    run=root/'artifacts.local/work/mz62-profile-coverage-20260911/run-v1'
    receipt=read(run/'receipt.json');assert receipt['status']=='PASS'
    assert sha(run/'receipt.json')=='c423418b7b54583b19a334d542d5f832780d7f40e23781b0d4ceb0eb52f985c8'
    path=run/'predictions.npz';assert sha(path)==receipt['outputs']['predictions.npz']
    design=read(design_path)
    cuts={k:np.load(design['cutoffs'][k]['path'],allow_pickle=False) for k in LOCAL}
    for k in LOCAL:assert sha(design['cutoffs'][k]['path'])==design['cutoffs'][k]['sha256']
    selected=np.array([0,1,1279,2559]);total=0
    with np.load(path,allow_pickle=False) as p:
        truth=p['mz55/truth'][selected];known=p['mz55/known'][selected]
        original={k:p['mz55/IDEAL/'+k][selected] for k in ('ranges','valid')}
        for profile in PROFILES[:3]:
            keys=set(METHODS)|{'ranges','valid'}
            for k in LOCAL:keys.update(k+'/'+s for s in ('raw','support','winner','candidate'))
            for k in HEADS:keys.update(k+'/'+s for s in ('anchor_available','anchor_vector'))
            saved={k:p['mz55/'+profile+'/'+k][selected] for k in keys}
            total+=check_candidates(saved,cuts,truth,known)
            check_packets(original,saved,profile)
        frame_ids=p['mz55/frame_ids'][selected].tolist()
    return dict(status='PASS',frames=4,frame_ids=frame_ids,profiles=3,methods=8,scalar_known_decisions=total,source='Sealed MZ62 saved output rows only',inputs={str(run/'receipt.json'):sha(run/'receipt.json'),str(path):sha(path)},new_inferences=0,new_source_decodes=0,not_new_MZ61_results=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--task',type=Path)
    parser.add_argument('--design',type=Path,default=Path(__file__).with_name('inputs.json'))
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--saved-parity',action='store_true')
    args=parser.parse_args()
    if args.self_test or args.saved_parity:
        assert args.task is None
        if args.self_test:print(self_test())
        if args.saved_parity:print(saved_parity(args.root,args.design.resolve(strict=True)))
    else:
        assert args.task is not None
        run(args.task.resolve(strict=True))
