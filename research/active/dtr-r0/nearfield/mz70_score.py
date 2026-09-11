"""MZ70 independent saved-output score; --selftest never opens experiment inputs.

Scientific execution requires root's score-go-v1.json binding run receipt SHA.
No fitting, model inference, selected cutoff, or aggregate promotion gate.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
import traceback
import numpy as np
from mz58_score import read, write, sha, load, metrics, scalar_metrics, paired
from mz59_score import scalar_cutoff, native_masks, assert_metric_equal
from mz64_score import partitions, compact_native, event_rows, summarize_margin, COHORTS, PROFILES, SOURCE_PROFILES, QUERIES
from mz68_score import reconstruct as reconstruct68
from mz63_score import check_geometry, check_packets, exchange_masks
from mz69_score import split, scalar_exchange

TASK='mz70-diverse-learning-20260911'
ARMS=('CONTROL','DIVERSE')
KEYS=tuple('MZ70/'+a for a in ARMS)
NEW=tuple(k+s for k in KEYS for s in ('/candidate','/UNION'))
BASE='OLD_NEG/UNION'
G='MZ64/GEOMETRY/UNION'
NULL='MZ68/NULL_COVERAGE/UNION'
SOURCE67_SHA='23aa6f552e38a0534052d2723b9a447ee91574bacacd503ecc64957c81b6560b'
INITIAL_SHA='fb318f03a598d00b89fc011eb5b1a368873f9ac801d1dc0710618ba119f4d4aa'
PRIOR={'mz68-missing-input-coverage-20260911':('2565c40a93862c6513962484fa45ebecb4e9aabec503f299c931c45e61f813eb','7fdbde94f6ee0cd713d47be07422a334b1a789ad9dedc41cb4834fb75df745ed'),
       'mz69-topology-transfer-20260911':('b65fa2e2c45c2014761fb03254267b38307c1ceaf49c846653da562e6ffdf259','37565ab3a6d254496f868768b4e19f3b50aee1099a37f3ddfce4f5df40905eeb')}
CLAIM='Matched 4096-step source-allocation contrast: DIVERSE versus CONTROL on consumed MZ67 HELD ALL_INVALID. Report per-query effects, native evidence, false-bit costs and ranking without selecting a threshold or model. Comparisons against shorter historical runs are descriptive, not source-allocation causal estimates.'


def ranking(scores, truth):
    assert scores.ndim==truth.ndim==1 and len(scores)==len(truth)
    assert truth.dtype==bool and np.isfinite(scores).all()
    pos=int(truth.sum());neg=len(truth)-pos
    if not pos or not neg:
        return dict(status='NOT_EVALUABLE',positive=pos,negative=neg,roc_auc=None,average_precision=None)
    order=np.argsort(-scores,kind='stable');s=scores[order];y=truth[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1];ends=np.r_[starts[1:],len(s)]
    cp=cn=0;wins=ap=0.
    for a,b in zip(starts,ends):
        pp=int(y[a:b].sum());nn=int(b-a-pp);wins+=cp*nn+.5*pp*nn;cp+=pp;cn+=nn
        ap+=(pp/pos)*(cp/(cp+cn))
    return dict(status='EVALUABLE',positive=pos,negative=neg,roc_auc=wins/(pos*neg),average_precision=ap,prevalence=pos/len(y),tied_groups=int(((ends-starts)>1).sum()))


def common_ranking(p,prefix,truth,known,ii,keys=KEYS):
    answer={};base=p[prefix+'MZ37'][ii]
    supports=[p[prefix+k+'/support'][ii] for k in keys]
    for q,name in enumerate(QUERIES):
        opportunity=known[ii,q]&(base[:,q]<0)
        common=opportunity&np.logical_and.reduce([v[:,q] for v in supports])
        row=dict(frames=len(ii),known=int(known[ii,q].sum()),known_positive=int((known[ii,q]&truth[ii,q]).sum()),
            known_negative=int((known[ii,q]&~truth[ii,q]).sum()),opportunity=int(opportunity.sum()),eligible=int(common.sum()),
            excluded_unknown=int((~known[ii,q]).sum()),excluded_mz37_positive=int((known[ii,q]&(base[:,q]>=0)).sum()),
            excluded_unsupported=int((opportunity&~common).sum()),support_counts={k:int((opportunity&v[:,q]).sum()) for k,v in zip(keys,supports)},
            support_difference_indices=ii[opportunity&np.any(np.stack(supports)[:,:,q]!=supports[0][:,q],axis=0)].tolist(),
            eligible_indices=ii[common].tolist(),models={k:ranking(p[prefix+k+'/raw'][ii[common],q],truth[ii[common],q]) for k in keys})
        assert row['eligible']+row['excluded_unknown']+row['excluded_mz37_positive']+row['excluded_unsupported']==len(ii)
        answer[name]=row
    return answer


def check_new(p,prefix,key,cut):
    base=p[prefix+'MZ37'];raw,sup,w,vec,av,cand=(p[prefix+key+'/'+f] for f in ('raw','support','winner','anchor_vector','anchor_available','candidate'))
    n=len(base)
    assert base.shape==raw.shape==sup.shape==w.shape==cand.shape==(n,4)
    assert sup.dtype==bool and av.dtype==bool and av.shape==(n,)
    assert np.isfinite(base).all() and np.isfinite(raw).all() and np.isfinite(cand).all()
    assert np.issubdtype(w.dtype,np.integer) and ((w>=0)&(w<3600)).all()
    assert vec.shape==(n,8) and np.isfinite(vec).all() and (vec>=0).all() and (vec[~av]==0).all()
    expected=np.empty((n,4),float)
    for i in range(n):
        for q in range(4):
            margin=float(raw[i,q])-float(cut[q])
            expected[i,q]=margin if base[i,q]<0 and sup[i,q] and margin>=0 else base[i,q]
    np.testing.assert_array_equal(cand,expected)
    union=np.maximum(p[prefix+BASE],cand)
    np.testing.assert_array_equal(p[prefix+key+'/UNION'],union)
    np.testing.assert_array_equal(union>=0,(p[prefix+BASE]>=0)|(cand>=0))
    np.testing.assert_array_equal(av,p[prefix+'MZ64/GEOMETRY/anchor_available'])
    if prefix.endswith('ALL_INVALID/'):
        assert not av.any() and not vec.any()
    return n*8


def descriptive(a,truth,known,ii,winners):
    c,d=(k+'/UNION' for k in KEYS)
    masks=exchange_masks(a[d][ii],a[c][ii],truth[ii],known[ii])
    gain=split(masks['TP_GAIN'],{k:v[ii] for k,v in winners[KEYS[1]].items()})
    loss=split(masks['TP_LOST'],{k:v[ii] for k,v in winners[KEYS[0]].items()})
    return dict(frames=len(ii),CONTROL=metrics(a[c][ii],truth[ii],known[ii]),DIVERSE=metrics(a[d][ii],truth[ii],known[ii]),
        exchanges={k:v.sum(0).tolist() for k,v in masks.items()},gained_winner=gain,lost_winner=loss,
        native_HEAD_NEAR_gain=bool(gain['native'][2]>0),native_HEAD_NEAR_net_gain=int(gain['native'][2])-int(loss['native'][2]),
        no_new_false_bits=not bool(masks['FP_ADDED'].any()),interpretation='Separate descriptive flags only; no overall gate or selected winner. Native gain credit is for actual newly positive branch decisions; lookup does not prove causal feature use.')


def prior_arrays(p,old,prefix=''):
    for key,v in old.items():
        actual=p[prefix+key]
        assert actual.dtype==v.dtype and actual.shape==v.shape and actual.tobytes()==v.tobytes(),prefix+key
    return {prefix+k for k in old}


def check_schedule(s,old,records):
    """Independent role-only exposure and deterministic shared-sequence audit."""
    np.testing.assert_array_equal(s['profile_index'],np.arange(4096)%4)
    for key in ('mz48','OLD_NEG','query'):
        np.testing.assert_array_equal(s[key],old[key][np.arange(4096)%1536])
    for key,width in (('CONTROL',4),('DIVERSE',4),('DIVERSE_source',4),('mz48',4),('OLD_NEG',8),('query',8)):
        assert s[key].dtype==np.int64 and s[key].shape==(4096,width)
    fits={c:np.array([i for i,r in enumerate(records[c]) if r['role']=='TRAIN_CANDIDATE'],np.int64) for c in ('mz61','mz67')}
    for c,v in fits.items():
        assert len(v)==2048 and all(records[c][i]['source_valid'] for i in v)
        np.testing.assert_array_equal(s[c+'_fit_ids'],v)
    rng=np.random.default_rng(170);tags=np.repeat(np.arange(2),2048)
    joined=np.r_[fits['mz61'],fits['mz67']];replacement=np.tile(fits['mz61'],2)
    for profile in range(4):
        perm=rng.permutation(4096);control=s['CONTROL'][profile::4].ravel();diverse=s['DIVERSE'][profile::4].ravel();tag=s['DIVERSE_source'][profile::4].ravel()
        np.testing.assert_array_equal(control,replacement[perm]);np.testing.assert_array_equal(diverse,joined[perm]);np.testing.assert_array_equal(tag,tags[perm])
        assert Counter(control)=={int(i):2 for i in fits['mz61']}
        for code,c in enumerate(('mz61','mz67')):assert Counter(diverse[tag==code])=={int(i):1 for i in fits[c]}
    return dict(status='PASS',steps_per_arm=4096,profile_steps=[1024]*4,CONTROL_mz61_each_id_each_profile=2,
        DIVERSE_each_source_id_each_profile=1,geometry_presentations_per_arm=16384,mz48_presentations_per_arm=16384,old_presentations_per_arm=32768,shared_replay_exact=True,seed=170)


def run(task):
    task=task.resolve(strict=True);assert task.name==TASK
    rd=task/'run-v1';out=task/'score-v1';work=task.parent
    go=read(task/'score-go-v1.json');assert go['status']=='ROOT_SCORE_GO' and go['run_exit_code']==0
    assert go['run_receipt_sha256']==sha(rd/'receipt.json') and go['scorer_sha256']==sha(__file__)
    rr=read(rd/'receipt.json');assert rr['status']=='PASS'
    assert not out.exists();out.mkdir();start=time.perf_counter();inputs={}
    try:
        def bind(path,expected=None):
            path=Path(path).resolve(strict=True);h=sha(path);assert expected is None or h==expected,str(path);inputs[str(path)]=h;return path
        for name in ('mz70_score.py','mz68_score.py','mz66_score.py','mz64_score.py','mz63_score.py','mz62_score.py','mz59_score.py','mz58_score.py','mz69_score.py'):bind(Path(__file__).with_name(name))
        bind(task/'score-go-v1.json');bind(rd/'receipt.json',go['run_receipt_sha256'])
        assert rr['steps_per_arm']==4096 and rr['total_steps']==8192 and rr['trainable_parameters_per_arm']==11020
        assert set(rr['fits'])==set(ARMS) and all(rr['fits'][a]['steps']==4096 for a in ARMS)
        assert rr['exact_initial_weights'] and rr['original_calibration_frames']==1256 and rr['new_cutoffs']==2 and rr['threshold_searches']==0
        assert rr['initial_checkpoint_sha256']==INITIAL_SHA
        assert set(rr['initial_parameter_sha256'])==set(ARMS) and len(set(rr['initial_parameter_sha256'].values()))==1
        assert rr['positive_pool']=='original_OPEN_max' and rr['loss_contract']=='original_MZ59_balanced_local_frame_loss'
        assert rr['training_profile_counts_per_arm']=={p:1024 for p in SOURCE_PROFILES}
        assert rr['all_invalid_training_steps_per_arm']==1024 and rr['all_invalid_training_presentations_per_arm']==16384
        assert rr['original_baseline_cohort_inferences']==rr['native_depth_reads']==rr['mz55_calibration_rows_used']==rr['mz61_calibration_rows_used']==rr['mz67_calibration_rows_used']==0
        assert rr['source_unknown_preserved'] and rr['optimizer_change_between_arms']=='none'
        for q,h in rr['inputs'].items():
            if q.endswith(('.py','.md','.json')):bind(q,h)
        for name in rr['outputs']:bind(rd/name,rr['outputs'][name])
        p=load(rd/'predictions.npz');groups=read(rd/'groups.json')
        previous={};preserved=set()
        for taskname,(rh,sh) in PRIOR.items():
            prior=work/taskname;pr=read(bind(prior/'run-v1/receipt.json',rh));ps=read(bind(prior/'score-v1/receipt.json',sh))
            old=load(bind(prior/'run-v1/predictions.npz',pr['outputs']['predictions.npz']))
            preserved|=prior_arrays(p,old,'mz67/' if taskname.startswith('mz69') else '');del old
            previous[taskname[:4]]=read(bind(prior/'score-v1/result.json',ps['outputs']['result.json']))
            if taskname.startswith('mz68'):
                prior_schedule=load(bind(prior/'run-v1/schedule.npz',pr['outputs']['schedule.npz']))
                prior_groups=read(bind(prior/'run-v1/groups.json',pr['outputs']['groups.json']))
                for key,value in prior_groups.items():assert groups[key]==value,key
        records={'mz48':groups['records'],**{c:groups[c+'_records'] for c in ('mz55','mz61','mz67')}}
        schedule=load(rd/'schedule.npz');schedule_audit=check_schedule(schedule,prior_schedule,records)
        parity=read(rd/'initialization-parity.json')
        assert parity['status']=='PASS' and parity['unique_train_frames']==32 and parity['baseline_cohort_replays']==0
        assert parity['atol']==2e-5 and parity['rtol']==1e-6
        for c in ('mz61','mz67'):np.testing.assert_array_equal(parity['selected'][c],schedule[c+'_fit_ids'][:16])
        assert len(parity['comparisons'])==16 and {(x['cohort'],x['profile'],x['arm']) for x in parity['comparisons']}=={(c,p,a) for c in ('mz61','mz67') for p in SOURCE_PROFILES for a in ARMS}
        for arm in ARMS:
            log=read(rd/(arm+'-training-steps.json'));assert log['status']=='PASS' and len(log['steps'])==4096 and log['profile_counts']=={p:1024 for p in SOURCE_PROFILES}
            for i,row in enumerate(log['steps']):
                assert row['step']==i+1 and row['profile']==SOURCE_PROFILES[i%4] and row['null_anchor_zero_checked']==(i%4==3)
                assert row['geometry_mz67']==(0 if arm=='CONTROL' else int(schedule['DIVERSE_source'][i].sum()))
                if i%4==3:assert row['native_valid_slots']==row['old_valid_slots']==0
        labels={};packets={}
        for c,rec in records.items():
            ref=rr['source_info']['label_refs'][c];lab=load(bind(ref['path'],ref['sha256']));labels[c]=lab
            np.testing.assert_array_equal(lab['frame_ids'],p[c+'/frame_ids']);np.testing.assert_array_equal(lab['frame_ids'],[r['frame_id'] for r in rec])
            counts,vc=lab['fullframe_event_counts'],lab['valid_counts'];n=len(rec)
            assert counts.shape==(n,45,80,4) and vc.shape==(n,45,80) and counts.dtype==vc.dtype==np.uint8
            assert (counts<=vc[...,None]).all() and (vc<=64).all()
            np.testing.assert_array_equal(counts.sum((1,2))>=3,p[c+'/truth'])
        source=work/'mz67-topology-source-20260911';si=read(bind(source/'source-index.json',SOURCE67_SHA))
        assert si['status']=='COMPLETE' and si['frames']==4096 and si['roles']==dict(TRAIN_CANDIDATE=2048,CALIBRATION=1024,HELDOUT_GEOMETRY=1024)
        ref=si['combined']['evaluator.npz'];ev=load(bind(source/ref['path'],ref['sha256']))
        for k in ('frame_ids','truth','known'):np.testing.assert_array_equal(p['mz67/'+k],ev[k])
        preserved.update(('mz67/truth','mz67/known'));assert sorted(rr['baseline_arrays_preserved'])==sorted(preserved)
        g48={k:np.array(v,int) for k,v in groups['groups'].items()};g48['nonfit']=np.sort(np.r_[g48['heldout_site'],g48['nonfit_family']])
        g55={role:np.array([i for i,r in enumerate(records['mz55']) if r['role']==role]) for role in ('TRAIN_CANDIDATE','CALIBRATION','HELDOUT_SITE')}
        g55.update(heldout_new_family=np.array([i for i in g55['HELDOUT_SITE'] if records['mz55'][i]['family']!='retained_rod']),nonfit=np.sort(np.r_[g55['CALIBRATION'],g55['HELDOUT_SITE']]))
        parts={c:partitions(rec,g48 if c=='mz48' else g55 if c=='mz55' else None) for c,rec in records.items()}
        for c in ('mz61','mz67'):
            assert [len(parts[c][r]) for r in ('TRAIN_CANDIDATE','CALIBRATION','HELDOUT_GEOMETRY')]==[2048,1024,1024]
            st=work/('mz61-geometry-source-20260911' if c=='mz61' else 'mz67-topology-source-20260911')
            idx=read(bind(st/'source-index.json',SOURCE67_SHA if c=='mz67' else '6f9e4be717b7fe23d66af58a9002a667d7acff304ec0c0af58497dfc502c72c7'))
            ref=idx['combined']['packets.npz'];packets[c]=load(bind(st/ref['path'],ref['sha256']))
            check_geometry({k[len(c)+1:]:v for k,v in p.items() if k.startswith(c+'/geometry/')})
        for name in ('crop_mask','sensor_coverage','rays'):np.testing.assert_array_equal(p['mz70/'+name],p['mz64/'+name])
        cuts={};authority={};cal=g48['calibration'];assert len(cal)==256
        t=np.concatenate([p['DEV/truth'],p['mz48/truth'][cal]]);kn=np.concatenate([p['DEV/known'],p['mz48/known'][cal]])
        for key,arm in zip(KEYS,ARMS):
            def calibration(name):return np.concatenate([p['DEV/DROP_CLOSE/'+name],p['mz48/DROP_CLOSE/'+name][cal]])
            cut=np.load(rd/(arm+'-cutoff.npy'),allow_pickle=False)
            rebuilt,auth=scalar_cutoff(calibration(key+'/raw'),calibration(key+'/support'),calibration('MZ37'),t,kn)
            np.testing.assert_array_equal(rebuilt,cut);assert np.isfinite(cut).all();cuts[key]=cut;authority[key]=auth
        scores={};changes={};attributes={};margins={};events={};contexts={};rankings={};flags={};cache={}
        scalar=prior_rows=native_count=0
        for c in COHORTS+('mz67',):
            truth,known,ids=(p[c+'/'+n] for n in ('truth','known','frame_ids'))
            if c=='mz36':truth,known,ids=(p[c+'/attempted_'+n] for n in ('truth','known','frame_ids'));assert int((~known).sum())==80
            partition=parts.get(c,{'all':np.arange(len(truth))})
            for dic in (scores,changes,attributes,margins,events,contexts,rankings,flags):dic[c]={}
            for profile in (SOURCE_PROFILES if c in ('mz61','mz67') else PROFILES):
                prefix=c+'/'+profile+'/'
                before=previous['mz69']['profiles'][profile] if c=='mz67' else previous['mz68']['conditions'][c][profile]
                oldmethods=tuple(before['all']['methods'] if c=='mz67' else before['all'])
                assert set(before)<=set(partition),c
                a={m:p[prefix+m] for m in oldmethods} if c=='mz67' else reconstruct68(p,prefix,oldmethods)
                for key in KEYS:
                    scalar+=check_new(p,prefix,key,cuts[key]);a[key+'/candidate']=p[prefix+key+'/candidate'];a[key+'/UNION']=p[prefix+key+'/UNION']
                if c in packets:check_packets(packets[c],{n:p[prefix+n] for n in ('ranges','valid')},profile)
                if c=='mz36':
                    for m,v in list(a.items()):a[m]=np.full((400,4),np.nan);a[m][p['mz36/admitted_index']]=v
                comparators=tuple(x for x in (BASE,G,NULL,'MZ66/PROJECT/UNION','MZ57/GLOBAL_ANCHOR/UNION',KEYS[0]+'/UNION',KEYS[1]+'/UNION') if x in a)
                for dic in (scores,changes,attributes,margins,events,contexts,rankings,flags):dic[c][profile]={}
                for group,ii in partition.items():
                    aa={m:v[ii] for m,v in a.items()};rows={m:metrics(v,truth[ii],known[ii]) for m,v in aa.items()};scores[c][profile][group]=rows
                    if group in before:
                        prior=before[group]['methods'] if c=='mz67' else before[group]
                        for m in oldmethods:assert_metric_equal(rows[m],prior[m]);prior_rows+=1
                    changes[c][profile][group]={m:{ref:paired(aa[m],aa[ref],truth[ii],known[ii]) for ref in comparators if ref!=m} for m in NEW}
                for m in NEW:assert_metric_equal(scalar_metrics(a[m],truth,known),scores[c][profile]['all'][m])
                cache.setdefault(profile,{})[c]=(a,truth,known)
                winners={};details={}
                if c in labels:
                    lab=labels[c]
                    for key in KEYS:
                        w,n,wk=native_masks(p,prefix,key,lab['fullframe_event_counts'],lab['valid_counts']);native_count+=n.size
                        winners[key]=dict(native=n,known=wk)
                        details[key]=(w,n,wk,p[prefix+key+'/raw'],cuts[key],p[prefix+key+'/support'],p[prefix+'MZ37'])
                        for group,ii in partition.items():
                            attributes[c][profile].setdefault(group,{})[key]={ref:compact_native(p,prefix,key,(w,n,wk),a,truth,known,cuts[key],ii,ref) for ref in comparators}
                            margins[c][profile].setdefault(group,{})[key]=summarize_margin(p[prefix+key+'/raw'],cuts[key],truth,known,n,ii)
                for method in NEW:
                    key=method.rsplit('/',1)[0]
                    events[c][profile][method]={ref:event_rows(a,truth,known,ids,method,ref,records.get(c),details.get(key)) for ref in comparators if ref!=method}
                if c in records:
                    pairs=defaultdict(list)
                    for i,r in enumerate(records[c]):pairs[r['pair_id']].append(i)
                    assert all(len(v)==2 for v in pairs.values());pi=np.array(list(pairs.values()));np.testing.assert_array_equal(truth[pi[:,0]],truth[pi[:,1]])
                    for m in tuple(oldmethods)+NEW:
                        changed=(a[m][pi[:,0]]>=0)!=(a[m][pi[:,1]]>=0)
                        by_role={}
                        for role in sorted({r['role'] for r in records[c]}):
                            mask=np.array([records[c][i]['role']==role for i in pi[:,0]]);by_role[role]=dict(pairs=int(mask.sum()),changed_pairs=int(changed[mask].any(1).sum()),changed_queries=changed[mask].sum(0).tolist())
                        contexts[c][profile][m]=dict(pairs=len(pi),changed_pairs=int(changed.any(1).sum()),changed_queries=changed.sum(0).tolist(),by_role=by_role,pair_ids=[k for k,v in zip(pairs,changed) if v.any()])
                if c in ('mz61','mz67'):
                    held=partition['HELDOUT_GEOMETRY'];flags[c][profile]=descriptive(a,truth,known,held,winners)
                    for role in ('TRAIN_CANDIDATE','HELDOUT_GEOMETRY'):
                        ii=partition[role];rankings[c][profile][role]=dict(matched=common_ranking(p,prefix,truth,known,ii),historical=common_ranking(p,prefix,truth,known,ii,(G[:-6],NULL[:-6])))
        aggregates={};aggregate_changes={}
        for profile in PROFILES:
            data=cache[profile];legacy=[(c,np.arange(len(data[c][1]))) for c in ('relation10000','distance5000','rich','mz36')]
            selections=dict(legacy_noncal=legacy,mz48_fit=[('mz48',g48['fit'])],mz48_nonfit=[('mz48',g48['nonfit'])])
            selections['all_old_noncal']=legacy+selections['mz48_fit']+selections['mz48_nonfit'];selections['mz55_held640']=[('mz55',parts['mz55']['HELDOUT_SITE'])]
            aggregates[profile]={};aggregate_changes[profile]={}
            for group,sel in selections.items():
                t=np.concatenate([data[c][1][ii] for c,ii in sel]);k=np.concatenate([data[c][2][ii] for c,ii in sel]);methods=tuple(data[sel[0][0]][0])
                aa={m:np.concatenate([data[c][0][m][ii] for c,ii in sel]) for m in methods}
                aggregates[profile][group]={m:metrics(v,t,k) for m,v in aa.items()}
                oldrows=previous['mz68']['aggregates'][profile].get(group)
                if oldrows:
                    for m,row in oldrows.items():assert_metric_equal(aggregates[profile][group][m],row);prior_rows+=1
                aggregate_changes[profile][group]={m:{ref:paired(aa[m],aa[ref],t,k) for ref in (BASE,G,NULL,KEYS[0]+'/UNION',KEYS[1]+'/UNION') if ref!=m} for m in NEW}
        result=dict(status='PASS',claim=CLAIM,conditions=scores,comparisons=changes,aggregates=aggregates,aggregate_comparisons=aggregate_changes,
            attribution=attributes,margins=margins,ranking=rankings,support_context_pairs=contexts,descriptive=flags,
            primary=flags['mz67']['ALL_INVALID'],cutoffs={k:v.tolist() for k,v in cuts.items()},cutoff_authority=authority,schedule_audit=schedule_audit,
            limitations='Consumed controlled sources. Native winner lookup does not recompute dense argmax or prove feature causality. AP groups exact ties; AUC gives half tie credit; no-positive/negative is NOT_EVALUABLE. Ranking uses matched common support with exclusions retained. No new cutoff, hardware, safety, promotion or trained-ceiling claim.')
        audit=dict(status='PASS',baseline_arrays_byte_exact=len(preserved),prior_metric_rows_exact=prior_rows,scalar_candidate_union_values=scalar,native_lookups=native_count,
            original_calibration_rows=1256,cutoff_vectors_rebuilt=2,mz36_UNKNOWN_bits=80,local_UNKNOWN_cells={c:int((lab['valid_counts']==0).sum()) for c,lab in labels.items()},dense_argmax_recomputed=False,new_fits=0,new_inference=0,new_cutoff_searches=0)
        for q,h in inputs.items():assert sha(q)==h,q
        write(out/'result.json',result);write(out/'audit.json',audit);write(out/'paired-events.json',events)
        report(out/'report.md',result)
        write(out/'receipt.json',dict(status='PASS',inputs=inputs,outputs={name:sha(out/name) for name in ('result.json','audit.json','paired-events.json','report.md')},seconds=time.perf_counter()-start,backend='FROZEN_PROTOCOL_CPU_ONLY',new_fit=0,new_inference=0,new_cutoff=0))
        print('PASS',result['primary'])
    except BaseException:write(out/'failure.json',dict(status='FAIL',inputs=inputs,error=traceback.format_exc()));raise


def report(path,r):
    lines=['# MZ70 matched source learning','',r['claim'],'','No overall retention gate or selected winner.','','| MZ67 HELD ALL_INVALID | CONTROL TP/FP/FN | DIVERSE TP/FP/FN | TP gain/loss | FP added/removed | Native gained/lost |','|---|---|---|---|---|---|']
    x=r['primary']
    for q,name in enumerate(QUERIES):
        lines.append('| '+name+' | '+' / '.join(str(x['CONTROL'][k][q]) for k in ('tp','fp','fn'))+' | '+' / '.join(str(x['DIVERSE'][k][q]) for k in ('tp','fp','fn'))+' | '+f"{x['exchanges']['TP_GAIN'][q]} / {x['exchanges']['TP_LOST'][q]} | {x['exchanges']['FP_ADDED'][q]} / {x['exchanges']['FP_REMOVED'][q]} | {x['gained_winner']['native'][q]} / {x['lost_winner']['native'][q]} |")
    lines+=['',r['limitations'],'','All old/new cohorts, original methods, four source profiles, three legacy profiles, native/known-wrong/UNKNOWN splits, exact exchange IDs, support pairs and unselected rankings are retained in result.json and paired-events.json.']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def selftest():
    y=np.array([True,True,False,False]);s=np.array([2.,1.,1.,0.]);v=ranking(s,y)
    assert v['roc_auc']==.875 and abs(v['average_precision']-5/6)<1e-15
    assert ranking(np.zeros(4),y)['roc_auc']==ranking(np.zeros(4),y)['average_precision']==.5
    assert ranking(s,np.zeros(4,bool))['status']=='NOT_EVALUABLE'
    raw=np.zeros((1256,4),np.float32);support=np.ones_like(raw,bool);base=np.full_like(raw,-1.);truth=np.zeros_like(raw,bool);known=np.ones_like(raw,bool)
    raw[8]=2.;truth[8]=True;raw[9]=1.
    cut,auth=scalar_cutoff(raw,support,base,truth,known);np.testing.assert_array_equal(cut,np.full(4,np.nextafter(1.,np.inf)))
    assert all(a['maximum_rows']==[9] for a in auth)
    before=np.array([[-1.,1.,-1.,1.]]);after=-before;t=np.array([[True,True,False,False]]);k=np.ones_like(t)
    masks=exchange_masks(after,before,t,k);assert {a:v.sum(0).tolist() for a,v in masks.items()}==scalar_exchange(after,before,t,k)
    assert masks['TP_GAIN'][0,0] and masks['TP_LOST'][0,1] and masks['FP_ADDED'][0,2] and masks['FP_REMOVED'][0,3]
    win={'native':np.array([[True,False,False,False]]),'known':np.array([[True,True,False,False]])}
    z=split(np.ones_like(t),win);assert z['native']==[1,0,0,0] and z['known_wrong']==[0,1,0,0] and z['unknown']==[0,0,1,1]
    # Both head contracts and eligibility disagreement are exercised without files.
    pre='mz67/ALL_INVALID/';base=np.full((3,4),-1.);base[0,0]=1.
    p={pre+'MZ37':base,pre+BASE:base.copy(),pre+'MZ64/GEOMETRY/anchor_available':np.zeros(3,bool)}
    for j,key in enumerate(KEYS):
        raw=np.broadcast_to(np.arange(3.)[:,None]-1,(3,4)).copy();sup=np.ones((3,4),bool)
        if j:sup[1,1]=False
        candidate=np.where((base<0)&sup&(raw>=0),raw,base)
        fields=dict(raw=raw,support=sup,winner=np.zeros((3,4),int),anchor_vector=np.zeros((3,8)),anchor_available=np.zeros(3,bool),candidate=candidate,UNION=np.maximum(base,candidate))
        p.update({pre+key+'/'+k:v for k,v in fields.items()});assert check_new(p,pre,key,np.zeros(4))==24
    tr=np.zeros((3,4),bool);tr[2]=True;kn=np.ones_like(tr);kn[0,2]=False
    rank=common_ranking(p,pre,tr,kn,np.arange(3))
    assert rank['BODY_FAR']['support_difference_indices']==[1] and rank['BODY_FAR']['eligible']==2
    assert rank['HEAD_NEAR']['excluded_unknown']==1 and rank['BODY_NEAR']['excluded_mz37_positive']==1
    p[pre+KEYS[0]+'/candidate']=p[pre+KEYS[0]+'/candidate'].copy();p[pre+KEYS[0]+'/candidate'][2,0]+=1
    try:check_new(p,pre,KEYS[0],np.zeros(4))
    except AssertionError:pass
    else:raise AssertionError('Corrupt candidate accepted')
    print('PASS: tied rankings, absent class, exact cutoff authority, exchanged bits, native/known-wrong/UNKNOWN partition, both-head scalar contracts, support/UNKNOWN exclusions, corruption rejection')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--task',type=Path);parser.add_argument('--selftest',action='store_true');args=parser.parse_args()
    if args.selftest:selftest()
    else:
        assert args.task;run(args.task if args.task.is_absolute() else args.root/args.task)
