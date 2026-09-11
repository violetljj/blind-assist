"""Independent MZ66 saved-output score; no model execution or cutoff search."""
import argparse
from pathlib import Path
import time
import traceback
from collections import defaultdict
import numpy as np
from mz58_score import read,write,sha,load,metrics,scalar_metrics,paired
from mz59_score import scalar_cutoff,validate_candidate,native_masks,assert_metric_equal
from mz64_score import partitions,compact_native,event_rows,summarize_margin,reconstruct as reconstruct64
from mz64_score import COHORTS,PROFILES,SOURCE_PROFILES,QUERIES,INITIAL_SHA
from mz63_score import check_packets,check_geometry

KEY='MZ66/PROJECT'
NEW=(KEY+'/candidate',KEY+'/UNION')
BASE='OLD_NEG/UNION'
REFERENCE='MZ64/GEOMETRY/UNION'
COMPARATORS=(BASE,'MZ57/GLOBAL_ANCHOR/UNION','MZ62/CONTROL/UNION','MZ64/CONTROL/UNION',REFERENCE)
OLD_GROUPS=('legacy_noncal','mz48_nonfit','mz55_held640')
PRIOR_RUN_SHA='3e917c4e2d6d56f134ab52633e572ae528e75487bc68405d2ce5a6a247d2508e'
PRIOR_SCORE_SHA='b3329f1a4155a60962baaeb906aa03a265fe5e460db9d0d9e8ae6198377cd116'
FROZEN_DEFINITION=('PROJECT versus sealed MZ64 GEOMETRY: total TP non-decreasing and total FP non-increasing in each of legacy_noncal, mz48_nonfit and mz55_held640, with aggregate old FP strictly reduced; on MZ61 DROP held1024 total TP non-decreasing, zero newly false bits, native BODY_NEAR additions beyond OLD_NEG non-decreasing, and held-awning native BODY_NEAR additions non-decreasing. Eleven clauses; report all effects and costs.')


def gate(old,held,truth,known,native,awning):
    clauses={};values={'old':{}};ref_fp=new_fp=0
    assert set(old)==set(OLD_GROUPS)
    for group,rows in old.items():
        actual,ref=rows[NEW[1]],rows[REFERENCE]
        clauses[group+'_tp_retained']=sum(actual['tp'])>=sum(ref['tp'])
        clauses[group+'_fp_not_increased']=sum(actual['fp'])<=sum(ref['fp'])
        ref_fp+=sum(ref['fp']);new_fp+=sum(actual['fp'])
        values['old'][group]={'PROJECT':actual,'MZ64_GEOMETRY':ref}
    clauses['summed_old_fp_strictly_lower']=new_fp<ref_fp
    actual,ref=(metrics(held[k],truth,known) for k in (NEW[1],REFERENCE))
    changes=paired(held[NEW[1]],held[REFERENCE],truth,known)
    clauses['held_total_tp_retained']=sum(actual['tp'])>=sum(ref['tp'])
    clauses['held_no_new_fp_bits']=sum(changes['fp_added'])==0
    clauses['held_native_body_near_retained']=int(native[KEY])>=int(native[REFERENCE[:-6]])
    clauses['held_awning_native_body_near_retained']=int(awning[KEY])>=int(awning[REFERENCE[:-6]])
    values.update(old_fp=dict(PROJECT=new_fp,MZ64_GEOMETRY=ref_fp),held=dict(PROJECT=actual,MZ64_GEOMETRY=ref,paired=changes),native_bn=native,awning_native_bn=awning)
    assert len(clauses)==11
    return dict(definition=FROZEN_DEFINITION,clauses=clauses,values=values,passes_primary=all(clauses.values()))


def reconstruct(p,prefix,oldmethods):
    a=reconstruct64(p,prefix,oldmethods)
    a[NEW[0]]=p[prefix+NEW[0]];a[NEW[1]]=np.maximum(a[BASE],a[NEW[0]])
    np.testing.assert_array_equal(a[NEW[1]],p[prefix+NEW[1]])
    return {k:a[k] for k in tuple(oldmethods)+NEW}


def projection_audit(log):
    """Check saved scalar bookkeeping; full parameter deltas are not saved."""
    rows=log['steps'];assert log['status']=='PASS' and len(rows)==1536
    for i,row in enumerate(rows):
        assert row['step']==i+1 and row['condition']==PROFILES[i%3]
        for key in ('actual_assigned_dot','post_dot','ideal_dot','residual_bound','allowed_rounding_bound','rounding_bound','arithmetic_roundoff_bound','negative','post_negative'):
            assert np.isfinite(row[key]),key
        assert row['rounding_bound']>=0 and row['arithmetic_roundoff_bound']>=0
        assert row['allowed_rounding_bound']==row['residual_bound']
        assert row['residual_bound']==np.nextafter(max(row['ideal_dot'],0.)+row['rounding_bound']+row['arithmetic_roundoff_bound'],np.inf)
        assert row['actual_assigned_dot']<=row['allowed_rounding_bound']
        assert abs(row['actual_assigned_dot']-row['post_dot'])<=row['arithmetic_roundoff_bound']
    projected=sum(bool(x['projected']) for x in rows);increases=sum(x['post_negative']>x['negative'] for x in rows)
    assert log['projected_steps']==projected and log['nonlinear_replay_increases']==increases
    return dict(steps=1536,projected_steps=projected,nonlinear_replay_increases=increases,
        maximum_nonlinear_increase=max(x['post_negative']-x['negative'] for x in rows),
        maximum_actual_dot_minus_bound=max(x['actual_assigned_dot']-x['allowed_rounding_bound'] for x in rows),
        parameter_dot_independently_recomputed=False,scope='Saved scalar identities/bounds; nonlinear loss descriptive, not a gate')


def vector_audit(vectors,log):
    """Independent NumPy FP64 closed form; at most one FP32 ULP plus propagated FP64 error."""
    expected=np.linspace(0,1535,16,dtype=np.int64)+1
    np.testing.assert_array_equal(vectors['step'],expected);assert vectors['step'].dtype==np.int64
    for name in ('before','proposed','gradient','assigned'):
        assert vectors[name].shape==(16,11020) and vectors[name].dtype==np.float32 and np.isfinite(vectors[name]).all()
    gamma=(4*11020+32)*2.**-53;gamma=gamma/(1-gamma);rows=[]
    for i,step in enumerate(expected):
        row=log['steps'][int(step)-1]
        b,a,g,x=(vectors[name][i].astype(np.float64) for name in ('before','proposed','gradient','assigned'))
        pre=float(np.dot(g,a-b));norm=float(np.dot(g,g));scale=float(np.sum(np.abs(g*(a-b))))
        pre_error=2*gamma*scale;norm_error=2*gamma*norm
        assert abs(pre-row['pre_dot'])<=pre_error
        assert abs(norm-row['gradient_norm_squared'])<=norm_error
        if abs(pre)>pre_error:assert row['projected']==bool(pre>0 and norm>0)
        if not row['projected']:
            assert vectors['assigned'][i].tobytes()==vectors['proposed'][i].tobytes()
            max_error=0.;max_bound=0.
        else:
            assert norm>norm_error
            coefficient=pre/norm;ideal=a-coefficient*g;expected32=ideal.astype(np.float32)
            coefficient_error=(pre_error+abs(coefficient)*norm_error)/(norm-norm_error)
            double_error=coefficient_error*np.abs(g)+2*np.finfo(np.float64).eps*(np.abs(a)+np.abs(coefficient*g))
            ulp=np.abs(np.spacing(expected32)).astype(np.float64)
            bound=ulp+double_error;error=np.abs(x-expected32.astype(np.float64))
            assert np.all(error<=bound)
            max_error=float(error.max());max_bound=float(bound.max())
        post=float(np.dot(g,x-b));post_error=2*gamma*float(np.sum(np.abs(g*(x-b))))
        assert abs(post-row['actual_assigned_dot'])<=post_error
        assert post<=row['allowed_rounding_bound']+post_error
        rows.append(dict(step=int(step),projected=bool(row['projected']),numpy_actual_dot=post,dot_error_bound=post_error,
            assigned_max_abs_error=max_error,assigned_max_bound=max_bound))
    return dict(status='PASS',steps=16,dimensions=11020,rows=rows,
        tolerance='Nonprojected exact bytes; projected one FP32 ULP plus explicitly propagated FP64 reduction error; no output-score tolerance',
        parameter_dot_independently_recomputed=True)


def run(task):
    task=task.resolve(strict=True);rd=task/'run-v1';out=task/'score-v1';work=task.parent
    assert not out.exists();out.mkdir();inputs={};start=time.perf_counter()
    try:
        def bind(path,expected=None):
            path=Path(path).resolve(strict=True);h=sha(path);assert expected is None or expected==h,str(path);inputs[str(path)]=h;return path
        for name in ('mz66_score.py','mz64_score.py','mz63_score.py','mz62_score.py','mz59_score.py','mz58_score.py'):bind(Path(__file__).with_name(name))
        rr=read(bind(rd/'receipt.json'));assert rr['status']=='PASS'
        def producer(path):
            path=Path(path).resolve(strict=True);hits=[h for q,h in rr['inputs'].items() if Path(q).resolve()==path]
            assert len(hits)==1,str(path);return bind(path,hits[0])
        definition=read(producer(task/'primary-definition.json'))
        assert definition==rr['primary_definition'] and definition['definition']==FROZEN_DEFINITION
        assert definition['status']=='FROZEN_BEFORE_FIT' and not definition['new_inference_or_fit_started']
        assert definition['baseline']==REFERENCE and definition['candidate']==NEW[1]
        assert definition['old_groups']==list(OLD_GROUPS) and definition['primary_profile']=='DROP_CLOSE'
        assert definition['primary_frames']==1024 and definition['native_addition_baseline']==BASE
        assert definition['initial_checkpoint_sha256']==INITIAL_SHA
        assert rr['steps_per_arm']==rr['total_steps']==1536 and rr['trainable_parameters']==11020
        assert rr['fits']['PROJECT']['steps']==1536 and set(rr['fits'])=={'PROJECT'}
        assert rr['initial_checkpoint_sha256']==INITIAL_SHA and rr['exact_initial_weights']
        assert rr['original_calibration_frames']==1256 and rr['new_cutoffs']==1 and rr['threshold_searches']==0
        assert rr['positive_pool']=='original_OPEN_max' and rr['loss_contract']=='original_MZ59_balanced_local_frame_loss'
        assert rr['original_baseline_cohort_inferences']==rr['native_depth_reads']==rr['mz55_calibration_rows_used']==rr['mz61_calibration_rows_used']==0
        assert rr['all_invalid_training_presentations']==0 and rr['initial_train_parity_unique_frames']==32
        assert rr['source_unknown_preserved']
        assert rr['optimizer_change']==definition['optimizer_change'] and rr['post_replay_measurements']==1536
        for q,h in rr['inputs'].items():
            if q.endswith('.py') or q.endswith('.md'):bind(q,h)
        for name in ('predictions.npz','groups.json','schedule.npz','initialization-parity.json','initial-missing.json','learned-missing.json','PROJECT.pt','PROJECT-cutoff.npy'):bind(rd/name,rr['outputs'][name])
        for name in ('initial-missing.json','learned-missing.json'):
            missing=read(rd/name);assert missing['status']=='PASS' and missing['frames_per_arm']==8 and missing['all_missing_context_exactly_zero']
        projection_log=read(bind(rd/'projection-steps.json',rr['outputs']['projection-steps.json']))
        projection=projection_audit(projection_log)
        projection['independent_vectors']=vector_audit(load(bind(rd/'projection-vectors.npz',rr['outputs']['projection-vectors.npz'])),projection_log)
        prior=work/'mz64-geometry-learning-20260911'
        pr=read(bind(prior/'run-v1/receipt.json',PRIOR_RUN_SHA));ps=read(bind(prior/'score-v1/receipt.json',PRIOR_SCORE_SHA))
        previous=read(bind(prior/'score-v1/result.json',ps['outputs']['result.json']))
        old=load(bind(prior/'run-v1/predictions.npz',pr['outputs']['predictions.npz']));p=load(rd/'predictions.npz')
        assert sorted(rr['baseline_arrays_preserved'])==sorted(old)
        for k,v in old.items():assert p[k].dtype==v.dtype and p[k].shape==v.shape and p[k].tobytes()==v.tobytes(),k
        preserved=len(old);del old
        for name in ('groups.json','schedule.npz'):
            bind(prior/'run-v1'/name,pr['outputs'][name]);assert sha(rd/name)==pr['outputs'][name]
        groups=read(rd/'groups.json');schedule=load(rd/'schedule.npz')
        parity=read(rd/'initialization-parity.json');assert parity['status']=='PASS' and parity['unique_train_frames']==32
        assert parity['atol']==2e-5 and parity['rtol']==1e-6 and parity['full_baseline_cohort_replays']==0
        assert len(parity['comparisons'])==7 and {(x['cohort'],x['profile'],x['arm']) for x in parity['comparisons']}=={(c,pr,'PROJECT') for c in ('mz48','mz61') for pr in (PROFILES if c=='mz48' else SOURCE_PROFILES)}
        for cohort,key in (('mz48','mz48'),('mz61','GEOMETRY')):np.testing.assert_array_equal(parity['ids'][cohort],np.unique(schedule[key])[:16])
        records={'mz48':groups['records'],'mz55':groups['mz55_records'],'mz61':groups['mz61_records']}
        g48={k:np.array(v,int) for k,v in groups['groups'].items()};g48['nonfit']=np.sort(np.r_[g48['heldout_site'],g48['nonfit_family']])
        g55={role:np.array([i for i,x in enumerate(records['mz55']) if x['role']==role]) for role in ('TRAIN_CANDIDATE','CALIBRATION','HELDOUT_SITE')}
        g55.update(heldout_new_family=np.array([i for i in g55['HELDOUT_SITE'] if records['mz55'][i]['family']!='retained_rod']),nonfit=np.sort(np.r_[g55['CALIBRATION'],g55['HELDOUT_SITE']]))
        g61={role:np.array([i for i,x in enumerate(records['mz61']) if x['role']==role]) for role in ('TRAIN_CANDIDATE','CALIBRATION','HELDOUT_GEOMETRY')}
        assert [len(g61[x]) for x in g61]==[2048,1024,1024]
        parts={c:partitions(records[c],g) for c,g in [('mz48',g48),('mz55',g55),('mz61',g61)]}
        labels={}
        for cohort in records:
            ref=rr['source_info']['label_refs'][cohort];labels[cohort]=load(bind(ref['path'],ref['sha256']))
            np.testing.assert_array_equal(labels[cohort]['frame_ids'],p[cohort+'/frame_ids'])
            counts,known=labels[cohort]['fullframe_event_counts'],labels[cohort]['valid_counts'];n=len(records[cohort])
            assert counts.shape==(n,45,80,4) and known.shape==(n,45,80) and counts.dtype==known.dtype==np.uint8
            assert (counts<=known[...,None]).all() and (known<=64).all()
            np.testing.assert_array_equal(counts.sum((1,2))>=3,p[cohort+'/truth'])
        source=work/'mz61-geometry-source-20260911';si=read(producer(source/'source-index.json'))
        assert si['status']=='COMPLETE' and si['frames']==4096
        ref=si['combined']['packets.npz'];packets=load(bind(source/ref['path'],ref['sha256']))
        check_geometry({k[len('mz61/'):]:v for k,v in p.items() if k.startswith('mz61/geometry/')})
        for name in ('crop_mask','sensor_coverage','rays'):np.testing.assert_array_equal(p['mz66/'+name],p['mz64/'+name])
        cut=np.load(rd/'PROJECT-cutoff.npy',allow_pickle=False);cal=g48['calibration']
        def calibration(name):return np.concatenate([p['DEV/DROP_CLOSE/'+name],p['mz48/DROP_CLOSE/'+name][cal]])
        t=np.concatenate([p['DEV/truth'],p['mz48/truth'][cal]]);k=np.concatenate([p['DEV/known'],p['mz48/known'][cal]])
        rebuilt,authority=scalar_cutoff(calibration(KEY+'/raw'),calibration(KEY+'/support'),calibration('MZ37'),t,k)
        np.testing.assert_array_equal(rebuilt,cut);assert np.isfinite(cut).all()
        scores={};changes={};attributes={};margins={};events={};contexts={};cache={};awning={}
        scalar=prior_rows=native_lookups=0
        for cohort in COHORTS:
            truth,known,ids=(p[cohort+'/'+n] for n in ('truth','known','frame_ids'))
            if cohort=='mz36':truth,known,ids=(p['mz36/attempted_'+n] for n in ('truth','known','frame_ids'));assert int((~known).sum())==80
            partition=parts.get(cohort,{'all':np.arange(len(truth))})
            for dic in (scores,changes,attributes,margins,events,contexts):dic[cohort]={}
            for profile in (SOURCE_PROFILES if cohort=='mz61' else PROFILES):
                prefix=cohort+'/'+profile+'/';before=previous['conditions'][cohort][profile];oldmethods=tuple(before['all'])
                a=reconstruct(p,prefix,oldmethods)
                validate_candidate(*(p[prefix+KEY+'/'+n] for n in ('raw','support','winner','anchor_vector','anchor_available','candidate')),p[prefix+'MZ37'],cut)
                np.testing.assert_array_equal(p[prefix+KEY+'/anchor_available'],p[prefix+'MZ64/GEOMETRY/anchor_available'])
                if cohort=='mz61':check_packets(packets,{n:p[prefix+n] for n in ('ranges','valid')},profile)
                if cohort=='mz36':
                    for name,values in list(a.items()):a[name]=np.full((400,4),np.nan);a[name][p['mz36/admitted_index']]=values
                for dic in (scores,changes,attributes,margins,events,contexts):dic[cohort][profile]={}
                assert set(before)==set(partition)
                for group,ii in partition.items():
                    aa={m:v[ii] for m,v in a.items()};rows={m:metrics(v,truth[ii],known[ii]) for m,v in aa.items()}
                    scores[cohort][profile][group]=rows
                    for m in oldmethods:assert_metric_equal(rows[m],before[group][m]);prior_rows+=1
                    changes[cohort][profile][group]={m:{ref:paired(aa[m],aa[ref],truth[ii],known[ii]) for ref in COMPARATORS} for m in NEW}
                for m in NEW:assert scalar_metrics(a[m],truth,known)==scores[cohort][profile]['all'][m];scalar+=int(known.sum())
                cache.setdefault(profile,{})[cohort]=(a,truth,known)
                detail=None
                if cohort in labels:
                    lab=labels[cohort];winning=native_masks(p,prefix,KEY,lab['fullframe_event_counts'],lab['valid_counts']);native_lookups+=winning[1].size
                    for group,ii in partition.items():
                        attributes[cohort][profile][group]={ref:compact_native(p,prefix,KEY,winning,a,truth,known,cut,ii,ref) for ref in COMPARATORS}
                        margins[cohort][profile][group]=summarize_margin(p[prefix+KEY+'/raw'],cut,truth,known,winning[1],ii)
                    detail=(*winning,p[prefix+KEY+'/raw'],cut,p[prefix+KEY+'/support'],p[prefix+'MZ37'])
                    if cohort=='mz61' and profile=='DROP_CLOSE':
                        ii=np.array([i for i,x in enumerate(records[cohort]) if x['role']=='HELDOUT_GEOMETRY' and x['family']=='shallow_awning' and truth[i,0] and known[i,0]])
                        assert len(ii)==64;w,n,wk=winning
                        awning=dict(counts=compact_native(p,prefix,KEY,winning,a,truth,known,cut,ii,BASE),positive_frames=64,
                            rows=[dict(index=int(i),frame_id=str(ids[i]),native=bool(n[i,0]),local_known=bool(wk[i,0]),winner=int(w[i,0]),raw=float(p[prefix+KEY+'/raw'][i,0]),cutoff=float(cut[0]),candidate_positive=bool(a[NEW[0]][i,0]>=0),union_positive=bool(a[NEW[1]][i,0]>=0),baseline_positive=bool(a[BASE][i,0]>=0)) for i in ii])
                for method in NEW:events[cohort][profile][method]={ref:event_rows(a,truth,known,ids,method,ref,records.get(cohort),detail) for ref in COMPARATORS}
                if cohort in records:
                    pairs=defaultdict(list)
                    for i,x in enumerate(records[cohort]):pairs[x['pair_id']].append(i)
                    assert len(pairs)==(2048 if cohort=='mz61' else 1280) and all(len(v)==2 for v in pairs.values())
                    pi=np.array(list(pairs.values()));np.testing.assert_array_equal(truth[pi[:,0]],truth[pi[:,1]])
                    for method in COMPARATORS+NEW:
                        changed=(a[method][pi[:,0]]>=0)!=(a[method][pi[:,1]]>=0)
                        contexts[cohort][profile][method]=dict(pairs=len(pi),changed_pairs=int(changed.any(1).sum()),changed_queries=changed.sum(0).tolist(),pair_ids=[key for key,v in zip(pairs,changed) if v.any()])
        aggregates={};aggregate_changes={}
        for profile in PROFILES:
            data=cache[profile];legacy=[(c,np.arange(len(data[c][1]))) for c in ('relation10000','distance5000','rich','mz36')]
            selections=dict(legacy_noncal=legacy,mz48_fit=[('mz48',g48['fit'])],mz48_nonfit=[('mz48',g48['nonfit'])])
            selections['all_old_noncal']=legacy+selections['mz48_fit']+selections['mz48_nonfit'];aggregates[profile]={};aggregate_changes[profile]={}
            for group,sel in selections.items():
                t=np.concatenate([data[c][1][ii] for c,ii in sel]);k=np.concatenate([data[c][2][ii] for c,ii in sel]);oldrows=previous['aggregates'][profile][group]
                aa={m:np.concatenate([data[c][0][m][ii] for c,ii in sel]) for m in tuple(oldrows)+NEW}
                aggregates[profile][group]={m:metrics(v,t,k) for m,v in aa.items()}
                for m,row in oldrows.items():assert_metric_equal(aggregates[profile][group][m],row);prior_rows+=1
                aggregate_changes[profile][group]={m:{ref:paired(aa[m],aa[ref],t,k) for ref in COMPARATORS} for m in NEW}
        held=g61['HELDOUT_GEOMETRY'];a,t,k=cache['DROP_CLOSE']['mz61']
        native={KEY:attributes['mz61']['DROP_CLOSE']['HELDOUT_GEOMETRY'][BASE]['native_added_tp'][0],REFERENCE[:-6]:previous['attribution']['mz61']['DROP_CLOSE']['HELDOUT_GEOMETRY'][REFERENCE[:-6]][BASE]['native_added_tp'][0]}
        aw={KEY:awning['counts']['native_added_tp'][0],REFERENCE[:-6]:previous['awning_held64_DROP'][REFERENCE[:-6]]['counts']['native_added_tp'][0]}
        primary=gate(dict(legacy_noncal=aggregates['DROP_CLOSE']['legacy_noncal'],mz48_nonfit=aggregates['DROP_CLOSE']['mz48_nonfit'],mz55_held640=scores['mz55']['DROP_CLOSE']['HELDOUT_SITE']),{m:v[held] for m,v in a.items()},t[held],k[held],native,aw)
        result=dict(status='PASS',primary_gate=primary,conditions=scores,comparisons=changes,aggregates=aggregates,aggregate_comparisons=aggregate_changes,
            attribution=attributes,margins=margins,support_context_pairs=contexts,awning_held64_DROP=awning,cutoff=cut.tolist(),cutoff_authority=authority,
            projection=projection,source_role='CONSUMED_DEVELOPMENT',limitation='One1536-step candidate versus sealed identical-budget GEOMETRY. Fixed baseline arrays retained. Native winner labels do not establish feature causality; dense argmax not independently regenerated. First-order replay projection is not a held-scene or nonlinear-loss guarantee. No hardware or trained-ceiling claim.')
        audit=dict(status='PASS',baseline_arrays_byte_exact=preserved,prior_metric_rows_exact=prior_rows,scalar_known_decisions=scalar,native_winner_lookups=native_lookups,
            schedule_groups_byte_identical=True,original_calibration_rows=1256,cutoff_vectors_recomputed=1,mz36_UNKNOWN_bits=80,
            local_UNKNOWN_cells={c:int((v['valid_counts']==0).sum()) for c,v in labels.items()},new_fits=0,new_inference=0,new_threshold_searches=0,RGB_reads=0,native_depth_reads=0,dense_argmax_recomputed=False)
        for q,h in inputs.items():assert sha(q)==h,q
        for name,value in [('result.json',result),('audit.json',audit),('paired-events.json',events),('awning-held64-drop.json',awning)]:write(out/name,value)
        report(out/'report.md',result)
        write(out/'receipt.json',dict(status='PASS',inputs=inputs,outputs={n:sha(out/n) for n in ('result.json','audit.json','paired-events.json','awning-held64-drop.json','report.md')},code_sha256=sha(__file__),seconds=time.perf_counter()-start,backend='CPU saved-output independent score',new_fits=0,new_inference=0))
        print('PASS',primary)
    except BaseException:write(out/'failure.json',dict(status='FAIL',inputs=inputs,error=traceback.format_exc()));raise


def report(path,result):
    pg=result['primary_gate'];lines=['# MZ66 replay projection','',f'All-clause retaining result: **{"PASS" if pg["passes_primary"] else "FAIL"}**.','','| Clause | Result |','| --- | --- |']
    lines += [f'| {k} | {v} |' for k,v in pg['clauses'].items()]
    lines += ['','| DROP group | Method | TP | FP | FN |','| --- | --- | ---: | ---: | ---: |']
    for group,rows in [('MZ61 held1024',result['conditions']['mz61']['DROP_CLOSE']['HELDOUT_GEOMETRY']),('MZ55 held640',result['conditions']['mz55']['DROP_CLOSE']['HELDOUT_SITE']),('legacy noncal',result['aggregates']['DROP_CLOSE']['legacy_noncal']),('MZ48 nonfit',result['aggregates']['DROP_CLOSE']['mz48_nonfit'])]:
        for m in COMPARATORS+NEW:lines.append('| '+group+' | '+m+' | '+' | '.join(str(sum(rows[m][k])) for k in ('tp','fp','fn'))+' |')
    lines += ['',result['limitation'],'','All profiles, original and new methods, query/group counts, native/known-wrong/UNKNOWN splits and gain/loss IDs remain in the result and paired-event files. No choice of mode or threshold from outcomes.']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def self_test():
    t=np.zeros((4,4),bool);t[0,0]=True;k=np.ones_like(t);base=np.full((4,4),-1.);base[0,0]=1;base[1,0]=1
    project=base.copy();project[1,0]=-1
    held={NEW[1]:project,REFERENCE:base}
    old={g:{m:metrics(v,t,k) for m,v in held.items()} for g in OLD_GROUPS}
    native={KEY:30,REFERENCE[:-6]:30};awning={KEY:5,REFERENCE[:-6]:5}
    assert gate(old,held,t,k,native,awning)['passes_primary']
    unchanged={m:base for m in held};same={g:{m:metrics(v,t,k) for m,v in unchanged.items()} for g in OLD_GROUPS}
    assert not gate(same,unchanged,t,k,native,awning)['clauses']['summed_old_fp_strictly_lower']
    swap=base.copy();swap[1,0]=-1;swap[2,0]=1
    assert not gate(old,{NEW[1]:swap,REFERENCE:base},t,k,native,awning)['clauses']['held_no_new_fp_bits']
    assert not gate(old,held,t,k,{KEY:29,REFERENCE[:-6]:30},awning)['passes_primary']
    raw=np.arange(5024,dtype=np.float32).reshape(1256,4);cut,_=scalar_cutoff(raw,np.ones_like(raw,bool),-np.ones_like(raw),np.zeros_like(raw,bool),np.ones_like(raw,bool))
    np.testing.assert_array_equal(cut,np.nextafter(raw[-1].astype(float),np.inf))
    assert 'torch' not in __import__('sys').modules
    return dict(status='PASS',checks=['11-clause improving case','unchanged FP strict fail','same-count false-bit swap fail','native retention fail','exact1256 cutoff'],new_models=0)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path('E:/linnan/linnan'));ap.add_argument('--task',type=Path);ap.add_argument('--self-test',action='store_true');ap.add_argument('--receipt',type=Path)
    args=ap.parse_args()
    if args.self_test:
        assert args.task is None;answer=dict(code_sha256=sha(__file__),synthetic=self_test())
        if args.receipt:write(args.receipt,answer)
        print(answer)
    else:
        assert args.task is not None;run(args.task if args.task.is_absolute() else args.root/args.task)
