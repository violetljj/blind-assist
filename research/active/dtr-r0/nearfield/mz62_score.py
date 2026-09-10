"""Independent MZ62 saved-output comparison; no image, depth or model execution."""
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
from mz58_score import read, write, sha, load, metrics, scalar_metrics, paired
from mz59_score import scalar_cutoff, validate_candidate, native_masks, assert_metric_equal

ARMS=('CONTROL','COVERAGE')
KEYS=tuple('MZ62/'+a for a in ARMS)
NEW=tuple(k+s for k in KEYS for s in ('/candidate','/UNION'))
BASE='OLD_NEG/UNION'
PROFILES=('IDEAL','MERGE_CLOSE','DROP_CLOSE')
COHORTS=('DEV','relation10000','distance5000','rich','mz36','mz48','mz55')
QUERIES=('BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR')
SCHEDULE_SHA='9424f5e85ef985ab999541cf75f8c4725cbff5237e0245bf4fa1b5946867d672'
FROZEN_DEFINITION=('COVERAGE versus matched1200-step CONTROL: more native-winning BODY_NEAR TP newly added over OLD_NEG/UNION on MZ55 new-family held480 DROP; final OLD_NEG/UNION OR candidate TP no lower and FP no higher on MZ55 held640, legacy noncal and MZ48 nonfit. All seven clauses must pass; original calibration only, no outcome-driven threshold search.')


def gate(native, held, legacy, nonfit):
    c,d=(k+'/UNION' for k in KEYS)
    values={'control_native_body_near':int(native[KEYS[0]][0]),'coverage_native_body_near':int(native[KEYS[1]][0])}
    clauses={'more_native_body_near':values['coverage_native_body_near']>values['control_native_body_near']}
    for name,rows in [('held640',held),('legacy_noncal',legacy),('mz48_nonfit',nonfit)]:
        for field in ('tp','fp'):
            values[name+'_'+field]={a:int(sum(rows[k][field])) for a,k in zip(ARMS,(c,d))}
        clauses[name+'_tp_retained']=sum(rows[d]['tp'])>=sum(rows[c]['tp'])
        clauses[name+'_fp_not_added']=sum(rows[d]['fp'])<=sum(rows[c]['fp'])
    return dict(definition=FROZEN_DEFINITION,values=values,clauses=clauses,passes_primary=all(clauses.values()))


def validate_schedule(s, old, train):
    assert set(ARMS).issubset(s) and len(train)==1600
    np.testing.assert_array_equal(s['mz48'],np.tile(old['shared'][:,:4],(2,1)))
    for k in ('OLD_NEG','query'):np.testing.assert_array_equal(s[k],np.tile(old[k],(2,1)))
    np.testing.assert_array_equal(s['profile_index'],np.arange(1200)%3)
    for arm in ARMS:
        assert s[arm].shape==(1200,4)
        ids,count=np.unique(s[arm],return_counts=True)
        np.testing.assert_array_equal(ids,train);np.testing.assert_array_equal(count,np.full(1600,3))
    for profile in range(3):
        np.testing.assert_array_equal(np.sort(s['COVERAGE'][s['profile_index']==profile].flatten()),train)
    return dict(steps_per_arm=1200,mz55_presentations_per_arm=4800,each_id_per_arm=3,
        coverage_each_profile_unique=1600,old_replay_exact_twice=True,
        control_profile_unique=[len(np.unique(s['CONTROL'][s['profile_index']==i])) for i in range(3)])


def reconstruct(p,prefix,refs):
    a={k:p[prefix+k] for k in refs if not k.endswith('/UNION')}
    for k in refs:
        if not k.endswith('/UNION'):continue
        family=k[:-6]
        if family in ('MZ50','NEW_NEG','OLD_NEG'):
            lead='' if family=='MZ50' else family+'/'
            a[k]=np.maximum(a[lead+'OPEN/candidate'],a[lead+'GATED/candidate'])
        else:
            candidate=family.replace('MZ57/','MZ56/')+'/candidate'
            a[k]=np.maximum(a[BASE],a[candidate])
        if prefix+k in p:np.testing.assert_array_equal(a[k],p[prefix+k])
    for key in KEYS:
        a[key+'/candidate']=p[prefix+key+'/candidate']
        a[key+'/UNION']=np.maximum(a[BASE],a[key+'/candidate'])
        np.testing.assert_array_equal(a[key+'/UNION'],p[prefix+key+'/UNION'])
    return a


def run(task):
    task=task.resolve(strict=True);rd=task/'run-v1';out=task/'score-v1';work=task.parent
    assert not out.exists();out.mkdir();inputs={};start=time.perf_counter()
    try:
        def bind(path,expected=None):
            path=Path(path).resolve(strict=True);h=sha(path)
            assert expected is None or h==expected,str(path)
            inputs[str(path)]=h;return path
        for n in ('mz62_score.py','mz59_score.py','mz58_score.py'):bind(Path(__file__).with_name(n))
        rr=read(bind(rd/'receipt.json'))
        assert rr['status']=='PASS' and rr['steps_per_arm']==1200 and rr['total_steps']==2400
        assert rr['trainable_parameters']==11020 and rr['new_cutoffs']==2 and rr['threshold_searches']==0
        assert rr['positive_pool']=='original_OPEN_max' and rr['loss_contract']=='original_MZ59_balanced_local_frame_loss'
        assert rr['mz55_calibration_rows_used']==rr['original_baseline_cohort_inferences']==rr['native_depth_reads']==0
        assert rr['source_unknown_preserved'] and rr['exact_initial_weights'] and set(rr['fits'])==set(ARMS)
        assert all(rr['fits'][a]['steps']==1200 for a in ARMS)
        assert rr['original_calibration_frames']==1256 and rr['initial_train_parity_unique_frames']==16
        def producer(path):
            path=Path(path).resolve(strict=True)
            matches=[h for p,h in rr['inputs'].items() if Path(p).resolve()==path]
            assert len(matches)==1,str(path);return bind(path,matches[0])
        producer(Path(__file__).with_name('MZ62_PROFILE_COVERAGE_20260911.md'))
        definition=read(producer(task/'primary-definition.json'))
        assert definition==rr['primary_definition'] and definition['definition']==FROZEN_DEFINITION
        assert definition['status']=='FROZEN_BEFORE_FIT' and not definition['new_inference_or_fit_started']
        for p,h in rr['inputs'].items():
            if p.endswith('.py'):bind(p,h)
        for name in ('predictions.npz','schedule.npz','groups.json','initialization-parity.json','initial-missing.json','learned-missing.json'):
            bind(rd/name,rr['outputs'][name])
        for a in ARMS:
            for suffix in ('.pt','-cutoff.npy'):bind(rd/(a+suffix),rr['outputs'][a+suffix])
        for name in ('initial-missing.json','learned-missing.json'):
            r=read(rd/name);assert r['status']=='PASS' and r['frames_per_arm']==8 and r['all_missing_context_exactly_zero']
        parity=read(rd/'initialization-parity.json')
        assert parity['status']=='PASS' and parity['unique_train_frames']==16 and parity['atol']==2e-5 and parity['rtol']==1e-6
        olddir=work/'mz60-native-query-20260911'
        r60=read(producer(olddir/'run-v1/receipt.json'));seal60=read(producer(olddir/'score-v1/receipt.json'))
        assert r60['status']==seal60['status']=='PASS'
        assert all(rr['initial_parameter_sha256'][a]==r60['initial_parameter_sha256']['WITNESS'] for a in ARMS)
        producer(work/'mz56-global-anchor-20260911/run-v2/GLOBAL_ANCHOR.pt')
        previous=read(bind(olddir/'score-v1/result.json',seal60['outputs']['result.json']))
        p=load(rd/'predictions.npz');old=load(bind(olddir/'run-v1/predictions.npz',r60['outputs']['predictions.npz']))
        for k,v in old.items():np.testing.assert_array_equal(p[k],v,err_msg=k)
        preserved=len(old);assert preserved==1696 and set(rr['baseline_arrays_preserved'])==set(old);del old
        groups=read(rd/'groups.json');assert groups==read(bind(olddir/'run-v1/groups.json',r60['outputs']['groups.json']))
        records={'mz48':groups['records'],'mz55':groups['mz55_records']}
        g48={k:np.array(v,int) for k,v in groups['groups'].items()}
        assert {k:len(v) for k,v in g48.items()}==dict(fit=1280,calibration=256,heldout_site=640,nonfit_family=384)
        g48['nonfit']=np.sort(np.r_[g48['heldout_site'],g48['nonfit_family']])
        g55={r:np.array([i for i,x in enumerate(records['mz55']) if x['role']==r],int) for r in ('TRAIN_CANDIDATE','CALIBRATION','HELDOUT_SITE')}
        assert {k:len(v) for k,v in g55.items()}==dict(TRAIN_CANDIDATE=1600,CALIBRATION=320,HELDOUT_SITE=640)
        g55['heldout_new_family']=np.array([i for i in g55['HELDOUT_SITE'] if records['mz55'][i]['family']!='retained_rod'])
        g55['nonfit']=np.sort(np.r_[g55['CALIBRATION'],g55['HELDOUT_SITE']]);assert len(g55['heldout_new_family'])==480
        s=load(bind(rd/'schedule.npz',SCHEDULE_SHA));old_s=load(bind(olddir/'run-v1/schedule.npz',r60['outputs']['schedule.npz']))
        schedule_audit=validate_schedule(s,old_s,g55['TRAIN_CANDIDATE'])
        np.testing.assert_array_equal(parity['ids'],np.unique(s['mz48'])[:16])
        def old_input(suffix):
            hits=[(q,h) for q,h in seal60['inputs'].items() if q.replace('\\','/').endswith(suffix)]
            assert len(hits)==1,suffix;return bind(*hits[0])
        labels={'mz48':load(old_input('mz52-full-frame-supervision-20260911/verified-v1/fullframe-cells.npz')),
                'mz55':load(old_input('mz55-diverse-mesh-source-20260911/source-v1/fullframe-cells.npz'))}
        for cohort,lab in labels.items():
            np.testing.assert_array_equal(lab['frame_ids'],p[cohort+'/frame_ids'])
            np.testing.assert_array_equal(lab['frame_ids'],[r['frame_id'] for r in records[cohort]])
            assert lab['fullframe_event_counts'].shape==(2560,45,80,4) and lab['valid_counts'].shape==(2560,45,80)
            assert lab['fullframe_event_counts'].dtype==lab['valid_counts'].dtype==np.uint8
            assert (lab['fullframe_event_counts']<=lab['valid_counts'][...,None]).all()
            np.testing.assert_array_equal(lab['fullframe_event_counts'].sum((1,2))>=3,p[cohort+'/truth'])
        for n in ('crop_mask','sensor_coverage','rays'):np.testing.assert_array_equal(p['mz62/'+n],p['mz60/'+n])
        cal=g48['calibration'];cuts={};authorities={}
        def calibration(name):return np.concatenate([p['DEV/DROP_CLOSE/'+name],p['mz48/DROP_CLOSE/'+name][cal]])
        tc=np.concatenate([p['DEV/truth'],p['mz48/truth'][cal]]);kc=np.concatenate([p['DEV/known'],p['mz48/known'][cal]])
        for arm,key in zip(ARMS,KEYS):
            cuts[arm]=np.load(rd/(arm+'-cutoff.npy'),allow_pickle=False)
            rebuilt,authorities[arm]=scalar_cutoff(calibration(key+'/raw'),calibration(key+'/support'),calibration('MZ37'),tc,kc)
            np.testing.assert_array_equal(cuts[arm],rebuilt);assert np.isfinite(rebuilt).all()
        scores={};changes={};attrs={};events={};contexts={};cached={};awning={};scalar_bits=reference_rows=native_checks=missing_rows=0
        refs=(BASE,'MZ37','MZ57/GLOBAL_ANCHOR/UNION','MZ59/DIVERSE/UNION','MZ60/WITNESS/UNION')+NEW
        compare=lambda aa,t,k:{name:{ref:paired(aa[name],aa[ref],t,k) for ref in refs if ref!=name} for name in NEW}
        for cohort in COHORTS:
            truth,known=p[cohort+'/truth'],p[cohort+'/known']
            if cohort=='mz36':
                truth,known=p['mz36/attempted_truth'],p['mz36/attempted_known'];admitted=p['mz36/admitted_index']
                assert truth.shape==known.shape==(400,4) and int((~known).sum())==80 and len(admitted)==380
            partitions={'all':np.arange(len(truth))}
            if cohort in records:
                partitions.update(g55 if cohort=='mz55' else g48)
                for field in ('role','family','site_id','relation','range','setting','support_context'):
                    for val in sorted({str(r[field]) for r in records[cohort]}):partitions[field+'/'+val]=np.array([i for i,r in enumerate(records[cohort]) if str(r[field])==val])
            scores[cohort]={};changes[cohort]={};attrs[cohort]={};events[cohort]={};contexts[cohort]={}
            for profile in PROFILES:
                prefix=cohort+'/'+profile+'/';oldmethods=tuple(previous['conditions'][cohort][profile]['all'])
                a=reconstruct(p,prefix,oldmethods)
                for arm,key in zip(ARMS,KEYS):
                    values=[p[prefix+key+'/'+n] for n in ('raw','support','winner','anchor_vector','anchor_available','candidate')]
                    validate_candidate(*values,p[prefix+'MZ37'],cuts[arm])
                    np.testing.assert_array_equal(values[4],p[prefix+'MZ60/WITNESS/anchor_available']);missing_rows+=int((~values[4]).sum())
                if cohort=='mz36':
                    expanded={}
                    for key,v in a.items():expanded[key]=np.full((400,4),np.nan);expanded[key][admitted]=v
                    a=expanded
                scores[cohort][profile]={};changes[cohort][profile]={}
                for group,ii in partitions.items():
                    aa={key:v[ii] for key,v in a.items()};rows={key:metrics(v,truth[ii],known[ii]) for key,v in aa.items()}
                    scores[cohort][profile][group]=rows;changes[cohort][profile][group]=compare(aa,truth[ii],known[ii])
                    for key in oldmethods:assert_metric_equal(rows[key],previous['conditions'][cohort][profile][group][key]);reference_rows+=1
                for key in NEW:
                    assert scalar_metrics(a[key],truth,known)==scores[cohort][profile]['all'][key];scalar_bits+=int(known.sum())
                cached.setdefault(profile,{})[cohort]=(a,truth,known)
                if cohort not in labels:continue
                attrs[cohort][profile]={g:{} for g in partitions};events[cohort][profile]={}
                lab=labels[cohort];crop=p['mz62/crop_mask'].flatten();sensor=p['mz62/sensor_coverage'].flatten()
                for arm,key in zip(ARMS,KEYS):
                    winner,native,wk=native_masks(p,prefix,key,lab['fullframe_event_counts'],lab['valid_counts']);native_checks+=native.size
                    added=(a[key+'/candidate']>=0)&(a[BASE]<0)&truth&known;miss=(a[key+'/UNION']<0)&truth&known
                    masks=dict(added_tp=added,native_added_tp=added&native,known_non_native_added_tp=added&wk&~native,unknown_winner_added_tp=added&~wk,
                        native_outside_crop_added_tp=added&native&~crop[winner],native_outside_sensor_added_tp=added&native&~sensor[winner],
                        missed_native_winner=miss&native,missed_known_non_native_winner=miss&wk&~native,missed_UNKNOWN_winner=miss&~wk,
                        final_native_winner_tp=(a[key+'/UNION']>=0)&truth&known&native)
                    for group,ii in partitions.items():attrs[cohort][profile][group][key]={n:v[ii].sum(0).tolist() for n,v in masks.items()}
                    if cohort=='mz55' and profile=='DROP_CLOSE':
                        awning[key]={}
                        for role,expected_n in [('TRAIN_CANDIDATE',100),('CALIBRATION',20),('HELDOUT_SITE',40)]:
                            ii=np.array([i for i,r in enumerate(records[cohort]) if r['role']==role and r['family']=='shallow_awning' and truth[i,0] and known[i,0]])
                            assert len(ii)==expected_n
                            raw=p[prefix+key+'/raw'][ii,0];accepted=a[key+'/UNION'][ii,0]>=0
                            rows=[dict(index=int(i),frame_id=records[cohort][i]['frame_id'],role=role,support_context=records[cohort][i]['support_context'],range=records[cohort][i]['range'],winner=int(winner[i,0]),native=bool(native[i,0]),local_known=bool(wk[i,0]),raw=float(p[prefix+key+'/raw'][i,0]),cutoff=float(cuts[arm][0]),margin=float(p[prefix+key+'/raw'][i,0])-float(cuts[arm][0]),candidate_positive=bool(a[key+'/candidate'][i,0]>=0),union_positive=bool(a[key+'/UNION'][i,0]>=0),baseline_positive=bool(a[BASE][i,0]>=0)) for i in ii]
                            awning[key][role]=dict(positive_frames=expected_n,final_or_tp=int(accepted.sum()),native_winners=int(native[ii,0].sum()),known_non_native_winners=int((wk[ii,0]&~native[ii,0]).sum()),UNKNOWN_winners=int((~wk[ii,0]).sum()),native_winners_below_cutoff=int((native[ii,0]&(raw<cuts[arm][0])).sum()),missed_native_winner=int((~accepted&native[ii,0]).sum()),missed_known_non_native_winner=int((~accepted&wk[ii,0]&~native[ii,0]).sum()),missed_UNKNOWN_winner=int((~accepted&~wk[ii,0]).sum()),native_winning_final_or_tp=int((accepted&native[ii,0]).sum()),rows=rows)
                    events[cohort][profile][key]=[dict(index=int(i),frame_id=records[cohort][i]['frame_id'],query=QUERIES[q],winner=int(winner[i,q]),native=bool(native[i,q]),local_known=bool(wk[i,q]),raw=float(p[prefix+key+'/raw'][i,q]),cutoff=float(cuts[arm][q]),**{f:records[cohort][i][f] for f in ('role','family','relation','range','site_id','setting','support_context','pair_id')}) for i,q in zip(*np.where(added))]
                pairs={}
                for i,r in enumerate(records[cohort]):pairs.setdefault(r['pair_id'],[]).append(i)
                assert len(pairs)==1280 and all(len(v)==2 for v in pairs.values());pi=np.array(list(pairs.values()))
                np.testing.assert_array_equal(truth[pi[:,0]],truth[pi[:,1]])
                contexts[cohort][profile]={key:dict(pairs=1280,changed_queries=((a[key][pi[:,0]]>=0)!=(a[key][pi[:,1]]>=0)).sum(0).tolist()) for key in (BASE,)+NEW}
        aggregates={};aggregate_changes={}
        for profile,data in cached.items():
            legacy=[(c,np.arange(len(data[c][1]))) for c in ('relation10000','distance5000','rich','mz36')]
            selections=dict(legacy_noncal=legacy,mz48_fit=[('mz48',g48['fit'])],mz48_nonfit=[('mz48',g48['nonfit'])])
            selections['all_old_noncal']=legacy+selections['mz48_fit']+selections['mz48_nonfit'];aggregates[profile]={};aggregate_changes[profile]={}
            for group,sel in selections.items():
                truth=np.concatenate([data[c][1][ii] for c,ii in sel]);known=np.concatenate([data[c][2][ii] for c,ii in sel])
                methods=tuple(previous['aggregates'][profile][group])+NEW
                aa={key:np.concatenate([data[c][0][key][ii] for c,ii in sel]) for key in methods}
                aggregates[profile][group]={key:metrics(v,truth,known) for key,v in aa.items()};aggregate_changes[profile][group]=compare(aa,truth,known)
                for key in previous['aggregates'][profile][group]:assert_metric_equal(aggregates[profile][group][key],previous['aggregates'][profile][group][key]);reference_rows+=1
        primary=gate({key:attrs['mz55']['DROP_CLOSE']['heldout_new_family'][key]['native_added_tp'] for key in KEYS},scores['mz55']['DROP_CLOSE']['HELDOUT_SITE'],aggregates['DROP_CLOSE']['legacy_noncal'],aggregates['DROP_CLOSE']['mz48_nonfit'])
        result=dict(status='PASS',primary_gate=primary,queries=QUERIES,new_methods=NEW,conditions=scores,comparisons=changes,aggregates=aggregates,aggregate_comparisons=aggregate_changes,attribution=attrs,support_context_pairs=contexts,cutoffs={a:v.tolist() for a,v in cuts.items()},cutoff_authority=authorities,source_role='CONSUMED_DEVELOPMENT',limitation='Matched total frame presentations and budget; profile assignment/order changes together. Original loss and calibration. One complete pass per frame/profile is not a trained ceiling. No MZ61 training or hardware claim.')
        audit=dict(status='PASS',exact_preserved_mz60_arrays=preserved,schedule=schedule_audit,exact_groups=True,prior_metric_rows_exact=reference_rows,scalar_known_decisions=scalar_bits,native_winner_lookups=native_checks,original_calibration_rows=1256,cutoff_vectors_recomputed=2,mz55_calibration_rows_used=0,mz36_UNKNOWN_bits=80,all_missing_anchor_rows_checked=missing_rows,local_UNKNOWN_cells={c:int((v['valid_counts']==0).sum()) for c,v in labels.items()},new_fits=0,new_inference_frames=0,RGB_reads=0,native_depth_reads=0,checkpoint_loads=0)
        for path,h in inputs.items():assert sha(path)==h,path
        result['awning_DROP_audit']={key:{role:{k:v for k,v in r.items() if k!='rows'} for role,r in roles.items()} for key,roles in awning.items()}
        write(out/'result.json',result);write(out/'audit.json',audit);write(out/'native-added-events.json',events)
        write(out/'awning-drop-audit.json',dict(arms=awning,scope='All100 TRAIN,20calibration,40held awning BODY_NEAR positives; saved outputs only. Native-winning finalOR TP is not independent branch credit for inherited positives.'))
        report(out/'report.md',result)
        write(out/'receipt.json',dict(status='PASS',inputs=inputs,outputs={n:sha(out/n) for n in ('result.json','audit.json','native-added-events.json','awning-drop-audit.json','report.md')},code_sha256=sha(__file__),seconds=time.perf_counter()-start,backend='FROZEN_PROTOCOL_CPU_ONLY saved-output score',new_fits=0,new_inference_frames=0,new_cutoffs=0))
        print('PASS',primary)
    except BaseException:write(out/'failure.json',dict(status='FAIL',inputs=inputs,error=traceback.format_exc()));raise


def report(path,result):
    primary=result['primary_gate'];lines=['# MZ62 complete profile coverage','',f'Primary retaining check: **{"PASS" if primary["passes_primary"] else "FAIL"}**.','','| Clause | Result |','| --- | --- |']
    lines += [f'| {k} | {"PASS" if v else "FAIL"} |' for k,v in primary['clauses'].items()]
    lines += ['',str(primary['values']),'','| DROP group | Method | TP | FP | FN | Exact |','| --- | --- | ---: | ---: | ---: | ---: |']
    for name,rows in [('MZ55 held640',result['conditions']['mz55']['DROP_CLOSE']['HELDOUT_SITE']),('Legacy noncal',result['aggregates']['DROP_CLOSE']['legacy_noncal']),('MZ48 nonfit',result['aggregates']['DROP_CLOSE']['mz48_nonfit'])]:
        for key in (BASE,'MZ59/DIVERSE/UNION','MZ60/WITNESS/UNION')+NEW:
            row=rows[key];lines.append(f'| {name} | {key} | '+' | '.join(str(sum(row[k])) for k in ('tp','fp','fn'))+f' | {row["exact_frames"]} |')
    lines += ['',result['limitation']];path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def self_test():
    native={KEYS[0]:[1,0,0,0],KEYS[1]:[2,0,0,0]}
    groups=[{k+'/UNION':dict(tp=[10,0,0,0],fp=[2,0,0,0]) for k in KEYS} for _ in range(3)]
    assert gate(native,*groups)['passes_primary']
    assert not gate({k:[1,0,0,0] for k in KEYS},*groups)['passes_primary']
    for i in range(3):
        for field,value in [('tp',9),('fp',3)]:
            import copy
            bad=copy.deepcopy(groups);bad[i][KEYS[1]+'/UNION'][field]=[value,0,0,0]
            r=gate(native,*bad);assert not r['passes_primary'] and sum(not v for v in r['clauses'].values())==1
    print('PASS independent seven-clause retaining boundaries')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task',type=Path);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:assert args.task is None;self_test()
    else:assert args.task is not None;run(args.task)
