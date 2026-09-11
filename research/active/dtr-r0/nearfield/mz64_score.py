"""Independent MZ64 saved-output scoring; no model/image/native-depth execution."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
import traceback
import numpy as np
from mz58_score import read, write, sha, load, metrics, scalar_metrics, paired
from mz59_score import scalar_cutoff, validate_candidate, native_masks, assert_metric_equal
from mz62_score import reconstruct as reconstruct62
from mz63_score import check_geometry, check_packets, exchange_masks

ARMS=('CONTROL','GEOMETRY')
KEYS=tuple('MZ64/'+a for a in ARMS)
NEW=tuple(k+s for k in KEYS for s in ('/candidate','/UNION'))
BASE='OLD_NEG/UNION'
FROZEN='MZ62/CONTROL/UNION'
PROFILES=('IDEAL','MERGE_CLOSE','DROP_CLOSE')
SOURCE_PROFILES=PROFILES+('ALL_INVALID',)
COHORTS=('DEV','relation10000','distance5000','rich','mz36','mz48','mz55','mz61')
QUERIES=('BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR')
INITIAL_SHA='b7106449e4246b7bd7933659cbf0dd815691072c03a8b522633b7aee9a456d2e'
FROZEN_DEFINITION='GEOMETRY versus matched CONTROL and frozen MZ62 CONTROL: greater native BODY_NEAR additions beyond OLD_NEG on DROP MZ61 HELD1024, including at least one held awning native addition; combined four-query TP retained and no newly false held bit versus either comparator; total TP retained and total FP not increased on legacy_noncal, MZ48 nonfit and MZ55 held640 versus either comparator. Report all effects and costs without outcome-driven selection.'
FIELDS=('role','family','site_id','relation','range','setting','support_context')
COMPARATORS=(BASE,'MZ57/GLOBAL_ANCHOR/UNION',FROZEN,KEYS[0]+'/UNION')


def gate(native, awning, held_values, held_truth, held_known, regressions):
    """Strict new FP IDs on held source; aggregate FP on each old cohort."""
    geometry=KEYS[1]+'/UNION'; control=KEYS[0]+'/UNION'
    values=dict(native_body_near={k:int(v[0]) for k,v in native.items()},
        held_awning_native_body_near=int(awning),held={},regressions={})
    clauses={}
    for ref,label in ((control,'matched_control'),(FROZEN,'frozen_mz62_control')):
        clauses['native_body_near_more_than_'+label]=int(native[KEYS[1]][0])>int(native[ref[:-6]][0])
        gm=metrics(held_values[geometry],held_truth,held_known);rm=metrics(held_values[ref],held_truth,held_known)
        change=paired(held_values[geometry],held_values[ref],held_truth,held_known)
        clauses['held_total_tp_retained_vs_'+label]=sum(gm['tp'])>=sum(rm['tp'])
        clauses['held_no_new_fp_bits_vs_'+label]=sum(change['fp_added'])==0
        values['held'][label]=dict(geometry=gm,comparator=rm,paired=change)
    clauses['held_awning_native_body_near_added']=int(awning)>0
    for group,rows in regressions.items():
        assert group in ('legacy_noncal','mz48_nonfit','mz55_held640')
        values['regressions'][group]={}
        for ref,label in ((control,'matched_control'),(FROZEN,'frozen_mz62_control')):
            gm,rm=rows[geometry],rows[ref]
            clauses[group+'_total_tp_retained_vs_'+label]=sum(gm['tp'])>=sum(rm['tp'])
            clauses[group+'_total_fp_not_higher_vs_'+label]=sum(gm['fp'])<=sum(rm['fp'])
            values['regressions'][group][label]=dict(geometry_tp=sum(gm['tp']),comparator_tp=sum(rm['tp']),geometry_fp=sum(gm['fp']),comparator_fp=sum(rm['fp']))
    assert len(regressions)==3 and len(clauses)==19
    return dict(definition=FROZEN_DEFINITION,values=values,clauses=clauses,passes_primary=all(clauses.values()))


def partitions(records, inherited=None):
    groups={'all':np.arange(len(records))}
    if inherited:groups.update({k:np.asarray(v,int) for k,v in inherited.items()})
    for field in FIELDS+tuple(f for f in ('geometry_recipe_id','site_replica') if all(f in r for r in records)):
        for value in sorted({str(r[field]) for r in records}):
            groups[field+'/'+value]=np.array([i for i,r in enumerate(records) if str(r[field])==value])
    for role in sorted({r['role'] for r in records}):
        groups[role]=np.array([i for i,r in enumerate(records) if r['role']==role])
        for family in sorted({r['family'] for r in records}):
            groups['role_family/'+role+'/'+family]=np.array([i for i,r in enumerate(records) if r['role']==role and r['family']==family])
    return groups


def validate_schedule(s,g48,g55,g61,records55,records61,old):
    np.testing.assert_array_equal(s['profile_index'],np.arange(1536)%3)
    for field in ('mz48','OLD_NEG','query'):
        np.testing.assert_array_equal(s[field],old[field][np.arange(1536)%600])
    assert s['mz48'].shape==(1536,4) and np.isin(s['mz48'],g48['fit']).all()
    assert s['OLD_NEG'].shape==s['query'].shape==(1536,8)
    assert ((s['query']>=0)&(s['query']<4)).all()
    for arm,fit,records in [('CONTROL',g55['TRAIN_CANDIDATE'],records55),('GEOMETRY',g61['TRAIN_CANDIDATE'],records61)]:
        assert s[arm].shape==(1536,4) and np.isin(s[arm],fit).all()
        assert all(records[i]['role']=='TRAIN_CANDIDATE' and records[i]['source_valid'] for i in fit)
        ids,n=np.unique(s[arm],return_counts=True);np.testing.assert_array_equal(ids,fit)
        if arm=='GEOMETRY':np.testing.assert_array_equal(n,np.full(2048,3))
        else:assert dict(Counter(n))=={3:256,4:1344}
        total=Counter(s[arm].ravel())
        for profile in range(3):
            ii=s[arm][s['profile_index']==profile].ravel();ids,n=np.unique(ii,return_counts=True)
            np.testing.assert_array_equal(ids,fit)
            if arm=='GEOMETRY':np.testing.assert_array_equal(n,np.ones(2048,int))
            else:assert dict(Counter(n))=={1:1152,2:448}
            per=Counter(ii);pairs=defaultdict(list)
            for i in fit:pairs[records[i]['pair_id']].append(i)
            assert all(len(v)==2 and per[v[0]]==per[v[1]] and total[v[0]]==total[v[1]] for v in pairs.values())
    return dict(steps_per_arm=1536,target_presentations=6144,mz48_presentations=6144,old_negative_presentations=12288,
        geometry_each_id_each_profile=1,control_rows_presented3=256,control_rows_presented4=1344,pair_balanced=True,shared_replay_arrays=True)


def reconstruct(p,prefix,oldmethods):
    refs=list(oldmethods)
    for method in oldmethods:
        if method.endswith('/UNION'):
            family=method[:-6]
            if family in ('MZ50','NEW_NEG','OLD_NEG'):
                lead='' if family=='MZ50' else family+'/'
                refs.extend(lead+x+'/candidate' for x in ('OPEN','GATED'))
            else:refs.append(family.replace('MZ57/','MZ56/')+'/candidate')
    a=reconstruct62(p,prefix,tuple(dict.fromkeys(refs)))
    # reconstruct62 also adds its two frozen heads, even when oldmethods is63's8.
    for key in KEYS:
        a[key+'/candidate']=p[prefix+key+'/candidate']
        a[key+'/UNION']=np.maximum(a[BASE],a[key+'/candidate'])
        np.testing.assert_array_equal(a[key+'/UNION'],p[prefix+key+'/UNION'])
    return {k:a[k] for k in tuple(oldmethods)+NEW}


def compact_native(p,prefix,key,winning,a,truth,known,cut,ii,comparator):
    w,native,wk=winning
    selected=np.zeros_like(truth,bool);selected[ii]=True
    positive=truth&known&selected
    candidate=a[key+'/candidate']>=0;final=a[key+'/UNION']>=0
    added=candidate&(a[comparator]<0)&positive
    margin=p[prefix+key+'/raw'].astype(float)-cut
    masks=dict(positive=positive,native_winner=positive&native,known_wrong_winner=positive&wk&~native,UNKNOWN_winner=positive&~wk,
        added_tp=added,native_added_tp=added&native,known_wrong_added_tp=added&wk&~native,UNKNOWN_added_tp=added&~wk,
        native_below_cutoff=positive&native&(margin<0),missed_native=positive&~final&native,
        missed_wrong=positive&~final&wk&~native,missed_UNKNOWN=positive&~final&~wk,
        above_cutoff_native=positive&native&(margin>=0),no_support=positive&~p[prefix+key+'/support'])
    return {k:v.sum(0).tolist() for k,v in masks.items()}


def event_rows(a,truth,known,frame_ids,method,comparator,records=None,detail=None):
    rows=[]
    for change,mask in exchange_masks(a[method],a[comparator],truth,known).items():
        for i,q in zip(*np.where(mask)):
            row=dict(index=int(i),frame_id=str(frame_ids[i]),query=QUERIES[q],change=change,truth=bool(truth[i,q]),known=bool(known[i,q]),
                before=None if not np.isfinite(a[comparator][i,q]) else float(a[comparator][i,q]),after=None if not np.isfinite(a[method][i,q]) else float(a[method][i,q]))
            if records:row.update({f:records[i][f] for f in FIELDS+('pair_id',)});row['geometry_id']=records[i].get('geometry_id')
            if detail is not None:
                w,native,wk,raw,cut,support,base=detail
                row.update(winner=int(w[i,q]),native=bool(native[i,q]),local_known=bool(wk[i,q]),raw=float(raw[i,q]),cutoff=float(cut[q]),margin=float(raw[i,q])-float(cut[q]),support=bool(support[i,q]),native_added_credit=bool(change=='TP_GAIN' and native[i,q] and support[i,q] and base[i,q]<0))
            rows.append(row)
    return rows


def summarize_margin(raw,cut,truth,known,native,ii):
    answer={}
    for q,name in enumerate(QUERIES):
        m=raw[ii,q].astype(float)-cut[q];pos=truth[ii,q]&known[ii,q]
        answer[name]={}
        for label,mask in [('positive',pos),('negative',~truth[ii,q]&known[ii,q]),('native_positive',pos&native[ii,q])]:
            v=m[mask];answer[name][label]=dict(n=len(v),quantiles=np.quantile(v,[0,.25,.5,.75,1]).tolist() if len(v) else [])
    return answer


def run(task):
    task=task.resolve(strict=True);rd=task/'run-v1';out=task/'score-v1';work=task.parent
    assert not out.exists();out.mkdir();inputs={};start=time.perf_counter()
    try:
        def bind(path,expected=None):
            path=Path(path).resolve(strict=True);h=sha(path);assert expected is None or h==expected,str(path)
            inputs[str(path)]=h;return path
        for name in ('mz64_score.py','mz63_score.py','mz62_score.py','mz59_score.py','mz58_score.py'):bind(Path(__file__).with_name(name))
        rr=read(bind(rd/'receipt.json'))
        assert rr['status']=='PASS' and rr['steps_per_arm']==1536 and rr['total_steps']==3072
        assert rr['trainable_parameters']==11020 and rr['new_cutoffs']==2 and rr['threshold_searches']==0
        assert rr['original_calibration_frames']==1256 and rr['original_baseline_cohort_inferences']==rr['native_depth_reads']==0
        assert rr['positive_pool']=='original_OPEN_max' and rr['loss_contract']=='original_MZ59_balanced_local_frame_loss'
        assert rr['exact_initial_weights'] and set(rr['fits'])==set(ARMS) and all(rr['fits'][a]['steps']==1536 for a in ARMS)
        def producer(path):
            path=Path(path).resolve(strict=True)
            matches=[h for p,h in rr['inputs'].items() if Path(p).resolve()==path]
            assert len(matches)==1,str(path);return bind(path,matches[0])
        definition=read(producer(task/'primary-definition.json'))
        producer(Path(__file__).with_name('MZ64_GEOMETRY_LEARNING_20260911.md'))
        assert definition==rr['primary_definition'] and definition['definition']==FROZEN_DEFINITION
        assert definition['status']=='FROZEN_BEFORE_FIT' and not definition['new_inference_or_fit_started']
        assert definition['initial_checkpoint_sha256']==INITIAL_SHA and definition['steps_per_arm']==1536
        assert definition['comparators']==[KEYS[0]+'/UNION',FROZEN]
        assert definition['held_tp_rule']=='sum_over_four_queries_non_decreasing'
        assert definition['held_fp_rule']=='zero_new_false_positive_bits_per_comparator'
        assert definition['new_source_calibration_rows_used']==0 and definition['all_invalid_training_presentations']==0
        for path,h in rr['inputs'].items():
            if path.endswith('.py'):bind(path,h)
        assert rr['source_unknown_preserved'] and rr['mz55_calibration_rows_used']==rr['mz61_calibration_rows_used']==rr['all_invalid_training_presentations']==0
        assert rr['initial_checkpoint_sha256']==INITIAL_SHA and len(set(rr['initial_parameter_sha256'].values()))==1
        for name in ('predictions.npz','groups.json','schedule.npz','initialization-parity.json','initial-missing.json','learned-missing.json'):
            bind(rd/name,rr['outputs'][name])
        for name in ('initial-missing.json','learned-missing.json'):
            missing=read(rd/name);assert missing['status']=='PASS' and missing['frames_per_arm']==8 and missing['all_missing_context_exactly_zero']
        for arm in ARMS:
            for suffix in ('.pt','-cutoff.npy'):bind(rd/(arm+suffix),rr['outputs'][arm+suffix])
        init=producer(work/'mz62-profile-coverage-20260911/run-v1/CONTROL.pt');assert sha(init)==INITIAL_SHA
        parity=read(rd/'initialization-parity.json');assert parity['status']=='PASS'
        # Detailed32-frame probe schema is checked against producer receipt below;
        # all baseline arrays additionally require exact independent preservation.
        assert rr['initial_train_parity_unique_frames']==32
        assert parity['unique_train_frames']==32 and parity['full_baseline_cohort_replays']==0
        assert parity['atol']==2e-5 and parity['rtol']==1e-6
        assert set(parity['ids'])=={'mz48','mz61'} and all(len(set(v))==16 for v in parity['ids'].values())
        expected_probes={(c,pr,a) for c in ('mz48','mz61') for pr in (PROFILES if c=='mz48' else SOURCE_PROFILES) for a in ARMS}
        assert len(parity['comparisons'])==14 and {(r['cohort'],r['profile'],r['arm']) for r in parity['comparisons']}==expected_probes
        r62=read(producer(work/'mz62-profile-coverage-20260911/run-v1/receipt.json'))
        seal62=read(producer(work/'mz62-profile-coverage-20260911/score-v1/receipt.json'))
        previous=read(bind(work/'mz62-profile-coverage-20260911/score-v1/result.json',seal62['outputs']['result.json']))
        r63=read(producer(work/'mz63-geometry-transfer-20260911/run-v1/receipt.json'))
        seal63=read(producer(work/'mz63-geometry-transfer-20260911/score-v1/receipt.json'))
        score63=read(bind(work/'mz63-geometry-transfer-20260911/score-v1/result.json',seal63['outputs']['result.json']))
        assert r62['status']==r63['status']==seal62['status']==seal63['status']=='PASS'
        p=load(rd/'predictions.npz');preserved={};preserved_names=[]
        for folder,rec,prefix in [('mz62-profile-coverage-20260911',r62,''),('mz63-geometry-transfer-20260911',r63,'mz61/')]:
            old=load(bind(work/folder/'run-v1/predictions.npz',rec['outputs']['predictions.npz']))
            for k,v in old.items():
                actual=p[prefix+k];assert actual.shape==v.shape and actual.dtype==v.dtype and actual.tobytes()==v.tobytes(),prefix+k
                preserved_names.append(prefix+k)
            preserved[folder]=len(old);del old
        groups=read(rd/'groups.json');oldgroups=read(bind(work/'mz62-profile-coverage-20260911/run-v1/groups.json',r62['outputs']['groups.json']))
        for k,v in oldgroups.items():assert groups[k]==v,k
        records={'mz48':groups['records'],'mz55':groups['mz55_records'],'mz61':groups['mz61_records']}
        g48={k:np.asarray(v,int) for k,v in groups['groups'].items()};g48['nonfit']=np.sort(np.r_[g48['heldout_site'],g48['nonfit_family']])
        g55={role:np.array([i for i,r in enumerate(records['mz55']) if r['role']==role]) for role in ('TRAIN_CANDIDATE','CALIBRATION','HELDOUT_SITE')}
        g55['heldout_new_family']=np.array([i for i in g55['HELDOUT_SITE'] if records['mz55'][i]['family']!='retained_rod'])
        g55['nonfit']=np.sort(np.r_[g55['CALIBRATION'],g55['HELDOUT_SITE']])
        g61={role:np.array([i for i,r in enumerate(records['mz61']) if r['role']==role]) for role in ('TRAIN_CANDIDATE','CALIBRATION','HELDOUT_GEOMETRY')}
        assert [len(g61[k]) for k in g61]==[2048,1024,1024]
        oldschedule=load(bind(work/'mz62-profile-coverage-20260911/run-v1/schedule.npz',r62['outputs']['schedule.npz']))
        schedule=load(rd/'schedule.npz');schedule_audit=validate_schedule(schedule,g48,g55,g61,records['mz55'],records['mz61'],oldschedule)
        for cohort,key in (('mz48','mz48'),('mz61','GEOMETRY')):
            np.testing.assert_array_equal(parity['ids'][cohort],np.unique(schedule[key])[:16])
        def prior_input(suffix):
            hits=[(q,h) for q,h in seal62['inputs'].items() if q.replace('\\','/').endswith(suffix)]
            assert len(hits)==1,suffix;return bind(*hits[0])
        labels={'mz48':load(prior_input('mz52-full-frame-supervision-20260911/verified-v1/fullframe-cells.npz')),
                'mz55':load(prior_input('mz55-diverse-mesh-source-20260911/source-v1/fullframe-cells.npz'))}
        source=work/'mz61-geometry-source-20260911'
        si=read(producer(source/'source-index.json'));assert sha(source/'source-index.json')==definition['source_index_sha256']
        assert si['status']=='COMPLETE' and si['frames']==4096
        def source_part(name):
            ref=si['combined'][name];path=(source/ref['path']).resolve(strict=True);assert path.is_relative_to(source.resolve())
            return bind(path,ref['sha256'])
        labels['mz61']=load(source_part('fullframe-cells.npz'));source_meta=read(source_part('metadata.json'))
        assert records['mz61']==source_meta['records'];source_eval=load(source_part('evaluator.npz'))
        for name in ('frame_ids','truth','known'):np.testing.assert_array_equal(source_eval[name],p['mz61/'+name])
        for name in ('truth','known'):
            assert source_eval[name].dtype==p['mz61/'+name].dtype and source_eval[name].tobytes()==p['mz61/'+name].tobytes()
            preserved_names.append('mz61/'+name)
        assert sorted(rr['baseline_arrays_preserved'])==sorted(preserved_names)
        preserved['mz61_source_truth_known']=2
        packet61=load(source_part('packets.npz'));collision=read(source_part('tensor-collision-audit.json'))
        for cohort,lab in labels.items():
            n=len(records[cohort]);np.testing.assert_array_equal(lab['frame_ids'],p[cohort+'/frame_ids'])
            np.testing.assert_array_equal(lab['frame_ids'],[r['frame_id'] for r in records[cohort]])
            c,v=lab['fullframe_event_counts'],lab['valid_counts'];assert c.shape==(n,45,80,4) and v.shape==(n,45,80)
            assert c.dtype==v.dtype==np.uint8 and (v<=64).all() and (c<=v[...,None]).all()
            np.testing.assert_array_equal(c.sum((1,2))>=3,p[cohort+'/truth'])
        geometry={k[len('mz61/'):]:v for k,v in p.items() if k.startswith('mz61/geometry/')};check_geometry(geometry)
        for name in ('crop_mask','sensor_coverage','rays'):np.testing.assert_array_equal(p['mz64/'+name],p['mz61/geometry/'+name])
        def cal(name):return np.concatenate([p['DEV/DROP_CLOSE/'+name],p['mz48/DROP_CLOSE/'+name][g48['calibration']]])
        ct=np.concatenate([p['DEV/truth'],p['mz48/truth'][g48['calibration']]])
        ck=np.concatenate([p['DEV/known'],p['mz48/known'][g48['calibration']]])
        cuts={};authorities={}
        for arm,key in zip(ARMS,KEYS):
            cuts[key]=np.load(rd/(arm+'-cutoff.npy'),allow_pickle=False)
            rebuilt,authorities[arm]=scalar_cutoff(cal(key+'/raw'),cal(key+'/support'),cal('MZ37'),ct,ck)
            np.testing.assert_array_equal(cuts[key],rebuilt);assert np.isfinite(rebuilt).all()
        path=bind(work/'mz62-profile-coverage-20260911/run-v1/CONTROL-cutoff.npy',r62['outputs']['CONTROL-cutoff.npy'])
        cuts['MZ62/CONTROL']=np.load(path,allow_pickle=False)
        scores={};changes={};attrs={};events={};contexts={};margins={};awning={};cached={}
        scalar_bits=reference_rows=native_checks=0
        for cohort in COHORTS:
            truth,known=p[cohort+'/truth'],p[cohort+'/known'];frame_ids=p[cohort+'/frame_ids']
            if cohort=='mz36':
                truth,known,frame_ids=(p['mz36/attempted_'+n] for n in ('truth','known','frame_ids'));admitted=p['mz36/admitted_index']
                assert len(truth)==400 and int((~known).sum())==80 and len(admitted)==380
            groups_here=partitions(records[cohort],{'mz48':g48,'mz55':g55,'mz61':g61}[cohort]) if cohort in records else {'all':np.arange(len(truth))}
            for mapping in (scores,changes,attrs,events,contexts,margins):mapping[cohort]={}
            for profile in (SOURCE_PROFILES if cohort=='mz61' else PROFILES):
                prefix=cohort+'/'+profile+'/'
                reference=score63['profiles'][profile] if cohort=='mz61' else previous['conditions'][cohort][profile]
                oldmethods=tuple(reference['all']['methods'] if cohort=='mz61' else reference['all'])
                a=reconstruct(p,prefix,oldmethods)
                for key in KEYS:
                    values=[p[prefix+key+'/'+n] for n in ('raw','support','winner','anchor_vector','anchor_available','candidate')]
                    validate_candidate(*values,p[prefix+'MZ37'],cuts[key])
                    np.testing.assert_array_equal(values[4],p[prefix+'MZ62/CONTROL/anchor_available'])
                if cohort=='mz61':check_packets(packet61,{k:p[prefix+k] for k in ('ranges','valid')},profile)
                if cohort=='mz36':
                    expanded={}
                    for k,v in a.items():expanded[k]=np.full((400,4),np.nan);expanded[k][admitted]=v
                    a=expanded
                for mapping in (scores,changes,attrs,events,contexts,margins):mapping[cohort][profile]={}
                for group,ii in groups_here.items():
                    aa={k:v[ii] for k,v in a.items()};rows={k:metrics(v,truth[ii],known[ii]) for k,v in aa.items()}
                    scores[cohort][profile][group]=rows
                    changes[cohort][profile][group]={k:{ref:paired(aa[k],aa[ref],truth[ii],known[ii]) for ref in COMPARATORS if ref!=k} for k in NEW}
                    if group in reference:
                        before=reference[group]['methods'] if cohort=='mz61' else reference[group]
                        for k in oldmethods:assert_metric_equal(rows[k],before[k]);reference_rows+=1
                assert set(reference).issubset(groups_here),set(reference)-set(groups_here)
                for k in NEW:assert scalar_metrics(a[k],truth,known)==scores[cohort][profile]['all'][k];scalar_bits+=int(known.sum())
                cached.setdefault(profile,{})[cohort]=(a,truth,known)
                winning={}
                if cohort in labels:
                    for key in KEYS+('MZ62/CONTROL',):
                        winning[key]=native_masks(p,prefix,key,labels[cohort]['fullframe_event_counts'],labels[cohort]['valid_counts']);native_checks+=winning[key][1].size
                    for group,ii in groups_here.items():
                        attrs[cohort][profile][group]={};margins[cohort][profile][group]={}
                        for key in winning:
                            attrs[cohort][profile][group][key]={ref:compact_native(p,prefix,key,winning[key],a,truth,known,cuts[key],ii,ref) for ref in (BASE,FROZEN)}
                            margins[cohort][profile][group][key]=summarize_margin(p[prefix+key+'/raw'],cuts[key],truth,known,winning[key][1],ii)
                    if cohort=='mz61' and profile=='DROP_CLOSE':
                        for key in winning:
                            ii=np.array([i for i,r in enumerate(records[cohort]) if r['role']=='HELDOUT_GEOMETRY' and r['family']=='shallow_awning' and truth[i,0] and known[i,0]])
                            assert len(ii)==64;ww,nn,kk=winning[key]
                            awning[key]=dict(positive_frames=64,counts=compact_native(p,prefix,key,winning[key],a,truth,known,cuts[key],ii,BASE),
                                rows=[dict(index=int(i),frame_id=str(frame_ids[i]),winner=int(ww[i,0]),native=bool(nn[i,0]),local_known=bool(kk[i,0]),raw=float(p[prefix+key+'/raw'][i,0]),cutoff=float(cuts[key][0]),margin=float(p[prefix+key+'/raw'][i,0])-float(cuts[key][0]),candidate_positive=bool(a[key+'/candidate'][i,0]>=0),union_positive=bool(a[key+'/UNION'][i,0]>=0),baseline_positive=bool(a[BASE][i,0]>=0),support_context=records[cohort][i]['support_context']) for i in ii])
                for method in NEW:
                    key=method.rsplit('/',1)[0];detail=None
                    if key in winning:detail=(*winning[key],p[prefix+key+'/raw'],cuts[key],p[prefix+key+'/support'],p[prefix+'MZ37'])
                    events[cohort][profile][method]={ref:event_rows(a,truth,known,frame_ids,method,ref,records.get(cohort),detail) for ref in COMPARATORS if ref!=method}
                if cohort in records:
                    pairs=defaultdict(list)
                    for i,r in enumerate(records[cohort]):pairs[r['pair_id']].append(i)
                    assert len(pairs)==(2048 if cohort=='mz61' else 1280) and all(len(v)==2 for v in pairs.values())
                    pi=np.array(list(pairs.values()));np.testing.assert_array_equal(truth[pi[:,0]],truth[pi[:,1]])
                    for k in (BASE,FROZEN)+NEW:
                        changed=((a[k][pi[:,0]]>=0)!=(a[k][pi[:,1]]>=0))
                        contexts[cohort][profile][k]=dict(pairs=len(pi),changed_pairs=int(changed.any(1).sum()),changed_queries=changed.sum(0).tolist(),pair_ids=[pair for pair,change in zip(pairs,changed) if change.any()])
        aggregates={};aggregate_changes={}
        for profile in PROFILES:
            data=cached[profile];legacy=[(c,np.arange(len(data[c][1]))) for c in ('relation10000','distance5000','rich','mz36')]
            sel=dict(legacy_noncal=legacy,mz48_fit=[('mz48',g48['fit'])],mz48_nonfit=[('mz48',g48['nonfit'])])
            sel['all_old_noncal']=legacy+sel['mz48_fit']+sel['mz48_nonfit'];aggregates[profile]={};aggregate_changes[profile]={}
            for group,selected in sel.items():
                t=np.concatenate([data[c][1][ii] for c,ii in selected]);k=np.concatenate([data[c][2][ii] for c,ii in selected])
                methods=tuple(previous['aggregates'][profile][group])+NEW
                aa={m:np.concatenate([data[c][0][m][ii] for c,ii in selected]) for m in methods}
                aggregates[profile][group]={m:metrics(v,t,k) for m,v in aa.items()}
                aggregate_changes[profile][group]={m:{ref:paired(aa[m],aa[ref],t,k) for ref in COMPARATORS if m!=ref} for m in NEW}
                for m,row in previous['aggregates'][profile][group].items():assert_metric_equal(aggregates[profile][group][m],row);reference_rows+=1
        held=g61['HELDOUT_GEOMETRY'];aa,tt,kk=cached['DROP_CLOSE']['mz61']
        native={k:attrs['mz61']['DROP_CLOSE']['HELDOUT_GEOMETRY'][k][BASE]['native_added_tp'] for k in KEYS+('MZ62/CONTROL',)}
        primary=gate(native,awning[KEYS[1]]['counts']['native_added_tp'][0],{k:v[held] for k,v in aa.items()},tt[held],kk[held],
            dict(legacy_noncal=aggregates['DROP_CLOSE']['legacy_noncal'],mz48_nonfit=aggregates['DROP_CLOSE']['mz48_nonfit'],mz55_held640=scores['mz55']['DROP_CLOSE']['HELDOUT_SITE']))
        result=dict(status='PASS',primary_gate=primary,queries=QUERIES,new_methods=NEW,conditions=scores,comparisons=changes,aggregates=aggregates,aggregate_comparisons=aggregate_changes,
            attribution=attrs,margins=margins,support_context_pairs=contexts,awning_held64_DROP={k:{x:v for x,v in row.items() if x!='rows'} for k,row in awning.items()},cutoffs={k:v.tolist() for k,v in cuts.items()},cutoff_authority=authorities,
            source_role='CONSUMED_DEVELOPMENT',source_collision_summary={m:{q:t['summary'] for q,t in tables.items()} for m,tables in collision['tables'].items()},
            limitation='Shared warm start already exposed to MZ55; matched total compute/replay, different sources and per-ID multiplicity. Native winner is not causal proof; dense argmax not independently recomputed. Geometry-label nonduplication is within MZ61, not independent of all historical data. No trained-ceiling or hardware claim.')
        audit=dict(status='PASS',baseline_arrays_exact=preserved,prior_metric_rows_exact=reference_rows,scalar_known_decisions=scalar_bits,native_winner_lookups=native_checks,
            schedule=schedule_audit,original_calibration_rows=1256,cutoff_vectors_recomputed=2,new_source_calibration_rows_used=0,mz36_UNKNOWN_bits=80,local_UNKNOWN_cells={c:int((v['valid_counts']==0).sum()) for c,v in labels.items()},new_fits=0,new_inference_frames=0,RGB_reads=0,native_depth_reads=0,checkpoint_loads=0,dense_argmax_recomputed=False)
        for path,h in inputs.items():assert sha(path)==h,path
        write(out/'result.json',result);write(out/'audit.json',audit);write(out/'paired-events.json',events);write(out/'awning-held64-drop.json',awning)
        report(out/'report.md',result)
        write(out/'receipt.json',dict(status='PASS',inputs=inputs,outputs={n:sha(out/n) for n in ('result.json','audit.json','paired-events.json','awning-held64-drop.json','report.md')},code_sha256=sha(__file__),seconds=time.perf_counter()-start,backend='FROZEN_PROTOCOL_CPU_ONLY saved-output audit',new_fits=0,new_inference_frames=0,new_cutoffs=0))
        print('PASS',primary)
    except BaseException:write(out/'failure.json',dict(status='FAIL',inputs=inputs,error=traceback.format_exc()));raise


def report(path,result):
    primary=result['primary_gate'];lines=['# MZ64 source geometry learning','',f'Primary retaining check: **{"PASS" if primary["passes_primary"] else "FAIL"}**.','','| Clause | Result |','| --- | --- |']
    lines += [f'| {k} | {"PASS" if v else "FAIL"} |' for k,v in primary['clauses'].items()]
    lines += ['','All19 clauses and all gains/costs retained; no winner selection.','','| DROP group | Method | TP | FP | FN | Exact |','| --- | --- | ---: | ---: | ---: | ---: |']
    for name,rows in [('MZ61 held1024',result['conditions']['mz61']['DROP_CLOSE']['HELDOUT_GEOMETRY']),('MZ55 held640',result['conditions']['mz55']['DROP_CLOSE']['HELDOUT_SITE']),('Legacy noncal',result['aggregates']['DROP_CLOSE']['legacy_noncal']),('MZ48 nonfit',result['aggregates']['DROP_CLOSE']['mz48_nonfit'])]:
        for key in (BASE,'MZ57/GLOBAL_ANCHOR/UNION',FROZEN)+NEW:
            row=rows[key];lines.append(f'| {name} | {key} | '+' | '.join(str(sum(row[k])) for k in ('tp','fp','fn'))+f' | {row["exact_frames"]} |')
    lines+=['',result['limitation'],'','Complete every-query/profile/group counts, UNKNOWN, native partitions, margins and equal-net-FP event IDs are retained in result.json and paired-events.json. Held awning64 rows are in awning-held64-drop.json.']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def self_test():
    import copy
    def rejects(fn):
        try:fn()
        except (AssertionError,ValueError):return
        raise AssertionError('Expected rejection')
    truth=np.zeros((3,4),bool);truth[0,0]=True;known=np.ones_like(truth)
    v=np.full((3,4),-1.);v[0,0]=1.;v[1,0]=1.
    a={k:v.copy() for k in (KEYS[0]+'/UNION',KEYS[1]+'/UNION',FROZEN)}
    native={KEYS[0]:[1,0,0,0],KEYS[1]:[2,0,0,0],'MZ62/CONTROL':[0,0,0,0]}
    regress={g:{k:metrics(x,truth,known) for k,x in a.items()} for g in ('legacy_noncal','mz48_nonfit','mz55_held640')}
    good=gate(native,1,a,truth,known,regress);assert good['passes_primary'] and len(good['clauses'])==19
    swap=copy.deepcopy(a);swap[KEYS[1]+'/UNION'][1,0]=-1;swap[KEYS[1]+'/UNION'][2,0]=1
    bad=gate(native,1,swap,truth,known,regress)
    assert not bad['clauses']['held_no_new_fp_bits_vs_matched_control']
    assert sum(metrics(swap[KEYS[1]+'/UNION'],truth,known)['fp'])==sum(metrics(v,truth,known)['fp'])
    assert not gate(native,0,a,truth,known,regress)['passes_primary']
    oldswap={g:{k:metrics(x,truth,known) for k,x in swap.items()} for g in regress}
    assert gate(native,1,a,truth,known,oldswap)['passes_primary']
    raw=np.arange(1256*4,dtype=np.float32).reshape(1256,4)/100
    cut,authority=scalar_cutoff(raw,np.ones_like(raw,bool),-np.ones_like(raw),np.zeros_like(raw,bool),np.ones_like(raw,bool))
    np.testing.assert_array_equal(cut,np.nextafter(raw[-1].astype(float),np.inf));assert all(r['maximum_rows']==[1255] for r in authority)
    raw[3,0]=1e6;unk=np.ones_like(raw,bool);unk[3,0]=False
    rejects(lambda:scalar_cutoff(raw,unk,-np.ones_like(raw),~unk,unk))
    row=metrics(v,truth,known);old=dict(row);old['attempted_frames']=old.pop('frames');assert_metric_equal(row,old)
    bad=dict(old,frames=99);rejects(lambda:assert_metric_equal(row,bad))
    bad=dict(old);bad['fp']=[0,0,0,0];rejects(lambda:assert_metric_equal(row,bad))
    prefix='x/';key=KEYS[1];t=np.ones((3,4),bool);kn=np.ones_like(t);kn[2,0]=False
    n=np.zeros_like(t);n[0,0]=True;wk=np.ones_like(t);wk[1,0]=False
    aa={BASE:-np.ones((3,4)),key+'/candidate':np.ones((3,4)),key+'/UNION':np.ones((3,4))}
    pp={prefix+key+'/raw':np.ones((3,4)),prefix+key+'/support':np.ones((3,4),bool)}
    attr=compact_native(pp,prefix,key,(np.zeros((3,4),int),n,wk),aa,t,kn,np.zeros(4),np.arange(3),BASE)
    assert attr['added_tp'][0]==2 and attr['native_added_tp'][0]==1 and attr['UNKNOWN_added_tp'][0]==1
    aa[BASE][0,0]=1
    attr=compact_native(pp,prefix,key,(np.zeros((3,4),int),n,wk),aa,t,kn,np.zeros(4),np.arange(3),BASE)
    assert attr['native_added_tp'][0]==0 # inherited OLD alarm is not new-head credit
    assert 'torch' not in __import__('sys').modules
    return dict(status='PASS',scope='Synthetic only',checks=['19 clauses','strict held FP identity swap','old aggregate FP semantics','awning required','1256 scalar cutoff','UNKNOWN rejection','legacy denominator aliases and conflicts','native versus UNKNOWN and inherited alarm credit'],model_runs=0)


def saved_compatibility(root):
    """Old arrays only; transient alias copies test schema, never new-model results."""
    work=root/'artifacts.local/work';rows=0;inputs={}
    oldgroups=read(work/'mz62-profile-coverage-20260911/run-v1/groups.json')
    meta=read(work/'mz61-geometry-source-20260911/combined-v2/metadata.json')
    r55=oldgroups['mz55_records'];r61=meta['records']
    g48={k:np.array(v,int) for k,v in oldgroups['groups'].items()};g48['nonfit']=np.sort(np.r_[g48['heldout_site'],g48['nonfit_family']])
    g55={r:np.array([i for i,x in enumerate(r55) if x['role']==r]) for r in ('TRAIN_CANDIDATE','CALIBRATION','HELDOUT_SITE')}
    g55.update(heldout_new_family=np.array([i for i in g55['HELDOUT_SITE'] if r55[i]['family']!='retained_rod']),nonfit=np.sort(np.r_[g55['CALIBRATION'],g55['HELDOUT_SITE']]))
    g61={r:np.array([i for i,x in enumerate(r61) if x['role']==r]) for r in ('TRAIN_CANDIDATE','CALIBRATION','HELDOUT_GEOMETRY')}
    gp={c:partitions(r,g) for c,r,g in [('mz48',oldgroups['records'],g48),('mz55',r55,g55),('mz61',r61,g61)]}
    path=work/'mz64-design-20260911/source-check-v1/schedule.npz'
    schedule=load(path);inputs[str(path)]=sha(path)
    oldschedule=load(work/'mz62-profile-coverage-20260911/run-v1/schedule.npz')
    schedule_check=validate_schedule(schedule,g48,g55,g61,r55,r61,oldschedule)
    for name,is63 in [('mz62-profile-coverage-20260911',False),('mz63-geometry-transfer-20260911',True)]:
        folder=work/name;r=read(folder/'run-v1/receipt.json');sr=read(folder/'score-v1/receipt.json')
        for path,h in [(folder/'run-v1/predictions.npz',r['outputs']['predictions.npz']),(folder/'score-v1/result.json',sr['outputs']['result.json'])]:
            assert sha(path)==h;inputs[str(path)]=h
        p=load(folder/'run-v1/predictions.npz');score=read(folder/'score-v1/result.json')
        for cohort in (('mz61',) if is63 else COHORTS[:-1]):
            prefix0='' if is63 else cohort+'/'
            for profile in (SOURCE_PROFILES if is63 else PROFILES):
                prefix=prefix0+profile+'/';previous=score['profiles'][profile]['all']['methods'] if is63 else score['conditions'][cohort][profile]['all']
                groups_before=score['profiles'][profile] if is63 else score['conditions'][cohort][profile]
                assert set(groups_before).issubset(gp.get(cohort,{'all':None})),set(groups_before)-set(gp.get(cohort,{'all':None}))
                for key in KEYS:
                    for suffix in ('candidate','UNION'):p[prefix+key+'/'+suffix]=p[prefix+'MZ62/CONTROL/'+suffix]
                a=reconstruct(p,prefix,tuple(previous))
                if is63:
                    source=work/'mz61-geometry-source-20260911';si=read(source/'source-index.json');ref=si['combined']['evaluator.npz'];path=source/ref['path']
                    assert sha(path)==ref['sha256'];inputs[str(path)]=ref['sha256'];ev=load(path);t,k=ev['truth'],ev['known']
                else:t,k=p[cohort+'/truth'],p[cohort+'/known']
                if cohort=='mz36':
                    t,k=p['mz36/attempted_truth'],p['mz36/attempted_known'];admitted=p['mz36/admitted_index']
                    for name2,values in list(a.items()):
                        a[name2]=np.full((400,4),np.nan);a[name2][admitted]=values
                for method,row in previous.items():assert_metric_equal(metrics(a[method],t,k),row);rows+=1
        del p
    return dict(status='PASS',scope='All old cohort/profile aggregate metrics and group keys; actual frozen source schedule; transient fixed-array aliases only, no new-model predictions',old_metric_rows_exact=rows,schedule=schedule_check,inputs=inputs,new_inference=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('E:/linnan/linnan'))
    parser.add_argument('--task',type=Path)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--saved-compatibility',action='store_true')
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    if args.self_test or args.saved_compatibility:
        assert args.task is None
        result=dict(code_sha256=sha(__file__))
        if args.self_test:result['synthetic']=self_test()
        if args.saved_compatibility:result['saved_compatibility']=saved_compatibility(args.root.resolve())
        if args.receipt:write(args.receipt,result)
        print(result)
    else:
        assert args.task is not None
        run(args.task if args.task.is_absolute() else args.root/args.task)
